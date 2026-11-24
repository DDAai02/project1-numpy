"""期末项目的图像编码器。

使用 NumPy 完成 TODO-01 到 TODO-06，模型参数由课程提供。
公开类沿用期末项目的接口，辅助函数来自课程提供的 helpers.py。
"""
import numpy as np
from helpers import LayerNorm, GELU, attention_weights, causal_mask, unit_vector


class MultiHeadAttention:
    def __init__(self, weights, prefix, heads):
        self.qkv_weight = weights[prefix + '.in_proj_weight']  # 合并的 Q、K、V 投影权重，形状为 [3D,D]
        self.qkv_bias = weights[prefix + '.in_proj_bias']      # 合并的投影偏置，形状为 [3D]
        self.out_weight = weights[prefix + '.out_proj.weight']  # 输出投影权重，形状为 [D,D]
        self.out_bias = weights[prefix + '.out_proj.bias']
        self.heads = heads
        self.width = self.out_bias.shape[0]
        self.head_dim = self.width // heads

    def split_heads(self, x):
        """TODO-02：将 [B,N,D] 转成 [B,H,N,D/H]。

        B 是批大小，N 是词元数，D 是特征维度，H 是注意力头数。
        图像模型中 D=768、H=12，每头 64 维。分头时保留词元顺序。
        """
        batch, tokens, width = x.shape
        # 先拆分每个词元的特征轴，再把每个头的所有词元排在一起。
        return x.reshape(batch, tokens, self.heads, self.head_dim).transpose(0, 2, 1, 3)

    def __call__(self, x, causal=False):
        # 线性层权重按 [输出维度,输入维度] 存储，相乘前先转置。参数保持固定。
        qkv = x @ self.qkv_weight.T + self.qkv_bias
        q = self.split_heads(qkv[:, :, :self.width])
        k = self.split_heads(qkv[:, :, self.width:2*self.width])
        v = self.split_heads(qkv[:, :, 2*self.width:])

        # TODO-03a：对查询和键做点积，再除以每头维度的平方根。
        # q、k、v 的形状都是 [B,H,N,64]，分数的形状为 [B,H,N,N]。
        # 只交换 k 最后的特征轴和词元轴，保留批次轴和头轴。
        scores = (q @ k.swapaxes(-1, -2)) / np.sqrt(np.float32(self.head_dim))

        # 因果掩码用于兼容后续的文本编码器，图像编码时不启用。
        if causal:
            scores = causal_mask(scores)
        coefficients = attention_weights(scores)  # 将每个查询的分数转换为注意力权重。

        # TODO-03b：按注意力权重对 v 加权求和，再合并各头的输出。
        # context 的形状为 [B,H,N,64]，merged 的形状为 [B,N,D]。
        context = coefficients @ v
        # 先换回分头前的轴顺序，再把同一词元的各头特征接在一起。
        merged = context.transpose(0, 2, 1, 3).reshape(x.shape)

        # TODO-03c：将合并结果乘以转置后的输出权重并加偏置，返回 [B,N,D]。
        return merged @ self.out_weight.T + self.out_bias


class TransformerBlock:
    def __init__(self, weights, prefix, heads):
        self.norm1 = LayerNorm(weights[prefix+'.ln_1.weight'], weights[prefix+'.ln_1.bias'])
        self.norm2 = LayerNorm(weights[prefix+'.ln_2.weight'], weights[prefix+'.ln_2.bias'])
        self.attention = MultiHeadAttention(weights, prefix+'.attn', heads)
        self.fc1_weight = weights[prefix+'.mlp.c_fc.weight']  # 升维权重，形状为 [4D,D]
        self.fc1_bias = weights[prefix+'.mlp.c_fc.bias']
        self.fc2_weight = weights[prefix+'.mlp.c_proj.weight']  # 降维权重，形状为 [D,4D]
        self.fc2_bias = weights[prefix+'.mlp.c_proj.bias']
        self.gelu = GELU()

    def __call__(self, x, causal=False):
        # 先对每个词元做层归一化，再计算注意力更新。
        update = self.attention(self.norm1(x), causal=causal)

        # TODO-04a：第一次残差相加，把注意力更新加回原输入。
        residual = x + update

        # 对第一次残差相加的结果做第二次层归一化。
        u = self.norm2(residual)

        # TODO-04b：先升维，用课程提供的精确 GELU 激活，再降回原维度。
        # hidden 的形状为 [B,N,3072]，mlp_update 的形状为 [B,N,768]。
        # 第二次残差相加，将 MLP 更新加回 residual，形状保持为 [B,N,D]。
        hidden = u @ self.fc1_weight.T + self.fc1_bias
        mlp_update = self.gelu(hidden) @ self.fc2_weight.T + self.fc2_bias
        return residual + mlp_update


class ImageEncoder:
    def __init__(self, weights):
        self.patch_weight = weights['visual.conv1.weight'].reshape(768, 3072)
        self.cls = weights['visual.class_embedding'].reshape(1, 1, 768)
        self.position = weights['visual.positional_embedding']  # 位置编码，形状为 [50,768]
        self.pre_norm = LayerNorm(weights['visual.ln_pre.weight'], weights['visual.ln_pre.bias'])
        self.blocks = [TransformerBlock(weights, 'visual.transformer.resblocks.' + str(i), 12)
                       for i in range(12)]
        self.post_norm = LayerNorm(weights['visual.ln_post.weight'], weights['visual.ln_post.bias'])
        self.projection = weights['visual.proj']  # 形状为 [768,512]，已按 [输入维度,输出维度] 存储。

    def patchify(self, images):
        """TODO-01：将 [B,3,224,224] 切分为 [B,49,3072]。

        图块按从左到右、从上到下的顺序排列。每个 32×32 的 RGB 图块
        按通道、高度、宽度的顺序展平，共有 3×32×32 个值。
        输入图像已经过预处理，切块只需重排像素。
        """
        # 将图块在网格中的行列与块内的行列拆开，再交换轴并展平。
        batch = images.shape[0]
        # 把网格行列移到通道轴之前，便于连续展平同一个图块。
        grid = images.reshape(batch, 3, 7, 32, 7, 32)
        return grid.transpose(0, 2, 4, 1, 3, 5).reshape(batch, 49, 3072)

    def embed(self, images):
        patches = self.patchify(images)  # 图块数组的形状为 [B,49,3072]。

        # TODO-05a：将图块投影到 [B,49,768]，这一层不加偏置。
        patch_tokens = patches @ self.patch_weight.T

        # 为批次中的每张图复制一个课程提供的 CLS 向量。
        cls = np.tile(self.cls, (images.shape[0], 1, 1))

        # TODO-05b：沿词元轴把 CLS 拼在开头，再加上学到的位置编码。
        # 位置编码沿批次轴广播，结果的形状为 [B,50,768]。
        return np.concatenate((cls, patch_tokens), axis=1) + self.position

    def encode_tokens(self, images):
        """对输入词元做层归一化，再依次通过 12 个 Transformer 块，得到 [B,50,768]。"""
        x = self.pre_norm(self.embed(images))
        for block in self.blocks:
            x = block(x)
        return x

    def __call__(self, images):
        """TODO-06：为每张图像返回一个 512 维单位向量，形状为 [B,512]。"""
        tokens = self.encode_tokens(images)

        # 取每张图的 CLS，做最后一次层归一化，再投影到 512 维。
        # 形状依次为 [B,50,768]、[B,768]、[B,512]；投影矩阵直接相乘。
        # 最后用课程提供的函数把每张图的向量缩放到单位长度。
        cls = tokens[:, 0, :]
        output = self.post_norm(cls) @ self.projection

        return unit_vector(output)
