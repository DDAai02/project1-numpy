# Project 1：以 NumPy 复现 OpenCLIP 图像编码器

使用课程提供的固定视觉参数，完成 ViT-B-32 图像编码器的六组 NumPy 实现。
输入为 float32 `[B,3,224,224]`，输出为 float32 `[B,512]` 单位向量。
不训练模型，也不依赖 PyTorch 进行编码器计算。

## 文件

- `image_encoder.py`：完成的六组计算，保留课程公共接口。
- `numpy_homework.py`：NP-E1..E8、NP-M1..M4 的实现、运行入口和必要边界检查。
- `results/`：逐阶段数值误差、额外检查和练习的真实运行结果。
- `tools/verify_extra.py`：使用新输入检查切块顺序、batch 一致性和注意力兼容性。
- `requirements.txt`、`environment.yml`：本项目的精简复现依赖。

教师要求不重复提交大体积参数和参考数据，因此运行需要课程提供的原始附件。

## 环境

`environment.yml` 定义了独立的 `project1-numpy` Conda 环境。

```bash
conda env create -f environment.yml
conda activate project1-numpy
```

环境已存在时直接激活。也可以在 Python 3.12 环境中安装：

```bash
python -m pip install -r requirements.txt
```

已验证版本：Python 3.12.12、NumPy 1.26.4、Pillow 12.1.1、imageio 2.37.2。
Pillow 使用 libjpeg-turbo。原 Anaconda base 的 JPEG 解码环境曾导致预处理与参考不一致，
因此应先运行预处理检查，且不要通过修改检查器或容差消除差异。

## 运行与验证

1. 解压教师提供的 `project1-numpy-student` 附件。
2. 将本仓库的 `image_encoder.py` 覆盖到附件目录，并将 `numpy_homework.py` 放入同一目录。
3. 保留教师提供的 `helpers.py`、检查器、预处理、参数、图片及参考数据。
4. 在附件目录运行：

```bash
python check.py --stage all
python run.py --output output_tokens.npz
python numpy_homework.py
```

处理自己的图片：

```bash
python run.py --images path/to/a.jpg path/to/b.png --output my_tokens.npz
```

从本仓库目录运行额外验证，把 `TEACHER_STARTER` 替换为实际附件目录：

```bash
python tools/verify_extra.py --starter "TEACHER_STARTER" --output extra_checks.json
```

## 验证结果

官方 assets、preprocess、patches、tokens、block0、last_tokens、output、batch 全部通过。
详细最大/平均绝对误差见 `results/check_all.txt` 和 `results/stage_errors.json`。
最终十张图片输出的最大绝对误差为 `5.811e-7`，平均绝对误差为 `8.099e-8`。
额外测试验证了新随机图片、单张与批量一致性、非参考 PNG、patch 顺序、通用头数与 causal 开关。
数值结果按各阶段的 `atol + rtol * abs(reference)` 判断，不要求逐位相等。

## 实现约定

- 每个 patch 保持 C/H/W 展平顺序，网格按行排列。
- 普通线性权重使用 `weight.T`；`visual.proj` 已经是 `[768,512]`，直接相乘。
- 多头注意力只交换 K 的最后两轴，合并时先恢复 token/head 轴顺序。
- 保留两次残差和提供的精确 GELU；图片不启用 causal mask。
- 编码器不读取参考输出；参考数据只供外部检查器使用。

## 来源与协助

框架、辅助函数、图片、参数和参考输出由课程提供；本次补全六组计算并完成配套练习。
使用 Codex 辅助实现、运行验证和定位环境差异。
模型结构参考 [OpenCLIP](https://github.com/mlfoundations/open_clip)。
