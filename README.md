# Project 1：用 NumPy 复现 ViT-B/32 图像编码器

用课程提供的预训练权重完成六组 TODO，将每张图片编码成一个归一化的 512 维向量。

## 如何复现

1. 下载并解压课程平台提供的 `project1-学生版.zip`。
2. 将仓库里的 `image_encoder.py` 复制到解压后含有 `check.py` 的目录，覆盖同名文件。
3. 在该目录打开终端，使用 Python 3.12 环境，依次运行：

```powershell
python -m pip install -r requirements.txt
python check.py --stage all
python run.py --output output_tokens.npz
```

这里的 `requirements.txt`、`check.py` 和 `run.py` 都由教师附件提供。复现时使用附件中的图片、权重和参考数据。

## 预期结果

- `check.py` 的 8 项检查均显示 `PASS`。本地实测最终输出的最大绝对误差为 `5.811e-7`。
- `run.py` 生成 `output_tokens.npz`，其中包含 10 个图片 ID 和形状为 `(10, 512)` 的特征向量数组。

若 `preprocess` 检查失败，可在独立的 Python 3.12 环境中安装已验证的依赖，再重跑检查：

```powershell
python -m pip install numpy==1.26.4 Pillow==12.1.1 imageio==2.37.2
python check.py --stage all
```

框架、权重和参考数据来自课程附件。使用 Codex 辅助实现和验证。
