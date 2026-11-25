# Project 1：用 NumPy 复现 ViT-B/32 图像编码器

用NumPy相关知识完成 `image_encoder.py` 里的六组共10个TODO，将每张图片编码成一个归一化的 512 维向量。
在 `image_encoder.py` 已经用*****框住了所有要求完成的内容，并附上了注释。

## 复现流程

1. 用仓库里的 `image_encoder.py` 覆盖同名文件。
2. 在该目录打开终端，依次运行：

```powershell
python check.py --stage all
python run.py --output output_tokens.npz
```

## 预期结果

- `check.py` 的 8 项检查均显示 `PASS`。本地实测最终输出的最大绝对误差为 `5.811e-7`。
- `run.py` 生成 `output_tokens.npz`，其中包含 10 个图片 ID 和形状为 `(10, 512)` 的特征向量数组。


### 配套练习

`numpy_homework.ipynb`为配套练习`project1_exercises.pdf`的完成内容。其原为.py文件，利用AI将其转为.ipynb文件，已人工核验核心内容未改变。
