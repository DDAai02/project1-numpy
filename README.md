# Project 1：用 NumPy 复现 OpenCLIP 图像编码器

用课程提供的固定权重运行 ViT-B-32 图像编码器。每张图片输出一个 512 维单位向量。

## 复现指引（Windows）

**1. 准备好两个文件夹。** 解压本仓库和教师发的 `project1-学生版.zip`。找到仓库中含 `environment.yml` 的目录，以及教师附件中含 `check.py`、`run.py`、`weights/` 和 `assets/` 的目录。因为完整检查需要教师附件里的权重和参考数据。

把下列终端指令前两行中的示例路径替换成实际路径，并运行命令。首次运行需创建环境。

```powershell
$codeDir = 'D:\course\project1-numpy'
$starterDir = 'D:\course\project1-numpy-student'
Set-Location -LiteralPath $codeDir
conda env create -f environment.yml
conda activate project1-numpy
```

**2. 复制一下实现。** 在同一个窗口运行下面两行。第一行用仓库里的 `image_encoder.py` 覆盖教师附件中的待完成版本，并把 `numpy_homework.py` 放到旁边。第二行切换到教师附件目录，后续命令都在这里运行。

```powershell
Copy-Item -LiteralPath "$codeDir\image_encoder.py", "$codeDir\numpy_homework.py" -Destination $starterDir -Force
Set-Location -LiteralPath $starterDir
```

**3. 检查并生成结果。**

```powershell
python check.py --stage all
python run.py --output output_tokens.npz
python numpy_homework.py
```

| 命令 | 成功时应看到 |
| --- | --- |
| `check.py` | 8 项检查均显示 `PASS`。十张图像最终输出的实测最大绝对误差为 `5.811e-7`。 |
| `run.py` | 显示 `Saved (10, 512) to output_tokens.npz`，文件生成在教师附件目录。 |
| `numpy_homework.py` | 输出 12 道题的结果，末行显示 `Required boundary checks: PASS`。 |

可对照[完整检查日志](results/check_all.txt)和[练习输出](results/numpy_homework_results.txt)。如果 `preprocess` 失败，先确认已激活上面的环境，然后运行 `python check.py --stage preprocess` 单独定位问题。
