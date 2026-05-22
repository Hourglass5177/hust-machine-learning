# 机器学习导论大作业：CIFAR-10 图像分类

本项目为《机器学习导论》课程大作业，选题为 **CIFAR-10 图像分类**。项目使用 PyTorch 实现了多层感知机基线模型和卷积神经网络模型，完成了图像数据处理、神经网络结构设计、模型训练与调参、分类结果评估等完整流程。

---

## 一、项目简介

本项目针对 CIFAR-10 十分类任务进行建模实验。CIFAR-10 数据集包含十类常见物体图像，每张图像为三通道彩色图像，尺寸为 32×32。项目目标是训练分类模型，使模型能够根据输入图像判断其所属类别。

本项目主要完成以下内容：

1. 自动下载并加载 CIFAR-10 数据集。
2. 实现训练集数据增强与测试集标准化处理。
3. 实现多层感知机基线模型。
4. 实现卷积神经网络主模型。
5. 对比多层感知机与卷积神经网络的分类性能。
6. 完成数据增强消融实验。
7. 完成学习率数值调参实验。
8. 使用准确率、分类报告、混淆矩阵和错误样本可视化进行分类结果评估。

---

## 二、运行环境

本项目实验环境如下：

```text
操作系统：WSL2 Ubuntu
图形界面：无 Ubuntu 图形系统，可使用 Windows 图形界面查看结果图像
显卡：NVIDIA GeForce RTX 4060 Laptop，8GB 显存
环境管理：Miniconda
环境名称：ml-cifar10
主要框架：PyTorch、TorchVision
```

主要依赖包括：

```text
torch
torchvision
numpy
pandas
matplotlib
scikit-learn
scipy
pillow
tqdm
seaborn
jupyterlab
ipykernel
tensorboard
torchinfo
thop
opencv-python
albumentations
rich
pyyaml
easydict
```

---

## 三、环境配置

### 3.1 创建环境

```bash
conda create -n ml-cifar10 python=3.11 pip -y
conda activate ml-cifar10
```

### 3.2 安装基础科学计算库

```bash
conda install -c conda-forge \
  numpy \
  pandas \
  matplotlib \
  scikit-learn \
  scipy \
  pillow \
  tqdm \
  seaborn \
  jupyterlab \
  ipykernel \
  tensorboard \
  -y
```

### 3.3 安装深度学习框架

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### 3.4 安装辅助工具库

```bash
pip install \
  torchinfo \
  thop \
  opencv-python \
  albumentations \
  rich \
  pyyaml \
  easydict
```

### 3.5 检查显卡是否可用

```bash
python - <<'PY'
import torch
import torchvision

print("PyTorch 版本：", torch.__version__)
print("TorchVision 版本：", torchvision.__version__)
print("是否可用 CUDA：", torch.cuda.is_available())

if torch.cuda.is_available():
    print("PyTorch 使用的 CUDA 版本：", torch.version.cuda)
    print("显卡名称：", torch.cuda.get_device_name(0))
PY
```

---

## 四、项目目录结构

```text
hust-machine-learning/
├── configs/
│   └── cifar10.yaml
├── models/
│   ├── __init__.py
│   ├── mlp.py
│   └── simple_cnn.py
├── utils/
│   ├── __init__.py
│   ├── data.py
│   ├── metrics.py
│   ├── seed.py
│   └── trainer.py
├── scripts/
│   ├── check_data.py
│   ├── train_mlp.py
│   ├── train_cnn.py
│   ├── evaluate_model.py
│   ├── compare_models.py
│   ├── summarize_experiments.py
│   └── summarize_lr_tuning.py
├── outputs/
│   ├── logs/
│   ├── results/
│   └── figures/
├── environment.yml
├── requirements.txt
├── .gitignore
└── README.md
```

目录说明：

```text
configs/：保存实验配置文件。
models/：保存模型结构定义。
utils/：保存数据加载、随机种子、指标计算和训练器代码。
scripts/：保存训练、评估、实验汇总脚本。
outputs/logs/：保存训练过程日志。
outputs/results/：保存实验结果表格。
outputs/figures/：保存样本图、训练曲线、混淆矩阵和错误样本图。
```

模型权重文件位于 `outputs/checkpoints/`，该目录默认不上传到仓库，可通过训练脚本重新生成。

---

## 五、数据集说明

本项目使用 CIFAR-10 数据集。该数据集包含十个类别：

```text
airplane
automobile
bird
cat
deer
dog
frog
horse
ship
truck
```

数据集基本信息：

```text
类别数量：10
图像尺寸：32×32
图像通道：三通道彩色图像
训练集数量：50000
测试集数量：10000
总图像数量：60000
```

本项目通过 `torchvision.datasets.CIFAR10` 自动下载并加载数据集。数据集不上传至仓库，运行数据加载脚本时会自动下载到本地数据目录。

---

## 六、图像数据处理

### 6.1 训练集处理

训练集使用随机裁剪、随机水平翻转、张量转换和归一化：

```python
transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])
```

处理目的如下：

```text
随机裁剪：增强模型对局部平移的鲁棒性。
随机水平翻转：扩充训练样本变化，提高模型泛化能力。
张量转换：将图像转换为 PyTorch 可处理的张量。
归一化：使不同通道的像素分布更加稳定，有利于模型训练。
```

### 6.2 测试集处理

测试集不使用随机增强，只进行张量转换和归一化：

```python
transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])
```

这样可以保证测试过程稳定、公平，避免随机增强影响测试结果。

### 6.3 检查数据加载流程

```bash
python scripts/check_data.py
```

该脚本会检查数据集加载、批量数据形状、类别名称，并保存样本图：

```text
outputs/figures/cifar10_samples.png
```

---

## 七、模型设计

### 7.1 多层感知机基线模型

多层感知机模型将输入图像从 `3×32×32` 展平成 3072 维向量，然后通过多层全连接网络进行分类。

结构如下：

```text
输入图像：[批量大小, 3, 32, 32]
展平向量：[批量大小, 3072]
全连接层
批归一化
激活函数
随机失活
全连接层
批归一化
激活函数
随机失活
全连接层
批归一化
激活函数
随机失活
输出层：[批量大小, 10]
```

该模型作为基线模型，用于说明直接展平图像会破坏图像的二维空间结构，因此难以充分利用局部纹理、边缘和形状信息。

### 7.2 卷积神经网络主模型

卷积神经网络使用卷积层提取局部空间特征，使用池化层降低空间分辨率，并结合批归一化和随机失活提高训练稳定性与泛化能力。

结构如下：

```text
输入图像：[批量大小, 3, 32, 32]

卷积块一：
卷积层
批归一化
激活函数
卷积层
批归一化
激活函数
最大池化
随机失活

卷积块二：
卷积层
批归一化
激活函数
卷积层
批归一化
激活函数
最大池化
随机失活

卷积块三：
卷积层
批归一化
激活函数
卷积层
批归一化
激活函数
最大池化
随机失活

分类器：
展平
全连接层
批归一化
激活函数
随机失活
输出层
```

卷积神经网络相比多层感知机更适合图像分类任务，因为卷积操作能够保留并利用图像局部空间结构。

---

## 八、训练方法

### 8.1 训练多层感知机基线模型

```bash
python scripts/train_mlp.py \
  --epochs 20 \
  --batch-size 128 \
  --lr 0.001 \
  --no-aug
```

输出文件：

```text
outputs/logs/mlp_history.csv
outputs/checkpoints/mlp_best.pth
outputs/figures/mlp_loss_curve.png
outputs/figures/mlp_accuracy_curve.png
```

### 8.2 训练卷积神经网络主模型

```bash
python scripts/train_cnn.py \
  --epochs 50 \
  --batch-size 128 \
  --lr 0.1 \
  --run-name cnn_main
```

输出文件：

```text
outputs/logs/cnn_main_history.csv
outputs/checkpoints/cnn_main_best.pth
outputs/figures/cnn_main_loss_curve.png
outputs/figures/cnn_main_accuracy_curve.png
```

### 8.3 训练无数据增强卷积神经网络

```bash
python scripts/train_cnn.py \
  --epochs 50 \
  --batch-size 128 \
  --lr 0.1 \
  --no-aug \
  --run-name cnn_no_aug
```

输出文件：

```text
outputs/logs/cnn_no_aug_history.csv
outputs/checkpoints/cnn_no_aug_best.pth
outputs/figures/cnn_no_aug_loss_curve.png
outputs/figures/cnn_no_aug_accuracy_curve.png
```

### 8.4 学习率调参实验

学习率调参实验固定模型结构、数据增强策略、优化器、批量大小和随机种子，只改变初始学习率。

```bash
python scripts/train_cnn.py \
  --epochs 30 \
  --batch-size 128 \
  --lr 0.01 \
  --run-name cnn_lr001_30ep
```

```bash
python scripts/train_cnn.py \
  --epochs 30 \
  --batch-size 128 \
  --lr 0.05 \
  --run-name cnn_lr005_30ep
```

```bash
python scripts/train_cnn.py \
  --epochs 30 \
  --batch-size 128 \
  --lr 0.1 \
  --run-name cnn_lr01_30ep
```

---

## 九、评估方法

### 9.1 评估卷积神经网络

```bash
python scripts/evaluate_model.py \
  --model cnn \
  --checkpoint outputs/checkpoints/cnn_main_best.pth \
  --batch-size 128
```

如果本地没有模型权重文件，需要先运行训练命令生成权重文件。

### 9.2 评估多层感知机

```bash
python scripts/evaluate_model.py \
  --model mlp \
  --checkpoint outputs/checkpoints/mlp_best.pth \
  --batch-size 128
```

### 9.3 汇总模型对比结果

```bash
python scripts/compare_models.py
```

### 9.4 汇总全部实验结果

```bash
python scripts/summarize_experiments.py
```

### 9.5 汇总学习率调参结果

```bash
python scripts/summarize_lr_tuning.py
```

---

## 十、实验结果

### 10.1 主实验结果

| 实验名称 | 实验说明 | 最佳测试准确率 |
|---|---|---:|
| 多层感知机基线模型 | 使用多层感知机，不使用数据增强 | 54.52% |
| 卷积神经网络主模型 | 使用卷积神经网络，使用数据增强，初始学习率为 0.1，训练 50 轮 | 92.78% |
| 无数据增强卷积神经网络 | 使用卷积神经网络，不使用数据增强，初始学习率为 0.1，训练 50 轮 | 90.58% |

### 10.2 学习率调参结果

| 实验名称 | 初始学习率 | 训练轮数 | 最佳测试准确率 |
|---|---:|---:|---:|
| 卷积神经网络，学习率 0.01 | 0.01 | 30 | 89.68% |
| 卷积神经网络，学习率 0.05 | 0.05 | 30 | 91.67% |
| 卷积神经网络，学习率 0.10 | 0.10 | 30 | 91.73% |

### 10.3 多层感知机与卷积神经网络对比

| 模型 | 最佳测试准确率 |
|---|---:|
| 多层感知机 | 54.52% |
| 卷积神经网络 | 92.78% |

卷积神经网络相比多层感知机：

```text
绝对提升：38.26 个百分点
相对提升：约 70.18%
```

实验结果表明，卷积神经网络能够更有效地利用图像局部空间结构，因此在 CIFAR-10 图像分类任务上显著优于多层感知机。

---

## 十一、结果文件

重要结果文件如下：

```text
outputs/results/model_comparison.csv
outputs/results/mlp_vs_cnn_improvement.csv
outputs/results/experiment_summary.csv
outputs/results/lr_tuning_summary.csv
outputs/results/cnn_classification_report.csv
outputs/results/cnn_confusion_matrix.csv
outputs/results/cnn_top_confusions.csv
outputs/results/cnn_eval_summary.csv
```

重要图像文件如下：

```text
outputs/figures/cifar10_samples.png
outputs/figures/mlp_accuracy_curve.png
outputs/figures/mlp_loss_curve.png
outputs/figures/cnn_accuracy_curve.png
outputs/figures/cnn_loss_curve.png
outputs/figures/cnn_confusion_matrix.png
outputs/figures/cnn_confusion_matrix_normalized.png
outputs/figures/cnn_error_samples.png
outputs/figures/model_test_accuracy_comparison.png
outputs/figures/model_best_accuracy_bar.png
outputs/figures/experiment_test_accuracy_comparison.png
outputs/figures/experiment_best_accuracy_bar.png
outputs/figures/lr_tuning_accuracy_curves.png
outputs/figures/lr_tuning_best_accuracy_bar.png
```

---

## 十二、实验分析

### 12.1 图像数据处理分析

训练集使用随机裁剪和随机水平翻转进行数据增强，能够增加训练样本的变化形式，提高模型对图像平移和水平翻转的鲁棒性。测试集不使用随机增强，只进行张量转换和归一化，以保证测试结果稳定。

### 12.2 模型结构分析

多层感知机直接将图像展平成一维向量进行分类，破坏了图像的二维空间结构，因此难以有效捕捉局部纹理和形状信息。卷积神经网络通过卷积核提取局部特征，并通过池化操作降低空间分辨率，在图像分类任务中具有明显优势。

### 12.3 数据增强分析

无数据增强的卷积神经网络最佳测试准确率为 90.58%，使用数据增强的卷积神经网络最佳测试准确率为 92.78%。说明数据增强能够提升模型泛化能力，并缓解过拟合现象。

### 12.4 学习率调参分析

在 30 轮训练条件下，初始学习率为 0.10 的模型最佳测试准确率为 91.73%，略高于学习率为 0.05 的 91.67%，明显高于学习率为 0.01 的 89.68%。说明较小学习率虽然训练较稳定，但收敛速度较慢；较大的初始学习率配合余弦退火策略能够取得更好的训练效果。

### 12.5 分类结果评估分析

项目使用准确率、精确率、召回率、调和平均值、混淆矩阵和错误样本可视化对分类结果进行评估。混淆矩阵可以显示模型容易混淆的类别，错误样本可视化可以进一步分析模型在具体图像上的失败原因。

---

## 十三、复现实验

### 13.1 激活环境

```bash
conda activate ml-cifar10
```

### 13.2 进入项目目录

```bash
cd ~/ML-work/ml-cifar10-project
```

### 13.3 检查数据

```bash
python scripts/check_data.py
```

### 13.4 训练多层感知机

```bash
python scripts/train_mlp.py \
  --epochs 20 \
  --batch-size 128 \
  --lr 0.001 \
  --no-aug
```

### 13.5 训练主卷积神经网络

```bash
python scripts/train_cnn.py \
  --epochs 50 \
  --batch-size 128 \
  --lr 0.1 \
  --run-name cnn_main
```

### 13.6 评估主卷积神经网络

```bash
python scripts/evaluate_model.py \
  --model cnn \
  --checkpoint outputs/checkpoints/cnn_main_best.pth \
  --batch-size 128
```

### 13.7 汇总实验结果

```bash
python scripts/summarize_experiments.py
python scripts/summarize_lr_tuning.py
```

---

## 十四、注意事项

1. 数据集文件不上传到仓库，运行脚本时会自动下载。
2. 模型权重文件不上传到仓库，可通过训练脚本重新生成。
3. 输出目录中的日志、结果表格和图像用于支撑实验报告。
4. 如果在 WSL2 中运行，可以通过以下命令在 Windows 文件资源管理器中查看图像结果：

```bash
explorer.exe outputs/figures
```

---

## 十五、结论

本项目完成了 CIFAR-10 图像分类任务的完整机器学习流程。实验结果表明，卷积神经网络显著优于多层感知机，说明卷积结构能够更有效地利用图像的局部空间信息。数据增强实验表明，随机裁剪和随机水平翻转能够提升模型泛化能力。学习率调参实验表明，合适的初始学习率配合学习率调度策略能够进一步提升训练效果。整体来看，本项目覆盖了图像数据处理、神经网络结构设计、模型训练与调参、分类结果评估等课程考察点。
