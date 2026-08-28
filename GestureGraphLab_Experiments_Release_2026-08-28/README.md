# GestureGraph Lab 统一谱位置编码实验

## 1. 包含内容

本发布包只包含：

- 训练与聚合结果所需的 Python 源码；
- 三个随机种子（42、43、44）的 15 个最佳模型 `best.pt`；
- 每个模型的 `model.json`、`history.json` 和 `test_confusion.json`；
- 三种子汇总结果；
- 实验 3 的自适应邻接矩阵与动态图；
- 数据转换脚本和 Python 依赖文件。

不包含 SHREC'17 原始数据、虚拟环境、缓存、旧实验归档、界面与摄像头功能。

## 2. 统一空间编码

ST-GCN 控制组不加入空间编码。实验 1–4 的每个空间特征提取层均采用相同的谱位置编码。

对 22 节点物理手骨架构建归一化拉普拉斯矩阵：

```text
L = I - D^(-1/2) A D^(-1/2)
L SE_k = alpha_k SE_k
```

去掉特征值为 0 的常量特征向量，使用后续 8 个特征对。每个特征向量先乘对应的固定特征值，再通过当前层独立的线性层映射到输入通道数：

```text
weighted_SE[:, k] = alpha_k * SE[:, k]
PE_l = weighted_SE @ W_pe,l
Z_l = X_l + PE_l
```

`alpha` 和 `SE` 由物理骨架固定确定，不参与训练；`W_pe,l` 是可学习参数。各模型的空间算子接收 `Z_l`。

## 3. 模型与实验对应关系

| 目录 | `--model` 名称 | 实验定义 |
|---|---|---|
| `00_stgcn_control` | `stgcn` | 原始固定邻接 ST-GCN；不加入谱位置编码，作为控制组。 |
| `01_spectral_pe` | `spectral_pe_stgcn` | 每层先加入统一谱位置编码，再使用固定物理邻接图卷积。 |
| `02_qkv_spatial_attention` | `spectral_pe_qkv` | 每层先加入统一谱位置编码，再使用 4 头、22×22 全关节 Q/K/V 空间注意力；物理邻接仅作为可学习软偏置，不使用硬掩码；保留原 TCN，不加入时间位置编码。 |
| `03_gwnet_adaptive_support` | `gwnet_adaptive_support` | 每层先加入统一谱位置编码，再使用 Graph WaveNet 双 support：固定物理图与完整自适应图分别进行二阶扩散。 |
| `04_agcrn_factorized_adjacency` | `agcrn_factorized_adjacency` | 实验 3 基础上，将共享图卷积权重分解为 AGCRN 风格的节点特异权重。 |

## 4. 实验设置

| 项目 | 设置 |
|---|---|
| 数据集 | SHREC'17 14 类，DD-Net 中位数滤波版本 |
| 官方划分 | 1960 个训练序列、840 个测试序列 |
| 验证集 | 从官方训练集按类别分层划分 20% |
| 输入 | 64 帧、22 关节、3 坐标 |
| Epoch | 40 |
| Batch size | 32 |
| 优化器 | AdamW |
| 初始学习率 | 0.001 |
| 权重衰减 | 0.0001 |
| 学习率调度 | CosineAnnealingLR |
| 损失 | CrossEntropyLoss，label smoothing = 0.05 |
| 随机种子 | 42、43、44 |
| 谱维度 | 8 |
| Q/K/V 头数 | 4 |
| 自适应节点嵌入维度 | 10 |
| 训练设备 | NVIDIA GeForce RTX 3090 |
| 复现实验环境 | Python 3.13.13、PyTorch 2.11.0+cu130 |
| 模型选择 | 仅根据最佳验证集准确率保存 checkpoint；官方测试集不参与选择 |

## 5. Python 依赖

建议使用独立虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` 包含：

```text
numpy>=2.0,<3
torch>=2.5,<3
pillow>=11,<13
matplotlib>=3.9,<4
```

GPU 训练需要安装与本机 NVIDIA 驱动匹配的 PyTorch CUDA 版本；CPU 也可运行，但训练时间会明显增加。

## 6. 数据准备

发布包不包含数据集。训练命令默认数据目录包含：

```text
data/shrec17_ddnet_npz/
├── train.npz
├── test.npz
└── provenance.json
```

如已有 DD-Net 的 `data/SHREC/train.pkl` 和 `test.pkl`，运行：

```powershell
python scripts\convert_ddnet_shrec.py `
  --source <DD-Net目录>\data\SHREC `
  --output data\shrec17_ddnet_npz
```

## 7. 训练指令

### 7.1 三种子完整训练

```powershell
python -m gesturegraph.improvement_benchmark --dataset shrec17_npz --data data\shrec17_ddnet_npz --output runs\retrain_seed42 --include-control --epochs 40 --batch-size 32 --frames 64 --seed 42 --device cuda --pe-dim 8 --attention-heads 4 --adaptive-dim 10

python -m gesturegraph.improvement_benchmark --dataset shrec17_npz --data data\shrec17_ddnet_npz --output runs\retrain_seed43 --include-control --epochs 40 --batch-size 32 --frames 64 --seed 43 --device cuda --pe-dim 8 --attention-heads 4 --adaptive-dim 10

python -m gesturegraph.improvement_benchmark --dataset shrec17_npz --data data\shrec17_ddnet_npz --output runs\retrain_seed44 --include-control --epochs 40 --batch-size 32 --frames 64 --seed 44 --device cuda --pe-dim 8 --attention-heads 4 --adaptive-dim 10
```

### 7.2 聚合三种子结果

```powershell
python -m gesturegraph.aggregate_improvement_runs `
  --runs runs\retrain_seed42 runs\retrain_seed43 runs\retrain_seed44 `
  --output runs\retrain_aggregate
```

### 7.3 单独训练一个实验

将 `--only` 设置为下列目录名之一：

```text
01_spectral_pe
02_qkv_spatial_attention
03_gwnet_adaptive_support
04_agcrn_factorized_adjacency
```

示例：

```powershell
python -m gesturegraph.improvement_benchmark --dataset shrec17_npz --data data\shrec17_ddnet_npz --output runs\qkv_seed42 --epochs 40 --batch-size 32 --frames 64 --seed 42 --device cuda --pe-dim 8 --attention-heads 4 --adaptive-dim 10 --only 02_qkv_spatial_attention
```

## 8. 正式实验结果

三种子均值 ± 样本标准差：

| 实验 | 参数量 | 官方测试准确率 | 相对同种子 ST-GCN |
|---|---:|---:|---:|
| ST-GCN 控制组 | 208,978 | 69.33% ± 1.20% | +0.00 ± 0.00 pp |
| 统一谱编码 + 固定 ST-GCN | 210,538 | 68.77% ± 1.41% | -0.56 ± 2.54 pp |
| 统一谱编码 + Q/K/V | 268,286 | 68.37% ± 7.10% | -0.95 ± 8.26 pp |
| 统一谱编码 + Graph WaveNet | 282,703 | 72.38% ± 1.73% | +3.06 ± 2.59 pp |
| 统一谱编码 + AGCRN 分解 W | 930,178 | **80.56% ± 0.50%** | **+11.23 ± 1.13 pp** |

各随机种子的官方测试准确率：

| 实验 | Seed 42 | Seed 43 | Seed 44 |
|---|---:|---:|---:|
| ST-GCN 控制组 | 70.60% | 68.21% | 69.17% |
| 统一谱编码 + 固定 ST-GCN | 67.14% | 69.52% | 69.64% |
| 统一谱编码 + Q/K/V | 60.24% | 73.33% | 71.55% |
| 统一谱编码 + Graph WaveNet | 70.71% | 72.26% | 74.17% |
| 统一谱编码 + AGCRN 分解 W | **80.95%** | **80.71%** | **80.00%** |

Q/K/V 对初始化较敏感。Graph WaveNet 双 support 提供稳定提升；加入 AGCRN 节点特异权重后获得最高且最稳定的准确率。

## 9. 训练模型与结果位置

```text
runs/
├── improvement_backbones_ddnet/          # seed 42
├── improvement_backbones_ddnet_seed43/   # seed 43
├── improvement_backbones_ddnet_seed44/   # seed 44
└── improvement_backbones_ddnet_aggregate/
```

每个模型目录包含：

```text
best.pt                 最佳验证集 checkpoint
model.json              实验设置与正式测试准确率
history.json            40 epoch 训练历史
test_confusion.json     14 类官方测试混淆矩阵
```

实验 3 另外包含 `adjacency_matrices.npz`。Seed 42 目录还包含学习动态图的 PNG、SVG、PDF 和元数据。

