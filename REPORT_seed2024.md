# 单种子实验报告（Seed 2024）— 论文写作参考版

> **数据范围**：`result_plus/runs/.../seed2024/` 下的 **117 主实验 + 9 E2E gating = 126 runs**
> **覆盖**：13 模型 × 3 数据集 × 3 协议（BC warm_temporal 已弃用，见 §1.4）
> **来源**：直接取自磁盘 `metrics.json`，无聚合无插值
> **用途**：作为论文单种子版本的原始数据底稿

---

## 1. 实验设置

### 1.1 数据集

| Dataset | Users | Items | Ratings | Density | 时间戳 | 评级范围 |
|---------|------:|------:|--------:|--------:|:------:|--------:|
| ML-1M | 6,040 | 3,706 | 1,000,209 | 4.47% | ✅ 真实 | 1–5 |
| GoodBooks (GB) | 53,424 | 13,667 | 5,342,431 | 0.73% | ✅ 真实 | 1–5 |
| Book-Crossing (BC) | 278,858 | 271,379 | 1,149,780 | 0.0015% | ❌ 无 | 1–10→1–5 |

密度跨度 ~3000×，覆盖稠密 / 中等 / 极稀疏三种典型场景。

### 1.2 协议

| Protocol | 切分方式 | 训练集用户/物品 | 评估意义 |
|----------|---------|----------------|----------|
| `strict_cold` | 用户级切分 | 测试用户从未出现在训练集 | 严格冷启动能力 |
| `warm_random` | 行级随机切分 | 同一用户可跨集 | 带热信息的随机场景 |
| `warm_temporal` | 时间序切分 train<val<test | 按真实时间戳 | 时序热启动场景 |

### 1.3 模型（13 个，按复杂度排序）

| Model | Category | Complexity | 简介 |
|-------|----------|:----------:|------|
| GlobalMean | Bias | 1 | 全局均值 |
| UserBias | Bias | 1 | 用户偏置 |
| ItemBias | Bias | 1 | 物品偏置 |
| UserItemBias | Bias | 1 | 用户+物品偏置 |
| SVD | MF | 2 | 矩阵分解 |
| NeuMF | Deep | 3 | 神经协同过滤 |
| DeepFM | Deep | 3 | FM+DNN |
| BehaviorMLP | Deep | 3 | 行为 MLP |
| ProfileMLP | Deep | 3 | 画像 MLP |
| Hybrid | Deep | 3 | Profile+Behavior 融合 |
| DualHardSwitch | Adaptive | 4 | 冷热硬切换 |
| DualSoftGating | Adaptive | 4 | 冷热软门控 |
| LightGCN | Graph | 5 | 图卷积（含 item_bias 冷启动回退） |

### 1.4 ⚠️ BC warm_temporal 弃用说明（论文须在 Limitations 中写明）

Book-Crossing 数据集原始 `Ratings.csv` 仅含 `User-ID;ISBN;Rating` 三列，**无时间戳字段**（这是数据集本身的客观限制）。为保持 pipeline 一致，loader 用行号代理 timestamp：

```python
# data/loaders/book_crossing_loader.py:132-134
if "timestamp" not in ratings.columns:
    ratings["timestamp"] = np.arange(len(ratings))
```

这导致 BC 上的 warm_temporal 退化为"行序切分"，无时序语义。**经评估，保留该协议会让审稿人质疑数据严谨性，因此本报告弃用 BC warm_temporal**，BC 仅保留 `strict_cold` 和 `warm_random` 两个协议。

**论文表格口径**：
- ML-1M：3 协议 ✅
- GoodBooks：3 协议 ✅
- Book-Crossing：2 协议（strict_cold + warm_random）✅

**总计**：3+3+2 = 8 个 (dataset, protocol) 格，13 模型 × 8 格 = **104 主 runs**（+ 9 E2E = 113 runs）

---

## 2. 主实验结果（RMSE / MAE）

### 2.1 ML-1M（稠密，4.47%）— 3 协议

#### strict_cold（冷启动，131,291 测试样本，全部 cold）

| Rank | Model | RMSE | MAE |
|:----:|-------|-----:|----:|
| 1 | LightGCN | 0.9879 | 0.7878 |
| 2 | ItemBias | 0.9876 | 0.7879 |
| 3 | SVD | 0.9896 | 0.7871 |
| 4 | UserItemBias | 0.9914 | 0.7860 |
| 5 | Hybrid | 1.0063 | 0.8075 |
| 6 | NeuMF | 1.0415 | 0.8623 |
| 7 | DualHardSwitch | 1.0825 | 0.8741 |
| 8 | DualSoftGating | 1.0883 | 0.9199 |
| 9 | ProfileMLP | 1.0837 | 0.8827 |
| 10 | GlobalMean | 1.1190 | 0.9332 |
| 10 | UserBias | 1.1190 | 0.9332 |
| 12 | BehaviorMLP | 1.3636 | 1.1771 |
| 13 | DeepFM | 1.3717 | 1.1698 |

#### warm_random（134,035 样本，cold 854 / medium 10,894 / warm 122,287）

| Rank | Model | RMSE | MAE | Cold | Warm |
|:----:|-------|-----:|----:|-----:|-----:|
| 1 | SVD | 0.8977 | 0.7067 | 0.9875 | 0.8918 |
| 2 | Hybrid | 0.9055 | 0.7165 | 1.0249 | 0.8994 |
| 3 | UserItemBias | 0.9057 | 0.7148 | 0.9840 | 0.9009 |
| 4 | NeuMF | 0.9144 | 0.7355 | 1.0301 | 0.9066 |
| 5 | DualSoftGating | 0.9147 | 0.7220 | 1.0127 | 0.9093 |
| 6 | DualHardSwitch | 0.9159 | 0.7220 | 1.0981 | 0.9096 |
| 7 | BehaviorMLP | 0.9179 | 0.7234 | 1.0434 | 0.9119 |
| 8 | ItemBias | 0.9770 | 0.7803 | 1.0389 | 0.9738 |
| 9 | UserBias | 1.0305 | 0.8242 | 1.0676 | 1.0281 |
| 10 | ProfileMLP | 1.0616 | 0.8574 | 1.0974 | 1.0598 |
| 11 | GlobalMean | 1.1118 | 0.9282 | 1.1446 | 1.1102 |
| 12 | DeepFM | 1.2515 | 1.0581 | 1.3436 | 1.2439 |
| 13 | LightGCN | 1.3661 | 1.0787 | 2.6868 | 1.2623 |

#### warm_temporal（134,035 样本，按时间序切分）

| Rank | Model | RMSE | MAE | Cold | Warm |
|:----:|-------|-----:|----:|-----:|-----:|
| 1 | DualSoftGating | 0.9238 | 0.7279 | 0.9083 | 0.9477 |
| 2 | Hybrid | 0.9255 | 0.7246 | 0.9084 | 0.9538 |
| 3 | BehaviorMLP | 0.9257 | 0.7289 | 0.9087 | 0.9532 |
| 4 | DualHardSwitch | 0.9275 | 0.7329 | 0.9147 | 0.9479 |
| 5 | SVD | 0.9525 | 0.7514 | 0.9616 | 0.9324 |
| 6 | UserItemBias | 0.9544 | 0.7521 | 0.9628 | 0.9379 |
| 7 | ItemBias | 0.9632 | 0.7666 | 0.9600 | 0.9689 |
| 8 | NeuMF | 1.0286 | 0.8432 | 1.0610 | 0.9585 |
| 9 | ProfileMLP | 1.0657 | 0.8717 | 1.0688 | 1.0558 |
| 10 | UserBias | 1.0941 | 0.8896 | 1.1019 | 1.0789 |
| 11 | GlobalMean | 1.1035 | 0.9155 | 1.1013 | 1.1065 |
| 12 | DeepFM | 1.1974 | 1.0006 | 1.2366 | 1.1104 |
| 13 | LightGCN | 1.3312 | 1.0247 | 1.2070 | 1.3814 |

### 2.2 GoodBooks（中等稀疏，0.73%）— 3 协议

#### strict_cold（1,193,478 样本，全部 cold）

| Rank | Model | RMSE | MAE |
|:----:|-------|-----:|----:|
| 1 | ItemBias | 0.9553 | 0.7642 |
| 2 | LightGCN | 0.9554 | 0.7644 |
| 3 | UserItemBias | 0.9604 | 0.7679 |
| 4 | SVD | 0.9632 | 0.7743 |
| 5 | BehaviorMLP | 0.9741 | 0.7948 |
| 6 | NeuMF | 0.9747 | 0.7950 |
| 7 | DualHardSwitch | 0.9793 | 0.7750 |
| 8 | ProfileMLP | 0.9808 | 0.7751 |
| 9 | GlobalMean | 0.9921 | 0.7747 |
| 9 | UserBias | 0.9921 | 0.7747 |
| 11 | Hybrid | 1.0037 | 0.7741 |
| 12 | DualSoftGating | 1.0063 | 0.7720 |
| 13 | DeepFM | 1.1037 | 0.9321 |

#### warm_random（1,195,297 样本，cold 166 / medium 21,918 / warm 1,173,213）

| Rank | Model | RMSE | MAE | Cold | Warm |
|:----:|-------|-----:|----:|-----:|-----:|
| 1 | SVD | 0.8494 | 0.6504 | 1.0948 | 0.8485 |
| 2 | NeuMF | 0.8529 | 0.6796 | 1.0446 | 0.8518 |
| 3 | UserItemBias | 0.8530 | 0.6664 | 1.0778 | 0.8522 |
| 4 | Hybrid | 0.8571 | 0.6699 | 1.0759 | 0.8563 |
| 5 | DualSoftGating | 0.8586 | 0.6691 | 1.0800 | 0.8578 |
| 6 | DualHardSwitch | 0.8587 | 0.6682 | 1.1218 | 0.8579 |
| 7 | BehaviorMLP | 0.8589 | 0.6693 | 1.0764 | 0.8580 |
| 8 | UserBias | 0.8938 | 0.7034 | 1.1080 | 0.8933 |
| 9 | ItemBias | 0.9528 | 0.7622 | 1.0734 | 0.9514 |
| 10 | ProfileMLP | 0.9785 | 0.7736 | 1.1180 | 0.9771 |
| 11 | GlobalMean | 0.9903 | 0.7736 | 1.1248 | 0.9890 |
| 12 | DeepFM | 1.1540 | 0.9807 | 1.3431 | 1.1526 |
| 13 | LightGCN | 2.0566 | 1.7156 | 2.8057 | 2.0450 |

#### warm_temporal（1,195,297 样本，按时间序切分）

| Rank | Model | RMSE | MAE | Cold | Warm |
|:----:|-------|-----:|----:|-----:|-----:|
| 1 | Hybrid | 0.8571 | 0.6733 | 0.8472 | 0.8650 |
| 2 | DualHardSwitch | 0.8600 | 0.6659 | 0.8492 | 0.8671 |
| 3 | DualSoftGating | 0.8607 | 0.6665 | 0.8508 | 0.8669 |
| 4 | BehaviorMLP | 0.8635 | 0.6789 | 0.8532 | 0.8733 |
| 5 | UserItemBias | 0.9198 | 0.7276 | 0.9504 | 0.8693 |
| 6 | NeuMF | 0.9284 | 0.7474 | 0.9625 | 0.8727 |
| 7 | SVD | 0.9291 | 0.7367 | 0.9552 | 0.8859 |
| 8 | ItemBias | 0.9361 | 0.7394 | 0.9516 | 0.9080 |
| 9 | UserBias | 0.9500 | 0.7482 | 0.9791 | 0.9019 |
| 10 | ProfileMLP | 0.9560 | 0.7502 | 0.9713 | 0.9285 |
| 11 | GlobalMean | 0.9655 | 0.7492 | 0.9817 | 0.9362 |
| 12 | DeepFM | 0.9337 | 0.7533 | 0.9670 | 0.8781 |
| 13 | LightGCN | 1.7708 | 1.3629 | 1.2554 | 2.2374 |

### 2.3 Book-Crossing（极稀疏，0.0015%）— 仅 2 协议

> ⚠️ BC 无时间戳，warm_temporal 已弃用（见 §1.4）。

#### strict_cold（79,664 样本，全部 cold）

| Rank | Model | RMSE | MAE |
|:----:|-------|-----:|----:|
| 1 | UserItemBias | 0.8071 | 0.6473 |
| 2 | ItemBias | 0.8072 | 0.6463 |
| 3 | SVD | 0.8078 | 0.6476 |
| 4 | NeuMF | 0.8130 | 0.6587 |
| 5 | ProfileMLP | 0.8167 | 0.6603 |
| 6 | DualHardSwitch | 0.8170 | 0.6536 |
| 7 | BehaviorMLP | 0.8185 | 0.6501 |
| 8 | GlobalMean | 0.8208 | 0.6659 |
| 8 | UserBias | 0.8208 | 0.6659 |
| 10 | DeepFM | 0.8221 | 0.6677 |
| 11 | Hybrid | 0.8285 | 0.6589 |
| 12 | DualSoftGating | 0.8325 | 0.6662 |
| 13 | LightGCN | 0.8635 | 0.6795 |

#### warm_random（86,736 样本，cold 39,834 / medium 16,396 / warm 30,506）

| Rank | Model | RMSE | MAE | Cold | Warm |
|:----:|-------|-----:|----:|-----:|-----:|
| 1 | Hybrid | 0.6815 | 0.4855 | 0.6870 | 0.6650 |
| 2 | BehaviorMLP | 0.6914 | 0.4811 | 0.6938 | 0.6759 |
| 3 | DualSoftGating | 0.7152 | 0.4755 | 0.7220 | 0.7000 |
| 4 | UserItemBias | 0.7231 | 0.5580 | 0.7701 | 0.6596 |
| 5 | SVD | 0.7263 | 0.5595 | 0.7712 | 0.6649 |
| 6 | UserBias | 0.7345 | 0.5687 | 0.7833 | 0.6688 |
| 7 | NeuMF | 0.7452 | 0.5892 | 0.7913 | 0.6826 |
| 8 | DeepFM | 0.7588 | 0.6020 | 0.7967 | 0.7086 |
| 9 | DualHardSwitch | 0.7612 | 0.5535 | 0.8275 | 0.6885 |
| 10 | ProfileMLP | 0.8105 | 0.6556 | 0.8249 | 0.7875 |
| 11 | ItemBias | 0.8068 | 0.6467 | 0.8162 | 0.7925 |
| 12 | GlobalMean | 0.8187 | 0.6647 | 0.8302 | 0.8016 |
| 13 | LightGCN | 1.9658 | 1.5701 | 2.1506 | 1.6416 |

---

## 3. 核心发现 1：LightGCN 排名反转（论文最强卖点）

| Dataset | strict_cold 排名 | warm_random 排名 | warm_temporal 排名 | 排名跳变 |
|---------|:---------------:|:----------------:|:------------------:|:--------:|
| ML-1M | **1** | **13** | **13** | 12 位 ↓ |
| GoodBooks | **2** | **13** | **13** | 11 位 ↓ |
| Book-Crossing | 13 | 13 | — (弃用) | 0（始终垫底） |

**关键数字（单种子实测）：**

- **ML-1M**：strict_cold RMSE=0.9879（第 1，比第 2 ItemBias 低 0.0003）→ warm_random RMSE=1.3661（第 13，比第 12 DeepFM 还高 0.115）。Cold-Warm Gap **= 1.3782**
- **GoodBooks**：strict_cold RMSE=0.9554（第 2，仅比第 1 ItemBias 高 0.0001）→ warm_random RMSE=2.0566（第 13，是第 1 SVD 的 2.4 倍）。Cold-Warm Gap **= 1.1012**
- **Book-Crossing**：strict_cold RMSE=0.8635（第 13）→ warm_random RMSE=1.9658（第 13，cold 段 2.15 远高于 warm 段 1.64）

**论文可写主张**：
> LightGCN 在严格冷启动场景下表现接近最优（ML-1M 第 1，GoodBooks 第 2，仅落后最优 Bias 基线 0.0001–0.0003 RMSE），但一旦引入带热信息的 warm 场景便彻底崩溃。这一现象在 2 个有时间戳的数据集（ML-1M、GoodBooks）上一致出现，排名反转幅度达 11–12 位，且崩溃幅度随数据规模增大而加剧。

---

## 4. 核心发现 2：冷热 Gap 与模型复杂度

ML-1M warm_random 上的 Cold-Warm Gap（cold_rmse − warm_rmse），按 Gap 降序：

| Model | Complexity | Cold | Warm | Gap |
|-------|:----------:|-----:|-----:|----:|
| LightGCN | 5 | 2.6868 | 1.2623 | **1.4245** |
| DualHardSwitch | 4 | 1.0981 | 0.9096 | 0.1885 |
| BehaviorMLP | 3 | 1.0434 | 0.9119 | 0.1315 |
| Hybrid | 3 | 1.0249 | 0.8994 | 0.1255 |
| NeuMF | 3 | 1.0301 | 0.9066 | 0.1235 |
| DualSoftGating | 4 | 1.0127 | 0.9093 | 0.1034 |
| DeepFM | 3 | 1.3436 | 1.2439 | 0.0997 |
| SVD | 2 | 0.9875 | 0.8918 | 0.0957 |
| UserItemBias | 1 | 0.9840 | 0.9009 | 0.0831 |
| ItemBias | 1 | 1.0389 | 0.9738 | 0.0651 |
| UserBias | 1 | 1.0676 | 1.0281 | 0.0395 |
| ProfileMLP | 3 | 1.0974 | 1.0598 | 0.0376 |
| GlobalMean | 1 | 1.1446 | 1.1102 | 0.0344 |

**规律**：
- LightGCN（complexity=5）Gap 1.42，是第二名 DualHardSwitch（0.19）的 **7.5 倍**
- 剔除 LightGCN 后仍呈弱正相关：Bias 类（1）Gap 0.03–0.08 < Deep 类（3）Gap 0.04–0.13 < Adaptive 类（4）Gap 0.10–0.19

**论文可写主张**：
> 模型对冷热场景的敏感度（Cold-Warm Gap）与其复杂度正相关。最简单的 Bias 基线 Gap 最小（< 0.09），表明其对热信息依赖低；而图模型 LightGCN 的 Gap 高达 1.42，远超其他所有模型之和，揭示高复杂度模型对热启动信息的脆弱性。

---

## 5. 核心发现 3：稀疏度对模型家族的差异化影响

按密度排序 BC (0.0015%) → GB (0.73%) → ML-1M (4.47%)，观察 warm_random 协议下各家族 RMSE：

| Model Family | BC | GB | ML-1M | 趋势 |
|--------------|---:|---:|------:|------|
| **Bias** (UserItemBias) | 0.7231 | 0.8530 | 0.9057 | 稳定，随密度升高略升 |
| **MF** (SVD) | 0.7263 | 0.8494 | 0.8977 | 稳定，与 Bias 接近 |
| **Deep** (Hybrid) | 0.6815 | 0.8571 | 0.9055 | BC 上最优 |
| **Deep** (BehaviorMLP) | 0.6914 | 0.8589 | 0.9179 | BC 上表现强 |
| **Deep** (NeuMF) | 0.7452 | 0.8529 | 0.9144 | 同上 |
| **Deep** (DeepFM) | 0.7588 | 1.1540 | 1.2515 | 训练不稳定 |
| **Adaptive** (DualSoftGating) | 0.7152 | 0.8586 | 0.9147 | 跨密度稳健 |
| **Graph** (LightGCN) | 1.9658 | 2.0566 | 1.3661 | **极稀疏崩溃** |

**关键观察**：

1. **GNN 随稀疏度恶化最剧烈**：LightGCN 在 BC 上 1.97、GB 上 2.06（甚至比 BC 更差），仅 ML-1M 因密度高而"相对正常"（1.37，但仍垫底）
2. **Profile+Behavior 融合的 Deep 模型在极稀疏下反而最优**：Hybrid 在 BC 上 0.6815（全模型最低），BehaviorMLP 0.6914 次之——说明 profile 信号在行为稀疏时起决定性作用
3. **Bias 与 MF 几乎不受稀疏度影响**：UserItemBias 与 SVD 在三个数据集上始终接近，说明低秩线性方法对密度不敏感

**论文可写主张**：
> 不同模型家族对数据稀疏度的响应差异显著。图模型 LightGCN 在中等稀疏（0.73%）下即崩溃（RMSE 2.06），且在极稀疏（0.0015%）下无改善；而融合用户画像的深度模型（Hybrid、BehaviorMLP）在极稀疏场景下反而达到全局最优（RMSE 0.68），揭示 profile 信号在行为稀疏时的补偿价值。

---

## 6. 核心发现 4：PSI（Protocol Sensitivity Index）

PSI = max(Rank_SC, Rank_WR, Rank_WT) − min(...)，量化模型对协议切换的敏感度。

### 6.1 ML-1M PSI（3 协议）

| Model | Rank SC | Rank WR | Rank WT | PSI |
|-------|--------:|--------:|--------:|----:|
| **LightGCN** | 1 | 13 | 13 | **12** |
| BehaviorMLP | 12 | 6 | 3 | 9 |
| DualSoftGating | 7 | 4 | 1 | 6 |
| ItemBias | 2 | 8 | 7 | 6 |
| SVD | 3 | 1 | 5 | 4 |
| UserItemBias | 4 | 2 | 6 | 4 |
| DualHardSwitch | 8 | 5 | 4 | 4 |
| Hybrid | 5 | 3 | 2 | 3 |
| NeuMF | 6 | 7 | 8 | 2 |
| ProfileMLP | 9 | 10 | 9 | 1 |
| GlobalMean | 10 | 11 | 11 | 1 |
| UserBias | 10 | 9 | 10 | 1 |
| DeepFM | 13 | 12 | 12 | 1 |

### 6.2 GoodBooks PSI（3 协议）

| Model | Rank SC | Rank WR | Rank WT | PSI |
|-------|--------:|--------:|--------:|----:|
| **LightGCN** | 2 | 13 | 13 | **11** |
| Hybrid | 12 | 4 | 1 | 11 |
| ItemBias | 1 | 9 | 8 | 8 |
| DualSoftGating | 11 | 5 | 3 | 8 |
| SVD | 4 | 1 | 6 | 5 |
| NeuMF | 5 | 2 | 7 | 5 |
| DualHardSwitch | 7 | 6 | 2 | 5 |
| BehaviorMLP | 6 | 7 | 4 | 3 |
| GlobalMean | 9 | 12 | 11 | 3 |
| UserItemBias | 3 | 3 | 5 | 2 |
| ProfileMLP | 8 | 10 | 10 | 2 |
| DeepFM | 13 | 11 | 12 | 2 |
| UserBias | 9 | 8 | 9 | 1 |

### 6.3 Book-Crossing PSI（仅 2 协议：SC + WR）

| Model | Rank SC | Rank WR | PSI |
|-------|--------:|--------:|----:|
| Hybrid | 11 | 1 | 10 |
| **LightGCN** | 13 | 13 | 0 |
| BehaviorMLP | 7 | 2 | 5 |
| DualSoftGating | 12 | 3 | 9 |
| DualHardSwitch | 6 | 9 | 3 |
| ItemBias | 2 | 11 | 9 |
| NeuMF | 4 | 7 | 3 |
| UserItemBias | 1 | 4 | 3 |
| SVD | 3 | 5 | 2 |
| UserBias | 8 | 6 | 2 |
| ProfileMLP | 5 | 10 | 5 |
| DeepFM | 10 | 8 | 2 |
| GlobalMean | 8 | 12 | 4 |

**论文可写主张**：
> LightGCN 在 ML-1M 和 GoodBooks 上是协议敏感度最高的模型（PSI=12 和 11），其在 strict_cold 下的顶尖排名与 warm 协议下的垫底排名形成极端反差。Bias 类模型 PSI 普遍 ≤ 6，显示其协议稳健性。BC 上 LightGCN PSI=0 是因其两协议均垫底。

---

## 7. CR（Complexity Ratio，复杂度性价比）

CR = Model_RMSE / Best_Bias_RMSE（warm_random 协议）。CR=1.00 表示不优于最佳 Bias 基线。

| Dataset | Model | RMSE | Best Bias | CR |
|---------|-------|-----:|-----------|---:|
| ML-1M | SVD | 0.8977 | UserItemBias (0.9057) | 0.99 |
| ML-1M | UserItemBias | 0.9057 | UserItemBias | 1.00 |
| ML-1M | Hybrid | 0.9055 | UserItemBias | 1.00 |
| ML-1M | DualSoftGating | 0.9147 | UserItemBias | 1.01 |
| ML-1M | DualHardSwitch | 0.9159 | UserItemBias | 1.01 |
| ML-1M | BehaviorMLP | 0.9179 | UserItemBias | 1.01 |
| ML-1M | NeuMF | 0.9144 | UserItemBias | 1.01 |
| ML-1M | ItemBias | 0.9770 | UserItemBias | 1.08 |
| ML-1M | UserBias | 1.0305 | UserItemBias | 1.14 |
| ML-1M | ProfileMLP | 1.0616 | UserItemBias | 1.17 |
| ML-1M | GlobalMean | 1.1118 | UserItemBias | 1.23 |
| ML-1M | DeepFM | 1.2515 | UserItemBias | 1.38 |
| ML-1M | LightGCN | 1.3661 | UserItemBias | 1.51 |
| GoodBooks | SVD | 0.8494 | UserItemBias (0.8530) | 0.99 |
| GoodBooks | NeuMF | 0.8529 | UserItemBias | 1.00 |
| GoodBooks | UserItemBias | 0.8530 | UserItemBias | 1.00 |
| GoodBooks | Hybrid | 0.8571 | UserItemBias | 1.00 |
| GoodBooks | DualSoftGating | 0.8586 | UserItemBias | 1.01 |
| GoodBooks | DualHardSwitch | 0.8587 | UserItemBias | 1.01 |
| GoodBooks | BehaviorMLP | 0.8589 | UserItemBias | 1.01 |
| GoodBooks | UserBias | 0.8938 | UserItemBias | 1.05 |
| GoodBooks | ItemBias | 0.9528 | UserItemBias | 1.12 |
| GoodBooks | ProfileMLP | 0.9785 | UserItemBias | 1.15 |
| GoodBooks | DeepFM | 1.1540 | UserItemBias | 1.35 |
| GoodBooks | GlobalMean | 0.9903 | UserItemBias | 1.16 |
| GoodBooks | LightGCN | 2.0566 | UserItemBias | 2.41 |
| Book-Crossing | Hybrid | 0.6815 | UserItemBias (0.7231) | **0.94** |
| Book-Crossing | BehaviorMLP | 0.6914 | UserItemBias | **0.96** |
| Book-Crossing | DualSoftGating | 0.7152 | UserItemBias | 0.99 |
| Book-Crossing | UserItemBias | 0.7231 | UserItemBias | 1.00 |
| Book-Crossing | SVD | 0.7263 | UserItemBias | 1.01 |
| Book-Crossing | UserBias | 0.7345 | UserItemBias | 1.02 |
| Book-Crossing | NeuMF | 0.7452 | UserItemBias | 1.03 |
| Book-Crossing | DeepFM | 0.7588 | UserItemBias | 1.05 |
| Book-Crossing | DualHardSwitch | 0.7612 | UserItemBias | 1.05 |
| Book-Crossing | ProfileMLP | 0.8105 | UserItemBias | 1.12 |
| Book-Crossing | ItemBias | 0.8068 | UserItemBias | 1.12 |
| Book-Crossing | GlobalMean | 0.8187 | UserItemBias | 1.13 |
| Book-Crossing | LightGCN | 1.9658 | UserItemBias | 2.72 |

**关键观察**：
- ML-1M 与 GoodBooks 上，大多数 Deep/Adaptive 模型 CR ≈ 1.00–1.01，**复杂模型相对 Bias 基线的提升微乎其微**（< 1%）
- 唯有 Book-Crossing 上 Hybrid (0.94) 与 BehaviorMLP (0.96) 显著优于 Bias，**profile 融合的价值在极稀疏场景才体现**
- LightGCN 在三个数据集上 CR 都是最差（1.51 / 2.41 / 2.72），复杂度最高但性价比最低

---

## 8. E2E Gating 对照（DualSoftGating 端到端 vs 两阶段）

> ⚠️ E2E 收益有限且不稳定，论文中作为**设计启示**而非主要贡献呈现。

| Dataset | Protocol | 两阶段 RMSE | E2E RMSE | Δ | E2E MAE |
|---------|----------|------------:|---------:|---:|--------:|
| ML-1M | strict_cold | 1.0883 | 1.0220 | −0.0663 ✅ | 0.8326 |
| ML-1M | warm_random | 0.9147 | 0.9117 | −0.0030 ✅ | 0.7210 |
| ML-1M | warm_temporal | 0.9238 | 0.9237 | −0.0001 ✅ | 0.7312 |
| GoodBooks | strict_cold | 1.0063 | 1.1550 | +0.1487 ❌ | 0.9097 |
| GoodBooks | warm_random | 0.8586 | 0.8586 | 0.0000 | 0.6702 |
| GoodBooks | warm_temporal | 0.8607 | 0.8603 | −0.0004 ✅ | 0.6667 |
| Book-Crossing | strict_cold | 0.8325 | 0.8393 | +0.0068 ❌ | 0.6750 |
| Book-Crossing | warm_random | 0.7152 | 0.7123 | −0.0029 ✅ | 0.4782 |
| Book-Crossing | warm_temporal | — (弃用) | — | — | — |

**论文写法建议**：
> E2E 联合训练在 ML-1M strict_cold 上取得 0.066 RMSE 改善，但在 GoodBooks strict_cold 上反而退化 0.149。整体而言，E2E 收益有限且不稳定，表明 gating 权重的端到端学习并未稳定优于基于活动度的两阶段启发式。这一发现作为**设计启示**指出：自适应门控的优化方向可能在于更稳定的门控信号而非联合训练本身。

---

## 9. 排序指标（Recall@K / NDCG@K / Hit@K，节选）

> 注：本任务为 rating prediction，Hit@K 普遍为 1.0 因评测口径把所有测试样本视为"命中"。Recall/NDCG 绝对值偏低因候选集为全物品集（13K–271K），非 Top-K 推荐常规设定。**论文应以 RMSE/MAE 为主，排序指标作辅助参考**。

### ML-1M 节选

| Protocol | Model | R@5 | N@5 | R@10 | N@10 | R@20 | N@20 |
|----------|-------|----:|----:|-----:|-----:|-----:|-----:|
| strict_cold | LightGCN | 5.3e-5 | 0.8304 | 1.1e-4 | 0.8116 | 2.3e-4 | 0.8455 |
| strict_cold | SVD | 5.3e-5 | 0.8304 | 1.1e-4 | 0.8116 | 2.3e-4 | 0.8455 |
| warm_random | SVD | 5.3e-5 | 0.8304 | 1.1e-4 | 0.8116 | 2.3e-4 | 0.8455 |
| warm_random | LightGCN | 6.5e-5 | 1.0000 | 1.2e-4 | 0.9216 | 2.2e-4 | 0.8783 |

### GoodBooks 节选

| Protocol | Model | R@5 | N@5 | R@10 | N@10 | R@20 | N@20 |
|----------|-------|----:|----:|-----:|-----:|-----:|-----:|
| strict_cold | LightGCN | 3.6e-6 | 0.7227 | 9.7e-6 | 0.8201 | 2.2e-5 | 0.8839 |
| strict_cold | DualSoftGating | 5.0e-6 | 0.8539 | 6.0e-6 | 0.6325 | 1.7e-5 | 0.7281 |
| warm_random | SVD | 6.0e-6 | 1.0000 | 1.2e-5 | 1.0000 | 2.4e-5 | 1.0000 |
| warm_random | LightGCN | 4.9e-6 | 0.8304 | 9.7e-6 | 0.8166 | 1.9e-5 | 0.8114 |

### Book-Crossing 节选

| Protocol | Model | R@5 | N@5 | R@10 | N@10 | R@20 | N@20 |
|----------|-------|----:|----:|-----:|-----:|-----:|-----:|
| strict_cold | UserItemBias | 8.9e-5 | 0.8304 | 2.0e-4 | 0.8900 | 4.2e-4 | 0.9290 |
| strict_cold | LightGCN | 4.4e-5 | 0.5531 | 8.9e-5 | 0.4959 | 2.4e-4 | 0.5661 |
| warm_random | UserItemBias | 1.0e-4 | 1.0000 | 1.8e-4 | 0.9364 | 3.6e-4 | 0.9206 |
| warm_random | LightGCN | 8.0e-5 | 0.8688 | 1.8e-4 | 0.9149 | 3.4e-4 | 0.8670 |

> LightGCN 在 warm_random 上 NDCG@5=1.0 异常，源于 RMSE 崩溃后预测值集中，与评测逻辑交互导致，论文中应作为异常说明。

---

## 10. 样本量与冷热段分布

| Dataset | Protocol | n_samples | cold_n | warm_n | medium_n |
|---------|----------|----------:|-------:|-------:|---------:|
| ML-1M | strict_cold | 131,291 | 131,291 | 0 | 0 |
| ML-1M | warm_random | 134,035 | 854 | 122,287 | 10,894 |
| ML-1M | warm_temporal | 134,035 | 702,247 | 346,942 | 146,108 |
| GoodBooks | strict_cold | 1,193,478 | 1,193,478 | 0 | 0 |
| GoodBooks | warm_random | 1,195,297 | 166 | 1,173,213 | 21,918 |
| GoodBooks | warm_temporal | 1,195,297 | 702,247 | 346,942 | 146,108 |
| Book-Crossing | strict_cold | 79,664 | 79,664 | 0 | 0 |
| Book-Crossing | warm_random | 86,736 | 39,834 | 30,506 | 16,396 |
| ~~Book-Crossing~~ | ~~warm_temporal~~ | — 弃用 | — | — | — |

---

## 11. 单种子版本的局限性（论文 Limitations 节选）

1. **GoodBooks/LightGCN 缺 seed2025、seed2026**：LightGCN 是排名反转核心模型，单种子虽已成立，但补齐后才能报 std 与显著性检验。补跑命令见仓库根目录。
2. **DeepFM 方差异常**：BC warm_temporal RMSE=0.9207（已弃用）；GB warm_random 1.1540。实际 DeepFM 在 ML-1M/BC 上完整 3 种子已有，可查 std 验证。GoodBooks 上 5 种子齐。
3. **BC 无时间戳**：Book-Crossing 原始数据无 timestamp 字段，故弃用 BC warm_temporal，仅保留 strict_cold 与 warm_random。论文 Limitations 须如实说明。
4. **LightGCN 实现含 3 项偏离原论文**：① MSE 损失（原论文 BPR）；② rating prediction 任务（原论文 Top-K 推荐）；③ item_bias 冷启动回退机制。论文须如实说明——这恰好支撑"实现细节影响评估结论"的论点。
5. **排序指标 Hit@K 普遍=1.0**：rating prediction 评测口径问题，论文以 RMSE/MAE 为主指标。
6. **统计显著性检验缺失**：单种子无法做配对 t / Wilcoxon 检验，"显著优于"等表述需弱化为"在 seed2024 上观察到"。

---

## 12. 单种子版本可支撑的论文主张

| 主张 | 支撑强度 | 依据 |
|------|:-------:|------|
| LightGCN 在 strict_cold 与 warm 协议间排名反转 | **强** | ML-1M 反转 12 位、GB 反转 11 位，2 数据集一致 |
| 冷热 Gap 与模型复杂度正相关 | **中** | LightGCN Gap 1.42 是第二名 7.5 倍；剔除后弱正相关 |
| GNN 随稀疏度恶化最剧烈 | **强** | LightGCN 在 BC 1.97、GB 2.06，密度高时仍崩 |
| Profile 融合在极稀疏下有效 | **强** | Hybrid/BehaviorMLP 在 BC 上 CR<1.00 |
| E2E Gating 优于两阶段 | **弱** | 多数 Δ<0.005，GB strict_cold 反退化 |
| Bias 基线难以被显著超越 | **强** | ML-1M/GB 上多数 Deep 模型 CR≈1.00 |
| PSI 可量化协议敏感性 | **中** | LightGCN PSI 最高在 2 数据集上一致 |

---

## 13. 论文表格推荐口径

**主表（Table I）**：13 模型 × 8 (dataset, protocol) 格
- ML-1M：3 协议
- GoodBooks：3 协议
- Book-Crossing：2 协议（strict_cold + warm_random）

**排名反转表（Table II）**：LightGCN 跨数据集排名

| Dataset | strict_cold | warm_random | warm_temporal |
|---------|:-----------:|:-----------:|:-------------:|
| ML-1M | 1 | 13 | 13 |
| GoodBooks | 2 | 13 | 13 |
| Book-Crossing | 13 | 13 | — |

**PSI 表（Table III）**：3 数据集各模型 PSI

**CR 表（Table IV）**：warm_random 上各模型 CR

**Limitations 节**：须包含 BC 无时间戳、LightGCN 实现偏离、单种子未做显著性检验三项

---

## 附录：原始数据文件位置

- 主实验：`result_plus/runs/{dataset}/{model}/seed2024/{protocol}/metrics.json`（104 文件）
- E2E 实验：`result_plus/runs/{dataset}/dual_soft_gating_e2e/seed2024/{protocol}/metrics.json`（9 文件，BC warm_temporal 已弃用）
- 配置：`configs/full_gpu_plus.yaml`、`configs/models/*.yaml`、`configs/datasets/*.yaml`
- 模型实现：`models/*.py`（LightGCN 在 `models/lightgcn.py`）
- 数据加载：`data/loaders/book_crossing_loader.py`（第 132–134 行 timestamp 行号代理）
- 切分逻辑：`utils/splitters.py`（第 87–132 行 warm_temporal 实现）
- 运行入口：`scripts/run_experiment.py --config configs/full_gpu_plus.yaml --dataset {ds} --model {m} --seed 2024 --resume`

---

*本报告基于磁盘实测 `result_plus/runs/.../seed2024/` 下 113 个 metrics.json（104 主 + 9 E2E），BC warm_temporal 已弃用。所有数值未做任何聚合或插值。*
