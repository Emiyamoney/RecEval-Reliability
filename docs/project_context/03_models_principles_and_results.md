# 03 — 模型原理与实现逻辑

> 项目路径: `E:\学习\推荐系统\project1_GPU\project1`
> 审计日期: 2026-07

---

## 1. 模型总览

| 模型 | 代码位置 | 主要思想 | 输入维度 | 输出 | 损失函数 | group_avg | activity_index | SVD 因子 | Occupation |
|------|----------|----------|:---:|------|----------|:---:|:---:|:---:|:---:|
| SVD baseline | `models/svd_model.py` | 用户偏置+电影偏置+隐因子点积 | N/A (直接计算) | r̂ ∈ [1,5] (via clip) | MSE + L2 reg | ❌ | ❌ | N/A (自身) | ❌ |
| SVD + MLP 残差 | `trainers/mlp_trainer.py` + `models/mlp_model.py` | SVD 预测 → MLP 学残差 | 129 | Δ (残差) | MSE(residual, y−svd) | ✅ (P0泄漏) | ❌ | ✅ (50+50) | ❌ |
| Activity MLP | `trainers/activity_trainer.py` + `models/mlp_model.py` | 活跃度加权 loss + 无 SVD | 30 | r̂ ∈ [1,5] | weighted MSE | ✅ (P0泄漏) | ✅ (特征+权重) | ❌ | ❌ |
| UserCF | `models/cf_model.py` | 用户余弦相似度 + TopK 邻居加权 | N/A | r̂ ∈ [1,5] (clip) | N/A (非参数) | ❌ | ❌ | ❌ | ❌ |

---

## 2. 每个模型的原理与实现

### 2.1 SVD Baseline

#### 原理简述

经典的偏置 SVD 协同过滤。将评分分解为全局均值 μ + 用户偏置 b_u + 电影偏置 b_i + 用户隐向量 p_u 与电影隐向量 q_i 的点积。SGD 优化，每次更新一个样本。

#### 具体实现

| 属性 | 值/说明 |
|------|--------|
| 代码位置 | `models/svd_model.py` |
| 数学公式 | r̂ = μ + b_u + b_i + p_u · q_i |
| k (隐因子维度) | 50 (`config.SVD_FACTORS`) |
| 学习率 | 0.01 (`config.SVD_LR`) |
| 正则化 | L2, λ=0.02 (`config.SVD_REG`) |
| Epochs | 5 (`config.SVD_EPOCHS`) |
| Batch | 无 (逐样本 SGD) |
| 优化器 | 手写 SGD |
| 预测时 unknown 处理 | 返回 `self.global_mean` |
| 是否 clip 输出 | ❌ (SVD.predict 不 clip；CF 和 MLP 中也不 clip) |
| 输出的用途 | (1) 直接预测 r̂_svd；(2) 作为 MLP 的特征（user/item 隐向量 + svd_base） |
| 训练数据 | `train.py:42`: `svd_model = train_svd(df)` — 当前 `df` 是全量/QUICK_SAMPLE，含测试用户 |
| Save/Load | pickle 完整对象 (`svd_model.pkl`, 3.2 MB) |

#### 泄漏影响

| 泄漏源 | 影响 | 等级 |
|--------|------|:---:|
| SVD 在含测试用户的全量 df 上训练 | 测试用户/电影的隐因子已见过测试评分 | P0 |
| 作为 MLP 特征生成器时传递泄漏 | MLP 接收的 SVD 因子也含测试信息 | P0 |

#### 论文角色

**必须保留的 baseline**。SVD 是推荐系统中最经典的协同过滤基线。保留理由：提供"纯协同过滤无内容特征"的性能下界，MLP 残差的效果必须相对于 SVD 来证明增益。

---

### 2.2 SVD + MLP 残差（主模型）

#### 原理简述

两阶段解耦训练：(1) SVD 学习评分矩阵的低秩结构；(2) MLP 接收内容特征 + SVD 隐因子，仅学习残差 Δ = r − r̂_svd。最终预测 r̂ = r̂_svd + Δ。

解耦设计避免 SVD 和 MLP 的梯度互相干扰，同时让 MLP 专注于 SVD 难以捕捉的非线性模式。

#### 具体实现

| 属性 | 值/说明 |
|------|--------|
| 代码位置 | `trainers/mlp_trainer.py` + `models/mlp_model.py` |
| 输入维度 | **129** |
| 特征构成 | 电影受众(5) + 用户偏好(18) + 性别(2) + 年龄(3) + group_avg(1) + SVD user factor(50) + SVD item factor(50) |
| 标签 | **残差** `y − svd_base`（非原始 Rating） |
| 网络结构 | Linear(129→128) → BN → ReLU → Drop(0.3) → Linear(128→64) → BN → ReLU → Drop(0.3) → Linear(64→1) → squeeze |
| 优化器 | Adam (lr=0.005) |
| 损失函数 | `nn.MSELoss()` |
| Batch Size | 2048 |
| Epochs | 1 (`config.EPOCHS`) |
| Dropout | 0.3 |
| 输出是否 clip | ❌ |
| 最终预测 | `r̂ = svd_base + residual`（在 evaluator.py:63 中相加） |
| 训练数据 | `train.py:45`: `train_mlp(df, svd_model, act_map)` — df 含测试用户 |
| Save/Load | `torch.save(model.state_dict(), MLP_PATH)` → `mlp_residual.pth` (106 KB) |

#### 泄漏影响

| 泄漏源 | 影响 | 等级 |
|--------|------|:---:|
| group_avg 含自身 | 训练时 MLP 学习到"用户评分 ≈ 组内均值"的捷径 | P0 |
| group_avg 评估时重算 | 测试集评分泄漏到推理输入 | P0 |
| SVD 因子含测试用户 | 输入特征已见过测试评分 | P0 |
| 用户重叠 | MLP 的 user_age_gen 对测试用户已构建完毕 | P1 |
| activity_map | 此模型不使用（无 activity_index 特征） | OK |

#### 论文角色

**主模型**。SVD+MLP 残差是项目最核心的架构创新点（两阶段解耦）。保留理由：它同时利用了协同信号（SVD）和内容信号（MLP），是 hybrid 推荐系统的典型代表。

---

### 2.3 Activity-Weighted MLP

#### 原理简述

不使用 SVD。以用户活跃度（评分次数）作为样本权重：高活跃用户的样本在 loss 中贡献更大，模型更关注高频用户的偏好模式。活跃度同时也作为特征拼接在输入中。

#### 具体实现

| 属性 | 值/说明 |
|------|--------|
| 代码位置 | `trainers/activity_trainer.py` + `models/mlp_model.py` |
| 输入维度 | **30** |
| 特征构成 | 电影受众(5) + 用户偏好(18) + 性别(2) + 年龄(3) + group_avg(1) + **activity_index(1)** |
| 标签 | 原始 Rating（非残差） |
| 网络结构 | 同 ResidualMLP (128→64→1) |
| 优化器 | Adam (lr=0.005) |
| 损失函数 | `(w * MSE(pred, y)).mean()` — 逐样本加权 |
| 权重 w | `self.activity_map.get(uid, 1.0)`（即 activity_index 的值） |
| Batch Size | 2048 |
| Epochs | 1 (`config.EPOCHS`) |
| 输出是否 clip | ❌ |
| 训练数据 | `train.py:53`: `train_activity_mlp(df, act_map)` — df 含测试用户 |

#### 泄漏影响

| 泄漏源 | 影响 | 等级 |
|--------|------|:---:|
| group_avg 含自身 + 评估重算 | 同 MLP 残差 | P0 |
| activity_map 全量计算 | 测试用户活跃度已知 | P1 |
| 用户重叠 | 人口特征 + group_avg 在测试时已存在 | P1 |
| SVD 因子 | 此模型不使用 | OK |

#### 论文角色

**当前最优 MSE (0.3121) 但严重受泄漏影响**。作为**消融模型**（ablation）保留：对比它和 SVD+MLP 残差可以回答"两阶段解耦是否优于纯加权 MLP"。但当前性能数字不可信。

> ⚠️ **关键判断**: Activity MLP 的 MSE=0.3121 很可能大部分来自 group_avg 泄漏（因为它的输入维度只有 30，模型容量小于 SVD+MLP 的 129 维，理论上不应显著优于 SVD+MLP）。修复泄漏后预期 MSE 可能 > 1.0。

---

### 2.4 UserCF

#### 原理简述

基于用户的协同过滤。构建 User×Movie 评分矩阵，中心化后用余弦相似度找 TopK 相似用户，用邻居的加权平均评分做预测。

#### 具体实现

| 属性 | 值/说明 |
|------|--------|
| 代码位置 | `models/cf_model.py` |
| K | 30 (`config.CF_K`) |
| 最小共同评分 | 5 (`config.CF_MIN_COMMON`) |
| 相似度 | 余弦相似度 |
| 预测公式 | r̂ = μ_u + Σ(sim(u,v)·(r_vi − μ_v)) / Σ\|sim(u,v)\| |
| 冷用户处理 | 返回 global_mean (~3.59) |
| 输出 clip | ✅ `np.clip(pred, 1.0, 5.0)` |
| 训练数据 | `train.py:61`: `train_cf(df)` — df 含测试用户 |
| Save/Load | pickle 字典（含完整评分矩阵），228 MB |

#### 泄漏影响

| 泄漏源 | 影响 | 等级 |
|--------|------|:---:|
| 评分矩阵含测试用户 | 预测时可在邻居中找到测试用户自己的历史评分 | P0 |
| group_avg | CF 不使用 | OK |
| activity_map | CF 不使用 | OK |

#### 论文角色

**传统 baseline**。UserCF 是推荐系统经典方法，适合作为"非深度学习"的性能参考点。保留。

---

## 3. 当前实验结果

| 模型 | MSE ↓ | MAE ↓ | RMSE ↓ | 训练数据 | 用户隔离 | 泄漏状态 | 可信度 |
|------|------:|------:|------:|------|:---:|------|:---:|
| SVD baseline | 0.7415 | 0.6833 | 0.8611 | 全量 670K | ❌ | P0 (用户重叠) | ❌ 不可信 |
| SVD + MLP 残差 | 0.4876 | 0.5555 | 0.6983 | 全量 670K | ❌ | P0×3 | ❌ 不可信 |
| **Activity MLP** | **0.3121** | **0.4016** | **0.5586** | 全量 670K | ❌ | P0×3 + P1 | ❌ 不可信 |
| UserCF | 0.7034 | 0.6468 | 0.8387 | 全量 670K | ❌ | P0 (评分矩阵泄漏) | ❌ 不可信 |

> **结论: 当前全部指标不可信，必须修复泄漏后重跑。**

---

## 4. 缺失的关键对比实验

| # | 缺失实验 | 目的 | 优先级 |
|---|----------|------|:---:|
| 1 | **MLP without activity weight** | 消融 activity weight 的效果：将 Activity MLP 的损失函数改为标准 MSE（保留 activity 作特征） | **Must** |
| 2 | **MLP with activity as feature, unweighted** | 区分"activity 作特征"和"activity 作权重"的贡献 | **Must** |
| 3 | **Activity MLP without group_avg** | 消融 group_avg 特征，验证泄漏修复前后差异 | **Must** |
| 4 | **Activity MLP with leakage-free group_avg** | 核心验证：修复泄漏后的模型是否能保持相对优势 | **Must** |
| 5 | **SVD+MLP without group_avg** | 消融 group_avg 对主模型的贡献 | Recommended |
| 6 | **Occupation ablation** | 加 Occupation 特征（当前未使用）看是否有增益 | Recommended |
| 7 | **NeuMF baseline** | 经典 neural CF baseline，论文中必须有 | Recommended |
| 8 | **Wide&Deep baseline** | Google 经典 hybrid 模型，适合对比 | Recommended |
| 9 | **低活跃用户分组评估** | 按用户评分次数分桶，看冷用户表现 | Recommended |
| 10 | **多 seed 重复** | 当前仅 seed=42，需 3-5 个 seed | Recommended |

---

## 5. 后续最小实验顺序 (6-10 步)

### Step 1: 修复所有数据泄漏

**做什么**: 按 `02_leakage_and_evaluation.md` 第 4 节的修复方案，逐一修复 P0 和 P1 泄漏。

**为什么**: 当前所有指标不可信。修复是后续所有实验的前提。

**产出**: 无泄漏版本的 train.py / evaluate.py / dataset.py。

### Step 2: 重跑全部 4 个模型

**做什么**: 在修复后的代码上重新训练 SVD / SVD+MLP / Activity MLP / UserCF，产出新的 results.json。

**为什么**: 建立修复后的 baseline 指标。预期所有模型的 MSE 上升，模型间的相对排序可能改变。

**产出**: 新结果表 (MSE/MAE/RMSE)，标注"泄漏修复后"。

### Step 3: Activity 消融实验

**做什么**: (a) Activity MLP without weighted loss (标准 MSE); (b) Activity MLP without group_avg; (c) Activity MLP 修复后 (leakage-free group_avg + train-only activity_map)。

**为什么**: 区分 activity 作为特征和作为权重的贡献，验证修复后 activity 模块的真实价值。

**产出**: 消融表 (3 行 × MSE/MAE/RMSE)。

### Step 4: Occupation 消融

**做什么**: 在 SVD+MLP 和 Activity MLP 中加入 Occupation one-hot 特征 (21 维)，对比有/无 Occupation 的指标变化。

**为什么**: 当前代码中 Occupation 被完全忽略。如果加入后有增益，论文可以声称"利用人口特征提升推荐精度"。

**产出**: 消融表 (有/无 Occupation 的对比)。

### Step 5: 补充 NeuMF baseline

**做什么**: 实现标准 NeuMF (GMF + MLP 双塔融合)，用相同 train/test split 训练评估。

**为什么**: 论文审稿人会期望有至少一个 neural CF baseline。NeuMF 是引用量最高的 baseline 之一。

**产出**: 一行新 baseline 加入结果表。

### Step 6: 补充 Wide&Deep baseline

**做什么**: 实现 Wide&Deep（wide: 线性组合 one-hot 特征; deep: MLP），对比 hybrid 效果。

**为什么**: Wide&Deep 是工业界 golden baseline，Google 2016 提出，适合与 SVD+MLP 对比。

**产出**: 一行新 baseline。

### Step 7: 低活跃用户分组评估

**做什么**: 按用户评分次数分桶（低: 1-10; 中: 11-50; 高: 51+），输出每个桶的 MSE/MAE/RMSE。

**为什么**: 冷启动是推荐系统的核心挑战。分组评估揭示模型在不同活跃度用户上的泛化能力。

**产出**: 分组结果表 (3 桶 × 4 模型)。

### Step 8: 多 seed 重复

**做什么**: 用 seed ∈ {42, 123, 456, 789, 1024} 重复 Step 2 的全部训练，报告均值 ± 标准差。

**为什么**: 审稿人会要求稳定性分析。单 seed 结果不可靠。

**产出**: 结果表带标准差。

---

## 6. 论文投稿价值判断

| 维度 | 评估 |
|------|------|
| **创新性** | SVD+MLP 两阶段解耦有一定新意，但不算突破性创新。需要更强的 motivation 和消融支持。 |
| **实验完整性** | 当前严重不足：泄漏修复 + baselines + 消融 + 分组评估均缺失。 |
| **baseline 覆盖** | 只有 UserCF 一个传统 baseline，缺少 NeuMF/Wide&Deep 等 neural baselines。 |
| **泄漏影响** | 当前最优结果完全不可信。修复后是否仍能保持相对排序是投稿可行性的关键。 |
| **建议** | 完成 Step 1-8 后重新评估。如果修复泄漏后 Activity MLP 或 SVD+MLP 残差仍显著优于 SVD baseline，且消融实验一致，则具备**中低档会议/workshop** 投稿价值。如果修复后所有模型退化到接近随机水平，则需要重新设计模型。 |
