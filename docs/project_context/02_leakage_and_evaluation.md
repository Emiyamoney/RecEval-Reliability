# 02 — 数据泄漏与评估审计

> 项目路径: `E:\学习\推荐系统\project1_GPU\project1`
> 审计日期: 2026-07

---

## 1. 数据划分方式

### 1.1 当前 Train/Test Split 流程

`train.py:22-71`：

```python
# train.py:22-28
df = pd.read_csv(DATA_PATH)        # 全量 670,168 条
if QUICK_RUN:
    df = df.sample(n=QUICK_SAMPLES, ...)
full_df = pd.read_csv(DATA_PATH)   # 再读一次全量

# ... 所有模型用 df 训练 ...

# train.py:71
test_df = full_df.sample(n=min(3000, len(full_df)), random_state=RANDOM_SEED+1)
```

`evaluate.py:21-22`：

```python
# evaluate.py:21-22
df = pd.read_csv(DATA_PATH)
test_df = df.sample(n=min(5000, len(df)), random_state=RANDOM_SEED+1)
```

### 1.2 划分特征审计

| 项目 | 当前做法 | 代码位置 | 是否合理 | 风险 |
|------|----------|----------|:---:|------|
| 划分方式 | 随机按**行**从全量数据 sample | `train.py:71` | ❌ | 同一用户的评分可能同时出现在 train 和 test |
| 测试集来源 | 与训练集**同一文件** | `train.py:22,71` | ❌ | 非独立测试集 |
| 用户重叠 | test_df 的 1,663 用户全部在 train_df 中 | `train.py:71` | ❌ | SVD/CF 训练时已见过测试用户 |
| 电影重叠 | test_df 的电影全部在 train_df 中（无独立检查但高概率重复） | — | ❌ | 冷启动测试不成立 |
| 固定 split 文件 | **无** | — | ❌ | 每次 `random_state+1` 但不可复现不同配置 |
| SVD 训练数据 | `df`（全量 sample 或 QUICK_SAMPLES） | `train.py:42` | ❌ | SVD 见过训练集的全部用户/电影 |
| MLP 训练数据 | `df`（同上） | `train.py:45` | ❌ | 同上 |
| CF 训练数据 | `df`（同上） | `train.py:61` | ❌ | 同上 |
| 评估时的 Dataset | **重新构造** `RatingDataset(test_df, ...)` | `evaluator.py:56` | ❌ | group_avg 从 test_df 重新计算 |
| Dataset 是否重算 Rating 统计 | ✅ — `group_avg` 从 test_df 重算 | `evaluator.py:56` → `dataset.py:47-55` | ❌ | **P0 泄漏** |

---

## 2. 泄漏风险逐项审计

### 2.1 group_avg 是否在训练时包含当前样本自己的 Rating

| 项目 | 内容 |
|------|------|
| **风险等级** | 🔴 **P0** |
| **涉及文件** | `data/dataset.py:47-55` |
| **关键代码** | `gr = df[df["UserID"].isin(gids)]` → `gr.groupby("MovieID")["Rating"].mean()` |

**泄漏机制:**

`RatingDataset.__init__` 接收训练 `df`，然后计算：

```python
# dataset.py:52-55
gr = df[df["UserID"].isin(gids)]
for mid, avg in gr.groupby("MovieID")["Rating"].mean().items():
    self.group_avg[(gpath, int(mid))] = float(avg)
```

用户 A 对电影 M 的评分 `r` 被包含在 `gr.groupby("MovieID")["Rating"].mean()`中。如果该组内有 `n` 个用户评价了 M，group_avg = (r + others) / n。

然后在 `__getitem__:82`：

```python
ga = self.group_avg.get((gn, mid), 3.0)
feats = np.concatenate([mv, up, ag, [ga]])
```

**用户 A 的评分 `r` 被编码进了输入特征 `ga`，用来预测用户 A 的评分 `r`。** MLP 学到的路径：当 `ga` 存在且偏向极端值时直接输出 `ga`，而非学习真实的协同/内容信号。

**对指标的影响**: 显著低估 MSE/MAE/RMSE。MLP 残差模型的 MSE=0.4876 和 Activity MLP 的 MSE=0.3121 可能大部分来自 group_avg 特征的信息泄漏。

**最小修复**: (1) 使用 Leave-One-Out: 计算 `group_avg` 时排除当前用户 → `ga = (sum - r_user) / (count - 1)`；(2) 或者使用交叉验证 fold 的方式预计算 group_avg，确保某个 fold 的 group_avg 不使用该 fold 的 Rating。

### 2.2 group_avg 是否在评估时由 test_df 重新计算

| 项目 | 内容 |
|------|------|
| **风险等级** | 🔴 **P0** |
| **涉及文件** | `evaluation/evaluator.py:56` → `data/dataset.py:47-55` |
| **关键代码** | `ds = RatingDataset(test_df, activity_map=self.activity_map, svd_model=self.svd)` |

**泄漏机制:**

`evaluator.py:56` 创建新的 `RatingDataset(test_df, ...)`。在 `dataset.py:52-55`，`group_avg` 基于 test_df 的 Rating 重新计算：

```python
gr = test_df[test_df["UserID"].isin(gids)]  # ← 测试集的评分！
for mid, avg in gr.groupby("MovieID")["Rating"].mean().items():
    self.group_avg[(gpath, int(mid))] = float(avg)  # ← 测试集评分泄漏
```

**测试集真实 Rating 通过 group_avg 被编码进 MLP 的输入特征**。MLP 在推理时"看到"了测试集评分的聚合信息，指标严重虚高。

**对指标的影响**: Activity MLP 的 MSE=0.3121 可能严重虚高。真实的泛化性能可能差 2-5 倍。

**最小修复**: 评估时不重新创建 RatingDataset 来算 group_avg，而是从训练阶段保存的预计算 `group_avg.pkl` 加载。

### 2.3 activity_index 是否从全量数据计算

| 项目 | 内容 |
|------|------|
| **风险等级** | 🟡 **P1** |
| **涉及文件** | `train.py:37` → `utils/activity_index.py:12-15` |
| **关键代码** | `act_map = compute_activity_index(DATA_PATH)` |

**泄漏机制:**

`compute_activity_index(DATA_PATH)` 读取完整的 `training_total.csv`（67 万条，包含后来 sample 为 test_df 的行），统计每个用户的评分次数。测试用户的活跃度被包含在内。

**注意**: activity_index 只使用评分**次数**（count），不使用评分**值**。泄漏的信息量较小（只是知道测试用户的历史活跃程度），但仍违反了"评估时不应知道测试集任何信息"的原则。

**对指标的影响**: 较小。活跃度高的用户可能评分更稳定，知道他们的活跃度对模型有些微帮助，但影响远小于 group_avg。

**最小修复**: 改为 `compute_activity_index_from_df(train_df)` 仅从训练集计算。`evaluate.py` 中从保存的 `activity_map.pkl` 加载。

### 2.4 用户偏好是否使用测试集 Rating

| 项目 | 内容 |
|------|------|
| **风险等级** | ⚪ **未确认** |
| **涉及文件** | `data/dataset.py:30-32` → `training_user_genre_preference.csv` |

**分析**: 用户偏好向量从 `training_user_genre_preference.csv` 直接读取。该文件为外部提供的预处理数据，项目的 Python 代码中不存在从 Rating 生成用户偏好的逻辑。但该文件在项目外的生成方式未知——如果原始生成过程使用了测试集评分，则存在泄漏。

**当前判断**: 在项目代码范围内，此特征 **不构成泄漏**（只是读取静态文件）。但需在论文中说明数据来源。

### 2.5 电影受众特征是否使用测试集 Rating

| 项目 | 内容 |
|------|------|
| **风险等级** | 🟢 **OK** |
| **涉及文件** | `data/dataset.py:24-26` → `movie_audience_vector_net_attitude.csv` |

**分析**: 电影受众向量（男性/女性/青年/中年/老年比例）描述了电影的受众构成，与具体用户的评分无关。从项目代码看，这是读取静态 CSV，不存在从 Rating 计算。

### 2.6 SVD 是否训练时见过测试评分

| 项目 | 内容 |
|------|------|
| **风险等级** | 🔴 **P0** |
| **涉及文件** | `train.py:22,42` |

**分析**: `train.py:22` 将 `df` 设为全量数据（或 QUICK_SAMPLES），然后 `train.py:42` 用 `df` 训练 SVD。而 `test_df` 在 `train.py:71` 从同一全量数据 sample —— **训练集和测试集用户/电影完全重叠**。SVD 在训练时已见过所有测试用户的隐因子和所有测试电影的隐因子。

**影响**: SVD baseline 的 MSE=0.7415 是在"见过测试用户"的条件下的表现，不能代表对冷用户的泛化。但 SVD 作为 MLP 的特征生成器，用过拟合的 SVD 因子会影响 MLP 的特征质量评估。

**最小修复**: SVD 必须仅用 `train_df`（用户隔离后）训练。

### 2.7 UserCF 是否预测时见过测试评分

| 项目 | 内容 |
|------|------|
| **风险等级** | 🔴 **P0** |
| **涉及文件** | `train.py:61` → `models/cf_model.py:27-47` |

**分析**: `cf_model.fit(df)` 用全量 `df` 构建 User×Movie 评分矩阵和用户相似度。评估时对 test_df 中的用户做预测——这些用户已存在于评分矩阵中。UserCF 的 MSE=0.7034 是在已知测试用户完整评分历史的条件下得到的。

**最小修复**: CF.fit() 仅用 `train_df`。对于 CF 中不存在于训练集的冷用户，predict 会 fallback 到 `global_mean`。

### 2.8 标准化、归一化、编码器是否 fit 了全量数据

| 项目 | 内容 |
|------|------|
| **风险等级** | 🟢 **OK** |

**分析**: 项目中不存在 StandardScaler、MinMaxScaler 等需要 fit 的变换器。唯一需要"fit"的是：
- SVD: 用 df fit → 如果 df=全量则 P0（已覆盖于 2.6）
- CF: 用 df fit → 同上（已覆盖于 2.7）
- group_avg: 从 df 计算 → P0（已覆盖于 2.1/2.2）

无额外的全局归一化泄漏。

---

## 3. 当前评估指标可信度

### 3.1 当前结果

| 模型 | MSE ↓ | MAE ↓ | RMSE ↓ | 训练数据 | 测试数据 | 受泄漏影响 |
|------|------:|------:|------:|------|------|:---:|
| SVD baseline | 0.7415 | 0.6833 | 0.8611 | 全量 df (QUICK_RUN=False) | 从同一 df sample 3,000 行 | 🔴 P0 |
| SVD + MLP 残差 | 0.4876 | 0.5555 | 0.6983 | 同上 | 同上 + group_avg 泄漏 | 🔴 P0+P0 |
| **Activity MLP** | **0.3121** | **0.4016** | **0.5586** | 同上 | 同上 + group_avg 泄漏 + activity 泄漏 | 🔴 P0+P0+P1 |
| UserCF | 0.7034 | 0.6468 | 0.8387 | 同上 | 同上 | 🔴 P0 |

### 3.2 可信度判断

| 问题 | 结论 |
|------|------|
| 当前最优 MSE (0.3121) 是否可信 | **不可信**。Activity MLP 同时受 group_avg 含自身泄漏 + 评估时 test_df 重算 group_avg + 用户重叠 + activity_map 全量计算，四重泄漏叠加。 |
| 哪些结果必须废弃 | 全部四个模型的当前 results.json 指标。修复泄漏后必须重跑。 |
| 哪些可作为参考 | 无。泄漏严重到无法判断任何模型的相对排序。 |
| 修复后必须重跑的模型 | 全部 4 个 |
| 修复后优先看哪些指标 | MSE → MAE → RMSE（推荐系统标准顺序），同时关注低活跃用户子集的指标 |
| 是否需要验证集 | **是**。当前只有 train/test，缺少 validation set 做超参选择。建议 train/val/test = 70/10/20（按用户 split） |
| 是否需要多随机种子 | **是**。当前只有 RandomState(42)，建议 3-5 个 seed 取均值±方差 |
| 是否需要分组评估 | **是**。低/中/高活跃用户分组可以揭示模型在不同用户群的泛化差异 |

---

## 4. 最小修复建议

| 优先级 | 问题 | 最小修复方案 | 涉及文件 | 修复后验证 |
|:---:|------|------------|----------|----------|
| **P0** | group_avg 含自身 + 评估时重算 | (1) 训练时用 LOO: `ga = (sum - rating) / (count - 1)`；(2) 评估时不创建 RatingDataset(test_df, ...)，改为传入训练集预计算的 `group_avg` | `data/dataset.py`, `evaluation/evaluator.py`, `train.py` | 对比修复前后 Activity MLP 的 MSE，预期显著升高（泄漏消失） |
| **P0** | train/test 用户重叠 | 按 **用户维度** 做 80/20 split，确保 0 用户重叠。SVD/CF/MLP 仅用 train_df 训练 | `train.py`, `evaluate.py` | 重跑确认 SVD baseline MSE 是否显著上升（冷用户变难） |
| **P0** | SVD 训练数据含测试用户 | SVD.fit(train_df)；CF.fit(train_df) | `train.py:42,61` | 同上 |
| **P1** | activity_map 含测试用户统计 | `compute_activity_index` 仅从 train_df 计算；保存为 `activity_map.pkl`；`evaluate.py` 加载而不重算 | `utils/activity_index.py`, `train.py`, `evaluate.py` | 重跑确认 Activity MLP MSE 略升 |
| **P2** | 无固定 split 文件 | 生成 `train_users.txt` / `test_users.txt` 保存分割结果 | `train.py` | 确保分割可复现 |
| **P2** | 无验证集 | 从 train_df 中再分出 10% 用户作 val | `train.py` | 用于早停和超参选择 |
