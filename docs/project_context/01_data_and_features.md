# 01 — 数据来源与特征工程

> 项目路径: `E:\学习\推荐系统\project1_GPU\project1`
> 审计日期: 2026-07
> 数据状态: QUICK_RUN=False（全量 67 万条，已训练并产出 checkpoint）

---

## 1. 数据文件总览

### 1.1 实际读取的数据文件

| 文件路径 | 用途 | 行数 | 列数 | 关键字段 | 含 Rating | 用于训练 | 用于评估 | 备注 |
|----------|------|-----:|-----:|----------|:---:|:---:|:---:|------|
| `training set/training_total.csv` | 主训练数据 | 670,168 | 10 | MovieID, UserID, Rating, Gender, Age, Occupation, Genres | ✅ | ✅ | ✅ (从中 sample test) | 训练+评估共享同一文件 |
| `test set/test_total.csv` | 独立测试集 | 330,041 | 8 | UserID, Rating, Gender, Age, Occupation, Genres | ✅ | ❌ | ❌ (未被使用) | 无 MovieID，无法用于当前模型 |
| `training set/movie_audience_vector_net_attitude.csv` | 电影受众特征 | 3,668 | 8 | MovieID, Vector_Male_Dim, Vector_Female_Dim, Vector_Age_1-34, Vector_Age_35-55, Vector_Age_56+ | ❌ | ✅ | ✅ | 静态特征，不含 Rating |
| `training set/training_user_genre_preference.csv` | 用户类型偏好 | 4,026 | 19 | UserID + 18 个类型偏好分 | ❌ | ✅ | ✅ | 静态特征，不含 Rating |
| `test set/test_user_genre_preference.csv` | 测试集用户偏好 | 2,012 | 18 | 18 个类型偏好分（无 UserID 列） | ❌ | ❌ | ❌ (未被使用) | 缺 UserID，无法直接匹配 |
| `group_0_data.csv` – `group_4_data.csv` | KMeans 用户分组 | 491+1277+525+922+811 = 4,026 | 20 | user_id, Group + 18 类型偏好 | ❌ | ✅ | ✅ | 用户聚类结果 |
| `group_0_test.csv` – `group_4_test.csv` | 测试集用户分组 | 804+213+463+202+330 = 2,012 | 3 | user_id, 0, Group | ❌ | ❌ | ❌ (未被使用) | 仅 3 列，似乎只含 group 编号 |
| `checkpoints/svd_model.pkl` | SVD checkpoint | — | — | — | — | — | — | pickle，3,213 KB |
| `checkpoints/mlp_residual.pth` | MLP 残差模型 | — | — | — | — | — | — | PyTorch state_dict，106 KB |
| `checkpoints/activity_mlp.pth` | Activity MLP | — | — | — | — | — | — | PyTorch state_dict，56 KB |
| `checkpoints/cf_model.pkl` | UserCF 模型 | — | — | — | — | — | — | pickle (含评分矩阵)，228 MB |
| `reports/meta.json` | 训练元信息 | — | — | — | — | — | — | 179 B |
| `reports/results.json` | 评估结果 | — | — | — | — | — | — | 349 B |

### 1.2 未被代码引用但位于项目中的文件

| 文件 | 是否被任何 .py 文件 import/read |
|------|:---:|
| `test set/test_total.csv` | ❌ (config.py 定义了 TEST_DATA_PATH 但 train.py/evaluate.py 均未使用) |
| `test set/test_user_genre_preference.csv` | ❌ (config.py 定义了 TEST_USER_VEC_PATH 但未使用) |
| `group_*_test.csv` | ❌ (config.py 定义了 TEST_GROUP_FILE_PATHS 但未使用) |
| `test_user_groups.csv` | ❌ 未使用 |
| `knn_model.pkl` | ❌ (config.py 定义了 KNN_PATH 但未使用) |

---

## 2. 数据字段说明

### 2.1 训练集字段 (`training_total.csv`)

| 字段名 | 含义 | 类型 | 作为模型输入 | 作为标签 | 由 Rating 派生 |
|--------|------|------|:---:|:---:|:---:|
| MovieID | 电影唯一 ID | int | ❌ (仅用于查找) | ❌ | ❌ |
| Title | 电影标题 | str | ❌ | ❌ | ❌ |
| Genres | 电影类型（管道分隔，如 `Animation\|Children's\|Comedy`） | str | ❌ (未直接使用) | ❌ | ❌ |
| UserID | 用户唯一 ID | int | ❌ (仅用于查找) | ❌ | ❌ |
| **Rating** | 评分 (1–5) | float | ❌ | ✅ (所有模型) | — |
| Gender | 性别 (F/M) | str | ✅ (→ 2 维 one-hot) | ❌ | ❌ |
| Age | 年龄 (1,18,25,35,45,50,56) | int | ✅ (→ 3 维 one-hot) | ❌ | ❌ |
| Occupation | 职业编码 (0–20) | int | ❌ (未使用!) | ❌ | ❌ |
| Timestamp | 时间戳 | int | ❌ | ❌ | ❌ |
| Zip-code | 邮编 | str | ❌ | ❌ | ❌ |

**关键发现：Occupation（职业）在 config.py 和 Dataset 中均未被使用作模型输入。** 代码 `data/dataset.py:49` 构建特征时只用了 Gender 和 Age，`Occupation` 和 `Zip-code` 被完全忽略。

### 2.2 独立测试集字段 (`test_total.csv`)

与训练集相比 **缺少 `MovieID` 和 `Title`**，这意味着当前依赖 MovieID 的模型（SVD/CF/MLP）无法直接使用该文件做评估。

### 2.3 电影受众特征 (`movie_audience_vector_net_attitude.csv`)

| 字段名 | 含义 |
|--------|------|
| MovieID | 电影 ID |
| Vector_Male_Dim | 男性受众比例 |
| Vector_Female_Dim | 女性受众比例 |
| Vector_Age_1-34 | 18–34 岁受众比例 |
| Vector_Age_35-55 | 35–55 岁受众比例 |
| Vector_Age_56+ | 56 岁以上受众比例 |

> 注意：这些是**电影层面的全局统计特征**，不依赖于某个特定用户的评分。它们是从外部数据（"网络态度"）预先计算的，**不包含本项目的 Rating 信息**。

### 2.4 用户类型偏好 (`training_user_genre_preference.csv`)

18 列，每列是一个电影类型（Action, Adventure, ..., Western），值为该用户对该类型的偏好分（归一化到 [0,1]）。

> 关键问题：此文件的来源和计算方式**未确认**。代码中只是 `pd.read_csv` 读取，没有在项目内重新计算。如果原始数据使用了测试集用户的评分来生成偏好向量，则存在泄漏。但从文件名和列结构看，这更可能是外部预处理的数据，**与 Rating 的关联性未确认**。

---

## 3. 特征工程总览

### 3.1 特征构成

模型输入特征由以下部分拼接，定义在 `data/dataset.py:78-91`：

#### 主模型 (SVD + MLP): 129 维

| 特征组 | 维度 | 代码位置 | 来源 | 计算方式 | 含 Rating | 训练时计算 | 评估时计算 | 泄漏风险 |
|--------|:---:|----------|------|----------|:---:|------|------|:---:|
| 电影受众向量 | 5 | `dataset.py:24-26` | `movie_audience_vector_net_attitude.csv` | 直接查表 | ❌ | 读 CSV | 读 CSV | OK |
| 用户类型偏好 | 18 | `dataset.py:30-32` | `training_user_genre_preference.csv` | 直接查表 | ❌ | 读 CSV | 读 CSV | **未确认** |
| 性别 one-hot | 2 | `dataset.py:38` | df["Gender"] | M→[1,0], F→[0,1] | ❌ | 从 df 取 | 从 test_df 取 | OK |
| 年龄 one-hot | 3 | `dataset.py:37` | df["Age"] | ≤34→[1,0,0], 35-55→[0,1,0], 56+→[0,0,1] | ❌ | 从 df 取 | 从 test_df 取 | OK |
| **group_avg** | 1 | `dataset.py:47-55` | df["Rating"] | 组内该电影平均评分 | ✅ | 从 df 计算 | **从 test_df 重新计算** | 🔴 **P0** |
| SVD user factor | 50 | `dataset.py:86` | SVD.P[u] | SVD fit 后查表 | ✅ (间接) | 从 pre-fit SVD | 从 pre-fit SVD | 取决于 SVD fit 数据 |
| SVD item factor | 50 | `dataset.py:86` | SVD.Q[i] | SVD fit 后查表 | ✅ (间接) | 从 pre-fit SVD | 从 pre-fit SVD | 取决于 SVD fit 数据 |

#### Activity MLP: 30 维

| 特征组 | 维度 | 说明 |
|--------|:---:|------|
| 基础特征 (同上 1-5 组) | 29 | 同上，但无 SVD 因子 |
| **activity_index** | 1 | 用户活跃度，`log2(1+n_ratings/20)` 归一化 |

### 3.2 Occupation（职业）未被使用

尽管训练数据包含 `Occupation` 字段（21 个不同值），**当前代码中没有任何地方将其编码为模型输入**。`data/dataset.py` 中构建特征时只取 Gender 和 Age。

---

## 4. 重点特征审计

### 4.1 group_avg

- **代码位置**: `data/dataset.py:47-55`
- **定义**: 用户所在 KMeans 组对该电影的平均评分
  ```python
  # dataset.py:47-55
  self.group_avg = {}
  for gpath in GROUP_FILE_PATHS:
      if os.path.exists(gpath):
          gdf = pd.read_csv(gpath)
          gids = set(gdf["user_id"])
          gr = df[df["UserID"].isin(gids)]  # ← df 是传入的 DataFrame
          for mid, avg in gr.groupby("MovieID")["Rating"].mean().items():
              self.group_avg[(gpath, int(mid))] = float(avg)
  ```
- **使用 Rating**: ✅。使用传入 df 的 Rating 列计算 groupby mean。
- **训练时**: 传入 `train_df`（与 SVD/MLP 训练用的同一份 df），当前样本自己的 Rating 被包含在 groupby mean 中 → **P0 泄漏**
- **评估时**: `evaluator.py:56` 创建 `RatingDataset(test_df, ...)`，其中 test_df 是评估用的测试集 → `self.group_avg` 被 test_df 的 Rating 重新计算 → **P0 泄漏**（测试集评分泄露到输入特征）
- **若移除 group_avg**: 模型仍可运行，特征维度从 129 → 128（或 30 → 29）

### 4.2 activity_index

- **代码位置**: `utils/activity_index.py:12-15`；调用处 `train.py:37`
- **计算方式**:
  ```python
  # activity_index.py:12-15
  def compute_activity_index(data_path: str) -> dict:
      df = pd.read_csv(data_path)  # ← DATA_PATH = training_total.csv (全量 67 万)
      counts = df.groupby("UserID").size()
      index_vals = np.log2(1.0 + counts.values / WEIGHT_SCALE)
      index_vals = index_vals / index_vals.max()
      return {int(uid): float(v) for uid, v in zip(counts.index, index_vals)}
  ```
- **是否从全量数据计算**: ✅。`train.py:37` 调用 `compute_activity_index(DATA_PATH)`，读取完整 `training_total.csv`（包含后来被 sample 为 test_df 的行）。`evaluate.py:28` 同样从全量计算。
- **是否含测试集用户统计**: ✅。test_df 的用户也在 `DATA_PATH` 中，其评分次数被计入 activity_index。
- **是否使用 Rating 值**: ❌。只使用评分次数（count），不使用评分值。
- **严格评估**: 应该仅从训练集统计。

### 4.3 用户偏好 / 电影受众特征

| 特征 | 来源 | Rating 相关 | Split 前预计算 | 泄漏评估 |
|------|------|:---:|:---:|------|
| 电影受众 (5维) | `movie_audience_vector_net_attitude.csv` | ❌ | ✅ (外部文件) | **OK** — 静态特征，与评分无关 |
| 用户类型偏好 (18维) | `training_user_genre_preference.csv` | **未确认** | ✅ (外部文件) | **未确认** — 源数据生成方式未在项目中说明；若偏好分来自评分统计，则取决于是否用了测试集评分 |
| 性别/年龄 (5维) | 传入 df | ❌ | ❌ (从 df 实时取) | **OK** — 人口特征，非目标变量 |

---

## 5. 数据覆盖度

| 检查项 | 结果 |
|--------|------|
| 训练用户数 | 4,027 |
| 独立测试集用户数 | 2,013 |
| 两集合用户重叠 | **0** |
| 用户偏好覆盖训练集用户 | 4,026 / 4,027（缺 1 人） |
| 电影向量覆盖训练集电影 | 3,668 / 3,625 + 21 部未覆盖 → 有 3,625 部训练电影中，部分不在向量文件中 |
| 分组覆盖训练集用户 | 4,026 / 4,027（缺 1 人） |
| 独立测试集是否有 MovieID | ❌ — 无法直接用于当前模型 |

> 独立测试集与训练集用户零重叠，且无 MovieID。这提示该测试集设计为"冷启动用户 + 仅凭 Genres 预测"场景，与当前模型的 MovieID 依赖不兼容。
