"""
DeepFM 训练器 — 优化版: create_dataloader, tqdm, AMP, non_blocking, 动态特征
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset
import numpy as np, pandas as pd
from utils.dataloader import create_dataloader, to_device
from utils.progress import make_epoch_pbar, make_model_pbar

_DEFAULT_MOVIE_VEC_PATHS = [
    "data/raw/ml1m/movie_audience_vector_net_attitude.csv",
    "training set/movie_audience_vector_net_attitude.csv",
]
_DEFAULT_USER_VEC_PATHS = [
    "data/raw/ml1m/training_user_genre_preference.csv",
    "training set/training_user_genre_preference.csv",
]
_DEFAULT_GROUP_FILE_PATHS = []


def _compute_movie_vec(train_df: pd.DataFrame) -> dict:
    movie_map = {}
    has_gender = "gender" in train_df.columns
    has_age = "age" in train_df.columns
    if not has_gender and not has_age:
        return movie_map
    cols = {"item_id": train_df["item_id"], "total": 1}
    if has_gender:
        cols["n_male"] = (train_df["gender"] == "M").astype(float)
        cols["n_female"] = (train_df["gender"] == "F").astype(float)
    else:
        cols["n_male"] = 0.0; cols["n_female"] = 0.0
    if has_age:
        age = pd.to_numeric(train_df["age"], errors="coerce").fillna(30)
        cols["n_age_1_34"] = (age <= 34).astype(float)
        cols["n_age_35_55"] = ((age > 34) & (age <= 55)).astype(float)
        cols["n_age_56"] = (age > 55).astype(float)
    else:
        cols["n_age_1_34"] = 0.0; cols["n_age_35_55"] = 0.0; cols["n_age_56"] = 0.0
    df_tmp = pd.DataFrame(cols)
    grp = df_tmp.groupby("item_id").sum()
    for iid, row in grp.iterrows():
        t = row["total"]
        if t > 0:
            movie_map[int(iid)] = np.array([
                row["n_male"] / t, row["n_female"] / t,
                row["n_age_1_34"] / t, row["n_age_35_55"] / t, row["n_age_56"] / t,
            ], dtype=np.float32)
    return movie_map


def _compute_user_pref(train_df: pd.DataFrame) -> tuple:
    user_pref = {}
    genre_cols = sorted([c for c in train_df.columns if c.startswith("genre_")])
    if genre_cols:
        grp = train_df.groupby("user_id")[genre_cols].mean()
        for uid, row in grp.iterrows():
            user_pref[int(uid)] = row.values.astype(np.float32)
        return user_pref, len(genre_cols)
    if "genre" in train_df.columns:
        genres_dummy = train_df["genre"].astype(str).str.get_dummies()
        genre_cols = sorted(genres_dummy.columns)
        df_tmp = pd.concat([train_df[["user_id"]], genres_dummy], axis=1)
        grp = df_tmp.groupby("user_id")[genre_cols].mean()
        for uid, row in grp.iterrows():
            user_pref[int(uid)] = row.values.astype(np.float32)
        return user_pref, len(genre_cols)
    return user_pref, 0


class DeepFMDataset(Dataset):
    def __init__(self, df, user2idx, item2idx, gender2idx, age2idx, occ2idx,
                 movie_map, user_pref, activity_map, group_stats, movie_mean_map,
                 global_mean, use_occupation=True, group_avg_mode="train_loo",
                 movie_vec_dim=5, user_pref_dim=0):
        self.df = df.reset_index(drop=True)
        self.user2idx = user2idx; self.item2idx = item2idx
        self.gender2idx = gender2idx; self.age2idx = age2idx; self.occ2idx = occ2idx
        self.movie_map = movie_map; self.user_pref = user_pref
        self.activity_map = activity_map; self.group_stats = group_stats
        self.movie_mean_map = movie_mean_map; self.global_mean = global_mean
        self.use_occupation = use_occupation
        self.group_avg_mode = group_avg_mode
        self.movie_vec_dim = movie_vec_dim; self.user_pref_dim = user_pref_dim
        self.user_group = {}
        for gpath in _DEFAULT_GROUP_FILE_PATHS:
            if os.path.exists(gpath):
                gdf = pd.read_csv(gpath)
                for uid in gdf["user_id"]:
                    self.user_group[int(uid)] = gpath

    def _get_group_avg(self, gn, mid, rating):
        key = (gn, mid)
        if self.group_avg_mode == "train_loo" and key in self.group_stats:
            s = self.group_stats[key]["sum"]; c = self.group_stats[key]["count"]
            if c > 1: return (s - rating) / (c - 1)
        if key in self.group_stats: return self.group_stats[key]["mean"]
        return self.movie_mean_map.get(mid, self.global_mean)

    def _get_age_bucket(self, age):
        if age <= 34: return 1
        if age <= 55: return 2
        return 3

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        uid, mid = int(row["user_id"]), int(row["item_id"])
        rating = float(row["rating"])
        gender = row.get("gender", "unknown")
        age = row.get("age", 30)
        occ = int(row.get("occupation", 0)) if not pd.isna(row.get("occupation", 0)) else 0
        sparse = [
            torch.tensor(self.user2idx.get(uid, 0), dtype=torch.long),
            torch.tensor(self.item2idx.get(mid, 0), dtype=torch.long),
            torch.tensor(self.gender2idx.get(gender, 0), dtype=torch.long),
            torch.tensor(self.age2idx.get(self._get_age_bucket(age), 0), dtype=torch.long),
        ]
        if self.use_occupation:
            sparse.append(torch.tensor(self.occ2idx.get(occ, 0), dtype=torch.long))
        mv = self.movie_map.get(mid, np.zeros(self.movie_vec_dim, dtype=np.float32))
        up = self.user_pref.get(uid, np.zeros(self.user_pref_dim, dtype=np.float32))
        gn = self.user_group.get(uid, "")
        ga = self._get_group_avg(gn, mid, rating) if gn else self.global_mean
        act = self.activity_map.get(uid, 0.0)
        dense = np.concatenate([mv, up, [ga], [act]]).astype(np.float32)
        return sparse, torch.tensor(dense, dtype=torch.float32), torch.tensor(rating, dtype=torch.float32)


def collate_deepfm(batch):
    n_sparse = len(batch[0][0])
    sparse_batch = [torch.stack([b[0][i] for b in batch]) for i in range(n_sparse)]
    return sparse_batch, torch.stack([b[1] for b in batch]), torch.stack([b[2] for b in batch])


def train_deepfm(train_df, val_df, activity_map, group_stats, movie_mean_map,
                 global_mean, device=None, verbose=True,
                 use_occupation=True, epochs=5, lr=0.005, batch_size=2048,
                 deep_layers=(128, 64), dropout=0.3, save_path=None,
                 runtime_cfg=None):
    from models.deepfm import DeepFM, FeaturesConfig
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rcfg = runtime_cfg or {}
    from utils.performance import get_amp_context

    all_users = sorted(train_df["user_id"].unique())
    all_items = sorted(train_df["item_id"].unique())
    user2idx = {int(u): i + 1 for i, u in enumerate(all_users)}
    item2idx = {int(m): i + 1 for i, m in enumerate(all_items)}
    gender2idx = {"F": 1, "M": 2}
    age2idx = {1: 1, 2: 2, 3: 3}
    occ2idx = {}
    if use_occupation and "occupation" in train_df.columns:
        all_occ = sorted(train_df["occupation"].dropna().astype(int).unique())
        occ2idx = {int(o): i + 1 for i, o in enumerate(all_occ)}
    else:
        use_occupation = False

    is_ml1m = any(c.startswith("genre_") for c in train_df.columns)
    movie_map = {}
    if is_ml1m:
        for mv_path in _DEFAULT_MOVIE_VEC_PATHS:
            if os.path.exists(mv_path):
                mv = pd.read_csv(mv_path)
                for _, r in mv.iterrows():
                    movie_map[int(r["MovieID"])] = np.array(
                        [r["Vector_Male_Dim"], r["Vector_Female_Dim"],
                         r["Vector_Age_1-34"], r["Vector_Age_35-55"], r["Vector_Age_56+"]],
                        dtype=np.float32)
                break
    if not movie_map:
        movie_map = _compute_movie_vec(train_df)
    movie_vec_dim = 5

    user_pref = {}; user_pref_dim = 0
    if is_ml1m:
        for uv_path in _DEFAULT_USER_VEC_PATHS:
            if os.path.exists(uv_path):
                uv = pd.read_csv(uv_path)
                for _, r in uv.iterrows():
                    user_pref[int(r["UserID"])] = r.drop("UserID").values.astype(np.float32)
                break
        if user_pref:
            user_pref_dim = len(next(iter(user_pref.values())))
    if not user_pref:
        user_pref, user_pref_dim = _compute_user_pref(train_df)

    feat_config = FeaturesConfig(
        len(user2idx), len(item2idx), use_occupation,
        movie_vec_dim=movie_vec_dim, user_pref_dim=user_pref_dim,
    )
    model = DeepFM(feat_config, deep_layers, dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    autocast_fn, scaler = get_amp_context(rcfg, device)

    ds_kwargs = dict(
        user2idx=user2idx, item2idx=item2idx,
        gender2idx=gender2idx, age2idx=age2idx, occ2idx=occ2idx,
        movie_map=movie_map, user_pref=user_pref,
        activity_map=activity_map, group_stats=group_stats,
        movie_mean_map=movie_mean_map, global_mean=global_mean,
        use_occupation=use_occupation,
        movie_vec_dim=movie_vec_dim, user_pref_dim=user_pref_dim,
    )

    train_ds = DeepFMDataset(train_df, group_avg_mode="train_loo", **ds_kwargs)
    train_loader = create_dataloader(train_ds, batch_size=batch_size, shuffle=True,
                                      runtime_cfg=rcfg, collate_fn=collate_deepfm)

    if verbose:
        print(f"[DeepFM] sparse={len(feat_config.sparse_fields)}, "
              f"dense_dim={feat_config.dense_dim}, deep={deep_layers}, "
              f"epochs={epochs}, batches={len(train_loader)}")

    model.train(); history = []; t0 = time.time()
    best_val = float("inf"); best_state = None; global_step = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        pbar = make_epoch_pbar(epoch, epochs, len(train_loader), rcfg,
                               desc_prefix="[DeepFM] ")
        for sparse, dense, r in train_loader:
            sparse = to_device(sparse, device)
            dense, r = dense.to(device, non_blocking=True), r.to(device, non_blocking=True)
            optimizer.zero_grad()
            with autocast_fn():
                pred = model(sparse, dense)
                loss = criterion(pred, r)
            if scaler:
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            else:
                loss.backward(); optimizer.step()
            history.append(loss.item()); epoch_loss += loss.item(); global_step += 1
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.3f}", avg=f"{epoch_loss/pbar.n:.3f}")
        pbar.close()

        avg = epoch_loss / max(len(train_loader), 1)
        if verbose:
            print(f"  Epoch {epoch+1}/{epochs}: loss={avg:.4f}")

        if val_df is not None and len(val_df) > 0:
            val_ds = DeepFMDataset(val_df, group_avg_mode="train_stats", **ds_kwargs)
            val_loader = create_dataloader(val_ds, batch_size=batch_size, shuffle=False,
                                            runtime_cfg=rcfg, collate_fn=collate_deepfm)
            model.eval(); val_loss = 0.0
            with torch.no_grad():
                for sparse, dense, r in val_loader:
                    sparse = to_device(sparse, device)
                    dense, r = dense.to(device), r.to(device)
                    val_loss += criterion(model(sparse, dense), r).item() * dense.size(0)
            val_loss /= len(val_ds); model.train()
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    if verbose:
        print(f"  DeepFM done: time={time.time()-t0:.0f}s, best_val={best_val:.4f}")

    if save_path:
        save_data = {
            "state_dict": model.state_dict(),
            "user2idx": user2idx, "item2idx": item2idx,
            "gender2idx": gender2idx, "age2idx": age2idx, "occ2idx": occ2idx,
            "use_occupation": use_occupation, "deep_layers": deep_layers, "dropout": dropout,
            "movie_vec_dim": movie_vec_dim, "user_pref_dim": user_pref_dim,
        }
        torch.save(save_data, save_path)

    return model, {
        "loss": sum(history) / max(len(history), 1), "history": history,
        "best_val_loss": best_val, "user2idx": user2idx, "item2idx": item2idx,
        "gender2idx": gender2idx, "age2idx": age2idx, "occ2idx": occ2idx,
        "use_occupation": use_occupation,
        "movie_vec_dim": movie_vec_dim, "user_pref_dim": user_pref_dim,
    }
