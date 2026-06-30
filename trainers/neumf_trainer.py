"""
NeuMF 训练器 — 使用统一列名 (user_id, item_id, rating)
支持: create_dataloader, tqdm, AMP, non_blocking
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset
import numpy as np
from utils.dataloader import create_dataloader, to_device
from utils.progress import make_epoch_pbar, make_model_pbar


class NeuMFDataset(Dataset):
    def __init__(self, df, user2idx, item2idx):
        self.users = torch.tensor([user2idx.get(int(uid), 0) for uid in df["user_id"]], dtype=torch.long)
        self.items = torch.tensor([item2idx.get(int(iid), 0) for iid in df["item_id"]], dtype=torch.long)
        self.ratings = torch.tensor(df["rating"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]


def collate_neumf(batch):
    u, i, r = zip(*batch)
    return torch.stack(u), torch.stack(i), torch.stack(r)


def train_neumf(train_df, val_df, device=None, verbose=True,
                epochs=5, lr=0.005, batch_size=2048,
                embedding_dim=32, mlp_layers=(64, 32, 16), dropout=0.2,
                save_path=None, runtime_cfg=None):
    from models.neumf import NeuMF
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rcfg = runtime_cfg or {}
    from utils.performance import get_amp_context

    # Build user/item mappings (reserve idx=0 for <UNK>)
    all_users = sorted(train_df["user_id"].unique())
    all_items = sorted(train_df["item_id"].unique())
    user2idx = {int(u): i + 1 for i, u in enumerate(all_users)}
    item2idx = {int(m): i + 1 for i, m in enumerate(all_items)}

    n_users = len(user2idx) + 1
    n_items = len(item2idx) + 1

    train_ds = NeuMFDataset(train_df, user2idx, item2idx)
    train_loader = create_dataloader(train_ds, batch_size=batch_size, shuffle=True,
                                      runtime_cfg=rcfg, collate_fn=collate_neumf)

    model = NeuMF(n_users, n_items, embedding_dim, mlp_layers, dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    autocast_fn, scaler = get_amp_context(rcfg, device)

    if verbose:
        print(f"[NeuMF] users={n_users}, items={n_items}, emb={embedding_dim}, "
              f"mlp={mlp_layers}, epochs={epochs}, batches={len(train_loader)}")

    model.train()
    history = []
    t0 = time.time()
    best_val = float("inf")
    best_state = None
    global_step = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        pbar = make_epoch_pbar(epoch, epochs, len(train_loader), rcfg,
                               desc_prefix="[NeuMF] ")
        for u, i, r in train_loader:
            u, i, r = to_device((u, i, r), device)
            optimizer.zero_grad()
            with autocast_fn():
                pred = model(u, i)
                loss = criterion(pred, r)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            history.append(loss.item())
            epoch_loss += loss.item()
            global_step += 1
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.3f}", avg=f"{epoch_loss/pbar.n:.3f}")
        pbar.close()

        avg_loss = epoch_loss / max(len(train_loader), 1)
        if verbose:
            print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")

        if val_df is not None and len(val_df) > 0:
            val_ds = NeuMFDataset(val_df, user2idx, item2idx)
            val_loader = create_dataloader(val_ds, batch_size=batch_size, shuffle=False,
                                            runtime_cfg=rcfg, collate_fn=collate_neumf)
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for u, i, r in val_loader:
                    u, i, r = to_device((u, i, r), device)
                    val_loss += criterion(model(u, i), r).item() * u.size(0)
            val_loss /= len(val_ds)
            model.train()
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    if verbose:
        print(f"  NeuMF done: time={time.time()-t0:.0f}s, best_val_loss={best_val:.4f}")

    if save_path:
        save_data = {
            "state_dict": model.state_dict(),
            "n_users": n_users, "n_items": n_items,
            "user2idx": user2idx, "item2idx": item2idx,
            "embedding_dim": embedding_dim, "mlp_layers": mlp_layers, "dropout": dropout,
        }
        torch.save(save_data, save_path)

    return model, {"loss": sum(history) / max(len(history), 1), "history": history,
                   "best_val_loss": best_val, "n_users": n_users, "n_items": n_items,
                   "user2idx": user2idx, "item2idx": item2idx}
