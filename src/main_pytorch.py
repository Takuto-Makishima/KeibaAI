import os
import random
import numpy as np
import pandas as pd
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import optuna

from src.creators.ai_creator import AiCreator

# ---------------------------
# Reproducibility helpers
# ---------------------------
def set_seed(seed:int=13):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)

def get_dataloader(dataset, batch_size=1, shuffle=False, seed=42, num_workers=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda x: x[0],
        worker_init_fn=seed_worker,
        generator=g,
        num_workers=num_workers
    )

# ---------------------------
# Ranking utilities
# ---------------------------
class RankingUtils:
    @staticmethod
    def neural_sort(scores, tau=1.0):
        single = False
        if scores.dim() == 1:
            scores = scores.unsqueeze(0)
            single = True
        b, n = scores.shape
        scores_i = scores.unsqueeze(2)
        scores_j = scores.unsqueeze(1)
        abs_diff = torch.abs(scores_i - scores_j)
        logits = -abs_diff
        P_hat = F.softmax(logits / (tau + 1e-12), dim=-1)
        return P_hat.squeeze(0) if single else P_hat

    @staticmethod
    def top_k_probabilities_from_P(P_hat, k=3):
        single = False
        if P_hat.dim() == 2:
            P_hat = P_hat.unsqueeze(0)
            single = True
        topk_mask = torch.zeros_like(P_hat)
        topk_mask[:, :, :k] = 1.0
        probs = (P_hat * topk_mask).sum(dim=-1)
        return probs.squeeze(0) if single else probs

    @staticmethod
    def top_k_probabilities(scores, k=3, tau=1.0):
        single = False
        if scores.dim() == 1:
            scores = scores.unsqueeze(0)
            single = True
        P_hat = RankingUtils.neural_sort(scores, tau=tau)
        probs = RankingUtils.top_k_probabilities_from_P(P_hat, k=k)
        return probs.squeeze(0) if single else probs

    @staticmethod
    def listmle_loss(scores, true_ranks):
        scores = scores.float()
        b, n = scores.shape
        losses = []
        for bi in range(b):
            s = scores[bi]
            ranks = true_ranks[bi]
            order = torch.argsort(ranks)
            s_ord = s[order]
            total = 0.0
            for i in range(n):
                denom = torch.logsumexp(s_ord[i:], dim=0)
                total = total + (denom - s_ord[i])
            losses.append(total)
        return torch.stack(losses).mean()

# ---------------------------
# Dataset for races
# ---------------------------
class RaceDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, numeric_cols: list, categorical_cols: list, target_col: str = "order"):
        self.df = dataframe
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.target_col = target_col
        self.groups = {rid: group.reset_index(drop=True) for rid, group in dataframe.groupby("race_id")}
        self.race_ids = list(self.groups.keys())

    def __len__(self):
        return len(self.race_ids)

    def __getitem__(self, idx):
        race_id = self.race_ids[idx]
        df_r = self.groups[race_id]

        # 数値特徴量
        if self.numeric_cols:
            numeric = df_r[self.numeric_cols].to_numpy(dtype=np.float32)
        else:
            numeric = np.zeros((len(df_r), 0), dtype=np.float32)

        # カテゴリ特徴量
        if self.categorical_cols:
            categorical = df_r[self.categorical_cols].to_numpy(dtype=np.int64)
        else:
            categorical = np.zeros((len(df_r), 1), dtype=np.int64)

        # 順位（ターゲット）
        ranks = df_r[self.target_col].to_numpy(dtype=np.int64)

        return {
            "numeric_feats": torch.from_numpy(numeric),       # float32
            "categorical_feats": torch.from_numpy(categorical), # int64
            "ranks": torch.from_numpy(ranks)                  # int64
        }

# ---------------------------
# 修正版 SetRankTransformer
# ---------------------------
class SetRankTransformer(nn.Module):
    def __init__(self, num_numeric, cat_cardinalities: list, cat_emb_dim=8,
                 hidden_dim=128, n_heads=4, n_layers=2, dropout=0.1, verbose=True):
        super().__init__()
        self.num_numeric = num_numeric
        self.cat_cardinalities = cat_cardinalities
        self.cat_emb_dim = cat_emb_dim

        self.cat_embs = nn.ModuleList([nn.Embedding(c, cat_emb_dim) for c in cat_cardinalities])
        cat_total_dim = cat_emb_dim * max(1, len(cat_cardinalities))
        self.num_proj = nn.Linear(num_numeric, cat_total_dim) if num_numeric > 0 else None
        self.d_model = cat_total_dim

        if self.d_model % n_heads != 0:
            if verbose:
                print(f"[Warning] d_model={self.d_model} は n_heads={n_heads} で割り切れないため n_heads=1 に変更します。")
            n_heads = 1
        assert self.d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "mha": nn.MultiheadAttention(embed_dim=self.d_model, num_heads=n_heads, batch_first=True, dropout=dropout),
                "ff": nn.Sequential(
                    nn.Linear(self.d_model, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, self.d_model)
                ),
                "ln1": nn.LayerNorm(self.d_model),
                "ln2": nn.LayerNorm(self.d_model)
            })
            for _ in range(n_layers)
        ])

        self.scorer = nn.Linear(self.d_model, 1)
        self.last_attn_weights = None

        if verbose and self.d_model > 1024:
            print(f"[Info] 特徴量次元 d_model={self.d_model} はかなり大きいです。計算・メモリ負荷が高くなる可能性があります。")

    def forward(self, numeric_feats, categorical_feats, return_attn=False):
        batch = categorical_feats.shape[0]
        n = categorical_feats.shape[1]
        if categorical_feats.dim() == 2:
            cat_feats = categorical_feats.unsqueeze(-1)
        else:
            cat_feats = categorical_feats

        embeds = []
        for col in range(cat_feats.shape[2]):
            emb = self.cat_embs[col](cat_feats[:, :, col])
            embeds.append(emb)
        cat_concat = torch.cat(embeds, dim=-1) if embeds else torch.zeros((batch, n, self.d_model), device=cat_feats.device)

        if self.num_proj is not None and numeric_feats is not None:
            num_proj = self.num_proj(numeric_feats)
            x = cat_concat + num_proj
        else:
            x = cat_concat

        attn_weights_all = []
        for layer in self.layers:
            residual = x
            attn_out, attn_weights = layer["mha"](x, x, x, need_weights=True, average_attn_weights=False)
            x = layer["ln1"](residual + attn_out)
            residual2 = x
            ff_out = layer["ff"](x)
            x = layer["ln2"](residual2 + ff_out)
            attn_weights_all.append(attn_weights)

        self.last_attn_weights = attn_weights_all
        scores = self.scorer(x).squeeze(-1)
        if return_attn:
            return scores, attn_weights_all
        return scores

    def get_last_attention(self):
        return self.last_attn_weights

# ---------------------------
# Evaluator
# ---------------------------
class Evaluator:
    @staticmethod
    def ndcg_at_k_single(pred_scores, true_ranks, k=3):
        n = pred_scores.shape[0]
        k = min(k, n)
        _, pred_idx = torch.topk(pred_scores, k=k, largest=True)
        relevance = (true_ranks <= k).float()
        gains = relevance[pred_idx]
        discounts = torch.log2(torch.arange(2, 2 + k, device=pred_scores.device).float())
        dcg = (gains / discounts).sum().item()
        ideal_rel = torch.sort(relevance, descending=True).values[:k]
        idcg = (ideal_rel / discounts).sum().item()
        return 0.0 if idcg == 0 else dcg / idcg

    @staticmethod
    def precision_at_k_single(pred_scores, true_ranks, k=3):
        _, pred_idx = torch.topk(pred_scores, k=k, largest=True)
        true_topk = (true_ranks <= k).nonzero(as_tuple=False).squeeze(-1)
        if true_topk.numel() == 0:
            return 0.0
        hits = sum([1 for i in pred_idx if i in true_topk])
        return float(hits) / k

# ---------------------------
# Trainer
# ---------------------------
class Trainer:
    def __init__(self, model: SetRankTransformer, optimizer, device="cpu", tau=1.0, patience=5, save_path="best_model.pt"):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.tau = tau
        self.patience = patience
        self.save_path = save_path
        self.train_loss_history = []
        self.val_loss_history = []
        self.val_ndcg_history = []
        self.val_prec_history = []
        self.best_val_loss = float("inf")
        self.epochs_no_improve = 0

    def train_one_epoch(self, train_loader, val_loader=None, k=3):
        self.model.train()
        total_loss, count = 0.0, 0
        for race in train_loader:
            numeric = race["numeric_feats"].unsqueeze(0).to(self.device) if race["numeric_feats"].numel() > 0 else None
            categorical = race["categorical_feats"].unsqueeze(0).to(self.device)
            ranks = race["ranks"].unsqueeze(0).to(self.device)
            scores = self.model(numeric, categorical)
            loss = RankingUtils.listmle_loss(scores, ranks)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            count += 1
        avg_train_loss = total_loss / max(1, count)
        self.train_loss_history.append(avg_train_loss)

        if val_loader is not None:
            self.model.eval()
            val_loss, ndcgs, precs, vcount = 0.0, [], [], 0
            with torch.no_grad():
                for race in val_loader:
                    numeric = race["numeric_feats"].unsqueeze(0).to(self.device) if race["numeric_feats"].numel() > 0 else None
                    categorical = race["categorical_feats"].unsqueeze(0).to(self.device)
                    ranks = race["ranks"].to(self.device)
                    scores = self.model(numeric, categorical).squeeze(0)
                    loss = RankingUtils.listmle_loss(scores.unsqueeze(0), ranks.unsqueeze(0))
                    val_loss += loss.item()
                    ndcgs.append(Evaluator.ndcg_at_k_single(scores, ranks, k=k))
                    precs.append(Evaluator.precision_at_k_single(scores, ranks, k=k))
                    vcount += 1
            avg_val_loss = val_loss / max(1, vcount)
            avg_ndcg = sum(ndcgs) / max(1, len(ndcgs)) if ndcgs else 0.0
            avg_prec = sum(precs) / max(1, len(precs)) if precs else 0.0
            self.val_loss_history.append(avg_val_loss)
            self.val_ndcg_history.append(avg_ndcg)
            self.val_prec_history.append(avg_prec)

            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.epochs_no_improve = 0
                torch.save(self.model.state_dict(), self.save_path)
            else:
                self.epochs_no_improve += 1

            if self.epochs_no_improve >= self.patience:
                print(f"Early stopping triggered after {self.patience} epochs without improvement.")
                return "early_stop"
        return avg_train_loss

    def fit(self, train_loader, val_loader=None, epochs=20, k=3):
        for epoch in range(epochs):
            res = self.train_one_epoch(train_loader, val_loader, k=k)
            if res == "early_stop":
                break
            train_loss = self.train_loss_history[-1]
            val_loss = self.val_loss_history[-1] if self.val_loss_history else None
            val_ndcg = self.val_ndcg_history[-1] if self.val_ndcg_history else None
            val_prec = self.val_prec_history[-1] if self.val_prec_history else None
            print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_ndcg@{k}={val_ndcg:.4f}, val_prec@{k}={val_prec:.4f}")
        return {
            "train_loss_history": self.train_loss_history,
            "val_loss_history": self.val_loss_history,
            "val_ndcg_history": self.val_ndcg_history,
            "val_prec_history": self.val_prec_history
        }

    def plot_metrics(self):
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.plot(self.train_loss_history, label="train_loss")
        plt.plot(self.val_loss_history, label="val_loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid(True)
        plt.subplot(1,2,2)
        plt.plot(self.val_ndcg_history, label="val_ndcg@3")
        plt.plot(self.val_prec_history, label="val_prec@3")
        plt.xlabel("Epoch"); plt.ylabel("Metric"); plt.legend(); plt.grid(True)
        plt.show()

# ---------------------------
# Predictor
# ---------------------------
class Predictor:
    def __init__(self, model: SetRankTransformer, device="cpu", tau=1.0):
        self.model = model.to(device)
        self.device = device
        self.tau = tau

    def predict(self, race, k=3):
        self.model.eval()
        with torch.no_grad():
            numeric = race.get("numeric_feats", None)
            if numeric is not None and numeric.numel() > 0:
                numeric = numeric.unsqueeze(0).to(self.device)
            categorical = race["categorical_feats"].unsqueeze(0).to(self.device)
            scores, attn_weights = self.model(numeric, categorical, return_attn=True)
            attn_weights = self.model.get_last_attention()
            scores = scores.squeeze(0)
            topk_probs = RankingUtils.top_k_probabilities(scores.unsqueeze(0), k=k, tau=self.tau).squeeze(0)
        return scores.cpu().numpy(), topk_probs.cpu().numpy(), attn_weights

    def visualize(self, race, horse_names=None, k=3):
        scores, probs, attn_weights = self.predict(race, k=k)
        if horse_names is None:
            horse_names = [f"Horse {i+1}" for i in range(len(scores))]
        df = pd.DataFrame({"Horse": horse_names, "Score": scores, "TopK_Prob": probs})
        df = df.sort_values(by="Score", ascending=False).reset_index(drop=True)
        print(df)
        plt.figure(figsize=(8, max(4, len(scores)*0.3)))
        plt.barh(df["Horse"], df["TopK_Prob"])
        plt.xlabel(f"Probability of Top-{k}")
        plt.title("Prediction: Top-K Probabilities")
        plt.gca().invert_yaxis()
        plt.grid(True)
        plt.show()

# ---------------------------
# OptunaSearch (uses get_dataloader with seed param)
# ---------------------------
class OptunaSearch:
    def __init__(self, train_dataset, val_dataset, num_numeric, cat_cardinalities, device="cpu", max_epochs=10, patience=3, seed=42):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.num_numeric = num_numeric
        self.cat_cardinalities = cat_cardinalities
        self.device = device
        self.max_epochs = max_epochs
        self.patience = patience
        self.seed = seed

    def objective(self, trial):
        set_seed(self.seed)

        cat_emb_dim = trial.suggest_int("cat_emb_dim", 4, 32)
        hidden_dim = trial.suggest_int("hidden_dim", 64, 256)
        n_heads = trial.suggest_int("n_heads", 1, 8)
        n_layers = trial.suggest_int("n_layers", 1, 3)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        tau = trial.suggest_float("tau", 0.1, 2.0)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)

        model = SetRankTransformer(
            num_numeric=self.num_numeric,
            cat_cardinalities=self.cat_cardinalities,
            cat_emb_dim=cat_emb_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout
        ).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            device=self.device,
            tau=tau,
            patience=self.patience,
            save_path=f"best_model_trial_{trial.number}.pt"
        )

        train_loader = get_dataloader(self.train_dataset, batch_size=1, shuffle=True, seed=self.seed)
        val_loader = get_dataloader(self.val_dataset, batch_size=1, shuffle=False, seed=self.seed)

        best_val_ndcg = 0.0
        for epoch in range(self.max_epochs):
            result = trainer.train_one_epoch(train_loader, val_loader)
            if result == "early_stop":
                break
            if trainer.val_ndcg_history:
                best_val_ndcg = max(best_val_ndcg, trainer.val_ndcg_history[-1])

        return best_val_ndcg  # maximize NDCGなので正の値を返す

    def run(self, n_trials=20):
        study = optuna.create_study(direction="maximize")  # maximizeに変更
        study.optimize(self.objective, n_trials=n_trials)
        print("Best trial:", study.best_trial.params)
        return study.best_trial.params

# ---------------------------
# Example toy usage (if run as script)
# ---------------------------
if __name__ == "__main__":

    START_YEAR = 2025
    END_YEAR = 2026
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    #objective = 'lambdarank'  # 'rank_xendcg', 'lambdarank'

    CREATE_LIST = [
        # [datetime(2025,1,5).date(), datetime(2025,1,6).date()],
        # [datetime(2025,1,11).date(), datetime(2025,1,12).date(), datetime(2025,1,13).date()],
        # [datetime(2025,1,18).date(), datetime(2025,1,19).date()],
        # [datetime(2025,1,25).date(), datetime(2025,1,26).date()],
        # [datetime(2025,2,1).date(), datetime(2025,2,2).date()],
        # [datetime(2025,2,8).date(), datetime(2025,2,9).date(), datetime(2025,2,10).date()],
        # [datetime(2025,2,15).date(), datetime(2025,2,16).date()],
        # [datetime(2025,2,22).date(), datetime(2025,2,23).date()],
        # [datetime(2025,3,1).date(), datetime(2025,3,2).date()],
        # [datetime(2025,3,8).date(), datetime(2025,3,9).date()],
        # [datetime(2025,3,15).date(), datetime(2025,3,16).date()],
        # [datetime(2025,3,22).date(), datetime(2025,3,23).date()],
        # [datetime(2025,3,29).date(), datetime(2025,3,30).date()],
        # [datetime(2025,4,5).date(), datetime(2025,4,6).date()],
        # [datetime(2025,4,12).date(), datetime(2025,4,13).date()],
        # [datetime(2025,4,19).date(), datetime(2025,4,20).date()],
        # [datetime(2025,4,26).date(), datetime(2025,4,27).date()],
        # [datetime(2025,5,3).date(), datetime(2025,5,4).date()],
        # [datetime(2025,5,10).date(), datetime(2025,5,11).date()],
        # [datetime(2025,5,17).date(), datetime(2025,5,18).date()],
        # [datetime(2025,5,24).date(), datetime(2025,5,25).date()],
        # [datetime(2025,5,31).date(), datetime(2025,6,1).date()],
        # [datetime(2025,6,7).date(), datetime(2025,6,8).date()],
        # [datetime(2025,6,14).date(), datetime(2025,6,15).date()],
        [datetime(2025,6,21).date(), datetime(2025,6,22).date()],
        [datetime(2025,6,28).date(), datetime(2025,6,29).date()],
        [datetime(2025,7,5).date(), datetime(2025,7,6).date()],
        [datetime(2025,7,12).date(), datetime(2025,7,13).date()],
        [datetime(2025,7,19).date(), datetime(2025,7,20).date()],
        # [datetime(2025,7,26).date(), datetime(2025,7,27).date()],
        # [datetime(2025,8,2).date(), datetime(2025,8,3).date()],
    ]

    set_seed(13)  # 再現性の固定

    AiCreator.create_setrank_transformer_model(CREATE_LIST, 2024, END_YEAR, objective='rank_xendcg')

    # Create toy dataframe: 200 races, each with 6 horses, 3 numeric features, 2 categorical cols
    # rng = np.random.RandomState(0)
    # rows = []
    # for race_id in range(200):
    #     n_horses = 6
    #     for i in range(n_horses):
    #         numeric1 = rng.randn()
    #         numeric2 = rng.randn()
    #         numeric3 = rng.randn()
    #         cat1 = rng.randint(0, 10)
    #         cat2 = rng.randint(0, 8)
    #         rows.append({
    #             "race_id": race_id,
    #             "num1": numeric1,
    #             "num2": numeric2,
    #             "num3": numeric3,
    #             "cat1": cat1,
    #             "cat2": cat2
    #         })
    # df = pd.DataFrame(rows)
    # df_out = []
    # for rid, g in df.groupby("race_id"):
    #     g2 = g.sample(frac=1, random_state=rng)  # shuffle
    #     g2 = g2.reset_index(drop=True)
    #     g2["order"] = np.arange(1, len(g2) + 1)
    #     df_out.append(g2)
    # df_all = pd.concat(df_out, ignore_index=True)

    # numeric_cols = ["num1", "num2", "num3"]
    # categorical_cols = ["cat1", "cat2"]

    # race_ids = df_all["race_id"].unique()
    # train_ids = race_ids[:160]
    # val_ids = race_ids[160:]
   
    # train_df = df_all[df_all["race_id"].isin(train_ids)].reset_index(drop=True)
    # val_df = df_all[df_all["race_id"].isin(val_ids)].reset_index(drop=True)

    # train_ds = RaceDataset(train_df, numeric_cols, categorical_cols, target_col="order")
    # val_ds = RaceDataset(val_df, numeric_cols, categorical_cols, target_col="order")

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # opt = OptunaSearch(train_ds, val_ds, num_numeric=len(numeric_cols), cat_cardinalities=[10, 8], device=device, max_epochs=3, patience=2, seed=13)
    # best_params = opt.run(n_trials=3)

    # print("Done. best params:", best_params)
