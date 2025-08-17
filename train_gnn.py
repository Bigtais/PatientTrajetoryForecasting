import os
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm

# only import wandb on rank 0
USE_WANDB = True
try:
    import wandb
except Exception:
    USE_WANDB = False

from gnn_model import RGCNEncoder
from scoring import DistMultScorer


# -------------------------- DDP helpers --------------------------
def setup_ddp():
    """Initialize distributed process group + device selection."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        # Fallback to single-process (useful for debugging without torchrun)
        rank, world_size, local_rank = 0, 1, 0

    torch.cuda.set_device(local_rank)
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return rank, world_size, local_rank


def cleanup_ddp():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """Average a scalar tensor across ranks."""
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return tensor


# -------------------------- Model wrapper --------------------------
class RGCNWithRelations(nn.Module):
    """
    Wraps the RGCN encoder + learnable relation embeddings
    so that DDP can synchronize all trainable parameters.
    """
    def __init__(self, in_dim, hidden_dim, out_dim, num_rels):
        super().__init__()
        self.rgcn = RGCNEncoder(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, num_rels=num_rels)
        self.rel_emb = nn.Embedding(num_rels, out_dim)
        nn.init.xavier_uniform_(self.rel_emb.weight)

    def forward(self, x, edge_index, edge_type):
        entity_emb = self.rgcn(x, edge_index, edge_type)  # [num_nodes, out_dim]
        return entity_emb

    def relation_weights(self):
        return self.rel_emb.weight


# -------------------------- Main script --------------------------
def main():
    # ============== DDP ==============
    rank, world_size, local_rank = setup_ddp()

    # ============== Reproducibility ==============
    seed = 213033
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # ============== Config ==============
    embedding_dim = 256
    hidden_dim = 128
    feat_dim = 128
    batch_size = 32
    epochs = 50
    k_folds = 5
    margin = 1.0
    lr = 1e-3

    out_dir = Path("trained_embeddings")
    if is_main_process(rank):
        out_dir.mkdir(parents=True, exist_ok=True)

    # ============== Load triples ==============
    # columns: subject, predicate, object
    df = pd.read_csv("UMLS/umls_triples.csv")

    # ============== Build vocabularies ==============
    entities = sorted(set(df["subject"]) | set(df["object"]))
    relations = sorted(set(df["predicate"]))
    entity2id = {e: i for i, e in enumerate(entities)}
    rel2id = {r: i for i, r in enumerate(relations)}

    # ============== Build graph edges ==============
    edge_index = []
    edge_type = []
    triplets_id = []

    for _, row in df.iterrows():
        s_id = entity2id[row["subject"]]
        o_id = entity2id[row["object"]]
        r_id = rel2id[row["predicate"]]

        edge_index.append([s_id, o_id])
        edge_type.append(r_id)
        triplets_id.append([s_id, r_id, o_id])

    # Add inverse edges
    edge_index += [[dst, src] for src, dst in edge_index]
    edge_type += [r + len(rel2id) for r in edge_type]
    num_rels = len(rel2id) * 2

    edge_index = torch.tensor(edge_index, dtype=torch.long, device=device).t().contiguous()
    edge_type = torch.tensor(edge_type, dtype=torch.long, device=device)

    # ============== Node features ==============
    num_entities = len(entity2id)
    # learnable node features (shared across ranks)
    x = nn.Embedding(num_entities, feat_dim, device=device).weight  # [num_entities, feat_dim]

    # ============== K-Fold ==============
    triplets_id = np.array(triplets_id)
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)

    # ============== wandb ==============
    use_wandb = USE_WANDB and is_main_process(rank)
    if use_wandb:
        wandb.init(
            project="umls-rgcn-cv-ddp",
            config=dict(
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                feat_dim=feat_dim,
                batch_size=batch_size,
                epochs=epochs,
                k_folds=k_folds,
                lr=lr,
                scorer="DistMult",
                world_size=world_size,
            ),
        )

    # ============== Scorer (stateless) ==============
    scorer = DistMultScorer()  # no parameters

    # Track best across all folds
    best_overall = {"loss": float("inf"), "state": None, "rel_state": None, "embeddings": None, "fold": None}

    # ============== Cross Validation ==============
    for fold, (train_idx, val_idx) in enumerate(kf.split(triplets_id)):
        if is_main_process(rank):
            print(f"\n===== Fold {fold + 1}/{k_folds} =====")

        train_triples = torch.tensor(triplets_id[train_idx], dtype=torch.long)
        val_triples = torch.tensor(triplets_id[val_idx], dtype=torch.long)

        # Datasets & Distributed Samplers
        train_ds = TensorDataset(train_triples)
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler, pin_memory=True)

        # Model (wrapped) + Optimizer
        model = RGCNWithRelations(in_dim=feat_dim, hidden_dim=hidden_dim, out_dim=embedding_dim, num_rels=num_rels).to(device)
        ddp_model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

        optimizer = torch.optim.Adam(ddp_model.parameters(), lr=lr)

        best_val_loss = float("inf")
        best_state_local = None  # this rank's best (we'll reduce)

        # Epoch loop
        epoch_iter = range(epochs)
        if is_main_process(rank):
            epoch_iter = tqdm(epoch_iter, desc=f"Fold {fold + 1}", unit="epoch")

        for epoch in epoch_iter:
            ddp_model.train()
            train_sampler.set_epoch(epoch)

            total_loss_local = 0.0
            seen_local = 0

            for (batch_pos,) in train_loader:
                batch_pos = batch_pos.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                entity_emb = ddp_model(x, edge_index, edge_type)  # [num_entities, embedding_dim]

                # Negative sampling (tails)
                b = batch_pos.size(0)
                random_tails = torch.randint(0, num_entities, (b,), device=device)
                # ensure negatives differ from true tail
                mask = random_tails == batch_pos[:, 2]
                if mask.any():
                    random_tails[mask] = torch.randint(0, num_entities, (int(mask.sum().item()),), device=device)
                batch_neg = torch.stack([batch_pos[:, 0], batch_pos[:, 1], random_tails], dim=1)

                # Scores
                rel_w = ddp_model.module.relation_weights()
                pos_scores = scorer.score(entity_emb, rel_w, batch_pos)
                neg_scores = scorer.score(entity_emb, rel_w, batch_neg)

                loss = F.relu(margin - pos_scores + neg_scores).mean()
                loss.backward()
                optimizer.step()

                total_loss_local += loss.item() * b
                seen_local += b

            # Average train loss across ranks
            total_loss_tensor = torch.tensor(total_loss_local, device=device)
            seen_tensor = torch.tensor(seen_local, device=device)
            dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(seen_tensor, op=dist.ReduceOp.SUM)
            avg_train_loss = (total_loss_tensor / torch.clamp_min(seen_tensor, 1)).item()

            # ---- Validation (rank-local calc, then average) ----
            ddp_model.eval()
            with torch.no_grad():
                entity_emb = ddp_model(x, edge_index, edge_type)

                # shard validation across ranks for speed
                val_shard = val_triples[rank::world_size].to(device, non_blocking=True)
                if val_shard.numel() == 0:
                    local_val_loss = torch.tensor(0.0, device=device)
                    local_count = torch.tensor(0, device=device)
                else:
                    b = val_shard.size(0)
                    rnd_tails = torch.randint(0, num_entities, (b,), device=device)
                    corrupt_val = torch.stack([val_shard[:, 0], val_shard[:, 1], rnd_tails], dim=1)

                    rel_w = ddp_model.module.relation_weights()
                    pos_val_scores = scorer.score(entity_emb, rel_w, val_shard)
                    neg_val_scores = scorer.score(entity_emb, rel_w, corrupt_val)
                    local_val_loss = F.relu(margin - pos_val_scores + neg_val_scores).mean()
                    local_count = torch.tensor(1, device=device)  # we use mean, so count=1 for averaging

                # average mean val loss across ranks
                val_loss_tensor = all_reduce_mean(local_val_loss.clone())
                # update bar/logs on main only
                if is_main_process(rank):
                    if isinstance(epoch_iter, tqdm):
                        epoch_iter.set_postfix({"train_loss": f"{avg_train_loss:.4f}",
                                                "val_loss": f"{val_loss_tensor.item():.4f}"})
                    if use_wandb:
                        wandb.log({"fold": fold, "epoch": epoch, "train_loss": avg_train_loss,
                                   "val_loss": val_loss_tensor.item()})

                # checkpoint best (main proc only)
                if is_main_process(rank) and val_loss_tensor.item() < best_val_loss:
                    best_val_loss = val_loss_tensor.item()
                    best_state_local = {
                        "state": {k: v.cpu() for k, v in ddp_model.module.rgcn.state_dict().items()},
                        "rel_state": {k: v.cpu() for k, v in ddp_model.module.rel_emb.state_dict().items()},
                        "embeddings": entity_emb.detach().cpu(),
                        "loss": best_val_loss,
                    }

        # Fold end
        if is_main_process(rank):
            print(f"Fold {fold + 1} finished | Best val loss: {best_val_loss:.4f}")
            if use_wandb:
                wandb.log({"fold": fold, "best_val_loss": best_val_loss})

            # compare to global best
            if best_state_local is not None and best_state_local["loss"] < best_overall["loss"]:
                best_overall = {
                    "loss": best_state_local["loss"],
                    "state": best_state_local["state"],
                    "rel_state": best_state_local["rel_state"],
                    "embeddings": best_state_local["embeddings"],
                    "fold": fold,
                }

        # sync before next fold
        dist.barrier()

    # ============== Save Best ==============
    if is_main_process(rank):
        torch.save(best_overall["state"], out_dir / "best_rgcn_model.pt")
        torch.save(best_overall["rel_state"], out_dir / "best_rel_emb.pt")
        np.save(out_dir / "best_umls_entity_embeddings.npy", best_overall["embeddings"].numpy())
        with open("cui2id.json", "w") as f:
            json.dump(entity2id, f)
        print(f"\nBest model from fold {best_overall['fold']} saved with val loss {best_overall['loss']:.4f}")

        if use_wandb:
            wandb.finish()

    cleanup_ddp()


if __name__ == "__main__":
    main()
