import argparse
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split

try:
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import global_mean_pool, global_add_pool
    from torch_geometric.utils import degree, softmax as segment_softmax
    from torch_geometric.typing import Adj
except Exception as e:
    raise ImportError("This script requires PyTorch Geometric. Install per https://pytorch-geometric.readthedocs.io/")

# plotting (headless)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------------------------
# Reproducibility
# -------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -------------------------
# Data & Graph building
# -------------------------
FEATURES = [
    "r_normalized",
    "kinetic_energy_normalized",
    "z_normalized",
    "y_normalized",
    "x_normalized",
    "time_normalized",
    "charge_normalized",
]

LABELS = {11: 0, 13: 1, 22: 2}
IDX2PDG = {v: k for k, v in LABELS.items()}
CLASS_NAMES = ["PDG 11", "PDG 13", "PDG 22"]


def _knn_edges_by_distance(dist: np.ndarray, k: int) -> torch.Tensor:
    """
    Make undirected edges by connecting each index to k neighbors before/after
    in the array after sorting by distance. Returns edge_index (2, E) tensor.
    """
    order = np.argsort(dist)
    n = len(order)
    senders = []
    receivers = []
    for rank, idx in enumerate(order):
        # connect to previous k
        for j in range(1, k + 1):
            if rank - j >= 0:
                nbr = int(order[rank - j])
                senders.append(idx); receivers.append(nbr)
                senders.append(nbr); receivers.append(idx)
        # connect to next k
        for j in range(1, k + 1):
            if rank + j < n:
                nbr = int(order[rank + j])
                senders.append(idx); receivers.append(nbr)
                senders.append(nbr); receivers.append(idx)

    if len(senders) == 0:
        return torch.empty((2, 0), dtype=torch.long)

    edge_index = torch.tensor([senders, receivers], dtype=torch.long)
    # remove self-loops if any
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]
    return edge_index


def load_graphs_from_parquet(parquet_path: str, k: int = 3) -> List[Data]:
    df = pd.read_parquet(parquet_path)
    missing = [c for c in FEATURES + ["pdg", "plane", "distance_normalized"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    # Map labels and drop rows with unknown pdg
    df = df[df["pdg"].isin(LABELS.keys())].copy()
    df["y_node"] = df["pdg"].map(LABELS).astype(np.int64)

    # Group by plane -> one graph per plane
    graphs: List[Data] = []
    for pid, sub in df.groupby("plane", sort=False):
        sub = sub.reset_index(drop=True)

        x = torch.tensor(sub[FEATURES].astype(np.float32).values, dtype=torch.float32)
        y = torch.tensor(sub["y_node"].values, dtype=torch.long)  # per-node class
        dist = sub["distance_normalized"].astype(np.float32).values
        edge_index = _knn_edges_by_distance(dist, k=k)

        # Optional: node degree as an extra structural feature (concat)
        if edge_index.numel() > 0:
            deg = degree(edge_index[0], num_nodes=x.size(0)).unsqueeze(1)
        else:
            deg = torch.zeros((x.size(0), 1), dtype=torch.float32)

        x_full = torch.cat([x, deg], dim=1)

        data = Data(
            x=x_full,
            edge_index=edge_index,
            y=y,                 # node-level labels
            plane=str(pid),      # keep id for reference
        )
        graphs.append(data)

    if len(graphs) == 0:
        raise RuntimeError("No graphs built (check your filters and columns).")
    return graphs

# -------------------------
# Attention-based GraphSAGE layer
# -------------------------
from torch_geometric.nn.conv import MessagePassing


class AttentionSAGEConv(MessagePassing):
    """
    SAGE-style update with learned attention on neighbor messages.
    h_i' = W_root h_i + Agg_j( alpha_{ij} * W_neigh h_j ),
    where alpha_{ij} = softmax_j( LeakyReLU( a^T [W_att h_i || W_att h_j] ) )
    """

    def __init__(self, in_channels: int, out_channels: int, heads: int = 1, dropout: float = 0.0):
        super().__init__(aggr='add', node_dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout

        self.lin_root = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_neigh = nn.Linear(in_channels, out_channels * heads, bias=False)
        self.lin_att = nn.Linear(in_channels, out_channels * heads, bias=False)
        self.att_vec = nn.Parameter(torch.Tensor(heads, out_channels * 2))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_root.weight)
        nn.init.xavier_uniform_(self.lin_neigh.weight)
        nn.init.xavier_uniform_(self.lin_att.weight)
        nn.init.xavier_uniform_(self.att_vec)

    def forward(self, x: torch.Tensor, edge_index: Adj):
        H = self.heads
        x_root = self.lin_root(x)  # [N, F_out]
        x_iq = self.lin_att(x)     # [N, H*F_out]
        x_jm = self.lin_neigh(x)   # [N, H*F_out]
        x_iq = x_iq.view(-1, H, self.out_channels)
        x_jm = x_jm.view(-1, H, self.out_channels)
        out = self.propagate(edge_index, x_root=x_root, x_iq=x_iq, x_jm=x_jm)
        return out

    def message(self, x_iq_j, x_jm_j, index, ptr, size_i):
        # x_iq_j, x_jm_j : [E, H, F_out]
        H = self.heads
        cat = torch.cat([x_iq_j, x_jm_j], dim=-1)  # [E, H, 2F]
        att = torch.einsum('ehf,hf->eh', self.leaky_relu(cat), self.att_vec)  # [E, H]
        # softmax over incoming edges of each i
        alpha = segment_softmax(att, index)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        msg = alpha.unsqueeze(-1) * x_jm_j  # [E, H, F]
        msg = msg.sum(dim=1)  # sum over heads -> [E, F]
        return msg

    def update(self, aggr_out, x_root):
        # SAGE update + nonlinearity
        out = x_root + aggr_out
        return out


class AttnGraphSAGE(nn.Module):
    def __init__(self, in_dim: int, hidden: Tuple[int, ...], dropout: float = 0.1, heads: int = 2):
        super().__init__()
        dims = [in_dim] + list(hidden)
        convs = []
        norms = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            convs.append(AttentionSAGEConv(d_in, d_out, heads=heads, dropout=dropout))
            norms.append(nn.BatchNorm1d(d_out))
        self.convs = nn.ModuleList(convs)
        self.norms = nn.ModuleList(norms)
        self.dropout = dropout

        last = dims[-1]
        self.node_head = nn.Linear(last, 3)  # PDG classes only

    def forward(self, data: Data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        node_logits = self.node_head(x)
        return node_logits


# -------------------------
# Metrics
# -------------------------
from sklearn.metrics import accuracy_score, f1_score, classification_report


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ce_loss = nn.CrossEntropyLoss()
    total_nodes = 0
    running_loss = 0.0
    all_true = []
    all_pred = []

    for batch in loader:
        batch = batch.to(device)
        node_logits = model(batch)
        node_loss = ce_loss(node_logits, batch.y)

        running_loss += node_loss.item() * batch.num_nodes

        y_true = batch.y.detach().cpu().numpy()
        y_pred = node_logits.argmax(dim=1).detach().cpu().numpy()
        all_true.append(y_true)
        all_pred.append(y_pred)
        total_nodes += batch.num_nodes

    y_true = np.concatenate(all_true) if len(all_true) else np.array([])
    y_pred = np.concatenate(all_pred) if len(all_pred) else np.array([])
    acc = accuracy_score(y_true, y_pred) if y_true.size else 0.0
    f1 = f1_score(y_true, y_pred, average='macro') if y_true.size else 0.0

    return {
        'loss': running_loss / max(total_nodes, 1),
        'node_acc': acc,
        'node_f1': f1,
    }


# -------------------------
# Helpers for counts and pretty printing
# -------------------------

def graph_counts_from_nodes(batch: Data) -> torch.Tensor:
    # [B, 3] counts from ground truth node labels
    y_onehot = F.one_hot(batch.y, num_classes=3).float()
    return global_add_pool(y_onehot, batch.batch)


def get_plane_from_batch(batch: Data) -> str:
    plane_attr = getattr(batch, 'plane', None)
    if plane_attr is None:
        return '<unknown>'
    if isinstance(plane_attr, (list, tuple)):
        return str(plane_attr[0])
    try:
        return str(plane_attr)
    except Exception:
        return '<unknown>'


@torch.no_grad()
def print_actual_vs_pred_counts_expected(model, dataset, device):
    # Use batch_size=1 for stable per-plane printing
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    model.eval()
    print("\n=== Actual vs Predicted (Expected counts from node probabilities) ===")
    print("plane\tactual_11\tactual_13\tactual_22\tpred_11\tpred_13\tpred_22")
    for batch in loader:
        batch = batch.to(device)
        plane = get_plane_from_batch(batch)
        node_logits = model(batch)
        node_probs = F.softmax(node_logits, dim=1)
        pred_counts = global_add_pool(node_probs, batch.batch).cpu().numpy()[0]
        true_counts = graph_counts_from_nodes(batch).cpu().numpy()[0]
        print(f"{plane}\t{int(true_counts[0])}\t{int(true_counts[1])}\t{int(true_counts[2])}\t"
              f"{pred_counts[0]:.1f}\t{pred_counts[1]:.1f}\t{pred_counts[2]:.1f}")

# -------------------------
# Plotting utilities
# -------------------------

@torch.no_grad()
def _counts_dataframe(model, dataset, device):
    """
    Returns a pandas DataFrame with rows: plane, class_idx, class_name, actual, predicted, diff
    Predicted are expected counts from node probabilities (sums within a plane).
    """
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    rows = []
    model.eval()
    for batch in loader:
        batch = batch.to(device)
        plane = get_plane_from_batch(batch)
        logits = model(batch)
        probs = F.softmax(logits, dim=1)

        true_counts = graph_counts_from_nodes(batch).cpu().numpy()[0]      # (3,)
        pred_counts = global_add_pool(probs, batch.batch).cpu().numpy()[0] # (3,)

        for ci, (t, p) in enumerate(zip(true_counts, pred_counts)):
            rows.append({
                "plane": str(plane),
                "class_idx": ci,
                "class_name": CLASS_NAMES[ci],
                "actual": float(t),
                "predicted": float(p),
                "diff": float(t - p),
            })
    return pd.DataFrame(rows)


def _plot_pdg_counts_overall(df: pd.DataFrame, outpath: Path, title: str):
    """
    Side-by-side bars of total actual vs predicted per PDG class.
    """
    if df.empty:
        return
    agg = df.groupby("class_name")[["actual", "predicted"]].sum().reindex(CLASS_NAMES)
    x = np.arange(len(agg))
    w = 0.35

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    ax.bar(x - w/2, agg["actual"].values, width=w, label="Actual")
    ax.bar(x + w/2, agg["predicted"].values, width=w, label="Predicted")
    ax.set_xticks(x)
    ax.set_xticklabels(agg.index)
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def _plot_plane_diff_heatmap(df: pd.DataFrame, outpath: Path, title: str):
    """
    Heatmap of (actual - predicted) per PDG class (rows) vs plane (columns, ascending).
    """
    if df.empty:
        return

    # Pivot so planes are columns (x-axis) and classes are rows (y-axis)
    pivot = (
        df.pivot_table(index="class_name", columns="plane", values="diff", aggfunc="sum")
        .reindex(index=CLASS_NAMES)
    )

    # Try to sort planes numerically if possible, else lexicographically
    try:
        pivot = pivot[sorted(pivot.columns, key=lambda x: float(x))]
    except Exception:
        pivot = pivot[sorted(pivot.columns)]

    planes = list(pivot.columns)
    data = pivot.values

    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(planes) + 2), 4))
    im = ax.imshow(data, aspect="auto", cmap="coolwarm", origin="upper")

    # X-axis → planes
    ax.set_xticks(np.arange(len(planes)))
    ax.set_xticklabels(planes, rotation=45, ha="right")
    ax.set_xlabel("Plane")

    # Y-axis → PDG classes
    ax.set_yticks(np.arange(len(CLASS_NAMES)))
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_ylabel("PDG Class")

    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Actual − Predicted")

    # Annotate each cell with the diff value
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8)

    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=200)
    plt.close(fig)



def save_plots_for_dataset(model, dataset, device, outdir: Path, tag: str):
    """
    tag: 'test' or 'all' (used in filenames/titles)
    """
    outdir.mkdir(parents=True, exist_ok=True)
    df = _counts_dataframe(model, dataset, device)

    _plot_pdg_counts_overall(
        df,
        outdir / f"pdg_counts_overall_{tag}.png",
        title=f"PDG Counts — Actual vs Predicted ({tag})"
    )
    _plot_plane_diff_heatmap(
        df,
        outdir / f"plane_diff_heatmap_{tag}.png",
        title=f"Per-Plane (Actual − Predicted) ({tag})"
    )

# -------------------------

# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parquet', required=True)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch_size', type=int, default=8, help='graphs per batch')
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--hidden', type=str, default='128,128')
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--heads', type=int, default=2)
    ap.add_argument('--k', type=int, default=3, help='neighbors in each direction along distance')
    ap.add_argument('--val_frac', type=float, default=0.1)
    ap.add_argument('--test_frac', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--outdir', type=Path, default=Path("plots"))
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build graphs
    graphs = load_graphs_from_parquet(args.parquet, k=args.k)

    # Train/val/test split (by graph/plane)
    n = len(graphs)
    n_test = int(n * args.test_frac)
    n_val = int(n * args.val_frac)
    n_train = n - n_val - n_test
    train_set, val_set, test_set = random_split(
        graphs, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)

    in_dim = len(FEATURES) + 1  # + degree feature
    hidden = tuple(int(x) for x in args.hidden.split(',') if x.strip())

    model = AttnGraphSAGE(in_dim=in_dim, hidden=hidden, dropout=args.dropout, heads=args.heads).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ce_loss = nn.CrossEntropyLoss()

    best_f1 = -1.0
    best_state = None
    patience = 20
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        total_nodes = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)

            node_logits = model(batch)
            node_loss = ce_loss(node_logits, batch.y)

            node_loss.backward()
            optimizer.step()

            running += node_loss.item() * batch.num_nodes
            total_nodes += batch.num_nodes

        train_loss = running / max(1, total_nodes)
        val_metrics = evaluate(model, val_loader, device)

        improved = val_metrics['node_f1'] > best_f1
        if improved:
            best_f1 = val_metrics['node_f1']
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | "
            f"val_node_acc={val_metrics['node_acc']:.4f} | val_node_f1={val_metrics['node_f1']:.4f}"
        )

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}. Best val node F1: {best_f1:.4f}")
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # Final evaluation
    test_metrics = evaluate(model, test_loader, device)
    print("\n=== Test metrics ===")
    print(f"node_acc={test_metrics['node_acc']:.4f} node_f1={test_metrics['node_f1']:.4f}")

    # (Optional) per-class report on the test set
    all_true = []
    all_pred = []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            node_logits = model(batch)
            all_true.append(batch.y.cpu().numpy())
            all_pred.append(node_logits.argmax(dim=1).cpu().numpy())
    if len(all_true):
        y_true = np.concatenate(all_true)
        y_pred = np.concatenate(all_pred)
        print("\nClassification report (node PDG):")
        print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    else:
        print("\n(No test examples for classification report.)")


    # Save plots for TEST set
    save_plots_for_dataset(model, test_set, device, args.outdir, tag="test")


    # Save plots for ALL data
    save_plots_for_dataset(model, graphs, device, args.outdir, tag="all")


if __name__ == '__main__':
    main()
