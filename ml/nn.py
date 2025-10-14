import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from visdom import Visdom
import matplotlib.pyplot as plt


# ============================================================
# Dataset
# ============================================================

class ShowerDataset(Dataset):
    """
    Dataset using hit-level parquet data with per-plane aggregation.
    Each event is composed of 24 planes, and the event-level regression
    target is per-plane PDG counts.
    """
    def __init__(self, hit_file, pdg_classes, n_planes=24, normalize_events=True):
        self.hits = pd.read_parquet(hit_file)

        # ensure plane column exists
        if "plane" not in self.hits.columns:
            raise ValueError("Input parquet must contain a 'plane' column for 24-plane output.")

        self.feature_cols = [
            "kinetic_energy", "primary_kinetic_energy",
            "X_transformed", "Y_transformed", "Z_transformed",
            "distance", # "time_transformed",
            "sin_azimuth", "cos_azimuth", "sin_zenith", "cos_zenith"
        ]

        # PDG mapping
        self.pdg_map = {pdg: i for i, pdg in enumerate(pdg_classes)}
        self.inv_pdg_map = {i: pdg for pdg, i in self.pdg_map.items()}

        self.hits["pdg_idx"] = self.hits["pdg"].map(self.pdg_map)

        # ------------------------------------------------------------
        # Compute per-event, per-plane PDG counts
        # ------------------------------------------------------------
        event_plane_counts = (
            self.hits.groupby(["event_id", "plane"])["pdg_idx"]
            .value_counts()
            .unstack(fill_value=0)
            .reindex(columns=range(len(pdg_classes)), fill_value=0)
        )

        event_plane_counts.columns = [
            f"count_{self.inv_pdg_map[c]}" for c in event_plane_counts.columns
        ]
        event_plane_counts = event_plane_counts.reset_index()

        # pivot to get a flat per-event structure
        event_counts_pivot = []
        for pdg in pdg_classes:
            colname = f"count_{pdg}"
            pivot = event_plane_counts.pivot(
                index="event_id", columns="plane", values=colname
            ).reindex(columns=range(n_planes), fill_value=0)
            pivot.columns = [f"{colname}_plane{p}" for p in pivot.columns]
            event_counts_pivot.append(pivot)

        self.event_data = pd.concat(event_counts_pivot, axis=1).fillna(0)

        # Normalize event-level targets
        self.normalize_events = normalize_events
        if normalize_events:
            self.scaler = StandardScaler()
            self.event_data = self.event_data.astype(float)
            self.event_data.loc[:, :] = self.scaler.fit_transform(self.event_data)
        else:
            self.scaler = None

        self.event_ids = self.hits["event_id"].unique().tolist()
        self.event_cols = list(self.event_data.columns)
        self.n_planes = n_planes

    def __len__(self):
        return len(self.event_ids)

    def __getitem__(self, idx):
        eid = self.event_ids[idx]
        df = self.hits[self.hits["event_id"] == eid]

        X_event = df[self.feature_cols].values.astype(float)
        y_pdg = df["pdg_idx"].values.astype(int)
        y_event = self.event_data.loc[eid].values.astype(float)

        return (
            torch.tensor(X_event, dtype=torch.float32),
            torch.tensor(y_pdg, dtype=torch.long),
            torch.tensor(y_event, dtype=torch.float32),
            eid
        )

# ============================================================
# Model
# ============================================================

class ShowerNetMultiTask(nn.Module):
    """
    Multi-task network with:
    - Per-hit PDG classification head
    - Per-hit existence logit s_i
    - Event-level regression head via weighted pooling
      producing per-plane outputs (24 x n_event_outputs)
    """
    def __init__(self, input_dim, hidden_dim, n_classes, n_event_outputs, n_planes=24):
        super().__init__()
        self.n_planes = n_planes
        self.n_event_outputs = n_event_outputs

        self.hit_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.pdg_head = nn.Linear(hidden_dim, n_classes)
        self.exist_head = nn.Linear(hidden_dim, 1)
        self.event_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_event_outputs * n_planes)
        )

    def forward(self, X_batch):
        pdg_logits_list, event_preds_list = [], []
        for X in X_batch:
            h = self.hit_encoder(X)
            pdg_logits = self.pdg_head(h)

            s = torch.sigmoid(self.exist_head(h))  # [N_hits, 1]
            h_weighted = (h * s).sum(dim=0) / (s.sum() + 1e-6)

            event_pred = self.event_head(h_weighted)
            event_pred = event_pred.view(self.n_planes, self.n_event_outputs)
            pdg_logits_list.append(pdg_logits)
            event_preds_list.append(event_pred)

        return pdg_logits_list, torch.stack(event_preds_list)

# ============================================================
# Training
# ============================================================

def train_multitask(train_dataset, val_dataset, input_dim, n_classes, n_event_outputs,
                    hidden_dim=128, epochs=10, batch_size=16, lr=1e-3, device="cuda",
                    lambda_event=1.0, n_planes=24):

    # !!!!!! use python -m visdom.server to start server
    viz = Visdom()
    win = None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=lambda x: list(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=lambda x: list(zip(*x)))

    model = ShowerNetMultiTask(input_dim, hidden_dim, n_classes, n_event_outputs, n_planes=n_planes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_pdg = nn.CrossEntropyLoss()
    loss_event = nn.MSELoss()

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for X_batch, y_pdg_batch, y_event_batch, _ in train_loader:
            X_batch = [x.to(device) for x in X_batch]
            y_event_batch = torch.stack(y_event_batch).to(device).view(len(X_batch), n_planes, n_event_outputs)

            pdg_logits_list, event_preds = model(X_batch)

            L_pdg = 0.0
            for logits, labels in zip(pdg_logits_list, y_pdg_batch):
                logits, labels = logits.to(device), labels.to(device)
                L_pdg += loss_pdg(logits, labels)
            L_pdg /= len(pdg_logits_list)

            L_event = loss_event(event_preds, y_event_batch)
            L_total = L_pdg + lambda_event * L_event

            optimizer.zero_grad()
            L_total.backward()
            optimizer.step()
            total_loss += L_total.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_pdg_batch, y_event_batch, _ in val_loader:
                X_batch = [x.to(device) for x in X_batch]
                y_event_batch = torch.stack(y_event_batch).to(device).view(len(X_batch), n_planes, n_event_outputs)
                pdg_logits_list, event_preds = model(X_batch)

                L_pdg = 0.0
                for logits, labels in zip(pdg_logits_list, y_pdg_batch):
                    logits, labels = logits.to(device), labels.to(device)
                    L_pdg += loss_pdg(logits, labels)
                L_pdg /= len(pdg_logits_list)
                L_event = loss_event(event_preds, y_event_batch)
                L_total = L_pdg + lambda_event * L_event
                val_loss += L_total.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss={avg_train_loss:.4f} | Val Loss={avg_val_loss:.4f}")

        # Visdom live plot update
        X = np.arange(1, epoch+2)
        Y = np.column_stack((train_losses, val_losses))
        if win is None:
            win = viz.line(
                Y=Y,
                X=X,
                opts=dict(
                    xlabel='Epoch',
                    ylabel='Loss',
                    title='Training and Validation Loss',
                    legend=['Train Loss', 'Val Loss']
                )
            )
        else:
            viz.line(
                Y=Y,
                X=X,
                win=win,
                update='replace'
            )

    return model

# ============================================================
# Evaluation
# ============================================================

def evaluate_multitask(model, dataset, n_planes, n_event_outputs, device="cuda"):
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        collate_fn=lambda x: list(zip(*x)))
    model.eval()
    correct, total = 0, 0
    mse_event = 0.0
    all_preds = {}

    with torch.no_grad():
        for X_batch, y_pdg_batch, y_event_batch, eids in loader:
            X_batch = [x.to(device) for x in X_batch]
            y_event_batch = torch.stack(y_event_batch).to(device).view(len(X_batch), n_planes, n_event_outputs)

            pdg_logits_list, event_preds = model(X_batch)

            for logits, labels, y_event, eid, event_pred in zip(
                pdg_logits_list, y_pdg_batch, y_event_batch, eids, event_preds
            ):
                preds = logits.argmax(dim=1).cpu()
                labels = labels.cpu()

                # hit-level accuracy
                correct += (preds == labels).sum().item()
                total += len(labels)

                # event-level mse
                mse_event += ((event_pred - y_event) ** 2).mean().item()

                all_preds[eid] = {
                    "hit_preds": preds.numpy(),
                    "hit_truth": labels.numpy(),
                    "event_pred": event_pred.detach().cpu().numpy(),
                    "event_truth": y_event.detach().cpu().numpy(),
                }

    hit_acc = correct / total if total > 0 else 0.0
    event_mse = mse_event / len(loader)
    print(f"\nHit Classification Accuracy: {hit_acc * 100:.2f}%")
    print(f"Event Regression MSE:        {event_mse:.4f}")

    return {
        "hit_accuracy": hit_acc,
        "event_mse": event_mse,
        "predictions": all_preds
    }

# ============================================================
# Usage Example
# ============================================================

if __name__ == "__main__":
    pdg_classes = [11, 13, 22]
    hit_file = "../ml/processed_events/normalized_features_z_3.parquet"
    n_planes = 24

    full_dataset = ShowerDataset(hit_file, pdg_classes, n_planes=n_planes, normalize_events=True)
    event_ids = full_dataset.event_ids

    test_eid = event_ids[0]
    remaining_ids = event_ids[1:]
    train_ids, val_ids = train_test_split(remaining_ids, test_size=0.1, random_state=42)

    id_to_idx = {eid: i for i, eid in enumerate(event_ids)}
    train_idx = [id_to_idx[eid] for eid in train_ids]
    val_idx = [id_to_idx[eid] for eid in val_ids]
    test_idx = [id_to_idx[test_eid]]

    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    test_dataset = Subset(full_dataset, test_idx)

    input_dim = len(full_dataset.feature_cols)
    n_classes = len(pdg_classes)
    n_event_outputs = len(pdg_classes)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    
    # model = torch.load("shower_net_multitask.pth", map_location=device) if False else None
    model = None
    if os.path.exists("../training_checkpoints/shower_net_multitask.pth"):
        model = ShowerNetMultiTask(input_dim, 256, n_classes, n_event_outputs, n_planes=n_planes).to(device)
        model.load_state_dict(torch.load("../training_checkpoints/shower_net_multitask.pth", map_location=device))
        model.eval()
        print("Loaded existing model from ../training_checkpoints/shower_net_multitask.pth")

    if model is None:
        print("Training Multi-Task Model...")

        model = train_multitask(train_dataset, val_dataset,
                                input_dim, n_classes, n_event_outputs,
                                hidden_dim=256, epochs=100, batch_size=500,
                                lr=1e-3, device=device, lambda_event=1.0, n_planes=n_planes)

        torch.save(model.state_dict(), "../training_checkpoints/shower_net_multitask.pth")

    print("\nEvaluating on Held-out Event...")
    test_metrics = evaluate_multitask(
        model, test_dataset,
        n_planes=n_planes,
        n_event_outputs=n_event_outputs,
        device=device
    )

    scaler = full_dataset.scaler
    predictions = test_metrics["predictions"]

    for i, (eid, res) in enumerate(predictions.items()):
        hit_preds = [full_dataset.inv_pdg_map[int(idx)] for idx in res["hit_preds"][:10]]
        hit_truth = [full_dataset.inv_pdg_map[int(idx)] for idx in res["hit_truth"][:10]]

        pred_event = np.atleast_2d(res["event_pred"])
        true_event = np.atleast_2d(res["event_truth"])

        # Flatten to match scaler shape (72 for 24x3)
        pred_event_flat = pred_event.reshape(1, -1)
        true_event_flat = true_event.reshape(1, -1)

        if scaler is not None:
            pred_event_flat = scaler.inverse_transform(pred_event_flat)
            true_event_flat = scaler.inverse_transform(true_event_flat)

        pred_event = pred_event_flat.reshape(n_planes, n_event_outputs)
        true_event = true_event_flat.reshape(n_planes, n_event_outputs)

        print("=" * 60)
        print(f"Event {eid}")
        print(f"  Hit Predictions (first 10): {hit_preds}")
        print(f"  Hit Truth       (first 10): {hit_truth}")
        for plane_idx in range(n_planes):
            pred_counts = [f"{x:.2f}" for x in pred_event[plane_idx]]
            true_counts = [f"{x:.2f}" for x in true_event[plane_idx]]
            print(f"  Plane {plane_idx:02d} | Predicted: {pred_counts} | Truth: {true_counts}")

        if i == 23:
            break

    print("\n===== SUMMARY =====")
    print(f"Hit Accuracy: {test_metrics['hit_accuracy']*100:.2f}%")
    print(f"Event MSE:    {test_metrics['event_mse']:.4f}")
