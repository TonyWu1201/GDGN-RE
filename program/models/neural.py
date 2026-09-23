"""Small independent cell/drug encoders used as the A1 neural anchor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from program.evaluation.metrics import evaluate_predictions
from program.models.dataset import PreparedData


class TwinTower(nn.Module):
    def __init__(self, n_expr: int, n_fp: int, dropout: float):
        super().__init__()
        def encoder(size: int) -> nn.Sequential:
            return nn.Sequential(nn.Linear(size, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(128, 64), nn.LayerNorm(64), nn.GELU())
        self.cell = encoder(n_expr)
        self.drug = encoder(n_fp)
        self.head = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, expression: torch.Tensor, fingerprint: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat((self.cell(expression), self.drug(fingerprint)), dim=-1)).squeeze(-1)


@dataclass
class TrainedMLP:
    model: TwinTower
    selected_epoch: int
    history: list[dict]
    parameter_count: int

    def predict(self, prepared: PreparedData, frame: pd.DataFrame, batch_size: int = 2048) -> np.ndarray:
        self.model.eval()
        ci, di, _ = prepared.arrays(frame)
        expr = torch.as_tensor(prepared.expression, dtype=torch.float32)
        fp = torch.as_tensor(prepared.fingerprints, dtype=torch.float32)
        chunks = []
        with torch.no_grad():
            for start in range(0, len(ci), batch_size):
                chunks.append(self.model(expr[ci[start:start + batch_size]], fp[di[start:start + batch_size]]).numpy())
        return np.concatenate(chunks).astype(float) if chunks else np.empty(0, dtype=float)


def fit_mlp(prepared: PreparedData, params: dict, seed: int,
            validation: pd.DataFrame | None, max_epochs: int = 200, patience: int = 20,
            fixed_epochs: int | None = None, min_cells: int = 10) -> TrainedMLP:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TwinTower(prepared.expression.shape[1], prepared.fingerprints.shape[1], params["dropout"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    ci, di, y = prepared.arrays(prepared.train)
    expr = torch.as_tensor(prepared.expression, dtype=torch.float32, device=device)
    fp = torch.as_tensor(prepared.fingerprints, dtype=torch.float32, device=device)
    cell_index = torch.as_tensor(ci, dtype=torch.long, device=device)
    drug_index = torch.as_tensor(di, dtype=torch.long, device=device)
    labels = torch.as_tensor(y, dtype=torch.float32, device=device)
    rng = np.random.default_rng(seed)
    history: list[dict] = []
    best_score = -np.inf
    best_rmse = np.inf
    best_state = None
    best_epoch = 0
    epochs = fixed_epochs if fixed_epochs is not None else max_epochs
    if epochs < 1 or epochs > max_epochs:
        raise ValueError("epoch count outside preregistered bounds")
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(ci))
        losses = []
        grad_norms = []
        for start in range(0, len(order), 128):
            batch = torch.as_tensor(order[start:start + 128], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(expr[cell_index[batch]], fp[drug_index[batch]])
            loss = nn.functional.mse_loss(pred, labels[batch])
            if not torch.isfinite(loss):
                raise ValueError("non-finite neural loss")
            loss.backward()
            norms = [p.grad.detach().norm(2) for p in model.parameters() if p.grad is not None]
            grad_norms.append(float(torch.linalg.vector_norm(torch.stack(norms)).item()))
            optimizer.step()
            losses.append(float(loss.item()))
        row = {"epoch": epoch, "train_mse": float(np.mean(losses)), "gradient_norm": float(np.mean(grad_norms))}
        if validation is not None and fixed_epochs is None:
            model.eval()
            vi, vdi, _ = prepared.arrays(validation)
            vi = torch.as_tensor(vi, dtype=torch.long, device=device)
            vdi = torch.as_tensor(vdi, dtype=torch.long, device=device)
            outputs = []
            with torch.no_grad():
                for start in range(0, len(vi), 2048):
                    outputs.append(model(expr[vi[start:start + 2048]], fp[vdi[start:start + 2048]]).cpu().numpy())
            pred_val = np.concatenate(outputs)
            val_frame = validation[["sample_id", "cell_id", "compound_id"]].copy()
            val_frame["y_true"] = validation["y"].to_numpy()
            val_frame["y_pred"] = pred_val
            metrics = evaluate_predictions(val_frame, min_cells=min_cells)
            score = metrics["macro_spearman_selection"]
            if score is None:
                raise ValueError("inner validation has no qualified drugs; freeze a viable split before training")
            row.update({"validation_macro_spearman": score, "validation_rmse": metrics["rmse"]})
            if score > best_score + 1e-8 or (abs(score - best_score) <= 1e-8 and metrics["rmse"] < best_rmse):
                best_score, best_rmse, best_epoch = score, metrics["rmse"], epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if epoch - best_epoch >= patience:
                row["early_stop"] = True
                history.append(row)
                break
        history.append(row)
    if best_state is not None:
        model.load_state_dict(best_state)
    model.cpu().eval()
    return TrainedMLP(model, best_epoch if best_state is not None else epochs,
                      history, sum(p.numel() for p in model.parameters()))
