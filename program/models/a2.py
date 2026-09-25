"""Small A2 mechanism models with explicit additive and pathway contributions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from torch import nn

from program.evaluation.metrics import evaluate_predictions
from program.models.dataset import PreparedData
from program.models.neural import TwinTower


A2_MODELS = (
    "r0", "additive", "r1_k16", "r1_k32", "r2_drug_equal", "r2_rank",
    "p", "p_random", "p_projection", "p_linear",
)
PATHWAY_MODELS = frozenset({"p", "p_random", "p_projection", "p_linear"})
R1_MODELS = frozenset({"additive", "r1_k16", "r1_k32", "r2_drug_equal", "r2_rank"})


def encoder(size: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(size, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(128, 64), nn.LayerNorm(64), nn.GELU(),
    )


class AdditiveInteraction(nn.Module):
    def __init__(self, n_expr: int, n_fp: int, dropout: float, rank: int = 0):
        super().__init__()
        self.cell = encoder(n_expr, dropout)
        self.drug = encoder(n_fp, dropout)
        self.mu = nn.Parameter(torch.zeros(()))
        self.cell_main = nn.Linear(64, 1)
        self.drug_main = nn.Linear(64, 1)
        self.rank = rank
        if rank:
            self.cell_factor = nn.Linear(64, rank, bias=False)
            self.drug_factor = nn.Linear(64, rank, bias=False)
        self.register_buffer("cell_reference", torch.zeros(64))
        self.register_buffer("drug_reference", torch.zeros(64))

    @torch.no_grad()
    def refresh_reference(self, expression: torch.Tensor, fingerprint: torch.Tensor) -> None:
        was_training = self.training
        self.eval()
        self.cell_reference.copy_(self.cell(expression).mean(dim=0))
        self.drug_reference.copy_(self.drug(fingerprint).mean(dim=0))
        self.train(was_training)

    def forward_parts(self, expression: torch.Tensor, fingerprint: torch.Tensor) -> dict[str, torch.Tensor]:
        hc, hd = self.cell(expression), self.drug(fingerprint)
        bc = self.cell_main(hc).squeeze(-1)
        bd = self.drug_main(hd).squeeze(-1)
        interaction = torch.zeros_like(bc)
        if self.rank:
            u = self.drug_factor(hd - self.drug_reference)
            v = self.cell_factor(hc - self.cell_reference)
            interaction = (u * v).sum(dim=-1)
        mu = self.mu.expand_as(bc)
        return {"mu": mu, "b_cell": bc, "b_drug": bd, "interaction": interaction,
                "y_pred": mu + bc + bd + interaction}

    def forward(self, expression: torch.Tensor, fingerprint: torch.Tensor) -> torch.Tensor:
        return self.forward_parts(expression, fingerprint)["y_pred"]


class ConcatAnchor(TwinTower):
    """Exact A1 TwinTower topology when R0 must be refitted inside A2."""

    def forward_parts(self, expression: torch.Tensor, fingerprint: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"y_pred": self.forward(expression, fingerprint)}


class PathwayAdditive(nn.Module):
    def __init__(self, n_fp: int, n_pathways: int, dropout: float):
        super().__init__()
        self.drug = encoder(n_fp, dropout)
        self.drug_main = nn.Linear(64, 1)
        self.coefficients = nn.Linear(64, n_pathways)
        self.pathway_slopes = nn.Parameter(torch.ones(n_pathways))

    def forward_parts(self, pathways: torch.Tensor, fingerprint: torch.Tensor) -> dict[str, torch.Tensor]:
        hd = self.drug(fingerprint)
        bd = self.drug_main(hd).squeeze(-1)
        contributions = self.coefficients(hd) * (pathways * self.pathway_slopes)
        return {"b_drug": bd, "contributions": contributions,
                "y_pred": bd + contributions.sum(dim=-1)}

    def forward(self, pathways: torch.Tensor, fingerprint: torch.Tensor) -> torch.Tensor:
        return self.forward_parts(pathways, fingerprint)["y_pred"]


def ranking_loss(
    prediction: torch.Tensor, label: torch.Tensor, drug_index: np.ndarray,
    noise_threshold: float, temperature: float, rng: np.random.Generator,
    max_pairs: int = 64,
) -> torch.Tensor:
    """Sample bounded same-drug pairs; smaller LN_IC50 is more sensitive."""
    if noise_threshold < 0 or temperature <= 0 or max_pairs < 1:
        raise ValueError("invalid frozen ranking parameters")
    chosen: list[tuple[int, int, float]] = []
    drugs = np.unique(drug_index)
    quota = max(1, max_pairs // max(1, len(drugs)))
    truth = label.detach().cpu().numpy()
    for drug in drugs:
        positions = np.flatnonzero(drug_index == drug)
        if len(positions) < 2:
            continue
        draws = rng.integers(0, len(positions), size=(quota * 8, 2))
        picked_this_drug = 0
        for left, right in draws:
            i, j = int(positions[left]), int(positions[right])
            difference = float(truth[j] - truth[i])
            if i != j and abs(difference) > noise_threshold:
                chosen.append((i, j, float(np.sign(difference))))
                picked_this_drug += 1
                if picked_this_drug >= quota or len(chosen) >= max_pairs:
                    break
        if len(chosen) >= max_pairs:
            break
    if not chosen:
        return prediction.sum() * 0.0
    i = torch.as_tensor([item[0] for item in chosen], dtype=torch.long, device=prediction.device)
    j = torch.as_tensor([item[1] for item in chosen], dtype=torch.long, device=prediction.device)
    signs = torch.as_tensor([item[2] for item in chosen], dtype=prediction.dtype, device=prediction.device)
    return nn.functional.softplus(-signs * (prediction[j] - prediction[i]) / temperature).mean()


def _network(name: str, prepared: PreparedData, dropout: float, n_pathways: int | None) -> nn.Module:
    if name == "r0":
        return ConcatAnchor(prepared.expression.shape[1], prepared.fingerprints.shape[1], dropout)
    if name in PATHWAY_MODELS - {"p_linear"}:
        if n_pathways is None or n_pathways < 1:
            raise ValueError("pathway model needs nonempty pathway features")
        return PathwayAdditive(prepared.fingerprints.shape[1], n_pathways, dropout)
    if name in R1_MODELS:
        rank = 0 if name == "additive" else 32 if name == "r1_k32" else 16
        return AdditiveInteraction(prepared.expression.shape[1], prepared.fingerprints.shape[1], dropout, rank)
    raise ValueError(f"unknown A2 neural model: {name}")


@dataclass
class TrainedA2:
    name: str
    model: nn.Module | None
    estimator: Ridge | None
    selected_epoch: int | None
    history: list[dict]
    parameter_count: int
    pathway_names: list[str]

    def predict_components(
        self, prepared: PreparedData, frame: pd.DataFrame, pathways: np.ndarray | None = None,
        batch_size: int = 2048,
    ) -> dict[str, np.ndarray]:
        ci, di, _ = prepared.arrays(frame)
        if self.estimator is not None:
            if pathways is None:
                raise ValueError("p_linear needs pathway features")
            values = np.concatenate((pathways[ci], prepared.fingerprints[di]), axis=1)
            return {"y_pred": np.asarray(self.estimator.predict(values), dtype=float)}
        if self.model is None:
            raise ValueError("fitted A2 model missing")
        self.model.cpu().eval()
        expression = torch.as_tensor(prepared.expression, dtype=torch.float32)
        fingerprint = torch.as_tensor(prepared.fingerprints, dtype=torch.float32)
        pathway_tensor = torch.as_tensor(pathways, dtype=torch.float32) if pathways is not None else None
        pieces: dict[str, list[np.ndarray]] = {}
        with torch.no_grad():
            for start in range(0, len(ci), batch_size):
                rows = slice(start, start + batch_size)
                if self.name in PATHWAY_MODELS:
                    if pathway_tensor is None:
                        raise ValueError("pathway model needs pathway features")
                    output = self.model.forward_parts(pathway_tensor[ci[rows]], fingerprint[di[rows]])
                else:
                    output = self.model.forward_parts(expression[ci[rows]], fingerprint[di[rows]])
                for key, value in output.items():
                    pieces.setdefault(key, []).append(value.numpy())
        return {key: np.concatenate(value).astype(float) for key, value in pieces.items()}

    def predict(self, prepared: PreparedData, frame: pd.DataFrame, pathways: np.ndarray | None = None) -> np.ndarray:
        return self.predict_components(prepared, frame, pathways)["y_pred"]


def fit_a2(
    name: str, prepared: PreparedData, params: dict, seed: int, validation: pd.DataFrame | None,
    pathways: np.ndarray | None = None, pathway_names: list[str] | None = None,
    fixed_epochs: int | None = None, max_epochs: int = 200, patience: int = 20,
    batch_size: int = 128, min_cells: int = 10,
    rank_noise_threshold: float | None = None, rank_temperature: float | None = None,
    rank_max_pairs: int = 64,
) -> TrainedA2:
    if name not in A2_MODELS:
        raise ValueError(f"unregistered A2 model: {name}")
    if name in PATHWAY_MODELS and (pathways is None or not np.isfinite(pathways).all()):
        raise ValueError("pathway features missing or non-finite")
    if name == "r2_rank" and (rank_noise_threshold is None or rank_temperature is None):
        raise ValueError("R2 ranking requires frozen train-derived noise threshold and temperature")
    names = pathway_names or []
    ci, di, y = prepared.arrays(prepared.train)
    if name == "p_linear":
        assert pathways is not None
        values = np.concatenate((pathways[ci], prepared.fingerprints[di]), axis=1)
        estimator = Ridge(alpha=float(params.get("alpha", 1.0)), solver="lsqr").fit(values, y)
        return TrainedA2(name, None, estimator, None, [], values.shape[1] + 1, names)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _network(name, prepared, float(params["dropout"]), pathways.shape[1] if pathways is not None else None).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    expression = torch.as_tensor(prepared.expression, dtype=torch.float32, device=device)
    fingerprint = torch.as_tensor(prepared.fingerprints, dtype=torch.float32, device=device)
    pathway_tensor = torch.as_tensor(pathways, dtype=torch.float32, device=device) if pathways is not None else None
    cindex = torch.as_tensor(ci, dtype=torch.long, device=device)
    dindex = torch.as_tensor(di, dtype=torch.long, device=device)
    labels = torch.as_tensor(y, dtype=torch.float32, device=device)
    rng = np.random.default_rng(seed)
    epochs = fixed_epochs if fixed_epochs is not None else max_epochs
    if epochs < 1 or epochs > max_epochs or batch_size < 1:
        raise ValueError("invalid epoch or batch count")
    best_score, best_rmse, best_epoch, best_state = -np.inf, np.inf, 0, None
    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        if isinstance(model, AdditiveInteraction):
            model.refresh_reference(expression[torch.unique(cindex)], fingerprint[torch.unique(dindex)])
        model.train()
        order = rng.permutation(len(ci))
        losses, gradients = [], []
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            batch = torch.as_tensor(indices, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            if isinstance(model, PathwayAdditive):
                assert pathway_tensor is not None
                pred = model(pathway_tensor[cindex[batch]], fingerprint[dindex[batch]])
            else:
                pred = model(expression[cindex[batch]], fingerprint[dindex[batch]])
            target = labels[batch]
            if name == "r2_drug_equal":
                per_drug = [nn.functional.mse_loss(pred[torch.as_tensor(di[indices] == drug, device=device)],
                                                  target[torch.as_tensor(di[indices] == drug, device=device)])
                            for drug in np.unique(di[indices])]
                loss = torch.stack(per_drug).mean()
            else:
                loss = nn.functional.mse_loss(pred, target)
            if name == "r2_rank":
                loss = loss + 0.1 * ranking_loss(pred, target, di[indices],
                                                float(rank_noise_threshold), float(rank_temperature), rng,
                                                max_pairs=rank_max_pairs)
            if not torch.isfinite(loss):
                raise ValueError("non-finite A2 neural loss")
            loss.backward()
            norms = [p.grad.detach().norm(2) for p in model.parameters() if p.grad is not None]
            gradients.append(float(torch.linalg.vector_norm(torch.stack(norms)).item()))
            optimizer.step()
            losses.append(float(loss.item()))
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)),
               "gradient_norm": float(np.mean(gradients))}
        if validation is not None and fixed_epochs is None:
            if isinstance(model, AdditiveInteraction):
                model.refresh_reference(expression[torch.unique(cindex)], fingerprint[torch.unique(dindex)])
            model.eval()
            vi, vd, _ = prepared.arrays(validation)
            outputs = []
            with torch.no_grad():
                for start in range(0, len(vi), 2048):
                    vcs = torch.as_tensor(vi[start:start + 2048], dtype=torch.long, device=device)
                    vds = torch.as_tensor(vd[start:start + 2048], dtype=torch.long, device=device)
                    if isinstance(model, PathwayAdditive):
                        assert pathway_tensor is not None
                        outputs.append(model(pathway_tensor[vcs], fingerprint[vds]).cpu().numpy())
                    else:
                        outputs.append(model(expression[vcs], fingerprint[vds]).cpu().numpy())
            check = validation[["sample_id", "cell_id", "compound_id"]].copy()
            check["y_true"], check["y_pred"] = validation["y"].to_numpy(), np.concatenate(outputs)
            metrics = evaluate_predictions(check, min_cells=min_cells)
            score = metrics["macro_spearman_selection"]
            if score is None:
                raise ValueError("inner validation has no qualified drugs")
            row.update({"validation_macro_spearman": score, "validation_rmse": metrics["rmse"]})
            if score > best_score + 1e-8 or (abs(score - best_score) <= 1e-8 and metrics["rmse"] < best_rmse):
                best_score, best_rmse, best_epoch = score, metrics["rmse"], epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            if epoch - best_epoch >= patience:
                row["early_stop"] = True
                history.append(row)
                break
        history.append(row)
    if best_state is not None:
        model.load_state_dict(best_state)
    if isinstance(model, AdditiveInteraction):
        model.refresh_reference(expression[torch.unique(cindex)], fingerprint[torch.unique(dindex)])
    model.cpu().eval()
    return TrainedA2(name, model, None, best_epoch if best_state is not None else epochs,
                     history, sum(p.numel() for p in model.parameters()), names)
