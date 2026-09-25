"""A1 controls and strong simple baselines with a shared fit/predict contract."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder

from program.models.dataset import PreparedData


NAIVE = ("global_mean", "drug_mean", "cell_mean", "two_way_additive")
DIAGNOSTIC = ("structure_only_ridge", "expression_only_ridge")
LEARNING = ("ridge", "elasticnet", "per_drug_elasticnet", "tree", "neighbor", "mlp")
ALL_MODELS = NAIVE + DIAGNOSTIC + LEARNING


def grid(name: str) -> list[dict]:
    if name in NAIVE:
        return [{}]
    if name in DIAGNOSTIC or name == "ridge":
        return [{"alpha": x} for x in (0.1, 1.0, 10.0)]
    if name in {"elasticnet", "per_drug_elasticnet"}:
        return [{"alpha": a, "l1_ratio": r} for a in (0.01, 0.1) for r in (0.1, 0.5)]
    if name == "tree":
        return [{"n_estimators": 64, "max_features": f, "max_depth": d}
                for f in (0.1, 0.3) for d in (12, 20)]
    if name == "neighbor":
        return [{"k": k} for k in (1, 3, 5)]
    if name == "mlp":
        return [{"lr": lr, "weight_decay": wd, "dropout": drop}
                for lr in (3e-4, 1e-3) for wd in (1e-4, 1e-3) for drop in (0.1, 0.2)]
    raise ValueError(name)


def deterministic(name: str) -> bool:
    return name not in {"tree", "mlp"}


@dataclass
class FittedBaseline:
    name: str
    estimator: object | None
    side: str = "both"
    global_mean: float = 0.0
    drug_means: dict[str, float] | None = None
    cell_means: dict[str, float] | None = None
    per_drug: dict[str, object] | None = None
    encoder: OneHotEncoder | None = None
    train: pd.DataFrame | None = None
    prepared: PreparedData | None = None
    k: int = 1
    low_coverage_drugs: int = 0

    def predict(self, prepared: PreparedData, frame: pd.DataFrame) -> np.ndarray:
        if self.name == "global_mean":
            return np.full(len(frame), self.global_mean)
        if self.name == "drug_mean":
            return frame["compound_id"].map(self.drug_means).fillna(self.global_mean).to_numpy(dtype=float)
        if self.name == "cell_mean":
            return frame["cell_id"].map(self.cell_means).fillna(self.global_mean).to_numpy(dtype=float)
        if self.name == "two_way_additive":
            categories = self.encoder.transform(frame[["compound_id", "cell_id"]])
            return self.estimator.predict(categories)
        if self.name == "per_drug_elasticnet":
            out = frame["compound_id"].map(self.drug_means).fillna(self.global_mean).to_numpy(dtype=float, copy=True)
            x = prepared.pair_matrix(frame, "expression")
            for drug, model in (self.per_drug or {}).items():
                ix = np.flatnonzero(frame["compound_id"].to_numpy() == drug)
                if len(ix):
                    out[ix] = model.predict(x[ix])
            return out
        if self.name == "neighbor":
            return self._neighbor_predict(prepared, frame)
        return np.asarray(self.estimator.predict(prepared.pair_matrix(frame, self.side)), dtype=float)

    def _neighbor_predict(self, prepared: PreparedData, frame: pd.DataFrame) -> np.ndarray:
        assert self.train is not None
        out = frame["compound_id"].map(self.drug_means).fillna(self.global_mean).to_numpy(dtype=float, copy=True)
        mids = {v: i for i, v in enumerate(prepared.model_ids)}
        expr = prepared.expression
        for drug, query in frame.groupby("compound_id", sort=False):
            known = self.train[self.train["compound_id"] == drug]
            if known.empty:
                continue
            x_train = expr[[mids[m] for m in known["profile_rna_model_id"]]]
            x_query = expr[[mids[m] for m in query["profile_rna_model_id"]]]
            nn = NearestNeighbors(n_neighbors=min(self.k, len(known)), metric="euclidean")
            nn.fit(x_train)
            neighbours = nn.kneighbors(x_query, return_distance=False)
            loc = frame.index.get_indexer(query.index)
            out[loc] = known["y"].to_numpy()[neighbours].mean(axis=1)
        return out


def fit_baseline(name: str, prepared: PreparedData, params: dict, seed: int = 42) -> FittedBaseline:
    if name not in ALL_MODELS or name == "mlp":
        raise ValueError(f"unsupported sklearn baseline {name}")
    train = prepared.train
    y = train["y"].to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError("non-finite training label")
    mean = float(y.mean())
    drug_mean = train.groupby("compound_id")["y"].mean().to_dict()
    cell_mean = train.groupby("cell_id")["y"].mean().to_dict()
    fitted = FittedBaseline(name, None, global_mean=mean, drug_means=drug_mean, cell_means=cell_mean)
    if name in {"global_mean", "drug_mean", "cell_mean"}:
        return fitted
    if name == "two_way_additive":
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)
        x = encoder.fit_transform(train[["compound_id", "cell_id"]])
        model = Ridge(alpha=0.01, solver="lsqr").fit(x, y)
        fitted.estimator, fitted.encoder = model, encoder
        return fitted
    if name == "per_drug_elasticnet":
        models = {}
        x = prepared.pair_matrix(train, "expression")
        drugs = train["compound_id"].to_numpy()
        for drug in sorted(set(drugs)):
            ix = np.flatnonzero(drugs == drug)
            if len(ix) < 30 or np.ptp(y[ix]) == 0:
                fitted.low_coverage_drugs += 1
                continue
            models[drug] = ElasticNet(alpha=params["alpha"], l1_ratio=params["l1_ratio"],
                                      max_iter=3000, random_state=seed).fit(x[ix], y[ix])
        fitted.per_drug = models
        return fitted
    if name == "neighbor":
        fitted.train, fitted.k = train.copy(), params["k"]
        return fitted
    side = "structure" if name == "structure_only_ridge" else "expression" if name == "expression_only_ridge" else "both"
    x = prepared.pair_matrix(train, side)
    if name in {"ridge", "structure_only_ridge", "expression_only_ridge"}:
        model = Ridge(alpha=params["alpha"], solver="lsqr")
    elif name == "elasticnet":
        model = ElasticNet(alpha=params["alpha"], l1_ratio=params["l1_ratio"],
                           max_iter=3000, random_state=seed)
    else:
        model = ExtraTreesRegressor(**params, n_jobs=-1, random_state=seed)
    model.fit(x, y)
    if name == "tree":
        # Parallel tree accumulation can differ in the last bit across calls.
        model.n_jobs = 1
    fitted.estimator, fitted.side = model, side
    return fitted
