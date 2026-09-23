"""Fit expression preprocessing on training entities and reuse its state unchanged."""

from __future__ import annotations

import numpy as np


def fit_preprocessing(
    train_expr: np.ndarray,
    impute: str = "median",
    scale: bool = True,
    min_variance: float = 1e-8,
    hvg_n: int | None = None,
    entity_ids: list[str] | None = None,
) -> dict:
    """Learn missingness, imputation, variance, HVG and scaling on unique train entities."""
    if impute != "median":
        raise ValueError("only median imputation is supported")
    expr = np.asarray(train_expr, dtype=float)
    if expr.ndim != 2 or expr.shape[0] < 2:
        raise ValueError("at least two training entities are required")
    if entity_ids is not None:
        if len(entity_ids) != expr.shape[0]:
            raise ValueError("entity_ids length does not match rows")
        first = {}
        for i, entity in enumerate(entity_ids):
            first.setdefault(entity, i)
        expr = expr[list(first.values())]
    if expr.shape[0] < 2:
        raise ValueError("at least two unique training entities are required")

    finite = np.isfinite(expr)
    missing_rate = (~finite).mean(axis=0)
    available = np.where(missing_rate < 0.5)[0]
    medians_all = np.full(expr.shape[1], np.nan, dtype=float)
    for col in available:
        medians_all[col] = np.median(expr[finite[:, col], col])
    filled = np.where(finite, expr, medians_all)
    variance = filled.var(axis=0, ddof=1)
    kept = np.where((missing_rate < 0.5) & np.isfinite(variance) & (variance > min_variance))[0]
    if not len(kept):
        raise ValueError("no expression genes passed training-only filters")

    if hvg_n is None:
        selected = kept
    else:
        order = np.argsort(-variance[kept], kind="stable")[: min(hvg_n, len(kept))]
        selected = np.sort(kept[order])
    train_selected = filled[:, selected]
    mean = train_selected.mean(axis=0)
    std = train_selected.std(axis=0, ddof=1)
    std = np.where(np.isfinite(std) & (std > min_variance), std, 1.0)

    return {
        "kept_columns": kept.tolist(),
        "selected_columns": selected.tolist(),
        "medians": medians_all[kept].tolist(),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "entity_scale": float(np.median(std)),
        "impute": impute,
        "scale": scale,
        "min_variance": min_variance,
        "hvg_n": hvg_n,
        "n_train_entities": int(expr.shape[0]),
        "n_input_columns": int(expr.shape[1]),
    }


def transform_with_state(expr: np.ndarray, state: dict) -> np.ndarray:
    """Apply only saved training statistics to any validation or test rows."""
    expr = np.asarray(expr, dtype=float)
    if expr.ndim != 2 or expr.shape[1] != state["n_input_columns"]:
        raise ValueError("expression column count differs from fitted state")
    selected = state["selected_columns"]
    kept_pos = {col: i for i, col in enumerate(state["kept_columns"])}
    medians = np.asarray([state["medians"][kept_pos[col]] for col in selected])
    out = expr[:, selected].copy()
    out = np.where(np.isfinite(out), out, medians)
    if state["scale"]:
        out = (out - np.asarray(state["mean"])) / np.asarray(state["std"])
    return out


def constant_label_guard(y: np.ndarray) -> bool:
    """Whether a label vector contains more than one distinct value."""
    return len(set(np.asarray(y).tolist())) > 1
