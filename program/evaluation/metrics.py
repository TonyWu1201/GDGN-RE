"""Strict, sample-aligned development metrics for Core LN_IC50 predictions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


REQUIRED = {"sample_id", "cell_id", "compound_id", "y_true", "y_pred"}


def _correlation(y: np.ndarray, pred: np.ndarray, method: str) -> float | None:
    if len(y) < 2 or np.ptp(y) == 0 or np.ptp(pred) == 0:
        return None
    value = spearmanr(y, pred).statistic if method == "spearman" else pearsonr(y, pred).statistic
    return float(value) if np.isfinite(value) else None


def _pair_accuracy(y: np.ndarray, pred: np.ndarray) -> float | None:
    # The metric is descriptive; pairwise training uses a separately frozen noise threshold.
    truth = np.sign(y[:, None] - y[None, :])
    guess = np.sign(pred[:, None] - pred[None, :])
    keep = np.triu(truth != 0, 1)
    return float(np.mean(guess[keep] == truth[keep])) if keep.any() else None


def evaluate_predictions(frame: pd.DataFrame, min_cells: int = 10) -> dict:
    missing = REQUIRED - set(frame)
    if missing:
        raise ValueError(f"prediction columns missing: {sorted(missing)}")
    if frame.empty or frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
        raise ValueError("predictions must have unique, non-null sample_id values")
    if frame[["cell_id", "compound_id"]].isna().any().any():
        raise ValueError("prediction entity ID missing")
    y = frame["y_true"].to_numpy(dtype=float)
    p = frame["y_pred"].to_numpy(dtype=float)
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("non-finite label or prediction: entire run must fail")
    if min_cells < 2:
        raise ValueError("min_cells must be at least 2")

    drug_rows = []
    scores = []
    constant_count = 0
    for drug, part in frame.groupby("compound_id", sort=True):
        yd = part["y_true"].to_numpy(dtype=float)
        pd_ = part["y_pred"].to_numpy(dtype=float)
        qualified = part["cell_id"].nunique() >= min_cells and np.ptp(yd) > 0
        rho = _correlation(yd, pd_, "spearman") if qualified else None
        pcc = _correlation(yd, pd_, "pearson") if qualified else None
        constant = bool(qualified and np.ptp(pd_) == 0)
        if qualified:
            scores.append(0.0 if rho is None else rho)
            constant_count += int(constant)
        drug_rows.append({
            "compound_id": str(drug), "n_samples": len(part), "n_cells": int(part["cell_id"].nunique()),
            "label_variance": bool(np.ptp(yd) > 0), "qualified": bool(qualified),
            "spearman_raw": rho, "spearman_selection": (0.0 if rho is None else rho) if qualified else None,
            "pcc_raw": pcc, "constant_prediction": constant,
            "pair_accuracy": _pair_accuracy(yd, pd_) if qualified else None,
        })
    err = p - y
    ss = float(np.sum((y - y.mean()) ** 2))
    result = {
        "n_samples": len(frame), "n_drugs": int(frame["compound_id"].nunique()),
        "n_qualified_drugs": len(scores), "qualified_coverage": len(scores) / frame["compound_id"].nunique(),
        "constant_qualified_drugs": constant_count,
        "macro_spearman_selection": float(np.mean(scores)) if scores else None,
        "rmse": float(np.sqrt(np.mean(err * err))), "mae": float(np.mean(np.abs(err))),
        "r2": float(1 - np.sum(err * err) / ss) if ss > 0 else None,
        "pooled_pcc": _correlation(y, p, "pearson"),
        "pooled_spearman": _correlation(y, p, "spearman"),
        "per_drug": drug_rows,
    }
    valid_pcc = [r["pcc_raw"] for r in drug_rows if r["qualified"] and r["pcc_raw"] is not None]
    result["macro_pcc_raw_available"] = float(np.mean(valid_pcc)) if valid_pcc else None
    pair_values = [r["pair_accuracy"] for r in drug_rows if r["qualified"] and r["pair_accuracy"] is not None]
    result["macro_pair_accuracy"] = float(np.mean(pair_values)) if pair_values else None
    cell_scores = []
    for _, part in frame.groupby("cell_id", sort=True):
        score = _correlation(part["y_true"].to_numpy(dtype=float), part["y_pred"].to_numpy(dtype=float), "spearman")
        if score is not None:
            cell_scores.append(score)
    result["cell_cross_drug_spearman_descriptive"] = float(np.mean(cell_scores)) if cell_scores else None
    result["n_cells_cross_drug_valid"] = len(cell_scores)
    return result


def summarize_strata(frame: pd.DataFrame, min_cells: int = 10) -> dict:
    out = {}
    for column in ("oncotree_lineage", "any_extrapolation", "chemical_group",
                   "train_cell_coverage_bin", "target_annotation_available"):
        if column in frame:
            out[column] = {}
            for value, part in frame.groupby(column, dropna=False, sort=True):
                metrics = evaluate_predictions(part, min_cells=min_cells)
                out[column][str(value)] = {key: metrics[key] for key in (
                    "n_samples", "n_qualified_drugs", "macro_spearman_selection", "rmse", "mae"
                )}
    return out


def evaluate_file(path: Path, min_cells: int = 10) -> dict:
    frame = pd.read_parquet(path)
    return {"overall": evaluate_predictions(frame, min_cells), "strata": summarize_strata(frame, min_cells)}


def write_metrics(path: Path, metrics: dict) -> None:
    path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
