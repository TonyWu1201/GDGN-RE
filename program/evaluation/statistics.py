"""Paired cluster bootstrap. Seeds are averaged before biological resampling."""

from __future__ import annotations

import numpy as np
import pandas as pd

from program.evaluation.metrics import evaluate_predictions


def align_pair(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    for name, df in (("left", left), ("right", right)):
        if df["sample_id"].isna().any() or df["sample_id"].duplicated().any():
            raise ValueError(f"{name} has missing or duplicate sample_id")
    if set(left["sample_id"]) != set(right["sample_id"]):
        raise ValueError("paired predictions must cover identical sample IDs")
    cols = [c for c in ("sample_id", "cell_id", "compound_id", "cell_group", "chemical_group", "y_true") if c in left]
    aligned = left[cols + ["y_pred"]].merge(
        right[["sample_id", "y_true", "y_pred"]], on="sample_id", validate="one_to_one", suffixes=("_a", "_b"), sort=False
    )
    if not np.array_equal(aligned["y_true_a"].to_numpy(), aligned["y_true_b"].to_numpy()):
        raise ValueError("paired predictions have different labels")
    return aligned.rename(columns={"y_true_a": "y_true"}).drop(columns="y_true_b")


def paired_cluster_bootstrap(
    left: pd.DataFrame, right: pd.DataFrame, protocol: str, n_boot: int = 1000,
    seed: int = 42, min_cells: int = 10,
) -> dict:
    paired = align_pair(left, right)
    group_cols = {"lco": ("cell_group",), "ldo_so": ("chemical_group",), "db": ("cell_group", "chemical_group")}
    if protocol not in group_cols:
        raise ValueError(f"unsupported bootstrap protocol {protocol}")
    for col in group_cols[protocol]:
        if col not in paired or paired[col].isna().any():
            raise ValueError(f"cluster group missing: {col}")
    rng = np.random.default_rng(seed)
    effects = []
    cell_parts = {g: part for g, part in paired.groupby("cell_group", sort=True)} if "cell_group" in paired else {}
    drug_parts = {g: part for g, part in paired.groupby("chemical_group", sort=True)} if "chemical_group" in paired else {}
    both_parts = {(c, d): part for (c, d), part in paired.groupby(["cell_group", "chemical_group"], sort=True)} if protocol == "db" else {}
    for _ in range(n_boot):
        fragments = []
        if protocol == "lco":
            groups = np.array(sorted(cell_parts))
            for draw, group in enumerate(rng.choice(groups, size=len(groups), replace=True)):
                part = cell_parts[group].copy()
                part["cell_id"] = part["cell_id"].astype(str) + f"#cell_draw{draw}"
                part["sample_id"] = part["sample_id"].astype(str) + f"#cell_draw{draw}"
                fragments.append(part)
        elif protocol == "ldo_so":
            groups = np.array(sorted(drug_parts))
            for draw, group in enumerate(rng.choice(groups, size=len(groups), replace=True)):
                part = drug_parts[group].copy()
                part["compound_id"] = part["compound_id"].astype(str) + f"#drug_draw{draw}"
                part["sample_id"] = part["sample_id"].astype(str) + f"#drug_draw{draw}"
                fragments.append(part)
        else:
            cells = np.array(sorted(cell_parts))
            drugs = np.array(sorted(drug_parts))
            for ci, cell_group in enumerate(rng.choice(cells, size=len(cells), replace=True)):
                for di, chemical_group in enumerate(rng.choice(drugs, size=len(drugs), replace=True)):
                    part = both_parts.get((cell_group, chemical_group))
                    if part is None:
                        continue
                    part = part.copy()
                    part["cell_id"] = part["cell_id"].astype(str) + f"#cell_draw{ci}"
                    part["compound_id"] = part["compound_id"].astype(str) + f"#drug_draw{di}"
                    part["sample_id"] = part["sample_id"].astype(str) + f"#cell_draw{ci}#drug_draw{di}"
                    fragments.append(part)
        if not fragments:
            continue
        sample = pd.concat(fragments, ignore_index=True)
        a = sample.rename(columns={"y_pred_a": "y_pred"}).drop(columns="y_pred_b")
        b = sample.rename(columns={"y_pred_b": "y_pred"}).drop(columns="y_pred_a")
        ma = evaluate_predictions(a, min_cells)["macro_spearman_selection"]
        mb = evaluate_predictions(b, min_cells)["macro_spearman_selection"]
        if ma is not None and mb is not None:
            effects.append(ma - mb)
    if not effects:
        raise ValueError("no valid bootstrap replicates")
    lo, hi = np.quantile(effects, [0.025, 0.975])
    return {"protocol": protocol, "n_boot_requested": n_boot, "n_boot_valid": len(effects),
            "difference_mean": float(np.mean(effects)), "ci95": [float(lo), float(hi)], "seed": seed}


def holm_adjust(p_values: list[float]) -> list[float]:
    """Family-wise correction for a predeclared exploratory comparison family."""
    if any(not np.isfinite(p) or p < 0 or p > 1 for p in p_values):
        raise ValueError("p-values must be finite in [0, 1]")
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted.tolist()
