"""A0 smoke and A1 selected-configuration fitting; outer test is used only after selection."""

from __future__ import annotations

import json
import pickle
import platform
import subprocess
import time
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from program.common import paths as P
from program.common.runlog import git_commit, sha256_file
from program.evaluation.metrics import evaluate_predictions, summarize_strata, write_metrics
from program.models.baselines import ALL_MODELS, DIAGNOSTIC, LEARNING, NAIVE, deterministic, fit_baseline, grid
from program.models.dataset import PreparedData, ProjectData, inner_split
from program.models.neural import fit_mlp
from program.models.run_registry import RunRecord, stable_hash, write_json


SEEDS = (42, 3407, 8128)
INNER_SEED = 20260934
MIN_CELLS = 10


@lru_cache(maxsize=1)
def _expression_sha256() -> str:
    return sha256_file(P.DEPMAP_EXPRESSION_CSV)


@lru_cache(maxsize=1)
def _core_sha256() -> str:
    return sha256_file(P.COHORTS_DIR / "core_samples.parquet")


def _git_dirty_hash() -> str:
    result = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"],
                            cwd=P.REPO_ROOT, capture_output=True, check=False)
    return stable_hash(result.stdout.decode("utf-8", errors="replace")) if result.returncode == 0 else "unknown"


def a1_plan() -> dict:
    budget = json.loads((P.REPO_ROOT / "configs" / "experiment_budget.json").read_text(encoding="utf-8"))
    entries = []
    for protocol in ("lco", "lpo"):
        fold = "fold_quick_screening_0" if protocol == "lco" else "fold_0"
        for name in ALL_MODELS:
            seeds = (42,) if deterministic(name) else SEEDS
            for seed in seeds:
                entries.append({"protocol": protocol, "fold": fold, "model": name, "seed": seed,
                                "max_trials": len(grid(name)), "class": "control" if name in NAIVE + DIAGNOSTIC else "learning"})
    naive = sum(e["class"] == "control" for e in entries)
    learning = len(entries) - naive
    limit = budget["rounds"]["a1"]
    if naive > limit["naive_and_diagnostic_runs"] or learning > limit["learning_models_max_runs"]:
        raise ValueError("A1 selected-configuration budget exceeded")
    if any(e["max_trials"] > budget["hyperparameter_trials_cap_per_family"] for e in entries):
        raise ValueError("A1 per-family hyperparameter budget exceeded")
    cohort_hash = (P.COHORTS_DIR / "cohort_hash.txt").read_text(encoding="utf-8").strip()
    split_dir = P.SPLITS_DIR / cohort_hash
    expression_cfg = json.loads((P.RUNS_DIR / "E1_build_expression_features" / "config_resolved.json").read_text(encoding="utf-8"))
    fingerprint_cfg = json.loads((P.RUNS_DIR / "E2_build_fingerprints" / "config_resolved.json").read_text(encoding="utf-8"))
    return {"status": "planned", "cohort_hash": cohort_hash,
            "feature_cache": {"expression_data_hash": expression_cfg["data_hash"],
                              "fingerprint_data_hash": fingerprint_cfg["data_hash"],
                              "fingerprint_preprocess_hash": fingerprint_cfg["preprocess_hash"]},
            "split_sha256": {p: sha256_file(split_dir / p / ("fold_quick_screening_0.csv" if p == "lco" else "fold_0.csv"))
                             for p in ("lco", "lpo")},
            "entries": entries, "selected_fit_count": len(entries),
            "control_count": naive, "learning_count": learning,
            "limits": {"controls": limit["naive_and_diagnostic_runs"], "learning": limit["learning_models_max_runs"]},
            "not_applicable": {"similar_drug_neighbor": "LDO-specific; LCO/LPO do not test new drugs",
                               "legacy_representative": "conditional historical bridge; not part of default A1"}}


def _a0_subset(frame: pd.DataFrame) -> pd.DataFrame:
    drugs = sorted(frame["compound_id"].unique())[:12]
    parts = []
    for role, n_cells in (("train", 100), ("test", 40)):
        side = frame[frame["role"] == role]
        cells = sorted(side["cell_id"].unique())[:n_cells]
        parts.append(side[side["cell_id"].isin(cells) & side["compound_id"].isin(drugs)])
    result = pd.concat(parts, ignore_index=True).sort_values("sample_id", kind="stable").reset_index(drop=True)
    if len(result) > 2000 or set(result["role"]) != {"train", "test"}:
        raise ValueError("A0 sampling failed size/role checks")
    return result


def _restrict_train(frame: pd.DataFrame, cell_fraction: float, drug_fraction: float) -> pd.DataFrame:
    if not 0 < cell_fraction <= 1 or not 0 < drug_fraction <= 1:
        raise ValueError("learning curve fractions must be in (0,1]")
    out = frame
    if cell_fraction < 1:
        groups = sorted(out["cell_group"].unique())
        out = out[out["cell_group"].isin(groups[:max(2, round(len(groups) * cell_fraction))])]
    if drug_fraction < 1:
        drugs = sorted(out["compound_id"].unique())
        out = out[out["compound_id"].isin(drugs[:max(2, round(len(drugs) * drug_fraction))])]
    if out.empty:
        raise ValueError("learning curve selected no training samples")
    return out.copy()


def _diagnostic_train(train: pd.DataFrame, diagnostic: str, seed: int) -> pd.DataFrame:
    if diagnostic != "label_shuffle":
        return train
    rng = np.random.default_rng(seed)
    result = train.copy()
    for _, idx in result.groupby("compound_id").groups.items():
        result.loc[idx, "y"] = rng.permutation(result.loc[idx, "y"].to_numpy())
    return result


def _prepared(data: ProjectData, train: pd.DataFrame, validation: pd.DataFrame, diagnostic: str) -> PreparedData:
    prepared = data.prepare(train, validation)
    if diagnostic == "expression_ablation":
        prepared.expression[:] = 0.0
    return prepared


def _fit(name: str, prepared: PreparedData, params: dict, seed: int,
         validation: pd.DataFrame | None, fixed_epochs: int | None = None):
    if name == "mlp":
        return fit_mlp(prepared, params, seed, validation=validation, fixed_epochs=fixed_epochs, min_cells=MIN_CELLS)
    return fit_baseline(name, prepared, params, seed)


def _config(data: ProjectData, stage: str, model: str, seed: int, diagnostic: str,
            cell_fraction: float, drug_fraction: float) -> dict:
    config_files = [P.REPO_ROOT / "configs" / x for x in ("cohort.json", "split_policy.json", "models.json", "baselines.json", "experiment_budget.json")]
    return {
        "stage": stage, "model": model, "protocol": data.protocol, "fold": data.fold, "seed": seed,
        "diagnostic": diagnostic, "cell_fraction": cell_fraction, "drug_fraction": drug_fraction,
        "cohort_hash": data.cohort_hash, "split_sha256": data.split_hash,
        "core_samples_sha256": _core_sha256(),
        "cell_master_sha256": sha256_file(P.ENTITIES_DIR / "cell_master.csv"),
        "compound_master_sha256": sha256_file(P.ENTITIES_DIR / "compound_master.csv"),
        "fingerprint_sha256": data.fingerprint_hash, "fingerprint_config": data.fingerprint_config,
        "expression_source_sha256": _expression_sha256(), "feature_schema": {"hvg": 1000, "fingerprint_bits": 2048},
        "implementation_sha256": {name: sha256_file(P.REPO_ROOT / name) for name in (
            "program/models/dataset.py", "program/models/baselines.py", "program/models/neural.py",
            "program/experiments/pipeline.py", "program/evaluation/metrics.py",
            "program/features/expression_preprocess.py")},
        "source_manifest_sha256": sha256_file(P.DATA_DIR / "manifests" / "source_manifest.csv"),
        "uv_lock_sha256": sha256_file(P.REPO_ROOT / "uv.lock"),
        "config_sha256": {path.name: sha256_file(path) for path in config_files},
        "git_commit": git_commit(), "python": platform.python_version(),
        "git_dirty_or_patch_hash": _git_dirty_hash(),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "inner_seed": INNER_SEED, "min_qualified_cells": MIN_CELLS,
        "selection_rule": "macro Spearman after RMSE <= global mean RMSE * 1.02; fallback minimum RMSE if none",
        "trial_grid": grid(model) if stage == "A1" else ([{"alpha": 1.0}] if model == "ridge" else [{}]),
    }


def run_selected(stage: str, model: str = "ridge", protocol: str = "lco", seed: int = 42,
                 resume: bool = False, diagnostic: str = "none", cell_fraction: float = 1.0,
                 drug_fraction: float = 1.0) -> Path:
    if stage not in {"A0", "A1"} or model not in ALL_MODELS:
        raise ValueError("M1–M3 support A0/A1 and listed baseline models only")
    if stage == "A0" and (model not in {"ridge", "global_mean"} or protocol != "lco" or diagnostic != "none"):
        raise ValueError("A0 is the fixed LCO global-mean/Ridge smoke")
    if stage == "A1" and seed not in ((42,) if deterministic(model) else SEEDS):
        raise ValueError("seed not registered for this A1 model")
    if diagnostic not in {"none", "label_shuffle", "expression_ablation"}:
        raise ValueError(diagnostic)
    data = ProjectData(protocol=protocol)
    config = _config(data, stage, model, seed, diagnostic, cell_fraction, drug_fraction)
    run = RunRecord(stage, model, protocol, data.fold, seed, config, resume=resume)
    try:
        run.status("running")
        frame = _a0_subset(data.frame) if stage == "A0" else data.frame
        outer_train = frame.loc[frame["role"] == "train"].copy()
        outer_test = frame.loc[frame["role"] == "test"].copy()
        outer_train = _restrict_train(outer_train, cell_fraction, drug_fraction)
        inner_train, inner_val = inner_split(outer_train, protocol, seed=INNER_SEED)
        inner_train = _diagnostic_train(inner_train, diagnostic, seed)
        inner = _prepared(data, inner_train, inner_val, diagnostic)
        baseline = float(np.sqrt(np.mean((inner_val["y"].to_numpy() - inner_train["y"].mean()) ** 2)))
        trials = []
        candidates = []
        for params in config["trial_grid"]:
            start = time.monotonic()
            try:
                with warnings.catch_warnings(record=True) as observed:
                    warnings.simplefilter("always")
                    fitted = _fit(model, inner, params, seed, inner_val)
            except BaseException as trial_error:
                trials.append({"params": params, "status": "failed", "error": str(trial_error),
                               "runtime_seconds": round(time.monotonic() - start, 3)})
                write_json(run.path / "trial_history.json", {"trials": trials})
                raise
            pred = fitted.predict(inner, inner_val)
            pred_frame = inner_val[["sample_id", "cell_id", "compound_id"]].copy()
            pred_frame["y_true"], pred_frame["y_pred"] = inner_val["y"].to_numpy(), pred
            metrics = evaluate_predictions(pred_frame, min_cells=MIN_CELLS)
            item = {"params": params, "inner_macro_spearman": metrics["macro_spearman_selection"],
                    "inner_rmse": metrics["rmse"], "selected_epoch": getattr(fitted, "selected_epoch", None),
                    "runtime_seconds": round(time.monotonic() - start, 3),
                    "submodels_fitted": len(getattr(fitted, "per_drug", None) or {}),
                    "warnings": [f"{w.category.__name__}: {w.message}" for w in observed]}
            trials.append(item)
            write_json(run.path / "trial_history.json", {"trials": trials})
            candidates.append((item, fitted))
        eligible = [(item, obj) for item, obj in candidates if item["inner_rmse"] <= baseline * 1.02]
        if eligible:
            chosen, _ = max(eligible, key=lambda pair: (
                -np.inf if pair[0]["inner_macro_spearman"] is None else pair[0]["inner_macro_spearman"],
                -pair[0]["inner_rmse"]
            ))
        else:
            chosen, _ = min(candidates, key=lambda pair: pair[0]["inner_rmse"])
        outer_train = _diagnostic_train(outer_train, diagnostic, seed)
        outer = _prepared(data, outer_train, outer_test, diagnostic)
        with warnings.catch_warnings(record=True) as final_observed:
            warnings.simplefilter("always")
            fitted = _fit(model, outer, chosen["params"], seed, None, fixed_epochs=chosen["selected_epoch"])
        pred = fitted.predict(outer, outer_test)
        checkpoint = run.path / "checkpoint.pkl"
        with checkpoint.open("wb") as handle:
            pickle.dump(fitted, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with checkpoint.open("rb") as handle:
            loaded = pickle.load(handle)
        if not np.array_equal(pred, loaded.predict(outer, outer_test)):
            raise ValueError("checkpoint reload changed predictions")
        prediction = outer_test[["sample_id", "cell_id", "compound_id", "normalized_parent_id", "scaffold_id",
                                 "cell_group", "chemical_group", "oncotree_lineage", "any_extrapolation"]].copy()
        train_cell_count = outer_train.groupby("compound_id")["cell_id"].nunique()
        quartiles = pd.qcut(train_cell_count, q=min(4, train_cell_count.nunique()), duplicates="drop") if train_cell_count.nunique() > 1 else None
        coverage = quartiles.astype(str).to_dict() if quartiles is not None else {drug: "single_bin" for drug in train_cell_count.index}
        prediction["train_cell_coverage_bin"] = prediction["compound_id"].map(coverage).fillna("unseen_drug")
        target = dict(zip(data.compound_master["compound_id"], data.compound_master["putative_target"].fillna("").ne("")))
        prediction["target_annotation_available"] = prediction["compound_id"].map(target).fillna(False).astype(bool)
        prediction["protocol"], prediction["fold"], prediction["seed"] = protocol, data.fold, seed
        prediction["split"], prediction["y_true"], prediction["y_pred"] = "test", outer_test["y"].to_numpy(), pred
        prediction.to_parquet(run.path / "predictions.parquet", index=False)
        metrics = {"overall": evaluate_predictions(prediction, min_cells=MIN_CELLS),
                   "strata": summarize_strata(prediction, min_cells=MIN_CELLS)}
        write_metrics(run.path / "metrics.json", metrics)
        write_json(run.path / "training_history.json", {"trials": trials, "final": getattr(fitted, "history", []),
                   "final_warnings": [f"{w.category.__name__}: {w.message}" for w in final_observed],
                   "rmse_guard_reference": baseline, "rmse_guard_passed": bool(eligible),
                   "low_coverage_drugs": getattr(fitted, "low_coverage_drugs", 0),
                   "final_submodels_fitted": len(getattr(fitted, "per_drug", None) or {})})
        write_json(run.path / "selected_epoch.json", {"selected_epoch": getattr(fitted, "selected_epoch", None),
                   "params": chosen["params"]})
        write_json(run.path / "preprocessing_state.json", outer.state)
        write_json(run.path / "gene_order.json", {"gene_pairs_in_order": outer.gene_order})
        write_json(run.path / "sample_manifest.json", {"train_ids": outer_train["sample_id"].tolist(),
                   "test_ids": outer_test["sample_id"].tolist(), "inner_train_count": len(inner_train),
                   "inner_validation_count": len(inner_val),
                   "train_cells": int(outer_train["cell_id"].nunique()), "train_drugs": int(outer_train["compound_id"].nunique()),
                   "test_cells": int(outer_test["cell_id"].nunique()), "test_drugs": int(outer_test["compound_id"].nunique())})
        (run.path / "error_log.txt").write_text("", encoding="utf-8")
        gpu_hours = (time.monotonic() - run.start_time) / 3600 if model == "mlp" and torch.cuda.is_available() else 0.0
        peak = torch.cuda.max_memory_allocated() if model == "mlp" and torch.cuda.is_available() else None
        run.status("smoke" if stage == "A0" else "succeeded", gpu_hours=gpu_hours, peak_gpu_bytes=peak,
                   n_train=len(outer_train), n_test=len(outer_test), n_trials=len(trials),
                   n_train_cells=int(outer_train["cell_id"].nunique()), n_train_drugs=int(outer_train["compound_id"].nunique()),
                   n_test_cells=int(outer_test["cell_id"].nunique()), n_test_drugs=int(outer_test["compound_id"].nunique()))
        return run.path
    except KeyboardInterrupt:
        run.status("incomplete", error_type="KeyboardInterrupt")
        raise
    except BaseException as exc:
        run.fail(exc)
        raise
