"""A2 development fits and preflight; formal runs are explicit and budgeted."""

from __future__ import annotations

import json
import pickle
import platform
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from program.common import paths as P
from program.common.runlog import git_commit, sha256_file
from program.evaluation.metrics import assert_metrics_close, evaluate_file, evaluate_predictions, summarize_strata, write_metrics
from program.experiments.pipeline import (
    INNER_SEED, MIN_CELLS, SEEDS, _a0_subset, _core_sha256, _expression_sha256, _git_dirty_hash,
)
from program.features.a2_pathways import PathwayFeatures, prepare_pathways
from program.models.a2 import A2_MODELS, PATHWAY_MODELS, fit_a2
from program.models.dataset import PreparedData, ProjectData, inner_split
from program.models.run_registry import RunRecord, stable_hash, write_json


def _models_config() -> dict:
    return json.loads((P.REPO_ROOT / "configs" / "models.json").read_text(encoding="utf-8"))


def _semantic_variant(name: str, ranking: dict | None = None,
                      trial_grid: list[dict] | None = None, feature_schema: dict | None = None) -> str:
    return stable_hash({"model": name, "ranking": ranking if name == "r2_rank" else None,
                        "trial_grid": trial_grid if trial_grid is not None else _grid(name),
                        "feature_schema": feature_schema if feature_schema is not None else
                        {"hvg": 1000, "fingerprint_bits": 2048, "pathway_mode": _mode(name)}})


def a2_plan() -> dict:
    models = _models_config()
    selected = models["a2_execution"]["selected_models"]
    budget = json.loads((P.REPO_ROOT / "configs" / "experiment_budget.json").read_text(encoding="utf-8"))
    if not isinstance(selected, list) or not selected or len(selected) != len(set(selected)):
        raise ValueError("A2 selected models must be a nonempty unique list")
    if any(name not in A2_MODELS for name in selected) or len(selected) > 4:
        raise ValueError("A2 selected model list exceeds four registered configurations")
    if "r0" in selected and selected[0] != "r0":
        raise ValueError("A2 refitted R0 must be the first selected configuration")
    historical: set[tuple[str, str, int]] = set()
    for path in (P.DATA_DIR / "runs" / "A2").rglob("config_resolved.json"):
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
            variant = _semantic_variant(old["model"], old.get("ranking"),
                                        old.get("trial_grid"), old.get("feature_schema"))
            historical.add((variant, old["protocol"], int(old["seed"])))
        except (OSError, ValueError, KeyError):
            continue
    rank_cfg = models["a2_execution"]["ranking"]
    if "r2_rank" in selected and (rank_cfg["noise_threshold"] is None or rank_cfg["temperature"] is None):
        raise ValueError("R2 ranking parameters must be frozen in models.json before selection")
    selected_variants = {name: _semantic_variant(name, rank_cfg) for name in selected}
    if len(set(selected_variants.values()) | {variant for variant, _, _ in historical}) > 4:
        raise ValueError("A2 historical plus selected configurations exceed four-config cap")
    entries = []
    for name in selected:
        seeds = (42,) if name == "p_linear" else SEEDS
        for protocol in ("lco", "lpo"):
            fold = "fold_quick_screening_0" if protocol == "lco" else "fold_0"
            for seed in seeds:
                entries.append({"model": name, "protocol": protocol, "fold": fold, "seed": seed,
                                "max_trials": 3 if name == "p_linear" else 8})
    if len(entries) > budget["rounds"]["a2"]["new_configs_max_runs"]:
        raise ValueError("A2 selected-fit budget exceeded")
    if len(historical | {(selected_variants[item["model"]], item["protocol"], item["seed"]) for item in entries}) > budget["rounds"]["a2"]["new_configs_max_runs"]:
        raise ValueError("A2 historical plus selected fits exceed 24-run cap")
    if any(item["max_trials"] > budget["hyperparameter_trials_cap_per_family"] for item in entries):
        raise ValueError("A2 trial budget exceeded")
    cohort_hash = (P.COHORTS_DIR / "cohort_hash.txt").read_text(encoding="utf-8").strip()
    split = {p: sha256_file(P.SPLITS_DIR / cohort_hash / p /
                            ("fold_quick_screening_0.csv" if p == "lco" else "fold_0.csv"))
             for p in ("lco", "lpo")}
    expression_cfg = json.loads((P.RUNS_DIR / "E1_build_expression_features" / "config_resolved.json").read_text(encoding="utf-8"))
    fingerprint_cfg = json.loads((P.RUNS_DIR / "E2_build_fingerprints" / "config_resolved.json").read_text(encoding="utf-8"))
    cache = {"expression_data_hash": expression_cfg["data_hash"],
             "fingerprint_data_hash": fingerprint_cfg["data_hash"],
             "fingerprint_preprocess_hash": fingerprint_cfg["preprocess_hash"]}
    manifest = {"cohort_hash": cohort_hash, "split_sha256": split, "selected_models": selected,
                "entries": entries, "feature_cache": cache,
                "hallmark_gmt_sha256": sha256_file(P.HALLMARK_ENTREZ_GMT) if any(name in PATHWAY_MODELS for name in selected) else None,
                "ranking": rank_cfg if "r2_rank" in selected else None}
    return {"status": "planned", **manifest, "manifest_hash": stable_hash(manifest),
            "selected_models": selected, "entries": entries, "selected_fit_count": len(entries),
            "limit": budget["rounds"]["a2"]["new_configs_max_runs"],
            "historical_entries": len(historical),
            "r0_source": "A1/mlp after preflight", "ranking": rank_cfg}


def _find_a1(name: str, protocol: str, seed: int) -> Path:
    root = P.DATA_DIR / "runs" / "A1" / name
    found = []
    for status_file in root.rglob("status.json"):
        directory = status_file.parent
        try:
            status = json.loads(status_file.read_text(encoding="utf-8"))
            config = json.loads((directory / "config_resolved.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (status.get("status") == "succeeded" and config.get("protocol") == protocol and
                config.get("seed") == seed and config.get("diagnostic") == "none" and
                config.get("cell_fraction") == 1.0 and config.get("drug_fraction") == 1.0):
            found.append(directory)
    if len(found) != 1:
        raise ValueError(f"expected one successful ordinary A1 {name}/{protocol}/{seed}, found {len(found)}")
    return found[0]


def preflight_a1(protocol: str, data: ProjectData | None = None, require_r0: bool = True) -> dict:
    """Verify that A1 R0 and simple comparator cover the active development split."""
    data = data or ProjectData(protocol)
    expected_train = set(data.frame.loc[data.frame["role"] == "train", "sample_id"])
    expected_test = set(data.frame.loc[data.frame["role"] == "test", "sample_id"])
    expected = data.frame.loc[data.frame["role"] == "test", ["sample_id", "cell_id", "compound_id", "y"]]
    simple = "per_drug_elasticnet" if protocol == "lco" else "ridge"
    records = {}
    sources = (("mlp", SEEDS), (simple, (42,))) if require_r0 else ((simple, (42,)),)
    for name, seeds in sources:
        for seed in seeds:
            directory = _find_a1(name, protocol, seed)
            config = json.loads((directory / "config_resolved.json").read_text(encoding="utf-8"))
            manifest = json.loads((directory / "sample_manifest.json").read_text(encoding="utf-8"))
            if (config["cohort_hash"] != data.cohort_hash or
                    set(manifest["train_ids"]) != expected_train or set(manifest["test_ids"]) != expected_test):
                raise ValueError(f"A1 {name}/{seed} sample split differs from active split")
            if config["fingerprint_sha256"] != data.fingerprint_hash:
                raise ValueError(f"A1 {name}/{seed} fingerprint differs")
            if config["expression_source_sha256"] != _expression_sha256():
                raise ValueError(f"A1 {name}/{seed} expression source differs")
            for key, path in (("core_samples_sha256", P.COHORTS_DIR / "core_samples.parquet"),
                              ("cell_master_sha256", P.ENTITIES_DIR / "cell_master.csv"),
                              ("compound_master_sha256", P.ENTITIES_DIR / "compound_master.csv")):
                if config[key] != sha256_file(path):
                    raise ValueError(f"A1 {name}/{seed} {key} differs")
            if name == "mlp":
                for implementation in ("program/models/neural.py", "program/models/dataset.py",
                                       "program/features/expression_preprocess.py"):
                    source = config["implementation_sha256"].get(implementation)
                    if source != sha256_file(P.REPO_ROOT / implementation):
                        raise ValueError(f"A1 {implementation} differs; R0 must be refitted within A2 budget")
                if config["feature_schema"] != {"hvg": 1000, "fingerprint_bits": 2048}:
                    raise ValueError("A1 R0 input feature schema differs")
            predictions = pd.read_parquet(directory / "predictions.parquet")
            paired = expected.merge(predictions[["sample_id", "cell_id", "compound_id", "y_true", "y_pred"]],
                                    on="sample_id", validate="one_to_one", suffixes=("_current", "_a1"))
            if (len(paired) != len(expected) or
                    not np.array_equal(paired["cell_id_current"], paired["cell_id_a1"]) or
                    not np.array_equal(paired["compound_id_current"], paired["compound_id_a1"]) or
                    not np.array_equal(paired["y"].to_numpy(dtype=float), paired["y_true"].to_numpy(dtype=float))):
                raise ValueError(f"A1 {name}/{seed} prediction identities or labels differ")
            assert_metrics_close(evaluate_file(directory / "predictions.parquet"),
                                 json.loads((directory / "metrics.json").read_text(encoding="utf-8")))
            records[f"{name}_{seed}"] = {"run_dir": directory.relative_to(P.DATA_DIR).as_posix(),
                                          "saved_split_sha256": config["split_sha256"],
                                          "current_split_sha256": data.split_hash,
                                          "sample_role_hash": stable_hash({"train": sorted(expected_train),
                                                                           "test": sorted(expected_test)})}
    return {"protocol": protocol, "simple_comparator": simple, "r0_verified": require_r0,
            "records": records}


def _grid(name: str) -> list[dict]:
    if name == "p_linear":
        return [{"alpha": alpha} for alpha in (0.1, 1.0, 10.0)]
    return [{"lr": lr, "weight_decay": wd, "dropout": drop}
            for lr in (3e-4, 1e-3) for wd in (1e-4, 1e-3) for drop in (0.1, 0.2)]


def _mode(name: str) -> str | None:
    return "random" if name == "p_random" else "projection" if name == "p_projection" else "hallmark" if name in PATHWAY_MODELS else None


def _config(data: ProjectData, name: str, seed: int, smoke: bool) -> dict:
    cfg_paths = [P.REPO_ROOT / "configs" / filename for filename in
                 ("cohort.json", "split_policy.json", "models.json", "experiment_budget.json")]
    implementations = ["program/models/a2.py", "program/models/dataset.py", "program/features/a2_pathways.py",
                       "program/features/pathway_features.py", "program/experiments/a2_pipeline.py",
                       "program/evaluation/metrics.py", "program/features/expression_preprocess.py"]
    config = {
        "stage": "A2_SMOKE" if smoke else "A2", "model": name, "protocol": data.protocol,
        "fold": data.fold, "seed": seed, "cohort_hash": data.cohort_hash,
        "split_sha256": data.split_hash, "core_samples_sha256": _core_sha256(),
        "expression_source_sha256": _expression_sha256(),
        "fingerprint_sha256": data.fingerprint_hash, "fingerprint_config": data.fingerprint_config,
        "source_manifest_sha256": sha256_file(P.DATA_DIR / "manifests" / "source_manifest.csv"),
        "feature_schema": {"hvg": 1000, "fingerprint_bits": 2048, "pathway_mode": _mode(name)},
        "hallmark_gmt_sha256": sha256_file(P.HALLMARK_ENTREZ_GMT) if name in PATHWAY_MODELS else None,
        "implementation_sha256": {path: sha256_file(P.REPO_ROOT / path) for path in implementations},
        "config_sha256": {path.name: sha256_file(path) for path in cfg_paths},
        "uv_lock_sha256": sha256_file(P.REPO_ROOT / "uv.lock"),
        "git_commit": git_commit(), "git_dirty_or_patch_hash": _git_dirty_hash(),
        "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "inner_seed": INNER_SEED, "min_qualified_cells": 2 if smoke else MIN_CELLS,
        "trial_grid": _grid(name)[:1] if smoke else _grid(name),
        "max_epochs": 2 if smoke else 200, "patience": 2 if smoke else 20,
        "batch_size": 128, "pathway_representation": "expression-derived, train-fold-only",
        "ranking": _models_config()["a2_execution"]["ranking"] if name == "r2_rank" else None,
    }
    return config


def _prepared_pathways(data: ProjectData, prepared: PreparedData, train: pd.DataFrame,
                       name: str, seed: int) -> PathwayFeatures | None:
    mode = _mode(name)
    return prepare_pathways(data, prepared, train, mode=mode, seed=seed) if mode else None


def _fit(name: str, prepared: PreparedData, params: dict, seed: int,
         validation: pd.DataFrame | None, pathway: PathwayFeatures | None,
         config: dict, fixed_epochs: int | None = None):
    ranking = config["ranking"] or {}
    return fit_a2(
        name, prepared, params, seed, validation,
        pathways=pathway.values if pathway is not None else None,
        pathway_names=pathway.names if pathway is not None else None,
        fixed_epochs=fixed_epochs, max_epochs=config["max_epochs"], patience=config["patience"],
        batch_size=config["batch_size"], min_cells=config["min_qualified_cells"],
        rank_noise_threshold=ranking.get("noise_threshold"), rank_temperature=ranking.get("temperature"),
        rank_max_pairs=int(ranking.get("max_pairs_per_batch", 64)),
    )


def run_a2(name: str, protocol: str = "lco", seed: int = 42,
           resume: bool = False, smoke: bool = False) -> Path:
    plan = a2_plan()
    if name not in A2_MODELS or (not smoke and name not in plan["selected_models"]):
        raise ValueError("A2 model is unregistered or absent from frozen selected matrix")
    if seed not in ((42,) if name == "p_linear" else SEEDS):
        raise ValueError("A2 seed is not registered for this model")
    data = ProjectData(protocol)
    if not smoke and name != "r0" and "r0" in plan["selected_models"]:
        anchor = [path for path in (P.DATA_DIR / "runs" / "A2" / "r0").rglob("status.json")
                  if path.parent.name == str(seed) and f"{protocol}_{data.fold}" in path.parts and
                  json.loads(path.read_text(encoding="utf-8")).get("status") == "succeeded"]
        if len(anchor) != 1:
            raise ValueError("refitted A2 R0 must succeed for this protocol and seed first")
    preflight = None if smoke else preflight_a1(protocol, data, require_r0="r0" not in plan["selected_models"])
    config = _config(data, name, seed, smoke)
    config["a2_manifest_hash"] = plan["manifest_hash"]
    config["a1_preflight"] = preflight
    run = RunRecord("A2_SMOKE" if smoke else "A2", name, protocol, data.fold, seed, config, resume=resume)
    try:
        run.status("running")
        frame = _a0_subset(data.frame) if smoke else data.frame
        outer_train = frame.loc[frame["role"] == "train"].copy()
        outer_test = frame.loc[frame["role"] == "test"].copy()
        inner_train, inner_val = inner_split(outer_train, protocol, seed=INNER_SEED)
        inner = data.prepare(inner_train, inner_val)
        inner_pathway = _prepared_pathways(data, inner, inner_train, name, seed)
        rmse_guard = float(np.sqrt(np.mean((inner_val["y"].to_numpy() - inner_train["y"].mean()) ** 2)))
        trials, candidates = [], []
        for params in config["trial_grid"]:
            started = time.monotonic()
            try:
                with warnings.catch_warnings(record=True) as observed:
                    warnings.simplefilter("always")
                    fitted = _fit(name, inner, params, seed, inner_val, inner_pathway, config)
                predicted = fitted.predict(inner, inner_val,
                                           inner_pathway.values if inner_pathway is not None else None)
                validation = inner_val[["sample_id", "cell_id", "compound_id"]].copy()
                validation["y_true"], validation["y_pred"] = inner_val["y"].to_numpy(), predicted
                metric = evaluate_predictions(validation, min_cells=config["min_qualified_cells"])
                entry = {"params": params, "status": "succeeded",
                         "inner_macro_spearman": metric["macro_spearman_selection"],
                         "inner_rmse": metric["rmse"], "selected_epoch": fitted.selected_epoch,
                         "parameter_count": fitted.parameter_count,
                         "runtime_seconds": round(time.monotonic() - started, 3),
                         "warnings": [f"{w.category.__name__}: {w.message}" for w in observed]}
                candidates.append(entry)
            except BaseException as exc:
                entry = {"params": params, "status": "failed", "error": str(exc),
                         "runtime_seconds": round(time.monotonic() - started, 3)}
                trials.append(entry)
                write_json(run.path / "trial_history.json", {"trials": trials})
                raise
            trials.append(entry)
            write_json(run.path / "trial_history.json", {"trials": trials})
        eligible = [item for item in candidates if item["inner_rmse"] <= rmse_guard * 1.02]
        if eligible:
            chosen = max(eligible, key=lambda item: (
                -np.inf if item["inner_macro_spearman"] is None else item["inner_macro_spearman"],
                -item["inner_rmse"]))
        else:
            chosen = min(candidates, key=lambda item: item["inner_rmse"])
        outer = data.prepare(outer_train, outer_test)
        outer_pathway = _prepared_pathways(data, outer, outer_train, name, seed)
        with warnings.catch_warnings(record=True) as observed_final:
            warnings.simplefilter("always")
            fitted = _fit(name, outer, chosen["params"], seed, None, outer_pathway,
                          config, fixed_epochs=chosen["selected_epoch"])
        pathway_values = outer_pathway.values if outer_pathway is not None else None
        components = fitted.predict_components(outer, outer_test, pathway_values)
        pred = components["y_pred"]
        if not np.isfinite(pred).all():
            raise ValueError("non-finite A2 prediction")
        if "interaction" in components:
            reconstructed = (components["mu"] + components["b_cell"] +
                             components["b_drug"] + components["interaction"])
            if not np.allclose(pred, reconstructed, rtol=1e-6, atol=1e-5):
                raise ValueError("R1 branch contributions do not sum to prediction")
        if "contributions" in components:
            reconstructed = components["b_drug"] + components["contributions"].sum(axis=1)
            if not np.allclose(pred, reconstructed, rtol=1e-6, atol=1e-5):
                raise ValueError("pathway contributions do not sum to prediction")
        checkpoint = run.path / "checkpoint.pkl"
        with checkpoint.open("wb") as handle:
            pickle.dump(fitted, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with checkpoint.open("rb") as handle:
            loaded = pickle.load(handle)
        loaded_pred = loaded.predict(outer, outer_test, pathway_values)
        if not np.array_equal(pred, loaded_pred):
            raise ValueError("A2 checkpoint reload changed predictions")
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
        for field in ("mu", "b_cell", "b_drug", "interaction"):
            if field in components:
                prediction[field] = components[field]
        if "contributions" in components:
            assert outer_pathway is not None
            prediction["pathway_total"] = components["contributions"].sum(axis=1)
            table = pd.DataFrame(components["contributions"], columns=outer_pathway.names)
            table.insert(0, "sample_id", prediction["sample_id"].to_numpy())
            table.to_parquet(run.path / "pathway_contributions.parquet", index=False)
        prediction.to_parquet(run.path / "predictions.parquet", index=False)
        metrics = {"overall": evaluate_predictions(prediction, min_cells=config["min_qualified_cells"]),
                   "strata": summarize_strata(prediction, min_cells=config["min_qualified_cells"])}
        write_metrics(run.path / "metrics.json", metrics)
        write_json(run.path / "training_history.json", {"trials": trials, "final": fitted.history,
                   "final_warnings": [f"{w.category.__name__}: {w.message}" for w in observed_final],
                   "rmse_guard_reference": rmse_guard, "rmse_guard_passed": bool(eligible)})
        write_json(run.path / "selected_epoch.json", {"selected_epoch": fitted.selected_epoch,
                   "params": chosen["params"], "parameter_count": fitted.parameter_count})
        write_json(run.path / "preprocessing_state.json", outer.state)
        write_json(run.path / "gene_order.json", {"gene_pairs_in_order": outer.gene_order})
        if outer_pathway is not None:
            write_json(run.path / "pathway_state.json", outer_pathway.state)
        write_json(run.path / "sample_manifest.json", {"train_ids": outer_train["sample_id"].tolist(),
                   "test_ids": outer_test["sample_id"].tolist(),
                   "inner_train_count": len(inner_train), "inner_validation_count": len(inner_val),
                   "train_cells": int(outer_train["cell_id"].nunique()),
                   "train_drugs": int(outer_train["compound_id"].nunique()),
                   "test_cells": int(outer_test["cell_id"].nunique()),
                   "test_drugs": int(outer_test["compound_id"].nunique())})
        (run.path / "error_log.txt").write_text("", encoding="utf-8")
        gpu_hours = (time.monotonic() - run.start_time) / 3600 if torch.cuda.is_available() and name != "p_linear" else 0.0
        peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() and name != "p_linear" else None
        run.status("smoke" if smoke else "succeeded", gpu_hours=gpu_hours, peak_gpu_bytes=peak,
                   n_train=len(outer_train), n_test=len(outer_test), n_trials=len(trials),
                   parameter_count=fitted.parameter_count)
        return run.path
    except KeyboardInterrupt:
        run.status("incomplete", error_type="KeyboardInterrupt")
        raise
    except BaseException as exc:
        run.fail(exc)
        raise
