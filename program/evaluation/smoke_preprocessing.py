"""Small, metric-free fit to verify sample, expression and drug feature alignment."""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from program.common import paths as P
from program.common.runlog import deterministic_json_bytes, resolve_config, save_run_record, sha256_file


def main() -> int:
    started = time.time()
    cohort_hash = (P.COHORTS_DIR / "cohort_hash.txt").read_text().strip()
    fold_path = P.SPLITS_DIR / cohort_hash / "lco" / "fold_candidate_0.csv"
    fold = pd.read_csv(fold_path, dtype=str)
    cells = pd.read_csv(P.ENTITIES_DIR / "cell_master.csv", dtype=str)
    cell_to_model = dict(zip(cells["cell_id"], cells["profile_rna_model_id"]))
    smoke_train_ids = fold.loc[fold["role"] == "train", "sample_id"].head(512).tolist()
    labels = pd.read_parquet(
        P.COHORTS_DIR / "core_samples.parquet",
        columns=["sample_id", "y"],
        filters=[("sample_id", "in", smoke_train_ids)],
    )
    if set(labels["sample_id"]) != set(smoke_train_ids):
        raise ValueError("smoke training labels incomplete")

    expression_config = json.loads((P.RUNS_DIR / "E1_build_expression_features" / "config_resolved.json").read_text(encoding="utf-8"))
    fingerprint_config = json.loads((P.RUNS_DIR / "E2_build_fingerprints" / "config_resolved.json").read_text(encoding="utf-8"))
    if expression_config["split_hash"] != fingerprint_config["split_hash"]:
        raise ValueError("feature split hashes differ")
    expression_dir = P.FEATURES_DIR / expression_config["data_hash"] / expression_config["split_hash"] / expression_config["preprocess_hash"] / "lco" / "fold_candidate_0"
    fingerprint_path = P.FEATURES_DIR / fingerprint_config["data_hash"] / fingerprint_config["split_hash"] / fingerprint_config["preprocess_hash"] / "fingerprints.npz"
    fingerprints = np.load(fingerprint_path)
    drug_index = {drug: i for i, drug in enumerate(fingerprints["compound_ids"])}

    def sample_matrix(role: str, limit: int) -> tuple[np.ndarray, np.ndarray | None]:
        selected = fold[fold["role"] == role].head(limit).copy()
        if role == "train":
            selected = selected.merge(labels, on="sample_id", validate="one_to_one")
        expression = np.load(expression_dir / f"expression_{role}.npz")
        cell_index = {model: i for i, model in enumerate(expression["rows"])}
        models = selected["cell_id"].map(cell_to_model)
        if models.isna().any() or not models.isin(cell_index).all() or not selected["compound_id"].isin(drug_index).all():
            raise ValueError("sample-to-feature alignment failed")
        expression_rows = expression["data"][[cell_index[model] for model in models]]
        fingerprint_rows = fingerprints["data"][[drug_index[drug] for drug in selected["compound_id"]]]
        features = np.concatenate((expression_rows, fingerprint_rows), axis=1)
        targets = selected["y"].to_numpy(dtype=float) if role == "train" else None
        if not np.isfinite(features).all() or (targets is not None and not np.isfinite(targets).all()):
            raise ValueError("nonfinite smoke input")
        return features, targets

    x_train, y_train = sample_matrix("train", 512)
    x_test, _ = sample_matrix("test", 128)
    model = Ridge(alpha=1.0, solver="lsqr")
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    if len(predictions) != len(x_test) or not np.isfinite(predictions).all():
        raise ValueError("smoke prediction alignment or finiteness failed")

    report = {
        "status": "smoke",
        "cohort_hash": cohort_hash,
        "split_hash": expression_config["split_hash"],
        "train_samples": len(y_train),
        "test_samples": len(predictions),
        "feature_columns": int(x_train.shape[1]),
        "feature_and_label_alignment": "pass",
        "finite_predictions": True,
        "sealed_labels_used": False,
        "scientific_metrics_reported": False,
    }
    output = P.RUNS_DIR / "A0_preprocess_smoke" / "smoke_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(deterministic_json_bytes(report))
    save_run_record(
        "A0_preprocess_smoke", resolve_config({"cohort_hash": cohort_hash, "split_hash": expression_config["split_hash"]}),
        input_hashes={str(fold_path): sha256_file(fold_path), str(fingerprint_path): sha256_file(fingerprint_path)},
        outputs={str(output): sha256_file(output)}, status="smoke", started_at=started,
    )
    print(f"[A0 smoke] {len(y_train)} train / {len(predictions)} test, {x_train.shape[1]} features; alignment and finite predictions passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        save_run_record("A0_preprocess_smoke", {"stage": "A0_preprocess_smoke"}, {}, {}, "failed", time.time(), error=str(exc))
        raise
