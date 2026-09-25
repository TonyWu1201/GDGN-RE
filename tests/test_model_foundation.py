"""Failure-prone model/evaluation contracts on small known-truth fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from program.evaluation.metrics import evaluate_predictions
from program.evaluation.statistics import holm_adjust, paired_cluster_bootstrap
from program.experiments.pipeline import _diagnostic_train, _restrict_train, a1_plan
from program.models.baselines import ALL_MODELS, fit_baseline, grid
from program.models.dataset import PreparedData, inner_split
from program.models.neural import fit_mlp
from program.models.run_registry import RunRecord


def _predictions() -> pd.DataFrame:
    return pd.DataFrame([
        {"sample_id": f"S{i}{d}", "cell_id": f"C{i}", "cell_group": f"G{i}",
         "compound_id": d, "y_true": float(i), "y_pred": float(i) if d == "D1" else 2.0}
        for d in ("D1", "D2") for i in range(4)
    ])


def test_macro_uses_same_label_qualified_drugs_and_constant_zero():
    pred = _predictions()
    result = evaluate_predictions(pred, min_cells=3)
    assert result["n_qualified_drugs"] == 2
    assert result["constant_qualified_drugs"] == 1
    assert result["macro_spearman_selection"] == pytest.approx(0.5)
    assert result["per_drug"][1]["spearman_raw"] is None
    assert result["per_drug"][1]["spearman_selection"] == 0
    with pytest.raises(ValueError, match="non-finite"):
        evaluate_predictions(pred.assign(y_pred=lambda x: x["y_pred"].mask(x.index == 0, np.nan)), min_cells=3)
    with pytest.raises(ValueError, match="unique"):
        evaluate_predictions(pd.concat([pred, pred.iloc[[0]]]), min_cells=3)


def test_paired_bootstrap_reorders_by_id_and_keeps_zero_effect():
    pred = _predictions()
    result = paired_cluster_bootstrap(pred, pred.sample(frac=1, random_state=3), "lco", n_boot=30, min_cells=2)
    assert result["ci95"] == pytest.approx([0.0, 0.0])
    with pytest.raises(ValueError, match="identical"):
        paired_cluster_bootstrap(pred, pred.iloc[1:], "lco", n_boot=3, min_cells=2)
    worse = pred.copy()
    worse["y_pred"] = -worse["y_true"]
    positive = paired_cluster_bootstrap(pred, worse, "lco", n_boot=30, min_cells=2)
    assert positive["difference_mean"] > 0
    assert positive["ci95"][0] >= 0
    assert holm_adjust([0.01, 0.04, 0.8]) == pytest.approx([0.03, 0.08, 0.8])


def test_inner_lco_group_isolation_and_label_invariance():
    frame = pd.DataFrame({"sample_id": [f"S{i}" for i in range(40)],
                          "cell_id": [f"C{i // 2}" for i in range(40)],
                          "cell_group": [f"G{i // 2}" for i in range(40)],
                          "compound_id": ["D1", "D2"] * 20, "y": range(40)})
    train, validation = inner_split(frame, "lco")
    changed = frame.assign(y=np.arange(40)[::-1])
    train2, validation2 = inner_split(changed, "lco")
    assert not set(train.cell_group) & set(validation.cell_group)
    assert set(validation.sample_id) == set(validation2.sample_id)
    assert set(train.sample_id) == set(train2.sample_id)


def test_additive_baseline_recovers_signal_and_is_order_invariant():
    cells = [f"C{i}" for i in range(6)]
    drugs = [f"D{j}" for j in range(4)]
    rows = [{"sample_id": f"{c}_{d}", "cell_id": c, "compound_id": d, "profile_rna_model_id": c,
             "y": float(2 * i + 3 * j)} for i, c in enumerate(cells) for j, d in enumerate(drugs)]
    train = pd.DataFrame(rows)
    prepared = PreparedData(train, train, np.arange(6, dtype=np.float32)[:, None],
                            np.arange(4, dtype=np.float32)[:, None], cells, drugs, {}, [])
    fitted = fit_baseline("two_way_additive", prepared, {})
    pred = fitted.predict(prepared, train)
    shuffled = train.sample(frac=1, random_state=42)
    shuffled_pred = fitted.predict(prepared, shuffled)
    assert np.sqrt(np.mean((pred - train.y.to_numpy()) ** 2)) < 0.5
    assert dict(zip(train.sample_id, pred)) == dict(zip(shuffled.sample_id, shuffled_pred))


def test_small_twin_tower_can_fit_interaction():
    cells = [f"C{i}" for i in range(6)]
    drugs = [f"D{j}" for j in range(6)]
    values = np.linspace(-1, 1, 6, dtype=np.float32)
    train = pd.DataFrame([
        {"sample_id": f"{c}_{d}", "cell_id": c, "compound_id": d, "profile_rna_model_id": c,
         "y": float(values[i] * values[j])}
        for i, c in enumerate(cells) for j, d in enumerate(drugs)
    ])
    prepared = PreparedData(train, train, values[:, None], values[:, None], cells, drugs, {}, [])
    fitted = fit_mlp(prepared, {"lr": 0.01, "weight_decay": 0.0, "dropout": 0.0},
                     seed=42, validation=None, fixed_epochs=100, max_epochs=100)
    predicted = fitted.predict(prepared, train)
    assert np.sqrt(np.mean((predicted - train.y.to_numpy()) ** 2)) < 0.2


def test_all_registered_sklearn_models_predict_finite_values():
    cells = [f"C{i}" for i in range(6)]
    drugs = [f"D{j}" for j in range(6)]
    frame = pd.DataFrame([
        {"sample_id": f"{c}_{d}", "cell_id": c, "compound_id": d, "profile_rna_model_id": c,
         "y": float(i + 0.2 * j + (i % 2) * j)}
        for i, c in enumerate(cells) for j, d in enumerate(drugs)
    ])
    expression = np.stack([np.arange(6), np.arange(6) ** 2], axis=1).astype(np.float32)
    fingerprints = np.eye(6, dtype=np.float32)
    prepared = PreparedData(frame, frame, expression, fingerprints, cells, drugs, {}, [])
    for name in ALL_MODELS:
        if name == "mlp":
            continue
        fitted = fit_baseline(name, prepared, grid(name)[0])
        prediction = fitted.predict(prepared, frame)
        assert len(prediction) == len(frame), name
        assert np.isfinite(prediction).all(), name


def test_tree_predictions_are_bitwise_stable_across_calls():
    rng = np.random.default_rng(0)
    expression = rng.normal(size=(40, 8)).astype(np.float32)
    fingerprints = rng.integers(0, 2, size=(6, 8)).astype(np.float32)
    cells = [f"C{i}" for i in range(40)]
    drugs = [f"D{j}" for j in range(6)]
    frame = pd.DataFrame([
        {"sample_id": f"{c}_{d}", "cell_id": c, "compound_id": d, "profile_rna_model_id": c,
         "y": float(expression[i, 0] + j)}
        for i, c in enumerate(cells) for j, d in enumerate(drugs)
    ])
    prepared = PreparedData(frame, frame, expression, fingerprints, cells, drugs, {}, [])
    params = grid("tree")[0]
    fitted = fit_baseline("tree", prepared, params)
    first = fitted.predict(prepared, frame)
    import pickle

    def _predict() -> np.ndarray:
        reloaded = pickle.loads(pickle.dumps(fitted))
        return reloaded.predict(prepared, frame)

    assert fitted.estimator.n_jobs == 1
    assert np.array_equal(first, fitted.predict(prepared, frame))
    assert np.array_equal(first, _predict())
    assert np.array_equal(first, _predict())


def test_inner_lpo_keeps_same_pair_together():
    frame = pd.DataFrame({"sample_id": [f"S{i}" for i in range(20)],
                          "cell_id": [f"C{i // 2}" for i in range(20)],
                          "compound_id": ["D0", "D1"] * 10})
    train, valid = inner_split(frame, "lpo")
    pair = lambda x: set(zip(x.cell_id, x.compound_id))
    assert not pair(train) & pair(valid)


def test_run_registry_refuses_overwrite_and_preserves_history(tmp_path, monkeypatch):
    from program.models import run_registry
    monkeypatch.setattr(run_registry.P, "DATA_DIR", tmp_path)
    cfg = {"model": "ridge", "cohort_hash": "abc"}
    record = RunRecord("A0", "ridge", "lco", "fold_0", 42, cfg)
    record.fail(ValueError("synthetic failure"))
    retry = RunRecord("A0", "ridge", "lco", "fold_0", 42, cfg, resume=True)
    retry.status("smoke")
    assert "failed" in (retry.path / "status_history.jsonl").read_text(encoding="utf-8")
    with pytest.raises(FileExistsError, match="immutable"):
        RunRecord("A0", "ridge", "lco", "fold_0", 42, cfg, resume=True)


def test_a1_plan_stays_within_registered_budget():
    plan = a1_plan()
    assert plan["control_count"] == 12
    assert plan["learning_count"] <= 36
    assert max(x["max_trials"] for x in plan["entries"]) <= 12


def test_a1_diagnostics_preserve_drug_means_and_fixed_entity_selection():
    frame = pd.DataFrame({"sample_id": [f"S{i}" for i in range(40)],
                          "cell_id": [f"C{i // 2}" for i in range(40)],
                          "cell_group": [f"G{i // 2}" for i in range(40)],
                          "compound_id": ["D0", "D1"] * 20, "y": np.arange(40, dtype=float)})
    shuffled = _diagnostic_train(frame, "label_shuffle", 42)
    assert shuffled.groupby("compound_id").y.mean().to_dict() == frame.groupby("compound_id").y.mean().to_dict()
    assert not np.array_equal(shuffled.y.to_numpy(), frame.y.to_numpy())
    reduced = _restrict_train(frame, 0.5, 0.5)
    assert reduced.cell_group.nunique() == 10
    assert reduced.compound_id.nunique() == 2
