"""A2 mechanisms and leakage-sensitive feature contracts on known-truth fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from program.evaluation.metrics import assert_metrics_close
from program.features.a2_pathways import prepare_pathways, rewire_memberships, transform_pathways
from program.models.a2 import AdditiveInteraction, ConcatAnchor, PathwayAdditive, fit_a2, ranking_loss
from program.models.dataset import PreparedData


def _fixture(interaction: bool = False) -> tuple[PreparedData, pd.DataFrame]:
    cells = [f"C{i}" for i in range(6)]
    drugs = [f"D{j}" for j in range(6)]
    c = np.linspace(-1, 1, 6, dtype=np.float32)
    d = np.linspace(-1, 1, 6, dtype=np.float32)
    frame = pd.DataFrame([
        {"sample_id": f"{cell}_{drug}", "cell_id": cell, "compound_id": drug,
         "profile_rna_model_id": cell, "y": float(c[i] * d[j] if interaction else c[i] + d[j])}
        for i, cell in enumerate(cells) for j, drug in enumerate(drugs)
    ])
    prepared = PreparedData(frame, frame, c[:, None], d[:, None], cells, drugs, {}, [])
    return prepared, frame


def test_a2_additive_and_interaction_fit_known_truth_and_preserve_row_identity():
    for interaction, model_name in ((False, "additive"), (True, "r1_k16")):
        prepared, frame = _fixture(interaction)
        fitted = fit_a2(model_name, prepared, {"lr": 0.01, "weight_decay": 0.0, "dropout": 0.0},
                        seed=42, validation=None, fixed_epochs=90, max_epochs=90)
        parts = fitted.predict_components(prepared, frame)
        pred = parts["y_pred"]
        assert np.sqrt(np.mean((pred - frame.y.to_numpy()) ** 2)) < 0.2
        np.testing.assert_allclose(pred, parts["mu"] + parts["b_cell"] +
                                   parts["b_drug"] + parts["interaction"], atol=1e-5)
        if not interaction:
            assert np.array_equal(parts["interaction"], np.zeros(len(frame)))
        shuffled = frame.sample(frac=1, random_state=11)
        assert dict(zip(frame.sample_id, pred)) == dict(zip(
            shuffled.sample_id, fitted.predict(prepared, shuffled)))


def test_r1_reference_uses_training_entities_and_stays_fixed_at_inference():
    model = AdditiveInteraction(1, 1, dropout=0.0, rank=16)
    cells = torch.tensor([[-1.0], [0.0], [1.0]])
    drugs = torch.tensor([[-2.0], [2.0]])
    model.refresh_reference(cells, drugs)
    before = model.cell_reference.clone(), model.drug_reference.clone()
    with torch.no_grad():
        _ = model(torch.tensor([[100.0]]), torch.tensor([[100.0]]))
    assert torch.equal(model.cell_reference, before[0])
    assert torch.equal(model.drug_reference, before[1])
    model.eval()
    with torch.no_grad():
        cell_factor = model.cell_factor(model.cell(cells) - model.cell_reference)
        drug_factor = model.drug_factor(model.drug(drugs) - model.drug_reference)
    assert torch.allclose(cell_factor.mean(dim=0), torch.zeros(16), atol=1e-6)
    assert torch.allclose(drug_factor.mean(dim=0), torch.zeros(16), atol=1e-6)


def test_refit_r0_keeps_a1_twin_tower_topology():
    from program.models.neural import TwinTower

    old = TwinTower(3, 4, 0.1).eval()
    replacement = ConcatAnchor(3, 4, 0.1).eval()
    replacement.load_state_dict(old.state_dict())
    cells, drugs = torch.ones((2, 3)), torch.ones((2, 4))
    torch.testing.assert_close(old(cells, drugs), replacement(cells, drugs))


def test_pathway_components_sum_and_random_membership_preserves_degrees():
    model = PathwayAdditive(n_fp=2, n_pathways=3, dropout=0.0).eval()
    parts = model.forward_parts(torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.0, 2.0]]),
                                torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
    torch.testing.assert_close(parts["y_pred"], parts["b_drug"] + parts["contributions"].sum(dim=1))
    members = [[0, 1, 2], [1, 3, 4], [0, 2, 4]]
    random = rewire_memberships(members, seed=42, swaps=30)
    assert random != members
    assert sorted(map(len, random)) == sorted(map(len, members))
    assert [sum(g in group for group in random) for g in range(5)] == [
        sum(g in group for group in members) for g in range(5)]


def test_pathway_state_is_train_only_and_intervention_recomputes(monkeypatch):
    from program.features import a2_pathways

    gene_pairs = [[f"G{i}", str(i)] for i in range(6)]
    raw = pd.DataFrame(np.arange(36, dtype=float).reshape(6, 6),
                       index=[f"C{i}" for i in range(6)])
    frame = pd.DataFrame({"sample_id": [f"S{i}" for i in range(6)],
                          "cell_id": [f"C{i}" for i in range(6)],
                          "compound_id": ["D0"] * 6,
                          "profile_rna_model_id": [f"C{i}" for i in range(6)], "y": range(6)})
    prepared = PreparedData(frame.iloc[:3], frame.iloc[3:], raw.to_numpy(dtype=np.float32),
                            np.ones((1, 2), dtype=np.float32), list(raw.index), ["D0"], {}, [])

    class Data:
        def raw_expression(self):
            return raw, gene_pairs

    monkeypatch.setattr(a2_pathways, "read_hallmark", lambda: {"H1": ["0", "1", "2"],
                                                                "H2": ["2", "3", "4"]})
    first = prepare_pathways(Data(), prepared, frame.iloc[:3], min_members=2)
    raw.iloc[3:, :] += 1000
    second = prepare_pathways(Data(), prepared, frame.iloc[:3], min_members=2)
    assert first.state == second.state
    changed = raw.to_numpy().copy()
    changed[0, 0] += 100
    transformed = transform_pathways(changed, first.state)
    assert transformed[0, 0] != first.values[0, 0]
    assert transformed[0, 1] == pytest.approx(first.values[0, 1])
    projection = prepare_pathways(Data(), prepared, frame.iloc[:3], mode="projection", min_members=2)
    assert projection.values.shape == (6, 2)
    assert projection.state["train_model_ids"] == ["C0", "C1", "C2"]
    random_control = prepare_pathways(Data(), prepared, frame.iloc[:3], mode="random", min_members=2)
    assert 0 <= random_control.state["membership_edge_overlap_fraction"] < 1


def test_r2_ranking_direction_and_noise_filter():
    correct = torch.tensor([0.0, 3.0], requires_grad=True)
    reversed_pred = torch.tensor([3.0, 0.0], requires_grad=True)
    truth = torch.tensor([0.0, 3.0])
    drugs = np.array([0, 0])
    good = ranking_loss(correct, truth, drugs, 0.0, 1.0, np.random.default_rng(3))
    bad = ranking_loss(reversed_pred, truth, drugs, 0.0, 1.0, np.random.default_rng(3))
    assert good < bad
    ignored = ranking_loss(correct, truth, drugs, 3.0, 1.0, np.random.default_rng(3))
    assert ignored == 0
    with pytest.raises(ValueError, match="invalid"):
        ranking_loss(correct, truth, drugs, -1.0, 1.0, np.random.default_rng(3))


def test_r2_loss_modes_fit_with_frozen_settings():
    prepared, frame = _fixture(interaction=True)
    for name in ("r2_drug_equal", "r2_rank"):
        fitted = fit_a2(name, prepared, {"lr": 0.01, "weight_decay": 0.0, "dropout": 0.0},
                        seed=42, validation=None, fixed_epochs=3, max_epochs=3,
                        rank_noise_threshold=0.0, rank_temperature=1.0)
        assert np.isfinite(fitted.predict(prepared, frame)).all()


def test_pathway_linear_control_uses_same_aligned_pathway_rows():
    prepared, frame = _fixture(interaction=False)
    pathways = np.linspace(-1, 1, 6, dtype=np.float32)[:, None]
    fitted = fit_a2("p_linear", prepared, {"alpha": 0.1}, seed=42, validation=None,
                    pathways=pathways, pathway_names=["synthetic_pathway"])
    first = fitted.predict(prepared, frame, pathways)
    shuffled = frame.sample(frac=1, random_state=7)
    assert np.isfinite(first).all()
    assert dict(zip(frame.sample_id, first)) == dict(zip(
        shuffled.sample_id, fitted.predict(prepared, shuffled, pathways)))


def test_metric_recompute_allows_only_tiny_float_roundoff():
    assert_metrics_close({"score": 0.5 + 1e-16, "n": 2, "drug": ["D1"]},
                         {"score": 0.5, "n": 2, "drug": ["D1"]})
    for changed in ({"score": 0.51, "n": 2, "drug": ["D1"]},
                    {"score": 0.5, "n": 3, "drug": ["D1"]},
                    {"score": 0.5, "n": 2, "drug": ["D2"]},
                    {"score": float("nan"), "n": 2, "drug": ["D1"]}):
        with pytest.raises(ValueError):
            assert_metrics_close(changed, {"score": 0.5, "n": 2, "drug": ["D1"]})


def test_a2_plan_refuses_more_than_four_configs_and_unfrozen_ranking(monkeypatch):
    from program.experiments import a2_pipeline

    monkeypatch.setattr(a2_pipeline, "_models_config", lambda: {"a2_execution": {
        "selected_models": ["additive", "r1_k16", "r1_k32", "p", "p_random"],
        "ranking": {"noise_threshold": None, "temperature": None}}})
    with pytest.raises(ValueError, match="four"):
        a2_pipeline.a2_plan()
    monkeypatch.setattr(a2_pipeline, "_models_config", lambda: {"a2_execution": {
        "selected_models": ["r2_rank"],
        "ranking": {"noise_threshold": None, "temperature": None}}})
    with pytest.raises(ValueError, match="frozen"):
        a2_pipeline.a2_plan()


def test_a2_plan_counts_historical_configs_across_manifest_edits(tmp_path, monkeypatch):
    import json
    from program.experiments import a2_pipeline

    monkeypatch.setattr(a2_pipeline.P, "DATA_DIR", tmp_path)
    for name in ("additive", "r1_k16", "p", "p_random"):
        directory = tmp_path / "runs" / "A2" / name
        directory.mkdir(parents=True)
        (directory / "config_resolved.json").write_text(
            json.dumps({"model": name, "protocol": "lco", "seed": 42}), encoding="utf-8")
    monkeypatch.setattr(a2_pipeline, "_models_config", lambda: {"a2_execution": {
        "selected_models": ["r1_k32"],
        "ranking": {"noise_threshold": None, "temperature": None}}})
    with pytest.raises(ValueError, match="historical"):
        a2_pipeline.a2_plan()
    assert a2_pipeline._semantic_variant("r2_rank", {"noise_threshold": 0.1, "temperature": 1.0}) != (
        a2_pipeline._semantic_variant("r2_rank", {"noise_threshold": 0.2, "temperature": 1.0}))


def test_smoke_evaluation_reuses_saved_qualification_rule(tmp_path, monkeypatch, capsys):
    from argparse import Namespace
    from program.cli import run_evaluate
    from program.common import paths as paths
    from program.evaluation.metrics import evaluate_file, write_metrics

    monkeypatch.setattr(paths, "DATA_DIR", tmp_path)
    run = tmp_path / "runs" / "A2_SMOKE" / "additive" / "v_test" / "lco_fold" / "42"
    run.mkdir(parents=True)
    frame = pd.DataFrame({"sample_id": ["S1", "S2", "S3", "S4"],
                          "cell_id": ["C1", "C2", "C1", "C2"],
                          "compound_id": ["D1", "D1", "D2", "D2"],
                          "y_true": [1.0, 2.0, 3.0, 4.0], "y_pred": [1.1, 2.1, 3.1, 4.1]})
    frame.to_parquet(run / "predictions.parquet", index=False)
    (run / "config_resolved.json").write_text('{"min_qualified_cells": 2}', encoding="utf-8")
    write_metrics(run / "metrics.json", evaluate_file(run / "predictions.parquet", min_cells=2))
    assert run_evaluate(Namespace(run_dir=str(run))) == 0
    assert "recomputed_equal" in capsys.readouterr().out


def test_a2_report_does_not_pick_one_of_duplicate_successful_variants(tmp_path, monkeypatch):
    from program.experiments import report_a2

    monkeypatch.setattr(report_a2.P, "GUIDANCE_DIR", tmp_path)
    monkeypatch.setattr(report_a2, "a2_plan", lambda: {
        "selected_models": ["additive"], "selected_fit_count": 6, "limit": 24,
        "entries": [{"protocol": "lco", "model": "additive", "seed": 42}],
    })
    item = {"status": "succeeded", "config": {"protocol": "lco", "model": "additive", "seed": 42},
            "metrics": {"macro_spearman_selection": 0.5, "rmse": 1.0},
            "parameter_count": 100, "runtime_seconds": 1.0, "error": ""}
    monkeypatch.setattr(report_a2, "_runs", lambda: [item, item.copy()])
    report = report_a2.build_report().read_text(encoding="utf-8")
    assert "多个成功变体" in report
    assert "配对比较暂不选取其中任何一条" in report
    assert "待完成；不得写作通过" in report
