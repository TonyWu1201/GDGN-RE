"""测试：通路分数重算与掩码（计划 §14 test_pathway_recompute，§7.4/§16.5-6）。"""

import numpy as np
import pytest

from program.features.pathway_features import (
    apply_intervention_recompute,
    compute_pathway_scores,
    pathway_coverage_mask,
)


def test_pathway_scores_recompute_after_expression_change():
    """表达扰动后通路分数同步重算（成员均值聚合的派生性质）。"""
    expr = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])  # 基因 0/1
    members = {"SET": [0, 1]}
    scores = compute_pathway_scores(expr, members)
    assert abs(scores[0, 0] - 5.5) < 1e-9  # (1+10)/2
    # 表达变化 → 分数同步变化
    expr2 = expr + 100.0
    scores2 = compute_pathway_scores(expr2, members)
    assert abs(scores2[0, 0] - 105.5) < 1e-9


def test_member_coverage_mask():
    """成员覆盖率与最低成员数掩码（cohort.json min_effective_members=10）。"""
    gmt = {"A": [str(i) for i in range(15)], "B": [str(i) for i in range(5)]}  # B 5 < 10
    measurable = {str(i) for i in range(15)}
    mask, eff = pathway_coverage_mask(gmt, measurable, min_members=10)
    assert mask["A"] and not mask["B"]
    assert eff["A"] == 15 and eff["B"] == 5


def test_scores_are_derived_not_independent():
    """通路分数 = 表达的派生表示：干预表达时同步重算，不是独立新增组学。"""
    rng = np.random.default_rng(0)
    expr = rng.normal(size=(6, 4))
    members = {"S1": [0, 1, 2], "S2": [2, 3]}
    s1 = compute_pathway_scores(expr, members)
    # 干预：把基因 2 全部置零 → S1、S2 分数必须变化
    expr_mod = expr.copy()
    expr_mod[:, 2] = 0.0
    s2 = compute_pathway_scores(expr_mod, members)
    assert not np.allclose(s1, s2)
    # apply_intervention_recompute 一致性
    s3 = apply_intervention_recompute(expr, members, lambda e: e.__setitem__((slice(None), 2), 0.0) or e)
    assert np.allclose(s2, s3)