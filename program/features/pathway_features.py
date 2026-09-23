"""阶段 E3：通路特征（program/features/pathway_features.py）。

§7.4：Hallmark 50 集（entrez GMT）；固定成员关系的均值聚合；
训练折标准化在调用方完成（这里提供原始分数）；分数=表达的派生表示。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from program.common import paths as P


def read_hallmark(path=P.HALLMARK_ENTREZ_GMT) -> dict[str, list[str]]:
    sets: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                sets[parts[0]] = parts[2:]
    return sets


def pathway_coverage_mask(
    gmt: dict[str, list[str]], measurable_genes: set[str], min_members: int = 10
) -> tuple[dict[str, bool], dict[str, int]]:
    """每集合有效成员数与掩码：有效成员 < min_members 掩掉（不按模型效果挑选）。"""
    mask: dict[str, bool] = {}
    effective: dict[str, int] = {}
    for name, members in gmt.items():
        eff = sum(1 for m in members if m in measurable_genes)
        effective[name] = eff
        mask[name] = eff >= min_members
    return mask, effective


def compute_pathway_scores(
    expr: np.ndarray, members_by_set: dict[str, list[int]], mask: dict[str, bool] | None = None
) -> np.ndarray:
    """固定成员关系的均值聚合。members_by_set 值为表达式矩阵列索引。

    分数 = 表达的派生表示（非独立组学）；干预表达时必须重算（见 apply_intervention_recompute）。
    """
    expr = np.asarray(expr, dtype=float)
    n = expr.shape[0]
    out = np.full((n, len(members_by_set)), np.nan)
    for j, (name, cols) in enumerate(members_by_set.items()):
        if mask is not None and not mask.get(name, False):
            continue
        out[:, j] = expr[:, cols].mean(axis=1)
    return out


def apply_intervention_recompute(expr, members_by_set, intervention) -> np.ndarray:
    """干预表达（如置零某基因）后同步重算通路分数（§16.5-6 测试点）。"""
    new_expr = np.asarray(expr, dtype=float).copy()
    new_expr = intervention(new_expr)
    return compute_pathway_scores(new_expr, members_by_set)


def set_overlap_jaccard(gmt: dict[str, list[str]]) -> pd.DataFrame:
    """集合间重叠（共享基因 Jaccard）报告。"""
    names = list(gmt)
    m = pd.DataFrame(index=names, columns=names, dtype=float)
    sets = {n: set(v) for n, v in gmt.items()}
    for a in names:
        for b in names:
            inter = len(sets[a] & sets[b])
            union = len(sets[a] | sets[b])
            m.loc[a, b] = inter / union if union else 0.0
    return m