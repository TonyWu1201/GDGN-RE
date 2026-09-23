"""阶段 E1 共享：表达预处理（program/features/expression_preprocess.py）。

铁律（§7.1）：确定实体与测量 → 冻结队列 → 划分 → 训练子集拟合预处理 → 原样变换验证/测试。
逐折流程：去高缺失/无变异基因 → 中位数插补 → 按唯一实体标准化（方差下限防零除）→ HVG。
"""

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
    """在训练子集上拟合预处理状态；返回 state 供 transform_with_state 使用。

    - 高缺失/无变异基因按训练统计筛除（kept_columns）；
    - 插补用训练中位数；
    - 标准化按唯一实体（entity_ids 提供时去重均值/方差，否则按行）；
    - HVG 按唯一实体方差选前 hvg_n 个（并列按列序打破）。
    """
    train_expr = np.asarray(train_expr, dtype=float)
    kept_all = list(range(train_expr.shape[1]))

    # 1) 高缺失/无变异筛除（训练统计）
    miss = np.isnan(train_expr).mean(axis=0)
    filled = np.where(np.isnan(train_expr), np.nanmedian(train_expr, axis=0), train_expr)
    if entity_ids is not None:
        uniq = sorted(set(entity_ids))
        first_idx = [next(i for i, c in enumerate(entity_ids) if c == u) for u in sorted(set(entity_ids), key=lambda u: entity_ids.index(u))]
        ent_mat = np.array([filled[entity_ids.index(u)] for u in uniq_ids(entity_ids)])
        var = ent_mat.var(axis=0, ddof=1)
    else:
        var = filled.var(axis=0, ddof=1)
    keep_mask = (miss < 0.5) & (var > min_variance)
    kept_columns = list(np.where(keep_mask)[0])

    # 2) 插补统计（训练中位数）
    medians = np.nanmedian(train_expr[:, kept_columns], axis=0) if kept_columns else np.array([])

    # 3) 唯一实体标准化统计
    entity_scale = 1.0
    if entity_ids is not None and len(kept_columns):
        ent_mean = ent_mat[:, kept_columns].mean(axis=0)
        ent_sd = ent_mat[:, kept_columns].std(axis=0, ddof=1)
        entity_scale = float(np.median(ent_sd[ent_sd > 0])) if (ent_sd > 0).any() else 1.0
        if entity_scale == 0:
            entity_scale = 1.0
        medians = ent_mean_per_col(ent_mat, kept_columns, medians)
    else:
        entity_scale = 1.0

    # 4) HVG（按唯一实体方差，训练折）
    selected = kept_columns
    if hvg_n is not None and kept_columns:
        ent_kept = ent_mat[:, kept_columns] if entity_ids is not None else filled[:, kept_columns]
        v = np.nanvar(ent_kept, axis=0, ddof=1)
        order = np.argsort(-v, kind="stable")  # 并列按列序打破
        selected = [kept_columns[i] for i in order[: min(hvg_n, len(kept_columns))]]
        selected = sorted(selected)

    return {
        "kept_columns": kept_columns,
        "selected_columns": selected,
        "medians": medians,
        "entity_scale": entity_scale,
        "impute": impute,
        "scale": scale,
        "min_variance": min_variance,
        "hvg_n": hvg_n,
    }


def uniq_ids(entity_ids):
    return sorted(set(entity_ids), key=lambda u: entity_ids.index(u))


def ent_mean_per_col(ent_mat, kept_columns, fallback):
    return np.nanmean(ent_mat[:, kept_columns], axis=0)


def transform_with_state(expr: np.ndarray, state: dict) -> np.ndarray:
    """原样变换验证/测试输入：只用训练统计，不重拟合。"""
    expr = np.asarray(expr, dtype=float)
    out = expr[:, state["selected_columns"]].copy()
    med = state["medians"]
    # medians 对应 kept_columns；需映射到 selected_columns 位置
    kept_pos = {c: i for i, c in enumerate(state["kept_columns"])}
    for j, c in enumerate(state["selected_columns"]):
        col = out[:, j]
        nan_mask = np.isnan(col)
        col[nan_mask] = med[kept_pos[c]]
        out[:, j] = col
    if state["scale"]:
        mu = out.mean(axis=0)
        sd = out.std(axis=0, ddof=1)
        sd = np.where(np.isfinite(sd) & (sd > state["min_variance"]), sd, 1.0)
        out = (out - mu) / sd
    return out


def constant_label_guard(y: np.ndarray) -> bool:
    """恒定预测处理占位：标签是否非恒定。"""
    return len(set(np.asarray(y).tolist())) > 1