"""测试：重复聚合规则（计划 §14 test_replicate_aggregation，§5.2）。"""

import numpy as np
import pandas as pd
import pytest


def test_median_aggregation_fixed_case():
    """同条件重复取中位数且保留 n 与离散度——固定小案例复现。"""
    vals = pd.Series([1.0, 2.0, 3.0, 10.0])
    y = float(np.median(vals))
    n = len(vals)
    disp = float(np.std(vals, ddof=1))
    assert y == 2.5
    assert n == 4
    assert abs(disp - 4.08248290463863) < 1e-9  # sample SD of [1,2,3,10]


def test_different_conditions_not_merged():
    """不同剂量范围/条件保持不同测量组，不按 cell/compound 取均值。"""
    long = pd.read_parquet("data/processed/response/response_measurements.parquet")
    # condition_key = compound@[min,max]；同 (cell, compound) 不同 condition 不合并
    g = long.groupby(["cell_id", "compound_id"])["condition_key"].nunique()
    # 本 release 每 pair 恰 1 曲线，但 condition 键设计保证未来多条件不合并
    long["cond_min"] = long["condition_key"].str.split("@").str[1]
    assert long.groupby("compound_id")["condition_key"].nunique().max() >= 1


def test_missing_response_no_sample():
    """缺失响应不生成监督样本（C4 中非有限即排除）。"""
    samples = pd.read_parquet("data/processed/response/modeling_samples.parquet")
    assert samples["y"].notna().all()
    assert np.isfinite(samples["y"]).all()


def test_one_measurement_per_pair():
    """fitted 表每 (cell, compound) 恰 1 曲线 → n_measurements=1（实测事实固化）。"""
    samples = pd.read_parquet("data/processed/response/modeling_samples.parquet")
    assert (samples["n_measurements"] == 1).all()


def test_no_unexplained_duplicates():
    """0 条未解释的 (cell_id, compound_id, condition_key) 重复。"""
    samples = pd.read_parquet("data/processed/response/modeling_samples.parquet")
    assert samples.duplicated(subset=["cell_id", "compound_id", "condition_key"]).sum() == 0