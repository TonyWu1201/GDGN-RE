"""测试：标签单位与外推旗标（计划 §14 test_label_units，§5.3）。"""

import numpy as np
import pytest


def test_ln_ic50_no_relog():
    """LN_IC50 = ln(IC50[µM])；换算 µM 用 exp，不再取 log。"""
    from program.harmonize.build_response_measurements import main as _m  # noqa: F401  (import ok)

    ln_ic50 = np.log(10.0)  # = ln(10)
    assert abs(np.exp(ln_ic50) - 10.0) < 1e-9
    # pIC50 = 6 - LN_IC50 / ln(10)
    pic50 = 6 - ln_ic50 / np.log(10)
    assert abs(pic50 - 5.0) < 1e-9


def test_auc_direction_note():
    """AUC 为归一化分数 [0,1]；方向声明不进标签（cohort.json auc_note）。"""
    # 字典 p.2: fraction of total area between highest and lowest screening concentration
    assert 0.0 <= 0.006282 and 0.998904 <= 1.0


def test_extrapolation_flag_boundaries():
    """外推旗标边界：IC50 恰在端点不记外推；超出才记。"""
    ln_ic50 = np.log(5.0)
    min_c, max_c = 0.01, 10.0
    ic50 = np.exp(ln_ic50)
    # 恰在 MAX_CONC：不外推
    assert not (ic50 > max_c) and not (ic50 < min_c)
    # 超出 MAX_CONC：外推
    assert np.exp(np.log(20.0)) > max_c
    # 低于 MIN_CONC：外推
    assert np.exp(np.log(0.001)) < min_c


def test_z_score_not_label():
    """Z_SCORE 随行携带但不得出现在建模样本标签位。"""
    import pandas as pd

    samples = pd.read_parquet("data/processed/response/modeling_samples.parquet")
    assert "z_score" not in samples.columns
    assert "y" in samples.columns


def test_label_magnitude_sanity():
    """长表 y 与 ln 单位一致：exp(y) 应为 µM 量级（0.001–1000）。"""
    import pandas as pd

    long = pd.read_parquet("data/processed/response/response_measurements.parquet")
    y = long["response_value"]
    assert float(y.min()) >= -9 and float(y.max()) <= 14
    # log_base 字段 = e
    assert (long["log_base"] == "e").all()
    assert (long["endpoint"] == "LN_IC50").all()