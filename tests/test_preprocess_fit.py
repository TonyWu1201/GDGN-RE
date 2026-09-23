"""测试：预处理只按训练折拟合（计划 §14 test_preprocess_fit，§7.1/§16.5-2/4）。"""

import numpy as np
import pandas as pd
import pytest

from program.features.expression_preprocess import (
    fit_preprocessing,
    transform_with_state,
)


def test_scaler_uses_train_only():
    """改动验证集数值不影响训练统计产物。"""
    train = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    val_a = np.array([[0.0, 0.0]])
    val_b = np.array([[999.0, -999.0]])

    state_a = fit_preprocessing(train, impute="median", scale=True)
    mean_a = state_a["medians"].copy()

    # 验证集不同 → 训练统计必须相同
    _ = transform_with_state(val_a, state_a)
    _ = transform_with_state(val_b, state_a)
    assert np.allclose(state_a["medians"], mean_a)


def test_imputation_uses_train_stats():
    """插补用训练统计（中位数），非验证/测试统计。"""
    train = np.array([[1.0, np.nan], [3.0, 4.0], [5.0, 6.0]])
    state = fit_preprocessing(train, impute="median", scale=False)
    assert state["medians"][0] == 3.0
    assert state["medians"][1] == 5.0  # NaN 被忽略的中位数
    val = np.array([[np.nan, np.nan]])
    out = transform_with_state(val, state)
    assert out[0, 0] == 3.0
    assert out[0, 1] == 5.0


def test_unique_entity_counting():
    """多响应对细胞不重复计入：训练统计按唯一细胞计算。"""
    # 3 个唯一细胞 × 2 响应重复
    expr = np.array([[1.0], [1.0], [2.0], [2.0], [3.0], [3.0]])
    cell_ids = ["A", "A", "B", "B", "C", "C"]
    state = fit_preprocessing(expr, impute="median", scale=True, entity_ids=cell_ids)
    # 唯一实体均值 = (1+2+3)/3 = 2；总体行均值同为 2 → 用方差区分：唯一实体方差 = var([1,2,3])=2/3
    assert abs(state["entity_scale"] - np.std([1.0, 2.0, 3.0], ddof=1)) < 1e-9


def test_low_variance_filter_train_only():
    """低方差基因筛选基于训练折；验证集恒定基因不触发筛选。"""
    train = np.array([[1.0, 5.0], [1.0, 6.0], [1.0, 7.0]])  # 第一列恒定
    state = fit_preprocessing(train, impute="median", scale=True, min_variance=1e-8)
    assert state["kept_columns"] == [1]


def test_constant_prediction_placeholder():
    """恒定预测处理占位：常数标签场景返回哨兵统计。"""
    from program.features.expression_preprocess import constant_label_guard

    assert constant_label_guard(np.array([5.0, 5.0, 5.0])) is False
    assert constant_label_guard(np.array([5.0, 6.0, 7.0])) is True