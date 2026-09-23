"""测试：特征哈希链目录规则（计划 §14 test_feature_hashes，§16.1）。"""

import shutil
from pathlib import Path

import pytest

from program.features.cache import feature_dir_for, hash_inputs


def test_config_change_new_directory_no_overwrite():
    """配置改变不得覆盖同一结果目录——新哈希新目录。"""
    h1 = hash_inputs({"rdkit": "2026.03.6", "hvg": 1000})
    h2 = hash_inputs({"rdkit": "2026.03.6", "hvg": 2000})
    h3 = hash_inputs({"rdkit": "2026.03.7", "hvg": 1000})
    assert h1 != h2, "不同 HVG 配置必须不同哈希"
    assert h1 != h3, "不同 RDKit 版本必须不同哈希"


def test_deterministic_hashing():
    """同输入同哈希（确定性）。"""
    a = hash_inputs({"a": 1, "b": [1, 2]})
    b = hash_inputs({"b": [1, 2], "a": 1})  # 键序不同
    assert a == b


def test_feature_dir_chain_layout():
    """目录结构 data/features/<data_hash>/<split_hash>/<preprocess_hash>/。"""
    base = Path("tmp/test_feature_chain")
    if base.exists():
        shutil.rmtree(base)
    d = feature_dir_for(base, data_hash="D1", split_hash="S1", preprocess_hash="P1")
    assert d == base / "D1" / "S1" / "P1"
    d.mkdir(parents=True, exist_ok=True)
    assert (d).exists()
    # 相同链路返回同一路径
    d2 = feature_dir_for(base, data_hash="D1", split_hash="S1", preprocess_hash="P1")
    assert d2 == d
    # 不同配置 → 不同目录
    d3 = feature_dir_for(base, data_hash="D1", split_hash="S1", preprocess_hash="P2")
    assert d3 != d
    shutil.rmtree(base)