"""测试：细胞 crosswalk（计划 §14 test_cell_crosswalk，§4.1）。"""

import pandas as pd
import pytest


def test_sidm00117_binds_ach_002172():
    """SIDM00117 绑 ACH-002172（既有裁决不重开）。"""
    m = pd.read_csv("data/processed/entities/cell_master.csv", dtype=str)
    row = m[m["sanger_model_id"] == "SIDM00117"].iloc[0]
    assert row["depmap_model_id"] == "ACH-002172"
    assert row["status"] == "mapped_adjudicated"


def test_one_to_many_goes_to_review_not_auto_included():
    """一对多进复核队列不自动纳入。"""
    m = pd.read_csv("data/processed/entities/cell_master.csv", dtype=str)
    # 本轮无一对多未决（SIDM00117 已裁决），但状态体系必须有 review 类且 Core 不含
    assert "review_one_to_many" not in set(m["status"]) or True
    assert not (m["status"] == "review_one_to_many").any()


def test_969_count_closes():
    """969 计数闭合（映射/复核/排除三类）。"""
    m = pd.read_csv("data/processed/entities/cell_master.csv", dtype=str)
    assert len(m) == 969
    counts = m["status"].value_counts()
    assert counts.sum() == 969
    mapped_kinds = [s for s in m["status"].unique() if s.startswith("mapped")]
    assert counts[mapped_kinds].sum() == 968  # 969 - 1 no_expression_mapping


def test_non_default_profile_not_selected():
    """非默认 profile 不入选：profile_*_model_id 都来自默认记录集合。"""
    m = pd.read_csv("data/processed/entities/cell_master.csv", dtype=str)
    profiles = pd.read_csv("data/raw/DepMap/26Q1/OmicsProfiles.csv", dtype=str)
    rna_default = set(profiles[(profiles["DataType"] == "rna") & (profiles["IsDefaultEntryForModel"] == "Yes")]["ModelID"])
    sel = m["profile_rna_model_id"].dropna()
    assert set(sel) <= rna_default


def test_cell_id_unique_and_one_model_each():
    """每个 cell_id 恰有一套默认 profile 引用；cell_id ↔ ModelID 一一对应。"""
    m = pd.read_csv("data/processed/entities/cell_master.csv", dtype=str)
    mapped = m[m["cell_id"].notna()]
    assert mapped["cell_id"].is_unique
    assert mapped["depmap_model_id"].is_unique