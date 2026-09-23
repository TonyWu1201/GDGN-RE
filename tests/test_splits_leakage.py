"""测试：划分泄漏审计（计划 §14 test_splits_leakage，§8.3/§16.5-1/7）。"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _cohort_hash() -> str:
    return Path("data/processed/cohorts/cohort_hash.txt").read_text().strip()


def _audit() -> dict:
    return json.load(open(f"data/splits/{_cohort_hash()}/split_audit.json", encoding="utf-8"))


def test_five_leakage_checks_all_pass():
    a = _audit()
    assert all(v["pass"] for v in a["leakage_audit"].values()), a["leakage_audit"]


def test_pairs_and_replicates_not_across_sets():
    """LPO：同一 pair 不跨 train/test。"""
    ch = _cohort_hash()
    fold = pd.read_csv(f"data/splits/{ch}/lpo/fold_0.csv", dtype=str)
    fold["pair"] = fold["cell_id"] + "|" + fold["compound_id"]
    tr = set(fold[fold["role"] == "train"]["pair"])
    te = set(fold[fold["role"] == "test"]["pair"])
    assert tr.isdisjoint(te)


def test_parent_same_group():
    """LDO-SO：同药物母体不跨 test/train。"""
    ch = _cohort_hash()
    comp = pd.read_csv("data/processed/entities/compound_master.csv", dtype=str)
    cid2parent = dict(zip(comp["compound_id"], comp["normalized_parent_id"]))
    for f in range(5):
        fold = pd.read_csv(f"data/splits/{ch}/ldo_so/fold_{f}.csv", dtype=str)
        fold["parent"] = fold["compound_id"].map(cid2parent)
        tr = set(fold[fold["role"] == "train"]["parent"])
        te = set(fold[fold["role"] == "test"]["parent"])
        assert tr.isdisjoint(te), f"fold {f} 母体跨集合"


def test_related_group_same_side():
    """LCO：细胞相关组不跨 train/test。"""
    ch = _cohort_hash()
    for name in ["fold_candidate_0", "fold_quick_screening_0"]:
        fold = pd.read_csv(f"data/splits/{ch}/lco/{name}.csv", dtype=str)
        tr = set(fold[fold["role"] == "train"]["cell_group"])
        te = set(fold[fold["role"] == "test"]["cell_group"])
        assert tr.isdisjoint(te)


def test_sealed_zero_leakage():
    """封存区样本零出现在任何开发折。"""
    ch = _cohort_hash()
    cell_assign = pd.read_csv(f"data/splits/{ch}/holdout/cell_group_assignment.csv", dtype=str)
    drug_assign = pd.read_csv(f"data/splits/{ch}/holdout/drug_group_assignment.csv", dtype=str)
    c_hold = set(cell_assign[cell_assign["region"] == "hold"]["cell_id"])
    d_hold = set(drug_assign[drug_assign["region"] == "hold"]["compound_id"])
    for fold_file in Path(f"data/splits/{ch}/lco").glob("fold_*.csv"):
        fold = pd.read_csv(fold_file, dtype=str)
        assert not fold["cell_id"].isin(c_hold).any()
        assert not fold["compound_id"].isin(d_hold).any()


def test_split_files_have_zero_response_columns():
    """split 文件 0 响应列。"""
    ch = _cohort_hash()
    forbidden = {"y", "response_value", "LN_IC50", "z_score", "auc"}
    for p in Path(f"data/splits/{ch}").rglob("*.csv"):
        df = pd.read_csv(p, nrows=0)
        assert forbidden.isdisjoint(df.columns), f"{p} 含响应列"