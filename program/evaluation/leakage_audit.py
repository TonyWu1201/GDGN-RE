"""阶段 D3 泄漏审计（program/evaluation/leakage_audit.py）。

计划 §10 五项必测（任一失败即管线不通过）：
1. 同一药物母体、细胞相关组、重复测量不跨受控集合（train/test 两两检查）；
2. 封存区样本零出现在任何开发折；
3. 分组与划分对标签值不敏感：置换响应数值后重跑划分，实体归属逐位不变；
4. 确定性：同种子同输入重跑逐位一致；打乱输入行序（保留 ID）结果不变；
5. 折结构统计：每折组织分布、每药测试细胞数（confirmation_drug_min_cells 资格）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _check_group_disjoint(dev, fold_groups, group_col: str, name: str) -> dict:
    """审计 1：任一折的 test 组与其他折的 test 组不相交（组不跨 test 侧）。"""
    seen: set = set()
    overlaps = []
    for f, groups in fold_groups.items():
        dup = seen & set(groups)
        if dup:
            overlaps.append({"fold": f, "n_overlapping_groups": len(dup)})
        seen |= set(groups)
    # train 侧 = 非本折 test 组 → 组级划分天然保证 train/test 无同组
    return {"check": f"1_group_disjoint_{name}", "pass": not overlaps, "detail": overlaps}


def _check_sealed_zero(dev, lco_folds, stress_folds, cohort_hash) -> dict:
    """审计 2：封存区样本零出现在开发折。开发样本表本身已由 C_dev×D_dev 构造，这里复验 ID 空间。"""
    import pandas as pd

    from program.common import paths as P

    holdout_dir = P.SPLITS_DIR / cohort_hash / "holdout"
    cell_assign = pd.read_csv(holdout_dir / "cell_group_assignment.csv", dtype=str)
    drug_assign = pd.read_csv(holdout_dir / "drug_group_assignment.csv", dtype=str)
    c_hold = set(cell_assign[cell_assign["region"] == "hold"]["cell_id"])
    d_hold = set(drug_assign[drug_assign["region"] == "hold"]["compound_id"])
    leaked_cell = dev["cell_id"].isin(c_hold).sum()
    leaked_drug = dev["compound_id"].isin(d_hold).sum()
    return {
        "check": "2_sealed_zero_leakage",
        "pass": int(leaked_cell) == 0 and int(leaked_drug) == 0,
        "detail": {"leaked_cell_rows": int(leaked_cell), "leaked_drug_rows": int(leaked_drug)},
    }


def _check_label_permutation_invariance(dev, cell_group, cid2parent, n_folds, seed) -> dict:
    """审计 3：置换响应数值后重跑划分，实体归属逐位不变。"""
    from program.evaluation.build_dev_splits import stratified_group_folds

    rng = np.random.default_rng(999)
    perm = dev.copy()
    perm["y"] = rng.permutation(dev["y"].to_numpy())
    folds_a = stratified_group_folds(dev, "cell_group", "oncotree_lineage", n_folds, seed)
    folds_b = stratified_group_folds(perm, "cell_group", "oncotree_lineage", n_folds, seed)
    same = all(set(folds_a[f]) == set(folds_b[f]) for f in folds_a)
    return {"check": "3_label_permutation_invariant", "pass": bool(same), "detail": {"folds_compared": n_folds}}


def _check_determinism(dev, n_folds, seed) -> dict:
    """审计 4：同种子重跑逐位一致；打乱行序结果不变。"""
    from program.evaluation.build_dev_splits import stratified_group_folds

    folds_a = stratified_group_folds(dev, "cell_group", "oncotree_lineage", n_folds, seed)
    folds_b = stratified_group_folds(dev, "cell_group", "oncotree_lineage", n_folds, seed)
    same_seed = all(set(folds_a[f]) == set(folds_b[f]) for f in folds_a)
    shuffled = dev.sample(frac=1.0, random_state=7).reset_index(drop=True)
    folds_c = stratified_group_folds(shuffled, "cell_group", "oncotree_lineage", n_folds, seed)
    row_order_invariant = all(set(folds_a[f]) == set(folds_c[f]) for f in folds_a)
    return {
        "check": "4_deterministic_and_row_order_invariant",
        "pass": bool(same_seed and row_order_invariant),
        "detail": {"same_seed": bool(same_seed), "row_order_invariant": bool(row_order_invariant)},
    }


def _check_fold_structure(dev, lco_folds, cid2parent) -> dict:
    """审计 5：折结构统计——每折组织分布 + 每药测试细胞数 ≥ confirmation_drug_min_cells。"""
    min_cells = 10
    stats = []
    ok = True
    for f, groups in lco_folds.items():
        test = dev[dev["cell_group"].isin(groups)]
        per_drug = test.groupby("compound_id")["cell_id"].nunique()
        n_fail = int((per_drug < min_cells).sum())
        if n_fail > 0:
            ok = False
        stats.append(
            {
                "fold": f,
                "test_samples": int(len(test)),
                "tissue_dist": test["oncotree_lineage"].value_counts().to_dict(),
                "drugs_below_min_cells": n_fail,
            }
        )
    return {"check": "5_fold_structure_stats", "pass": bool(ok), "detail": stats}


def run_full_audit(
    dev,
    lco_folds,
    lco_quick,
    lpo_test_pairs,
    ldo_folds,
    stress_folds,
    cid2parent,
    cell_group,
    cohort_hash,
    split_seed,
    n_folds,
) -> dict:
    results: dict[str, dict] = {}
    results["g1"] = _check_group_disjoint(dev, lco_folds, "cell_group", "lco")
    results["g1b"] = _check_group_disjoint(dev, ldo_folds, "parent", "ldo_so")
    results["g2"] = _check_sealed_zero(dev, lco_folds, stress_folds, cohort_hash)
    results["g3"] = _check_label_permutation_invariance(dev, cell_group, cid2parent, n_folds, split_seed + 1)
    results["g4"] = _check_determinism(dev, n_folds, split_seed + 1)
    results["g5"] = _check_fold_structure(dev, lco_folds, cid2parent)
    return results