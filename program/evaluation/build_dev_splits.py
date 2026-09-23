"""阶段 D3：开发划分与泄漏审计（program/evaluation/build_dev_splits.py）。

计划 §8.3。开发区 = C_dev × D_dev（151,710 - 确认区样本）。

协议（split_policy.json，本轮所需）：
- LCO（主任务）：按细胞相关组 GroupKFold，组织分层；quick_screening 1 outer fold
  + candidate_confirmation 5 outer folds（各折独立落盘，测试侧组不与训练侧重叠）。
  OncotreeSubtype 分组压力测试作为独立附加文件；真正的表达近邻测试留待实现。
- LPO（补充）：规范化样本对随机留出，同一 pair 不跨集合（1 outer fold）。
- LDO-SO（次任务）：按标准化母体或 scaffold 的连通组分组（5 outer folds）。

泄漏审计（leakage_audit.py，5 项必测）：
1. 母体/相关组/重复测量不跨集合；
2. 封存区样本零出现在开发折；
3. 置换响应后划分不变；
4. 同种子同输入确定性 + 行序不变；
5. 折结构统计。

输出：data/splits/<cohort_hash>/{lco,lpo,ldo_so}/fold_*.csv + split_audit.json
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

from program.common import paths as P
from program.common.runlog import (
    deterministic_json_bytes,
    resolve_config,
    save_config_resolved,
    save_run_record,
    sha256_file,
    sha256_frame,
)
from program.evaluation.chemical_groups import chemical_clusters

BASE_SPLIT_SEED = 20260923
N_FOLDS_CANDIDATE = 5
N_FOLDS_QUICK = 1
QUICK_PARTITIONS = 5
EXPR_NEIGHBOR_K = 10


def stratified_group_folds(
    samples: pd.DataFrame,
    group_col: str,
    tissue_col: str,
    n_folds: int,
    seed: int,
) -> dict[int, set[str]]:
    """按组划分 n_folds，组织分布平衡：贪心把（组 → 主组织）按组织分层分配到折。"""
    rng = np.random.default_rng(seed)
    group_tissue: dict[str, str] = {}
    group_size: dict[str, int] = defaultdict(int)
    for g, t in zip(samples[group_col], samples[tissue_col]):
        group_size[g] += 1
        group_tissue.setdefault(g, t)

    layer_groups: dict[str, list[str]] = defaultdict(list)
    for g, t in group_tissue.items():
        layer_groups[t].append(g)

    fold_groups: dict[int, set[str]] = {f: set() for f in range(n_folds)}
    fold_size = [0] * n_folds
    for t in sorted(layer_groups):
        gs = sorted(layer_groups[t], key=lambda g: (-group_size[g], g))
        rng.shuffle(gs)
        target_total = sum(group_size[g] for g in gs) / n_folds
        acc = 0
        f_order = sorted(range(n_folds), key=lambda f: fold_size[f])
        fi = 0
        for g in gs:
            # 分配到当前最空折，同组织尽量同折轮转
            f = f_order[fi % n_folds]
            fold_groups[f].add(g)
            fold_size[f] += group_size[g]
            acc += group_size[g]
            fi += 1
    return {f: groups for f, groups in fold_groups.items()}


def main() -> int:
    t0 = time.time()
    stage = "D3_build_dev_splits"

    cohort_hash = (P.COHORTS_DIR / "cohort_hash.txt").read_text().strip()
    core = pd.read_parquet(P.COHORTS_DIR / "core_samples.parquet")

    holdout_dir = P.SPLITS_DIR / cohort_hash / "holdout"
    cell_assign = pd.read_csv(holdout_dir / "cell_group_assignment.csv", dtype=str)
    drug_assign = pd.read_csv(holdout_dir / "drug_group_assignment.csv", dtype=str)

    c_dev = set(cell_assign[cell_assign["region"] == "dev"]["cell_id"])
    d_dev = set(drug_assign[drug_assign["region"] == "dev"]["compound_id"])
    dev = core[core["cell_id"].isin(c_dev) & core["compound_id"].isin(d_dev)].copy()
    assert not dev.empty, "开发区为空"

    cell_group = dict(zip(cell_assign["cell_id"], cell_assign["group_id"]))
    sid2cell = dict(zip(core["cell_id"], core["sanger_model_id"]))
    dev = dev.assign(cell_group=dev["cell_id"].map(cell_group))

    comp = pd.read_csv(P.ENTITIES_DIR / "compound_master.csv", dtype=str)
    cid2parent = dict(zip(comp["compound_id"], comp["normalized_parent_id"]))
    cid2scaf = dict(zip(comp["compound_id"], comp["scaffold_id"]))
    cid2cluster = chemical_clusters(comp)

    split_dir = P.SPLITS_DIR / cohort_hash
    audit: dict = {"stage": stage, "cohort_hash": cohort_hash, "dev_samples": len(dev)}

    # ---- LCO：候选确认 5 folds + 快速筛选 1 fold ----
    lco_folds = stratified_group_folds(dev, "cell_group", "oncotree_lineage", N_FOLDS_CANDIDATE, BASE_SPLIT_SEED + 1)
    lco_quick = {0: stratified_group_folds(dev, "cell_group", "oncotree_lineage", QUICK_PARTITIONS, BASE_SPLIT_SEED + 4)[0]}

    lco_dir = split_dir / "lco"
    lco_dir.mkdir(parents=True, exist_ok=True)
    fold_meta = []
    for f in range(N_FOLDS_CANDIDATE):
        test_groups = lco_folds[f]
        is_test = dev["cell_group"].isin(test_groups)
        out = dev[["sample_id", "cell_id", "compound_id", "cell_group", "oncotree_lineage"]].copy()
        out["role"] = np.where(is_test, "test", "train")
        out.to_csv(lco_dir / f"fold_candidate_{f}.csv", index=False)
        fold_meta.append(
            {
                "fold": f,
                "test_groups": len(test_groups),
                "test_samples": int(is_test.sum()),
                "train_samples": int((~is_test).sum()),
                "test_tissue_dist": dev[is_test]["oncotree_lineage"].value_counts().to_dict(),
                "test_drugs": int(dev[is_test]["compound_id"].nunique()),
                "min_drug_test_cells": int(dev[is_test].groupby("compound_id")["cell_id"].nunique().min()) if is_test.any() else None,
            }
        )
    # 快速筛选 1 fold
    test_groups_q = lco_quick[0]
    is_test_q = dev["cell_group"].isin(test_groups_q)
    assert is_test_q.any() and (~is_test_q).any(), "快速筛选折必须同时有训练和测试样本"
    out_q = dev[["sample_id", "cell_id", "compound_id", "cell_group", "oncotree_lineage"]].copy()
    out_q["role"] = np.where(is_test_q, "test", "train")
    out_q.to_csv(lco_dir / "fold_quick_screening_0.csv", index=False)
    fold_meta.append({"fold": "quick_0", "test_samples": int(is_test_q.sum()), "train_samples": int((~is_test_q).sum())})

    # 组织亚型压力测试；不能称为表达近邻聚类
    lco_nn_dir = split_dir / "lco_subtype_stress"
    lco_nn_dir.mkdir(parents=True, exist_ok=True)
    model_csv = pd.read_csv(P.DEPMAP_MODEL_CSV, dtype=str)
    mid2sub = dict(zip(model_csv["ModelID"], model_csv["OncotreeSubtype"]))
    cell_master = pd.read_csv(P.ENTITIES_DIR / "cell_master.csv", dtype=str)
    cell2mid = dict(zip(cell_master["cell_id"], cell_master["depmap_model_id"]))
    dev_stress = dev.assign(subtype=dev["cell_id"].map(cell2mid).map(mid2sub).fillna("UNKNOWN"))
    stress_folds = stratified_group_folds(dev_stress, "subtype", "oncotree_lineage", 5, BASE_SPLIT_SEED + 100)
    stress_meta = []
    for f in range(5):
        test_subtypes = stress_folds[f]
        is_test = dev_stress["subtype"].isin(test_subtypes)
        out = dev_stress[["sample_id", "cell_id", "compound_id", "subtype", "oncotree_lineage"]].copy()
        out["role"] = np.where(is_test, "test", "train")
        out.to_csv(lco_nn_dir / f"fold_stress_{f}.csv", index=False)
        stress_meta.append({"fold": f, "test_subtypes": len(test_subtypes), "test_samples": int(is_test.sum())})

    # ---- LPO：样本对随机留出（规范化 pair = cell×drug；本表 1 行/pair） ----
    lpo_dir = split_dir / "lpo"
    lpo_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(BASE_SPLIT_SEED + 2)
    pair_keys = dev["cell_id"] + "|" + dev["compound_id"]
    unique_pairs = sorted(pair_keys.unique())
    rng.shuffle(unique_pairs)
    n_test = int(0.2 * len(unique_pairs))
    test_pairs = set(unique_pairs[:n_test])
    out_lpo = dev[["sample_id", "cell_id", "compound_id"]].copy()
    out_lpo["pair_key"] = pair_keys
    out_lpo["role"] = np.where(out_lpo["pair_key"].isin(test_pairs), "test", "train")
    out_lpo.drop(columns=["pair_key"]).to_csv(lpo_dir / "fold_0.csv", index=False)

    # ---- LDO-SO：同母体或同 scaffold 的连通化学簇整体划分 ----
    ldo_dir = split_dir / "ldo_so"
    ldo_dir.mkdir(parents=True, exist_ok=True)
    dev_ldo = dev.assign(parent=dev["compound_id"].map(cid2parent), scaffold=dev["compound_id"].map(cid2scaf), chemical_group=dev["compound_id"].map(cid2cluster), drug_stratum="NA")
    ldo_folds = stratified_group_folds(dev_ldo, "chemical_group", "drug_stratum", 5, BASE_SPLIT_SEED + 3)
    ldo_meta = []
    for f in range(5):
        test_groups = ldo_folds[f]
        is_test = dev_ldo["chemical_group"].isin(test_groups)
        out = dev_ldo[["sample_id", "cell_id", "compound_id", "parent", "scaffold", "chemical_group", "oncotree_lineage"]].copy()
        out["role"] = np.where(is_test, "test", "train")
        out.to_csv(ldo_dir / f"fold_{f}.csv", index=False)
        ldo_meta.append({"fold": f, "test_chemical_groups": len(test_groups), "test_parents": int(dev_ldo[is_test]["parent"].nunique()), "test_drugs": int(dev_ldo[is_test]["compound_id"].nunique()), "test_samples": int(is_test.sum())})

    audit["lco"] = fold_meta
    audit["lco_subtype_stress"] = stress_meta
    audit["lpo"] = {"test_pairs": n_test, "unique_pairs": len(unique_pairs)}
    audit["ldo_so"] = ldo_meta

    # ---- 泄漏审计 5 项 ----
    from program.evaluation.leakage_audit import run_full_audit

    audit_results = run_full_audit(
        dev=dev,
        lco_folds=lco_folds,
        lco_quick=lco_quick,
        lpo_test_pairs=test_pairs,
        ldo_folds=ldo_folds,
        stress_folds=stress_folds,
        cid2parent=cid2parent,
        cid2scaf=cid2scaf,
        cid2cluster=cid2cluster,
        cell_group=cell_group,
        cohort_hash=cohort_hash,
        split_seed=BASE_SPLIT_SEED,
        n_folds=N_FOLDS_CANDIDATE,
    )
    audit["leakage_audit"] = audit_results
    all_pass = all(v["pass"] for v in audit_results.values())
    assert all_pass, f"泄漏审计未全绿: { {k: v['pass'] for k, v in audit_results.items()} }"

    out_audit = split_dir / "split_audit.json"
    out_audit.write_bytes(deterministic_json_bytes(audit))

    base_cfg = {
        "stage": stage,
        "implementation_sha256": {
            "build_dev_splits": sha256_file(P.REPO_ROOT / "program" / "evaluation" / "build_dev_splits.py"),
            "leakage_audit": sha256_file(P.REPO_ROOT / "program" / "evaluation" / "leakage_audit.py"),
            "chemical_groups": sha256_file(P.REPO_ROOT / "program" / "evaluation" / "chemical_groups.py"),
        },
        "split_seed_base": BASE_SPLIT_SEED,
        "folds": {"lco_candidate": N_FOLDS_CANDIDATE, "lco_quick": N_FOLDS_QUICK, "lpo": 1, "ldo_so": 5},
        "protocols": ["lco", "lpo", "ldo_so", "lco_subtype_stress"],
        "leakage_audit_all_pass": all_pass,
    }
    cfg = resolve_config(base_cfg)
    save_config_resolved(stage, cfg)
    save_run_record(
        stage,
        cfg,
        input_hashes={str(p): sha256_file(p) for p in (P.COHORTS_DIR / "core_samples.parquet", P.ENTITIES_DIR / "compound_master.csv", holdout_dir / "cell_group_assignment.csv", holdout_dir / "drug_group_assignment.csv")},
        outputs={str(out_audit): sha256_file(out_audit)},
        status="succeeded",
        started_at=t0,
    )
    print(f"[D3] LCO 5+1 folds；LPO {n_test} test pairs；LDO-SO 5 folds；压力测试 5 folds")
    print(f"[D3] 泄漏审计：{ {k: v['pass'] for k, v in audit_results.items()} }")
    return 0


if __name__ == "__main__":
    sys.exit(main())
