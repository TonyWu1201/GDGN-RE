"""阶段 D2：封存确认区（program/evaluation/build_sealed_regions.py）。

计划 §8.2。先成组后抽样：
- 细胞侧组 = related_cell_groups（Core 内仅 ML-1/ML-2 一组，其余单实体自组）；
- 药物侧组 = 标准化母体或 scaffold 相同者构成的化学连通组；
- 分组不使用响应数值或模型误差（split_policy.json）；
- 组织分布作平衡约束（每组织在封存/开发侧占比偏差受控）；
- 15%/15% 细胞组/药物组（预注册）。

四象限：C_dev×D_dev（开发）/ C_hold×D_dev（LCO 确认）/ C_dev×D_hold（LDO-SO 确认）/ C_hold×D_hold（DB 确认）。
split 文件只存实体 ID 与组归属，0 响应列。

划分种子（split_seed）：本轮选定 20260923 并登记 DECISION_LOG（独立于训练种子）。

输出：data/splits/<cohort_hash>/holdout/{cell_group_assignment.csv, drug_group_assignment.csv,
      region_manifest.csv, split_config_resolved.json} + sealed_coverage_report.json
"""

from __future__ import annotations

import sys
import time

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

SPLIT_SEED = 20260923
CELL_GROUP_RATIO = 0.15
DRUG_GROUP_RATIO = 0.15
TISSUE_TOLERANCE = 0.05


def assign_groups_with_balance(
    members: list[str],
    member_groups: dict[str, str],
    member_tissue: dict[str, str],
    ratio: float,
    seed: int,
) -> tuple[dict[str, str], dict[str, str]]:
    """按组为单位留出 ratio 比例，组织内分层抽样（每组织内按实体数比例留组）。返回 (member->region, group->region)。"""
    rng = np.random.default_rng(seed)
    group_members: dict[str, list[str]] = {}
    for m in members:
        group_members.setdefault(member_groups[m], []).append(m)

    # 每组织的组集合；组内组织不一致记 MIXED（Core 内仅可能跨组织组）
    group_tissue: dict[str, str] = {}
    for g, ms in group_members.items():
        ts = {member_tissue[m] for m in ms}
        group_tissue[g] = next(iter(ts)) if len(ts) == 1 else "MIXED"

    # 按组织分层：目标 = 每组织留 ratio；组分配到其主组织层
    layer_groups: dict[str, list[str]] = {}
    for g, ms in group_members.items():
        layer_groups.setdefault(group_tissue[g], []).append(g)

    hold_groups: set[str] = set()
    for t, gs in layer_groups.items():
        if t == "MIXED":
            # 混合组：并入整体池，按比例留
            n_members = sum(len(group_members[g]) for g in gs)
            k = max(1, round(ratio * n_members)) if n_members else 0
            order = sorted(gs, key=lambda g: (-len(group_members[g]), g))
            rng.shuffle(order)
            acc = 0
            for g in order:
                if acc >= k:
                    break
                hold_groups.add(g)
                acc += len(group_members[g])
            continue
        layer_members = [m for m in members if member_tissue[m] == t]
        n_members = len(layer_members)
        target = ratio * n_members
        # 组织内组大小尽量均匀：先小组后大组混洗，使占比接近 target 且不超
        order = sorted(gs, key=lambda g: (-len(group_members[g]), g))
        rng.shuffle(order)
        acc = 0
        for g in order:
            if acc + len(group_members[g]) <= target + 0.5 * max(len(group_members[x]) for x in gs):
                hold_groups.add(g)
                acc += len(group_members[g])
            if acc >= target:
                break

    member_region = {m: ("hold" if member_groups[m] in hold_groups else "dev") for m in members}
    group_region = {g: ("hold" if g in hold_groups else "dev") for g in group_members}
    return member_region, group_region


def main() -> int:
    t0 = time.time()
    stage = "D2_build_sealed_regions"

    cohort_hash = (P.COHORTS_DIR / "cohort_hash.txt").read_text().strip()
    core = pd.read_parquet(P.COHORTS_DIR / "core_samples.parquet")
    cell = pd.read_csv(P.ENTITIES_DIR / "cell_master.csv", dtype=str)
    related = pd.read_csv(P.ENTITIES_DIR / "related_cell_groups.csv", dtype=str)

    cell_ids = sorted(core["cell_id"].unique())
    drug_ids = sorted(core["compound_id"].unique())

    # ---- 细胞组：related_cell_groups（RG001 两人）；无组实体自组 ----
    sid2cell = dict(zip(cell["cell_id"], cell["sanger_model_id"]))
    related_pairs = related[related["group_id"].notna()][["sanger_model_id", "group_id"]]
    sid2group = dict(zip(related_pairs["sanger_model_id"], related_pairs["group_id"]))
    cell_group = {c: sid2group.get(sid2cell[c], f"CG_{c}") for c in cell_ids}
    cell_tissue = dict(zip(core["cell_id"], core["oncotree_lineage"]))

    # ---- 药物组：标准化母体或 scaffold 相同者必须同侧 ----
    comp = pd.read_csv(P.ENTITIES_DIR / "compound_master.csv", dtype=str)
    cid2parent = dict(zip(comp["compound_id"], comp["normalized_parent_id"]))
    clusters = chemical_clusters(comp)
    drug_group = {d: clusters[d] for d in drug_ids}

    cell_region, cell_group_region = assign_groups_with_balance(cell_ids, cell_group, cell_tissue, CELL_GROUP_RATIO, SPLIT_SEED)
    drug_region, _ = assign_groups_with_balance(drug_ids, drug_group, {d: "NA" for d in drug_ids}, DRUG_GROUP_RATIO, SPLIT_SEED + 1)

    # ---- 四象限 ----
    c_hold = {c for c in cell_ids if cell_region[c] == "hold"}
    d_hold = {d for d in drug_ids if drug_region[d] == "hold"}
    c_dev = set(cell_ids) - c_hold
    d_dev = set(drug_ids) - d_hold

    quad = {
        "dev": (c_dev, d_dev),
        "lco_confirmation": (c_hold, d_dev),
        "ldo_so_confirmation": (c_dev, d_hold),
        "db_confirmation": (c_hold, d_hold),
    }
    # 并集 = Core，两两不相交（样本级）
    seen = set()
    for name, (cs, ds) in quad.items():
        s = {(c, d) for c in cs for d in ds}
        assert s.isdisjoint(seen), f"象限 {name} 重叠"
        seen |= s
    assert seen == {(c, d) for c in cell_ids for d in drug_ids}, "象限并集 != Core"

    # ---- 资格清单：封存区每药 ≥10 细胞且标签有方差 ----
    quad_lco_drug_n = core[core["compound_id"].isin(d_dev) & core["cell_id"].isin(c_hold)].groupby("compound_id")["cell_id"].nunique()
    quad_lco_drug_var = core[core["compound_id"].isin(d_dev) & core["cell_id"].isin(c_hold)].groupby("compound_id")["y"].nunique()
    eligibility = pd.DataFrame(
        {
            "compound_id": sorted(d_dev),
        }
    )
    eligibility["lco_confirmation_cells"] = eligibility["compound_id"].map(quad_lco_drug_n).fillna(0).astype(int)
    eligibility["has_label_variance"] = eligibility["compound_id"].map(quad_lco_drug_var).fillna(0) > 1
    eligibility["meets_min10"] = eligibility["lco_confirmation_cells"] >= 10
    eligibility = eligibility.merge(comp[["compound_id", "source_name"]], on="compound_id", how="left")

    # ---- 输出（0 响应列） ----
    out_dir = P.SPLITS_DIR / cohort_hash / "holdout"
    out_dir.mkdir(parents=True, exist_ok=True)

    cell_assign = pd.DataFrame(
        {
            "cell_id": cell_ids,
            "sanger_model_id": [sid2cell[c] for c in cell_ids],
            "group_id": [cell_group[c] for c in cell_ids],
            "region": [cell_region[c] for c in cell_ids],
            "oncotree_lineage": [cell_tissue[c] for c in cell_ids],
        }
    )
    drug_assign = pd.DataFrame(
        {
            "compound_id": drug_ids,
            "normalized_parent_id": [cid2parent[d] for d in drug_ids],
            "chemical_group_id": [drug_group[d] for d in drug_ids],
            "region": [drug_region[d] for d in drug_ids],
        }
    )
    forbidden_cols = {"y", "response", "LN_IC50", "z_score"}
    assert forbidden_cols.isdisjoint(cell_assign.columns) and forbidden_cols.isdisjoint(drug_assign.columns), "split 文件含响应列"

    manifest = pd.DataFrame(
        [
            {"region": name, "cells": len(cs), "drugs": len(ds), "samples": int(((core["cell_id"].isin(cs)) & (core["compound_id"].isin(ds))).sum())}
            for name, (cs, ds) in quad.items()
        ]
    )

    cfg_resolved = {
        "stage": stage,
        "implementation_sha256": {
            "build_sealed_regions": sha256_file(P.REPO_ROOT / "program" / "evaluation" / "build_sealed_regions.py"),
            "chemical_groups": sha256_file(P.REPO_ROOT / "program" / "evaluation" / "chemical_groups.py"),
        },
        "cohort_hash": cohort_hash,
        "split_seed": SPLIT_SEED,
        "cell_group_ratio": CELL_GROUP_RATIO,
        "drug_group_ratio": DRUG_GROUP_RATIO,
        "tissue_balance_tolerance": TISSUE_TOLERANCE,
        "grouping_rules": {"use_response_values": False, "use_model_errors": False},
        "drug_grouping": "connected components of standardized parent and Bemis-Murcko scaffold",
        "naming": "internal confirmation (entities participated in legacy project development); NOT independent data",
    }
    cfg = resolve_config(cfg_resolved)

    cell_assign.to_csv(out_dir / "cell_group_assignment.csv", index=False)
    drug_assign.to_csv(out_dir / "drug_group_assignment.csv", index=False)
    manifest.to_csv(out_dir / "region_manifest.csv", index=False)
    (out_dir / "split_config_resolved.json").write_bytes(deterministic_json_bytes(cfg))
    eligibility.to_csv(out_dir / "sealed_region_drug_eligibility.csv", index=False)

    # 覆盖报告
    coverage = {
        "regions": manifest.to_dict("records"),
        "cell_region_counts": cell_assign["region"].value_counts().to_dict(),
        "drug_region_counts": drug_assign["region"].value_counts().to_dict(),
        "lco_confirmation": {
            "drugs_meeting_min10": int(eligibility["meets_min10"].sum()) if "meets_min10" in eligibility else int(eligibility["lco_confirmation_cells"].ge(10).sum()),
            "drugs_with_label_variance": int(eligibility["has_label_variance"].sum()),
            "total_drugs_in_region": len(eligibility),
        },
        "tissue_distribution_hold_vs_dev": {
            t: {
                "hold": int(cell_assign[(cell_assign["region"] == "hold") & (cell_assign["oncotree_lineage"] == t)].shape[0]),
                "dev": int(cell_assign[(cell_assign["region"] == "dev") & (cell_assign["oncotree_lineage"] == t)].shape[0]),
            }
            for t in sorted(cell_assign["oncotree_lineage"].unique())
        },
    }
    (out_dir / "sealed_coverage_report.json").write_bytes(deterministic_json_bytes(coverage))

    # ---- 验收 ----
    assert set(cell_assign["region"].unique()) == {"hold", "dev"}
    # 同组实体零跨侧
    g2r = cell_assign.groupby("group_id")["region"].nunique()
    assert (g2r == 1).all(), "细胞组跨侧"
    gp2r = drug_assign.groupby("chemical_group_id")["region"].nunique()
    assert (gp2r == 1).all(), "药物组跨侧"

    save_run_record(
        stage,
        cfg,
        input_hashes={str(p): sha256_file(p) for p in (P.COHORTS_DIR / "core_samples.parquet", P.ENTITIES_DIR / "compound_master.csv", P.ENTITIES_DIR / "related_cell_groups.csv")},
        outputs={
            str(out_dir / "cell_group_assignment.csv"): sha256_file(out_dir / "cell_group_assignment.csv"),
            str(out_dir / "drug_group_assignment.csv"): sha256_file(out_dir / "drug_group_assignment.csv"),
        },
        status="succeeded",
        started_at=t0,
    )
    print(f"[D2] C_hold {len(c_hold)} 细胞 / D_hold {len(d_hold)} 药；四象限不相交且并集=Core")
    print(manifest.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
