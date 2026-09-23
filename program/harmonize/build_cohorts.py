"""阶段 D1：队列构建与瀑布审计（program/harmonize/build_cohorts.py）。

计划 §6。瀑布链逐级输出计数与主因归类：
GDSC2 242,036 → 细胞映射 → 表达可用 → 结构核验 → 聚合 → Core 门槛 → Extended-M 覆盖。

Core 门槛（cohort.json）：单药 ≥30 细胞、单细胞 ≥10 药；稀疏另表保存不删除。
Extended-M：Core 中额外有突变（默认 profile）的样本；覆盖表对照 0.80 掩码偏好线。
Legacy-Bridge：旧 404 细胞 × 184 药经新主表重新对齐（旧表只作候选证据）。

输出：data/processed/cohorts/{core,extended_m,extended_m_masked_full_core,legacy_bridge}_samples.parquet
      + cohort_waterfall.json/md + drug/cell_coverage.csv + sparse_*.csv + tissue_comparison.csv
      + cohort_hash.txt
"""

from __future__ import annotations

import sys
import time

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

LEGACY_REPO_ROOT = P.Path(
    P.REPO_ROOT.parent.parent / "模型改进" / "GDGN"
)  # 旧仓库只读引用；路径经环境差异时在此调整
LEGACY_COMMON_CELL_LINES = LEGACY_REPO_ROOT / "data" / "common_cell_lines.csv"
LEGACY_COMMON_DRUGS = LEGACY_REPO_ROOT / "data" / "common_drugs_pubchem.csv"

MIN_CELLS_PER_DRUG = 30
MIN_DRUGS_PER_CELL = 10

SAMPLE_COLS = [
    "sample_id",
    "cell_id",
    "compound_id",
    "condition_key",
    "y",
    "n_measurements",
    "replicate_dispersion",
    "any_extrapolation",
    "sanger_model_id",
    "cell_line_name",
    "oncotree_lineage",
    "gdsc_cancer_type",
    "source_drug_id",
    "source_name",
    "normalized_parent_id",
    "scaffold_id",
]


def main() -> int:
    t0 = time.time()
    stage = "D1_build_cohorts"

    samples = pd.read_parquet(P.RESPONSE_DIR / "modeling_samples.parquet")
    cell = pd.read_csv(P.ENTITIES_DIR / "cell_master.csv", dtype=str)
    comp = pd.read_csv(P.ENTITIES_DIR / "compound_master.csv", dtype=str)
    comp["source_drug_id"] = comp["source_drug_id"].astype(str)

    w = samples.merge(
        comp[["compound_id", "source_drug_id", "source_name", "scaffold_id", "normalized_parent_id"]],
        on="compound_id",
        how="left",
    ).merge(
        cell[["cell_id", "sanger_model_id", "cell_line_name", "oncotree_lineage", "gdsc_cancer_type", "status"]],
        on="cell_id",
        how="left",
    )

    waterfall: list[dict] = []

    def record(level: str, df: pd.DataFrame, reason: str = "") -> None:
        waterfall.append(
            {
                "level": level,
                "reason": reason,
                "measurements": int(len(df)),
                "unique_cells": int(df["cell_id"].nunique()),
                "unique_drugs": int(df["compound_id"].nunique()),
                "unique_pairs": int(df[["cell_id", "compound_id"]].drop_duplicates().shape[0]),
            }
        )

    record("W1_valid_measurements_after_source_qc", w)

    # ---- W2：细胞稳定 ID 映射 ----
    w2 = w[w["sanger_model_id"].notna()].copy()
    record("W2_after_cell_mapping", w2, "per C2; RH-1 rows lost here (no_expression_mapping, expected)")

    # ---- W3：基础表达可用（RNA 默认 profile） ----
    rna_cells = set(cell[cell["profile_rna_model_id"].notna()]["cell_id"])
    w3 = w2[w2["cell_id"].isin(rna_cells)].copy()
    record("W3_after_basic_expression", w3, "RNA default profile required (C2)")

    # ---- W4：小分子结构核验（242 core 内） ----
    w4 = w3[w3["source_drug_id"].notna()].copy()
    record("W4_after_structure_verification", w4, "242 core_candidate compounds (C3); 47+2+4 dispositioned out")

    # ---- W5：合法重复聚合（fitted 表 1 曲线/pair → 恒等） ----
    record("W5_after_replicate_aggregation", w4, "1 measurement per (cell,drug); median aggregation identity")

    # ---- W6：Core 门槛（迭代至稳定） ----
    core = w4.copy()
    for _ in range(10):
        dc = core.groupby("compound_id")["cell_id"].nunique()
        cd = core.groupby("cell_id")["compound_id"].nunique()
        keep_drugs = set(dc[dc >= MIN_CELLS_PER_DRUG].index)
        keep_cells = set(cd[cd >= MIN_DRUGS_PER_CELL].index)
        new = core[core["compound_id"].isin(keep_drugs) & core["cell_id"].isin(keep_cells)]
        if len(new) == len(core):
            break
        core = new
    record("W6_core_after_thresholds", core, f"min_cells_per_drug={MIN_CELLS_PER_DRUG}, min_drugs_per_cell={MIN_DRUGS_PER_CELL}")

    # ---- 覆盖与稀疏另表 ----
    dc_all = w4.groupby("compound_id")["cell_id"].nunique()
    cd_all = w4.groupby("cell_id")["compound_id"].nunique()
    drug_cov = dc_all.sort_values(ascending=False).rename("n_cells").reset_index()
    cell_cov = cd_all.sort_values(ascending=False).rename("n_drugs").reset_index()
    sparse_drugs = dc_all[dc_all < MIN_CELLS_PER_DRUG].rename("n_cells").sort_values(ascending=False).reset_index()
    sparse_cells = cd_all[cd_all < MIN_DRUGS_PER_CELL].rename("n_drugs").sort_values().reset_index()
    sparse_drugs = sparse_drugs.merge(comp[["compound_id", "source_name"]], on="compound_id", how="left")
    sparse_cells = sparse_cells.merge(cell[["cell_id", "cell_line_name"]], on="cell_id", how="left")
    n_cells_all, n_drugs_all = int(w4["cell_id"].nunique()), int(w4["compound_id"].nunique())
    sparsity = 1 - len(w4) / (n_cells_all * n_drugs_all)

    # ---- 组织分布对比（筛前 vs Core） ----
    t_before = w4["oncotree_lineage"].value_counts().rename("before_core")
    t_after = core["oncotree_lineage"].value_counts().rename("core")
    tissue = pd.concat([t_before, t_after], axis=1).fillna(0).astype(int)
    tissue["loss_ratio"] = (1 - tissue["core"] / tissue["before_core"].replace(0, pd.NA)).astype(float).round(4)

    # ---- Core 样本 ----
    core_samples = (
        core[SAMPLE_COLS].sort_values("sample_id").reset_index(drop=True)
    )

    # ---- Extended-M ----
    mut_cells = set(cell[cell["profile_mut_model_id"].notna()]["cell_id"])
    extended_m = core_samples.copy()
    extended_m["has_mutation"] = extended_m["cell_id"].isin(mut_cells)
    extended_m = extended_m[extended_m["has_mutation"]].reset_index(drop=True)
    core_mut_coverage = sum(1 for c in core_samples["cell_id"].unique() if c in mut_cells) / max(1, int(core_samples["cell_id"].nunique()))

    # ---- Legacy-Bridge ----
    legacy_out = None
    legacy_counts = None
    if LEGACY_COMMON_CELL_LINES.exists() and LEGACY_COMMON_DRUGS.exists():
        old_cells = pd.read_csv(LEGACY_COMMON_CELL_LINES, dtype=str)
        old_drugs = pd.read_csv(LEGACY_COMMON_DRUGS, dtype=str)
        mid2cid = dict(zip(cell["depmap_model_id"], cell["cell_id"]))
        legacy_cell_ids = {mid2cid[m] for m in old_cells["depMapID"].astype(str).str.strip() if m in mid2cid}
        old_drug_ids = set(old_drugs["GDSC_DRUG_ID"].astype(str).str.strip())
        legacy_comp = set(comp[comp["source_drug_id"].isin(old_drug_ids)]["compound_id"])
        legacy = core[core["cell_id"].isin(legacy_cell_ids) & core["compound_id"].isin(legacy_comp)]
        legacy_out = legacy[SAMPLE_COLS].sort_values("sample_id").reset_index(drop=True)
        legacy_counts = {
            "legacy_cells_aligned": int(legacy_out["cell_id"].nunique()),
            "legacy_drugs_aligned": int(legacy_out["compound_id"].nunique()),
            "legacy_samples": len(legacy_out),
            "legacy_cells_total_404": len(old_cells),
            "legacy_drugs_total_184": len(old_drugs),
        }

    # ---- cohort_hash ----
    cohort_hash = sha256_frame(core_samples)

    out_core = P.COHORTS_DIR / "core_samples.parquet"
    out_ext = P.COHORTS_DIR / "extended_m_samples.parquet"
    out_masked = P.COHORTS_DIR / "extended_m_masked_full_core.parquet"
    out_legacy = P.COHORTS_DIR / "legacy_bridge_samples.parquet"
    core_samples.to_parquet(out_core, index=False)
    extended_m.to_parquet(out_ext, index=False)
    core_samples.assign(has_mutation=core_samples["cell_id"].isin(mut_cells)).to_parquet(out_masked, index=False)
    if legacy_out is not None:
        legacy_out.to_parquet(out_legacy, index=False)

    # ---- 验收：三条 0 线 ----
    assert core_samples["y"].notna().all(), "非有限标签 0 线失败"
    dup_unexplained = int(core_samples.duplicated(subset=["cell_id", "compound_id", "condition_key"]).sum())
    assert dup_unexplained == 0, f"未解释重复 {dup_unexplained} != 0"
    core_statuses = set(cell[cell["cell_id"].isin(core_samples["cell_id"])]["status"])
    allowed = {"mapped", "mapped_adjudicated", "mapped_via_cvcl_rrid", "mapped_via_cmp_broad_id_evidence"}
    assert core_statuses <= allowed, f"Core 存在未解决映射歧义: {core_statuses}"

    wf_json = {
        "stage": stage,
        "waterfall": waterfall,
        "core_thresholds": {"min_cells_per_drug": MIN_CELLS_PER_DRUG, "min_drugs_per_cell": MIN_DRUGS_PER_CELL},
        "core_stats": {
            "samples": len(core_samples),
            "cells": int(core_samples["cell_id"].nunique()),
            "drugs": int(core_samples["compound_id"].nunique()),
            "tissues": int(core_samples["oncotree_lineage"].nunique()),
            "matrix_sparsity": round(float(sparsity), 4),
            "scaffolds": int(core_samples["scaffold_id"].nunique()),
            "normalized_parents": int(core_samples["normalized_parent_id"].nunique()),
            "y_mean": round(float(core_samples["y"].mean()), 4),
            "y_std": round(float(core_samples["y"].std()), 4),
        },
        "extended_m": {
            "samples": len(extended_m),
            "cells_with_mutation": int(extended_m["cell_id"].nunique()),
            "core_cell_coverage": round(core_mut_coverage, 4),
            "masked_model_preference_line": 0.80,
        },
        "legacy_bridge": legacy_counts,
        "zero_lines": {
            "nonfinite_labels": 0,
            "unexplained_duplicates": dup_unexplained,
            "unresolved_mapping_ambiguity": int((cell[cell["cell_id"].isin(core_samples["cell_id"])]["status"] == "review_one_to_many").sum()),
        },
    }
    out_wf_json = P.COHORTS_DIR / "cohort_waterfall.json"
    out_wf_json.write_bytes(deterministic_json_bytes(wf_json))

    md_lines = ["# D1 队列瀑布审计（2026-09-23）", ""]
    for wlv in waterfall:
        md_lines.append(
            f"- **{wlv['level']}**: {wlv['measurements']} 测量 / {wlv['unique_cells']} 细胞 / {wlv['unique_drugs']} 药 / {wlv['unique_pairs']} 对{(' — ' + wlv['reason']) if wlv['reason'] else ''}"
        )
    md_lines += [
        "",
        "## Core 门槛（冻结：查看模型结果前一次性调整并登记 DECISION_LOG）",
        f"- min_cells_per_drug={MIN_CELLS_PER_DRUG}, min_drugs_per_cell={MIN_DRUGS_PER_CELL}（cohort.json 预注册）",
        f"- Core：{len(core_samples)} 样本 / {core_samples['cell_id'].nunique()} 细胞 / {core_samples['compound_id'].nunique()} 药 / {core_samples['oncotree_lineage'].nunique()} 组织 / {core_samples['scaffold_id'].nunique()} scaffolds",
        f"- y: mean={core_samples['y'].mean():.4f} std={core_samples['y'].std():.4f}",
        f"- 矩阵稀疏度 {sparsity:.4f}；稀疏药 {len(sparse_drugs)}、稀疏细胞 {len(sparse_cells)}（另表保存，不删除）",
        f"- Extended-M（突变）：{len(extended_m)} 样本；Core 细胞突变覆盖率 {core_mut_coverage * 100:.1f}%（0.80 掩码偏好线）",
        f"- Legacy-Bridge：{legacy_counts['legacy_cells_aligned']} 细胞 × {legacy_counts['legacy_drugs_aligned']} 药 = {legacy_counts['legacy_samples']} 样本（旧 404×184 对齐；四格对照在实验轮执行）",
        f"- 三条 0 线：非有限标签 0 / 未解释重复 {dup_unexplained} / 未解决歧义 {wf_json['zero_lines']['unresolved_mapping_ambiguity']}",
        f"- cohort_hash = `{cohort_hash}`",
    ]
    (P.COHORTS_DIR / "cohort_waterfall.md").write_text("\n".join(md_lines), encoding="utf-8")

    drug_cov.to_csv(P.COHORTS_DIR / "drug_coverage.csv", index=False)
    cell_cov.to_csv(P.COHORTS_DIR / "cell_coverage.csv", index=False)
    sparse_drugs.to_csv(P.COHORTS_DIR / "sparse_drugs.csv", index=False)
    sparse_cells.to_csv(P.COHORTS_DIR / "sparse_cells.csv", index=False)
    tissue.reset_index(names="oncotree_lineage").to_csv(P.COHORTS_DIR / "tissue_comparison.csv", index=False)
    (P.COHORTS_DIR / "cohort_hash.txt").write_text(cohort_hash, encoding="utf-8")

    base_cfg = {
        "stage": stage,
        "thresholds": {"min_cells_per_drug": MIN_CELLS_PER_DRUG, "min_drugs_per_cell": MIN_DRUGS_PER_CELL},
        "counts": {wlv["level"]: wlv["measurements"] for wlv in waterfall},
        "cohort_hash": cohort_hash,
    }
    cfg = resolve_config(base_cfg)
    save_config_resolved(stage, cfg)
    save_run_record(
        stage,
        cfg,
        input_hashes={str(P.RESPONSE_DIR / "modeling_samples.parquet"): sha256_file(P.RESPONSE_DIR / "modeling_samples.parquet")},
        outputs={str(out_core): sha256_file(out_core), str(out_wf_json): sha256_file(out_wf_json)},
        status="succeeded",
        started_at=t0,
    )
    print(f"[D1] Core: {len(core_samples)} 样本 / {core_samples['cell_id'].nunique()} 细胞 / {core_samples['compound_id'].nunique()} 药")
    print(f"[D1] Extended-M: {len(extended_m)}；Legacy-Bridge: {len(legacy_out)}")
    print(f"[D1] cohort_hash={cohort_hash}")
    return 0


if __name__ == "__main__":
    sys.exit(main())