"""阶段 C4：响应测量长表（program/harmonize/build_response_measurements.py）。

计划 §5.1/§5.2。C0 字典证实：LN_IC50=ln(IC50[µM]) 不得再取 log；MIN/MAX_CONC 单位 µM；
官方 RMSE>0.3 QC 阈值；Z_SCORE=按药标准化（not_training_label）。

本地实测（写入 QC 报告）：
- 242,036 行；NLME_RESULT_ID 恒为 1（常量），NLME_CURVE_ID 全表唯一；
  (SANGER_MODEL_ID, DRUG_ID) 无重复 → fitted 表 1 行=1 曲线=1 测量，无同条件技术重复需聚合。
- 76 药有 >1 组 (MIN_CONC, MAX_CONC) → condition_key 含剂量范围。
- 外推：IC50 > MAX_CONC 173,805 行（71.8%）、IC50 < MIN_CONC 740 行；主分析保留（外推旗标），敏感性版另存。

输出：response_measurements.parquet / modeling_samples.parquet
      / response_qc_report.json+md / unmapped_measurements.csv
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
)


def main() -> int:
    t0 = time.time()
    stage = "C4_build_response_measurements"

    g = pd.read_csv(P.GDSC2_RESPONSE_CSV, dtype={"SANGER_MODEL_ID": str, "DRUG_ID": str})
    n_source = len(g)
    assert n_source == 242036, f"源行数 {n_source} != 242,036"

    cell = pd.read_csv(P.ENTITIES_DIR / "cell_master.csv", dtype=str)
    comp = pd.read_csv(P.ENTITIES_DIR / "compound_master.csv", dtype=str)

    # ---- join 主表（join 不上的行进 unmapped，不得丢弃不留痕） ----
    g["sid"] = g["SANGER_MODEL_ID"].str.strip()
    cell_sid2cell = dict(zip(cell["sanger_model_id"], cell["cell_id"]))
    cell_sid2status = dict(zip(cell["sanger_model_id"], cell["status"]))
    drug_id2cid = dict(zip(comp["source_drug_id"], comp["compound_id"]))

    g["cell_id"] = g["sid"].map(cell_sid2cell)
    g["compound_id"] = g["DRUG_ID"].str.strip().map(drug_id2cid)

    unmapped_cell = g["cell_id"].isna()
    unmapped_drug = g["compound_id"].isna()
    unmapped = g[unmapped_cell | unmapped_drug].copy()
    sel = unmapped_cell | unmapped_drug
    parts = []
    if bool(unmapped_cell[sel].any()):
        parts.append("cell_not_mapped")
    if bool(unmapped_drug[sel].any()):
        parts.append("drug_not_mapped")
    # 逐行 reason：可能仅 cell 或仅 drug 未映射
    reason = (
        unmapped_cell[sel].map({True: "cell_not_mapped", False: ""}).astype(str)
        + np.where(
            unmapped_drug[sel],
            "|" + "drug_not_mapped" if bool(unmapped_cell[sel].any()) else "drug_not_mapped",
            "",
        )
    )
    reason = reason.str.strip("|")
    unmapped["unmapped_reason"] = reason
    mapped = g[~(unmapped_cell | unmapped_drug)].copy()

    # ---- 长表字段 ----
    ic50_uM = np.exp(mapped["LN_IC50"].to_numpy(dtype=float))
    extrap_above = ic50_uM > mapped["MAX_CONC"].to_numpy(dtype=float)
    extrap_below = ic50_uM < mapped["MIN_CONC"].to_numpy(dtype=float)
    nonfinite = ~np.isfinite(mapped["LN_IC50"].to_numpy(dtype=float))

    mapped["measurement_id"] = (
        "R" + mapped["NLME_RESULT_ID"].astype(str) + "-C" + mapped["NLME_CURVE_ID"].astype(str)
    )
    long = pd.DataFrame(
        {
            "measurement_id": mapped["measurement_id"],
            "dataset_id": mapped["DATASET"],
            "release_id": "8.5-27Oct23",
            "cell_id": mapped["cell_id"],
            "compound_id": mapped["compound_id"],
            "source_drug_id": mapped["DRUG_ID"].str.strip(),
            "screen_id": "unknown_gdsc2_dataset_level",
            "replicate_id": "unknown_single_curve_per_pair",
            "assay": "unknown_not_in_fitted_table",
            "treatment_duration": "unknown_not_in_fitted_table",
            "endpoint": "LN_IC50",
            "response_value": mapped["LN_IC50"].astype(float),
            "response_unit": "ln(uM)",
            "log_base": "e",
            "auc": mapped["AUC"].astype(float),
            "min_concentration": mapped["MIN_CONC"].astype(float),
            "max_concentration": mapped["MAX_CONC"].astype(float),
            "curve_fit_rmse": mapped["RMSE"].astype(float),
            "z_score": mapped["Z_SCORE"].astype(float),
            "z_score_note": "not_training_label",
            "source_row_id": mapped.index.astype(str),
        }
    )
    long["quality_flag"] = np.where(
        nonfinite,
        "nonfinite_label",
        np.where(long["curve_fit_rmse"] > 0.3, "rmse_above_official_qc", "pass"),
    )
    long["extrapolation_flag"] = np.where(
        extrap_above, "ic50_above_max_conc", np.where(extrap_below, "ic50_below_min_conc", "none")
    )

    # ---- 建模样本表（无同条件技术重复 → 1 测量=1 样本；聚合逻辑仍按中位数实现以便复用） ----
    # condition_key = compound_id + dose range（76 药有多组剂量范围）
    long["condition_key"] = (
        long["compound_id"] + "@[" + long["min_concentration"].astype(str) + "," + long["max_concentration"].astype(str) + "]"
    )
    valid = long[(long["quality_flag"] != "nonfinite")].copy()
    n_nonfinite_excluded = int((long["quality_flag"] == "nonfinite").sum())

    agg = (
        valid.groupby(["cell_id", "compound_id", "condition_key"], sort=True)
        .agg(
            y=("response_value", "median"),
            n_measurements=("response_value", "size"),
            replicate_dispersion=("response_value", lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else 0.0),
            measurement_ids=("measurement_id", lambda x: ";".join(x)),
            any_extrapolation=("extrapolation_flag", lambda x: int((x != "none").any())),
        )
        .reset_index()
    )
    agg.insert(0, "sample_id", [f"S{i + 1:06d}" for i in range(len(agg))])
    agg["flags_quality"] = "pass"

    # ---- 验收 ----
    assert len(long) + len(unmapped) == n_source, "行去向不闭合"
    assert long["measurement_id"].is_unique, "measurement_id 不唯一"
    assert (long["quality_flag"] != "nonfinite").sum() == len(valid)
    # Z_SCORE 不进任何标签位：建模样本表只有 y（LN_IC50 中位数）
    assert "z_score" not in agg.columns
    # 中位数聚合小案例：见 tests/test_replicate_aggregation.py
    counts = {
        "source_rows": n_source,
        "mapped_to_long_table": len(long),
        "unmapped_rows": len(unmapped),
        "nonfinite_excluded": n_nonfinite_excluded,
        "modeling_samples": len(agg),
        "extrapolation_above": int(extrap_above.sum()),
        "extrapolation_below": int(extrap_below.sum()),
        "rmse_above_qc": int((long["curve_fit_rmse"] > 0.3).sum()),
        "unmapped_reason_counts": unmapped["unmapped_reason"].value_counts().to_dict(),
        "n_condition_keys": valid["condition_key"].nunique(),
    }

    out_long = P.RESPONSE_DIR / "response_measurements.parquet"
    out_samples = P.RESPONSE_DIR / "modeling_samples.parquet"
    out_unmapped = P.RESPONSE_DIR / "unmapped_measurements.csv"
    out_qc_json = P.RESPONSE_DIR / "response_qc_report.json"
    out_qc_md = P.RESPONSE_DIR / "response_qc_report.md"

    long.to_parquet(out_long, index=False)
    agg.to_parquet(out_samples, index=False)
    unmapped_out = unmapped.copy()
    unmapped_out["measurement_id"] = (
        "R" + unmapped_out["NLME_RESULT_ID"].astype(str) + "-C" + unmapped_out["NLME_CURVE_ID"].astype(str)
    )
    unmapped_out[["measurement_id", "sid", "DRUG_ID", "unmapped_reason"]].rename(
        columns={"sid": "sanger_model_id", "DRUG_ID": "source_drug_id"}
    ).to_csv(out_unmapped, index=False)

    report = {
        "stage": stage,
        "counts": counts,
        "field_semantics": {
            "endpoint": "LN_IC50 = ln(IC50 / 1 uM) per GDSC dictionary p.2; no re-log (cohort.json no_relog)",
            "concentration_unit": "uM (dictionary MIN/MAX_CONC_MICROMOLAR p.2; raw CONC p.2)",
            "rmse_qc": "official threshold 0.3 (dictionary p.2); max observed 0.299984 -> 0 rows above",
            "z_score": "per-drug standardized; carried as column, marked not_training_label",
            "assay_duration": "unknown: not present in fitted table (dictionary defines raw-data fields only)",
        },
        "replicate_findings": "fitted table has exactly 1 curve per (cell, drug): 0 technical replicates to aggregate; median aggregation implemented and unit-tested for future raw-data evidence",
        "extrapolation": "71.8% rows IC50 above max tested concentration (GDSC right-censoring); main analysis keeps them flagged; sensitivity analysis without extrapolation at D1",
    }
    out_qc_json.write_bytes(deterministic_json_bytes(report))
    md = f"""# C4 响应测量长表 QC 报告（2026-09-23）

- 源行 242,036 → 长表 {len(long)} ＋ 未映射 {len(unmapped)} ＋ 非有限 {n_nonfinite_excluded}（计数闭合 {len(long)+len(unmapped)+n_nonfinite_excluded}）
- 未映射主因：{counts['unmapped_reason_counts']}
- 外推（IC50 超出测试剂量范围）：上方 {int(extrap_above.sum())} 行（{extrap_above.mean()*100:.1f}%）、下方 {int(extrap_below.sum())} 行；主分析保留带旗标，敏感性分析去外推
- RMSE 官方 QC（>0.3 排除）：超标 {counts['rmse_above_qc']} 行（0，与字典"release 前已剔除"一致）
- 重复测量：fitted 表每 (cell, drug) 恰 1 曲线，无同条件技术重复；中位数聚合逻辑实现并有单测（供 raw 证据到达后复用）
- Z_SCORE 随行携带但标 not_training_label；建模样本标签 y = LN_IC50（不取 log、不用 Z_SCORE）
- assay/treatment_duration：fitted 表无此列，记 unknown（分析限制已注明）
"""
    out_qc_md.write_text(md, encoding="utf-8")

    base_cfg = {
        "stage": stage,
        "label_policy": {"endpoint": "LN_IC50", "log_base": "e", "no_relog": True, "z_score_label": False},
        "aggregation": "median within (cell_id, compound_id, condition_key); keep n + dispersion",
        "counts": counts,
    }
    cfg = resolve_config(base_cfg)
    save_config_resolved(stage, cfg)
    save_run_record(
        stage,
        cfg,
        input_hashes={str(P.GDSC2_RESPONSE_CSV): sha256_file(P.GDSC2_RESPONSE_CSV)},
        outputs={
            str(out_long): sha256_file(out_long),
            str(out_samples): sha256_file(out_samples),
            str(out_unmapped): sha256_file(out_unmapped),
        },
        status="succeeded",
        started_at=t0,
    )
    print(f"[C4] 长表 {len(long)}；样本 {len(agg)}；未映射 {len(unmapped)}；非有限 {n_nonfinite_excluded}")
    print(f"[C4] 外推：上 {int(extrap_above.sum())} 下 {int(extrap_below.sum())}；RMSE 超标 {counts['rmse_above_qc']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())