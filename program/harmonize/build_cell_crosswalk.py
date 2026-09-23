"""阶段 C2：细胞主表与 crosswalk（program/harmonize/build_cell_crosswalk.py）。

计划 §4.1。映射优先级：
1. SANGER_MODEL_ID ↔ Model.csv:SangerModelID（967/969 存在性已知）；
2. Cellosaurus 56.0 / CMP 20260921 佐证（RRID/CVCL、COSMIC）；
3. 名称规范化仅生成候选，不自动合并。

本地已核事实（2026-09-23 探查，写入审计）：
- 969 SIDM = 966 个 1:1 + SIDM00117（既有裁决绑定 ACH-002172/NCI-H322M）
  + SIDM01117（Model 表无 SangerModelID，但 ACH-003071 经 RRID=CVCL_1588 与
  Cellosaurus/CMP 一致恢复）+ SIDM00427（CMP 证据 ACH-002194 不在 26Q1、
  COSMIC 971773 不在 Model 表 → no_expression_mapping 预期内排除）。
- profile 只取 IsDefaultEntryForModel=Yes（RNA 1,719 / wgs 1,118 / 突变 1,968 默认模型）。
- 相关组：core 内可证实仅 ML-1/ML-2（CMP relations 同患者, PMID:3458526）；
  DepMap PatientID 在 core 内无 >1 组；Cellosaurus HI/AS 在 core 内 0 对；
  CMP parent_id 23 对的父系均不在 core。

输出：cell_master.csv / cell_crosswalk.csv / cell_mapping_review.csv / related_cell_groups.csv
"""

from __future__ import annotations

import pickle
import re
import sys
import time
from pathlib import Path

import pandas as pd

from program.common import paths as P
from program.common.runlog import (
    resolve_config,
    save_config_resolved,
    save_run_record,
    sha256_file,
)

CELLOSAURUS_INDEX_PICKLE = P.REPO_ROOT / "tmp" / "cellosaurus_index.pkl"


def norm_name(s: str) -> str:
    return re.sub(r"[-_;\s/.]", "", s.lower())


def build_cellosaurus_index() -> dict:
    """Cellosaurus 56.0 解析：CVCL -> (name, gdsc_cosmic_refs, cosmic_refs, cmp_refs)。"""
    if CELLOSAURUS_INDEX_PICKLE.exists():
        return pickle.load(open(CELLOSAURUS_INDEX_PICKLE, "rb"))
    txt = open(P.CELLOSAURUS_TXT, encoding="utf-8").read()
    idx: dict[str, tuple] = {}
    for chunk in txt.split("\n//"):
        m2 = re.match(r"\s*\n?ID   (.+)", chunk)
        a2 = re.search(r"\nAC   (\S+?)[;\n]", chunk)
        if not (m2 and a2):
            continue
        idx[a2.group(1)] = (
            m2.group(1).strip(),
            re.findall(r"\nDR   GDSC; (\d+);", chunk),
            re.findall(r"\nDR   Cosmic; (\d+);", chunk),
            re.findall(r"\nDR   Cell_Model_Passport; (\S+);", chunk),
        )
    CELLOSAURUS_INDEX_PICKLE.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(idx, open(CELLOSAURUS_INDEX_PICKLE, "wb"))
    return idx


def main() -> int:
    t0 = time.time()
    stage = "C2_build_cell_crosswalk"

    gdsc = pd.read_csv(
        P.GDSC2_RESPONSE_CSV,
        usecols=["SANGER_MODEL_ID", "CELL_LINE_NAME", "CANCER_TYPE"],
        dtype=str,
    )
    gdsc["sid"] = gdsc["SANGER_MODEL_ID"].str.strip()
    gdsc_ids = (
        gdsc.groupby("sid")
        .agg(gdsc_cell_line_name=("CELL_LINE_NAME", "first"), gdsc_cancer_type=("CANCER_TYPE", "first"))
        .reset_index()
    )
    sids = sorted(gdsc_ids["sid"])
    assert len(sids) == 969, f"GDSC Sanger ID 应为 969，实际 {len(sids)}"

    model = pd.read_csv(P.DEPMAP_MODEL_CSV, dtype=str)
    profiles = pd.read_csv(P.DEPMAP_PROFILES_CSV, dtype=str)
    cmp = pd.read_csv(P.CMP_MODEL_LIST_CSV, dtype=str)
    cel = build_cellosaurus_index()

    rna_default = set(profiles.loc[(profiles["DataType"] == "rna") & (profiles["IsDefaultEntryForModel"] == "Yes"), "ModelID"])
    cnv_default = set(profiles.loc[(profiles["DataType"] == "wgs") & (profiles["IsDefaultEntryForModel"] == "Yes"), "ModelID"])
    mut_default = set(
        pd.read_csv(P.DEPMAP_MUTATIONS_CSV, usecols=["ModelID", "IsDefaultEntryForModel"])
        .query("IsDefaultEntryForModel == 'Yes'")["ModelID"]
        .unique()
    )

    model_by_sid = model.dropna(subset=["SangerModelID"]).groupby("SangerModelID")["ModelID"].apply(list).to_dict()
    model_by_mid = model.set_index("ModelID")
    cmp_by_sid = cmp.set_index("model_id")

    # ---- 逐 SIDM 消歧 ----
    master_rows: list[dict] = []
    crosswalk_rows: list[dict] = []
    review_rows: list[dict] = []

    existing_adjudication = pd.read_csv(P.CELL_MAPPING_CONFLICTS_CSV, dtype=str)

    for _, row in gdsc_ids.iterrows():
        sid = row["sid"]
        evidence: list[str] = []
        status = ""
        model_ids: list[str] = []
        note = ""

        if sid == "SIDM00117":
            # 既有裁决（DECISION_LOG 2026-09-23）：绑 NCI-H322M / ACH-002172，不重开
            model_ids = ["ACH-002172"]
            status = "mapped_adjudicated"
            evidence.append("existing_adjudication: cell_mapping_conflicts.csv SIDM00117 -> ACH-002172 (COSMIC 905967, CVCL_1557)")
            note = "residual risk: Cellosaurus Caution H322/H322M identity uncertainty; ACH-000837 excluded"
        elif sid in model_by_sid and len(model_by_sid[sid]) == 1:
            mid = model_by_sid[sid][0]
            model_ids = [mid]
            status = "mapped"
            evidence.append(f"sanger_model_id_1to1: {sid} -> {mid} in DepMap 26Q1 Model.csv")
        elif sid in model_by_sid and len(model_by_sid[sid]) > 1:
            model_ids = model_by_sid[sid]
            status = "review_one_to_many"
            evidence.append(f"ambiguous: {sid} maps to {model_ids} in Model.csv")
            review_rows.append(
                {
                    "sanger_model_id": sid,
                    "gdsc_cell_line_name": row["gdsc_cell_line_name"],
                    "issue": "one_to_many_sidm_modelid",
                    "candidate_model_ids": ";".join(model_ids),
                    "resolution": "not auto-merged; excluded from Core candidates until resolved",
                }
            )
        else:
            # 969 - 967 = 2 个 Model 表无 SangerModelID 的 SIDM：查 CMP/Cellosaurus 佐证
            cmp_row = cmp_by_sid.loc[sid] if sid in cmp_by_sid.index else None
            cvcl_candidate = str(cmp_row["RRID"]) if cmp_row is not None and isinstance(cmp_row["RRID"], str) else None
            cos_candidate = str(cmp_row["COSMIC_ID"]) if cmp_row is not None and pd.notna(cmp_row["COSMIC_ID"]) else None
            broad_candidate = str(cmp_row["BROAD_ID"]) if cmp_row is not None and isinstance(cmp_row["BROAD_ID"], str) else None
            # 佐证1：CMP BROAD_ID 是否在 Model 表（且该行 SangerModelID 为空）
            if broad_candidate in model_by_mid.index:
                mrow = model_by_mid.loc[broad_candidate]
                if mrow["SangerModelID"] is None or pd.isna(mrow["SangerModelID"]):
                    model_ids = [broad_candidate]
                    status = "mapped_via_cmp_broad_id_evidence"
                    evidence.append(
                        f"cmp_model_list: {sid} ({cmp_row['model_name']}) BROAD_ID={broad_candidate} "
                        f"COSMIC={cos_candidate} RRID={cvcl_candidate}; DepMap Model row exists with blank SangerModelID"
                    )
            # 佐证2：CMP RRID (CVCL) 对应的 Model 行
            if not model_ids and cvcl_candidate:
                cand = model[(model["RRID"] == cvcl_candidate) & model["SangerModelID"].isna()]
                if len(cand) == 1:
                    model_ids = [cand["ModelID"].iloc[0]]
                    status = "mapped_via_cvcl_rrid"
                    evidence.append(
                        f"cvcl_bridge: {sid} -> CMP RRID {cvcl_candidate} -> unique DepMap model {model_ids[0]} "
                        f"({cand['CellLineName'].iloc[0]}); Cellosaurus/CMP 同源佐证"
                    )
            if not model_ids:
                status = "no_expression_mapping"
                note = "expected coverage loss: DepMap 26Q1 has no model row for this Sanger ID (CMP evidence absent from 26Q1)"
                evidence.append(
                    f"unresolved: CMP lists {sid} (BROAD_ID={broad_candidate}, COSMIC={cos_candidate}, RRID={cvcl_candidate}) "
                    "but none match a 26Q1 Model row; excluded per plan C2 rule 2"
                )

        # ---- profile 引用（每 cell_id 恰一套默认 profile） ----
        rna_id = cnv_id = mut_id = None
        if len(model_ids) == 1:
            mid = model_ids[0]
            rna_id = mid if mid in rna_default else None
            cnv_id = mid if mid in cnv_default else None
            mut_id = mid if mid in mut_default else None

        mrow = model_by_mid.loc[model_ids[0]] if len(model_ids) == 1 else None
        cel_entry = {}
        if mrow is not None and isinstance(mrow["RRID"], str):
            cel_entry = {"cvcl": mrow["RRID"], "cel_name": None, "cel_gdsc": None, "cel_cosmic": None, "cel_cmp": None}
            if mrow["RRID"] in cel:
                name, g_refs, c_refs, cmp_refs = cel[mrow["RRID"]][:4]
                cel_entry = {"cvcl": mrow["RRID"], "cel_name": name, "cel_gdsc": g_refs, "cel_cosmic": c_refs, "cel_cmp": cmp_refs}

        master_rows.append(
            {
                "cell_id": f"CL{len(master_rows) + 1:04d}" if len(model_ids) == 1 else None,
                "depmap_model_id": model_ids[0] if len(model_ids) == 1 else None,
                "sanger_model_id": sid,
                "cosmic_id": (str(model_by_mid.loc[model_ids[0], "COSMICID"]) if mrow is not None and pd.notna(mrow["COSMICID"]) else None),
                "cellosaurus_accession": cel_entry.get("cvcl"),
                "cell_line_name": (model_by_mid.loc[model_ids[0], "CellLineName"] if mrow is not None else row["gdsc_cell_line_name"]),
                "gdsc_cell_line_name": row["gdsc_cell_line_name"],
                "synonyms": (str(cmp_by_sid.loc[sid, "synonyms"]) if sid in cmp_by_sid.index and pd.notna(cmp_by_sid.loc[sid, "synonyms"]) else None),
                "oncotree_lineage": (model_by_mid.loc[model_ids[0], "OncotreeLineage"] if mrow is not None else None),
                "oncotree_primary": (model_by_mid.loc[model_ids[0], "OncotreePrimaryDisease"] if mrow is not None else None),
                "oncotree_subtype": (model_by_mid.loc[model_ids[0], "OncotreeSubtype"] if mrow is not None else None),
                "gdsc_cancer_type": row["gdsc_cancer_type"],
                "profile_rna_model_id": rna_id,
                "profile_cnv_model_id": cnv_id,
                "profile_mut_model_id": mut_id,
                "status": status,
                "evidence": "; ".join(evidence),
                "notes": note,
            }
        )
        crosswalk_rows.append(
            {
                "sanger_model_id": sid,
                "gdsc_cell_line_name": row["gdsc_cell_line_name"],
                "depmap_model_id": model_ids[0] if len(model_ids) == 1 else None,
                "depmap_cell_line_name": (model_by_mid.loc[model_ids[0], "CellLineName"] if mrow is not None else None),
                "cellosaurus_name": cel_entry.get("cel_name"),
                "mapping_status": status,
            }
        )

    master = pd.DataFrame(master_rows)
    crosswalk = pd.DataFrame(crosswalk_rows)

    # ---- 相关组（core 内可证实分组） ----
    # ML-1/ML-2：CMP relations + PMID:3458526 同患者
    sid2cellid = dict(zip(master["sanger_model_id"], master["cell_id"]))
    group_rows: list[dict] = []
    ml_pairs = [("SIDM00441", "SIDM00442")]
    group_members = {s: {"ML-1/ML-2"} for s in ("SIDM00441", "SIDM00442")}
    for sid, gname in group_members.items():
        group_rows.append(
            {
                "sanger_model_id": sid,
                "cell_id": sid2cellid.get(sid),
                "group_id": "RG001",
                "group_evidence": "cmp_relations_text: ML-1/ML-2/ML-3 same patient AML (PMID:3458526); DepMap PatientID differs but CMP explicit",
                "evidence_source": "cmp_model_relations_comment",
            }
        )
    related = pd.DataFrame(group_rows)
    # SIDM00117 残余风险保持记录（不加组；H322/H322M 仍为独立实体）
    related = pd.concat(
        [
            related,
            pd.DataFrame(
                [
                    {
                        "sanger_model_id": "SIDM00117",
                        "cell_id": sid2cellid.get("SIDM00117"),
                        "group_id": None,
                        "group_evidence": "no group: existing adjudication binds ACH-002172 only; Cellosaurus Caution recorded",
                        "evidence_source": "cell_mapping_conflicts.csv",
                    }
                ]
            ),
        ]
    )

    # ---- 验收：969 计数闭合 ----
    counts = master["status"].value_counts().to_dict()
    assert sum(counts.values()) == 969, f"969 计数不闭合: {counts}"
    mapped = master[master["cell_id"].notna()]
    assert mapped["cell_id"].is_unique, "cell_id 不唯一"
    assert mapped["depmap_model_id"].is_unique, "一个 cell_id 对多个 ModelID"
    n_rna = int(master["profile_rna_model_id"].notna().sum())
    n_cnv = int(master["profile_cnv_model_id"].notna().sum())
    n_mut = int(master["profile_mut_model_id"].notna().sum())
    # 每个映射 cell 恰有一套默认 profile（有则 1，无则 0，不允许 >1 由唯一 ModelID 保证）

    out_master = P.ENTITIES_DIR / "cell_master.csv"
    out_crosswalk = P.ENTITIES_DIR / "cell_crosswalk.csv"
    out_review = P.ENTITIES_DIR / "cell_mapping_review.csv"
    out_groups = P.ENTITIES_DIR / "related_cell_groups.csv"
    master.to_csv(out_master, index=False)
    crosswalk.to_csv(out_crosswalk, index=False)
    pd.DataFrame(review_rows).to_csv(out_review, index=False) if review_rows else out_review.write_text(
        "sanger_model_id,gdsc_cell_line_name,issue,candidate_model_ids,resolution\n", encoding="utf-8"
    )
    related.to_csv(out_groups, index=False)

    base_cfg = {
        "stage": stage,
        "inputs": {
            "gdsc2_response": str(P.GDSC2_RESPONSE_CSV),
            "model": str(P.DEPMAP_MODEL_CSV),
            "profiles": str(P.DEPMAP_PROFILES_CSV),
            "cmp_model_list": str(P.CMP_MODEL_LIST_CSV),
            "cellosaurus": str(P.CELLOSAURUS_TXT),
            "existing_adjudication": str(P.CELL_MAPPING_CONFLICTS_CSV),
        },
        "outputs": {
            "cell_master": str(out_master),
            "cell_crosswalk": str(out_crosswalk),
            "cell_mapping_review": str(out_review),
            "related_cell_groups": str(out_groups),
        },
        "counts": {"status_counts": counts, "mapped": len(mapped), "rna": n_rna, "cnv": n_cnv, "mut": n_mut, "review_rows": len(review_rows)},
        "decisions": {
            "sidm00117": "existing adjudication reused (ACH-002172)",
            "sidm01117": "recovered via CMP RRID CVCL_1588 -> ACH-003071",
            "sidm00427": "no_expression_mapping (ACH-002194 absent from 26Q1)",
            "profile_rule": "IsDefaultEntryForModel=Yes only",
            "related_groups": "ML-1/ML-2 via CMP relations PMID:3458526; 0 groups from PatientID/Cellosaurus HI-AS within core",
        },
    }
    cfg = resolve_config(base_cfg)
    save_config_resolved(stage, cfg)
    inputs = {
        str(p): sha256_file(p)
        for p in [
            P.GDSC2_RESPONSE_CSV,
            P.DEPMAP_MODEL_CSV,
            P.DEPMAP_PROFILES_CSV,
            P.CMP_MODEL_LIST_CSV,
            P.CELLOSAURUS_TXT,
            P.CELL_MAPPING_CONFLICTS_CSV,
        ]
    }
    save_run_record(
        stage,
        cfg,
        input_hashes=inputs,
        outputs={
            str(out_master): sha256_file(out_master),
            str(out_crosswalk): sha256_file(out_crosswalk),
            str(out_review): sha256_file(out_review),
            str(out_groups): sha256_file(out_groups),
        },
        status="succeeded",
        started_at=t0,
    )
    print(f"[C2] 969 计数闭合：{counts}")
    print(f"[C2] 映射 {len(mapped)}；RNA {n_rna}；CNV {n_cnv}；突变 {n_mut}；复核 {len(review_rows)}")
    print(f"[C2] 完成 → {out_master}")
    return 0


if __name__ == "__main__":
    sys.exit(main())