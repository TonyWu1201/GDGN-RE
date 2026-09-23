"""阶段 C3：药物主表定稿（program/harmonize/build_compound_master.py）。

计划 §4.2。起点 = compound_structure_adjudication.csv 的 242 个 core_candidate；
低置信度 4 药、Rapamycin(1084)/KRAS(G12C) Inhibitor-12(1855) 维持排除待复核；
47 个 no_public_structure 排除照实报告。

处理：
- 字段按计划 §4.2 全量；compound_id 按 (DRUG_ID) 排序确定性编号。
- RDKit 标准化（固定版本）：保留 original_smiles 不动，另产
  standardized_isomeric_smiles；金属配合物（Cisplatin/Oxaliplatin）不拆片段，
  保留原始 SMILES 标 metal_complex_untouched（DECISION_LOG 2026-09-22 规则）。
- 重复分组：同 InChIKey 药物赋同一 normalized_parent_id（10 对，Core 内全部成对）。
- Bemis–Murcko scaffold 预注册化学分组表（供 D2/D3 LDO-SO 用）。

输出：compound_master.csv / compound_scaffolds.csv / compound_disposition.csv
      / compound_standardization_report.json
"""

from __future__ import annotations

import sys
import time

import pandas as pd
from rdkit import Chem, __version__ as rdkit_version
from rdkit.Chem.Scaffolds import MurckoScaffold

from program.common import paths as P
from program.common.runlog import (
    deterministic_json_bytes,
    resolve_config,
    save_config_resolved,
    save_run_record,
    sha256_file,
)

METAL_SCAFFOLD_ID = "SC_METAL"
FAIL_SCAFFOLD_ID = "SC_FAIL"


def standardize_smiles(smi: str) -> tuple[str | None, str | None]:
    """固定 RDKit 流程：解析 → 最大片段为母体（单片段不动）→ 立体保留 → isomeric canonical。"""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None, "rdkit_parse_failed"
    try:
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        parent_mol = max(frags, key=lambda m: m.GetNumHeavyAtoms()) if len(frags) > 1 else mol
        return Chem.MolToSmiles(parent_mol, isomericSmiles=True, canonical=True), None
    except Exception as e:  # noqa: BLE001
        return None, f"standardize_error:{e}"


def main() -> int:
    t0 = time.time()
    stage = "C3_build_compound_master"

    adj = pd.read_csv(P.COMPOUND_ADJUDICATION_CSV, dtype=str).fillna("")
    raw = pd.read_csv(P.COMPOUND_MASTER_RAW_CSV, dtype=str).fillna("")
    screened = pd.read_csv(P.GDSC_SCREENED_COMPOUNDS_INCHI_CSV, dtype=str).fillna("")
    screened_plain = pd.read_csv(P.GDSC_SCREENED_COMPOUNDS_CSV, dtype=str).fillna("")

    core = adj[adj["core_eligibility"] == "core_candidate"].copy()
    low_conf = adj[adj["core_eligibility"] == "low_confidence_candidate_only"]
    review = adj[adj["core_eligibility"] == "review_required_not_core"]
    excluded = adj[adj["core_eligibility"] == "excluded_from_core"]
    assert len(core) == 242, f"core_candidate 应为 242，实际 {len(core)}"

    raw_by_id = raw.set_index("source_drug_id")
    screened_by_id = screened.set_index("DRUG_ID")
    plain_by_id = screened_plain.set_index("DRUG_ID")

    # ---- 逐药构建主表（compound_id 按 DRUG_ID 数值排序确定性编号） ----
    core = core.sort_values("source_drug_id", key=lambda s: s.astype(int)).reset_index(drop=True)
    rows: list[dict] = []
    std_failures: list[dict] = []
    for i, r in core.iterrows():
        did = r["source_drug_id"]
        rr = raw_by_id.loc[did]
        sc = screened_by_id.loc[did]
        salt_policy = rr["salt_policy"]
        original_smiles = rr["original_smiles"]

        if salt_policy == "metal_complex_untouched":
            std_smiles, std_err = original_smiles, None
        else:
            std_smiles, std_err = standardize_smiles(original_smiles)
        if std_err:
            std_failures.append({"source_drug_id": did, "source_name": r["source_name"], "error": std_err})

        ikey = ""
        if std_smiles:
            mol = Chem.MolFromSmiles(std_smiles)
            if mol is not None:
                ikey = Chem.MolToInchiKey(mol)

        rows.append(
            {
                "compound_id": f"CMP{i + 1:04d}",
                "source_drug_id": did,
                "source_name": r["source_name"],
                "inchikey_adjudicated": r["candidate_inchikey"],
                "inchikey_recomputed": ikey,
                "original_smiles": original_smiles,
                "standardized_isomeric_smiles": std_smiles,
                "salt_policy": salt_policy,
                "stereochemistry_status": rr["stereochemistry_status"],
                "structure_source": "legacy_seed_cid_pubchem_verified_adjudicated",
                "mapping_method": rr["mapping_method"],
                "mapping_confidence": rr["mapping_confidence"],
                "review_note": r["adjudication_note"],
                "putative_target": sc["TARGET"],
                "target_pathway": sc["TARGET_PATHWAY"],
                "chembl_id": sc["CHEMBL_ID"],
                "screening_site": sc["SCREENING_SITE"],
            }
        )

    master = pd.DataFrame(rows)

    # ---- InChIKey 重算一致性 ----
    # 重算键来自去盐母体；审定键（candidate_inchikey）可能对应含盐形式（PubChem CID 键）。
    # HCl/Br-/甲磺酸盐等会改变 InChIKey 骨架段（质子化态），故第一段不一致者按盐策略归因，逐条列出，
    # 不得静默；盐策略类（largest_fragment_as_parent）为预期内差异。
    mism = master[master["inchikey_adjudicated"] != master["inchikey_recomputed"]].copy()
    mism["skeleton_mismatch"] = [
        k.split("-")[0] != v.split("-")[0]
        for k, v in zip(mism["inchikey_adjudicated"], mism["inchikey_recomputed"])
    ]
    skel_mism = mism[mism["skeleton_mismatch"]].copy()
    # 盐差异归因：这些药物 original_smiles 含酸/碱反离子，审定键为含盐形式键
    salt_attributed_ids: set[str] = set()
    unattributed_skel_mism: list[dict] = []
    for _, r in skel_mism.iterrows():
        rr = raw_by_id.loc[r["source_drug_id"]]
        if rr["salt_policy"] == "largest_fragment_as_parent":
            salt_attributed_ids.add(r["source_drug_id"])
        else:
            unattributed_skel_mism.append(
                {"source_drug_id": r["source_drug_id"], "source_name": r["source_name"], "adjudicated": r["inchikey_adjudicated"], "recomputed": r["inchikey_recomputed"]}
            )
    assert not unattributed_skel_mism, f"未归因的骨架层 InChIKey 差异: {unattributed_skel_mism}"
    mism_out = mism[["source_drug_id", "source_name", "inchikey_adjudicated", "inchikey_recomputed", "skeleton_mismatch"]].copy()
    mism_out["attribution"] = [
        "salt_form_inchikey_difference (parent vs salt-form CID key)" if d in salt_attributed_ids else "layer_difference"
        for d in mism_out["source_drug_id"]
    ]

    # ---- normalized_parent_id：同 InChIKey 同 parent ----
    parent_ids: dict[str, str] = {}
    for ikey, grp in master.groupby("inchikey_adjudicated"):
        pid = f"NP{len(parent_ids) + 1:04d}"
        for cid in grp["compound_id"]:
            parent_ids[cid] = pid
    master["normalized_parent_id"] = master["compound_id"].map(parent_ids)

    # ---- Bemis–Murcko scaffold（预注册化学分组；金属配合物单独组） ----
    scaf_rows: list[dict] = []
    scaf_ids: dict[str, str] = {}
    for _, r in master.iterrows():
        if r["salt_policy"] == "metal_complex_untouched":
            scaf_rows.append(
                {"compound_id": r["compound_id"], "source_name": r["source_name"], "scaffold_smiles": "METAL_COMPLEX", "scaffold_id": METAL_SCAFFOLD_ID}
            )
            continue
        mol = Chem.MolFromSmiles(r["standardized_isomeric_smiles"])
        s = MurckoScaffold.MurckoScaffoldSmiles(mol=mol) if mol is not None else None
        if s is None:
            scaf_rows.append(
                {"compound_id": r["compound_id"], "source_name": r["source_name"], "scaffold_smiles": "PARSE_FAILED", "scaffold_id": FAIL_SCAFFOLD_ID}
            )
            continue
        if s not in scaf_ids:
            scaf_ids[s] = f"SC{len(scaf_ids) + 1:04d}"
        scaf_rows.append({"compound_id": r["compound_id"], "source_name": r["source_name"], "scaffold_smiles": s, "scaffold_id": scaf_ids[s]})
    scaffolds = pd.DataFrame(scaf_rows)
    master = master.merge(scaffolds[["compound_id", "scaffold_id"]], on="compound_id", how="left")

    # ---- 295 药去向登记（响应行流向在 D1 瀑布用） ----
    disposition = pd.concat(
        [
            core.assign(disposition="core_included"),
            low_conf.assign(disposition="low_confidence_not_core"),
            review.assign(disposition="review_required_not_core"),
            excluded.assign(disposition="excluded_no_public_structure"),
        ]
    )[["source_drug_id", "source_name", "core_eligibility", "decision", "disposition", "adjudication_note"]]

    # ---- 验收 ----
    assert master["compound_id"].is_unique, "compound_id 不唯一"
    assert (master["standardized_isomeric_smiles"] != "").all(), "存在无标准化 SMILES 的药物"
    parse_fail = [c for c in master["standardized_isomeric_smiles"] if Chem.MolFromSmiles(c) is None]
    assert not parse_fail, f"RDKit 解析失败 {len(parse_fail)}"
    same_key_counts = master["inchikey_adjudicated"].value_counts()
    multi = same_key_counts[same_key_counts > 1]
    assert len(multi) == 10 and (multi == 2).all(), f"同 InChIKey 对不闭合: {multi.to_dict()}"
    assert len(disposition) == 295, "295 药去向应闭合"

    out_master = P.ENTITIES_DIR / "compound_master.csv"
    out_scaffolds = P.ENTITIES_DIR / "compound_scaffolds.csv"
    out_disp = P.ENTITIES_DIR / "compound_disposition.csv"
    out_report = P.ENTITIES_DIR / "compound_standardization_report.json"
    master.to_csv(out_master, index=False)
    scaffolds.to_csv(out_scaffolds, index=False)
    disposition.to_csv(out_disp, index=False)

    same_key_groups = master["inchikey_adjudicated"].value_counts()
    multi_keys = same_key_groups[same_key_groups > 1]
    report = {
        "rdkit_version": rdkit_version,
        "n_drugs_total_gdsc2": 295,
        "n_core_included": len(master),
        "n_low_confidence_excluded": len(low_conf),
        "n_review_required_excluded": len(review),
        "n_no_public_structure_excluded": len(excluded),
        "standardization_failures": std_failures,
        "inchikey_mismatches_attributed": mism_out.to_dict("records"),
        "same_inchikey_pairs": {
            k: v.to_dict("records") for k, v in master.groupby("inchikey_adjudicated") if len(v) > 1
        },
        "n_normalized_parents": master["normalized_parent_id"].nunique(),
        "n_scaffolds": int((~scaffolds["scaffold_id"].isin([METAL_SCAFFOLD_ID, FAIL_SCAFFOLD_ID])).sum()),
        "metal_complex_untouched": master[master["salt_policy"] == "metal_complex_untouched"]["source_name"].tolist(),
        "inchikey_multiset": multi_keys.to_dict(),
    }
    out_report.write_bytes(deterministic_json_bytes(report))

    base_cfg = {
        "stage": stage,
        "rdkit_version": rdkit_version,
        "salt_policy_rule": "largest_fragment_as_parent; metal complexes untouched (DECISION_LOG 2026-09-22)",
        "compound_id_rule": "sorted by integer DRUG_ID, CMP0001..",
        "counts": {
            "core": len(master),
            "low_confidence": len(low_conf),
            "review_required": len(review),
            "excluded": len(excluded),
            "std_failures": len(std_failures),
            "skeleton_inchikey_mismatches_attributed_to_salt": int(mism_out["skeleton_mismatch"].sum()),
            "same_inchikey_pairs": int((multi_keys == 2).sum()),
        },
    }
    cfg = resolve_config(base_cfg)
    save_config_resolved(stage, cfg)
    save_run_record(
        stage,
        cfg,
        input_hashes={
            str(p): sha256_file(p)
            for p in [P.COMPOUND_ADJUDICATION_CSV, P.COMPOUND_MASTER_RAW_CSV, P.GDSC_SCREENED_COMPOUNDS_INCHI_CSV]
        },
        outputs={
            str(out_master): sha256_file(out_master),
            str(out_scaffolds): sha256_file(out_scaffolds),
            str(out_disp): sha256_file(out_disp),
        },
        status="succeeded",
        started_at=t0,
    )
    print(f"[C3] 242 core + {len(low_conf)} low-conf + {len(review)} review + {len(excluded)} excluded = 295")
    print(f"[C3] 标准化失败 {len(std_failures)}；同 InChIKey {int((multi_keys == 2).sum())} 对；parent {master['normalized_parent_id'].nunique()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())