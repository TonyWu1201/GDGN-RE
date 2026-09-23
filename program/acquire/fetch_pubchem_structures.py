"""PubChem 批量结构核验（rebuild-2026 计划 4.2）。

流程：
1. 从 GDSC2 拟合表提取全部药物（DRUG_ID, DRUG_NAME, PUTATIVE_TARGET）。
2. 用旧仓库 gdsc_to_cid.csv 与本地 compound_cid_smiles.csv 作候选 CID 证据（仅候选，非白名单）。
3. 对有候选 CID 的药物经 pubchempy 按 CID 直接核验；无候选的按名称检索产生候选。
4. 拉取 IsomericSMILES / InChIKey / InChI；用 RDKit 做结构可解析性检查并标准化为母体 SMILES。
5. 产出：
   - data/canonical/compound_master_raw.csv   候选药物主表（含映射证据与置信度）
   - data/canonical/compound_unmatched.csv    未匹配药物清单（不强行映射，进人工复核）
   - data/canonical/pubchem_fetch_report.json 拉取统计

注意：PubChem 的 ConnectivitySMILES 不含立体化学，此处取 IsomericSMILES。
限速：串行请求，每次间隔 REQUEST_INTERVAL 秒，失败重试 RETRIES 次。
"""

from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pubchempy as pcp
from rdkit import Chem, RDLogger

from program.acquire.paths import DATA_DIR, get_gdsc2_drugs

# 关闭 RDKit 的解析告警噪声（未匹配/坏结构在报告中显式列出）
RDLogger.DisableLog("rdApp.*")

RAW_STRUCT_DIR = DATA_DIR / "raw" / "legacy_structure_seed" / "unverified"
CANONICAL_DIR = DATA_DIR / "canonical"
OLD_SEED_CSV = RAW_STRUCT_DIR / "gdsc_to_cid.csv"
LOCAL_SEED_CSV = RAW_STRUCT_DIR / "compound_cid_smiles.csv"

REQUEST_INTERVAL = 0.35
RETRIES = 3
RETRY_WAIT = 5.0

# GDSC2 内部代号药物（IAP_5620 等）在 PubChem 名称空间无命中；这些是筛选库内部编号，
# 名称检索无法产生候选，直接进入未匹配清单走人工复核，不做模糊猜测。
# 同一母体不同批号（LMB_AB1/2/3）也不能靠名称合并，需人工核验。
ALIAS_HINTS: dict[str, str] = {
    "Picolinici-acid": "picolinic acid",
}


def load_seed_cids() -> dict[str, list[dict]]:
    """加载两个种子文件的候选 CID：gdsc_name -> [{cid, source, smiles?}]。"""
    seeds: dict[str, list[dict]] = {}

    if OLD_SEED_CSV.exists():
        with OLD_SEED_CSV.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                name = row["gdsc_name"].strip()
                cid = row["cid"].strip()
                if name and cid and cid.lower() not in {"nan", ""}:
                    cid = cid.split(".")[0]
                    seeds.setdefault(name, []).append(
                        {"cid": cid, "source": "legacy_gdsc_to_cid", "smiles": None}
                    )

    if LOCAL_SEED_CSV.exists():
        with LOCAL_SEED_CSV.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cid = row["cid"].strip()
                smi = row["smiles"].strip()
                if cid and cid.lower() not in {"nan", ""}:
                    seeds.setdefault(row.get("gdsc_name", ""), []).append(
                        {"cid": cid, "source": "local_compound_cid_smiles", "smiles": smi}
                    )

    return seeds


def fetch_by_cid(cid: str) -> dict | None:
    """按 CID 拉取 IsomericSMILES/InChIKey/InChI；失败返回 None。"""
    for attempt in range(RETRIES):
        try:
            results = pcp.Compound.from_cid(
                int(cid), properties=["IsomericSMILES", "InChIKey", "InChI", "MolecularFormula"]
            )
            return {
                "pubchem_cid": str(results.cid),
                "isomeric_smiles": results.isomeric_smiles,
                "inchikey": results.inchikey,
                "inchi": results.inchi,
                "molecular_formula": results.molecular_formula,
            }
        except Exception as exc:  # noqa: BLE001 - 需要区分网络/解析错误后重试
            if attempt == RETRIES - 1:
                print(f"    [warn] CID {cid} 拉取失败: {type(exc).__name__}: {exc}")
                return None
            time.sleep(RETRY_WAIT)


def search_by_name(name: str) -> list[dict]:
    """按名称检索候选（compound 命名空间），仅产生候选证据。"""
    for attempt in range(RETRIES):
        try:
            results = pcp.get_compounds(
                name, namespace="name", listkey_count=5,
                properties=["IsomericSMILES", "InChIKey", "InChI", "MolecularFormula"],
            )
            out = []
            for c in results:
                if c.cid and c.isomeric_smiles:
                    out.append(
                        {
                            "pubchem_cid": str(c.cid),
                            "isomeric_smiles": c.isomeric_smiles,
                            "inchikey": c.inchikey,
                            "inchi": c.inchi,
                            "molecular_formula": c.molecular_formula,
                        }
                    )
            return out
        except Exception as exc:  # noqa: BLE001
            if attempt == RETRIES - 1:
                print(f"    [warn] 名称 {name!r} 检索失败: {type(exc).__name__}: {exc}")
                return []
            time.sleep(RETRY_WAIT)


def rdkit_check(smiles: str) -> dict:
    """RDKit 可解析性检查 + 母体标准化（盐/fragment 去除，保留立体化学）。

    计划 4.2 规则 2：金属配合物不适用"最大片段作母体"，单独标注为
    metal_complex（不强行当作普通小分子拆分）。
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "rdkit_parse_ok": False,
            "parent_smiles": None,
            "n_fragments": None,
            "salt_policy_applied": "none_parse_failed",
        }

    METALS = {
        "Pt", "Pd", "Ni", "Co", "Cu", "Zn", "Fe", "Mn", "Cr", "Ti", "V", "Ru",
        "Rh", "Ir", "Os", "Mo", "W", "Re", "Au", "Ag", "Hg", "Cd", "Pb", "Al",
        "Ga", "In", "Sn", "Sb", "Bi", "Zr", "Hf", "Ta", "Nb", "Y", "La", "Ce",
        "Gd", "Lu", "Yb", "Er", "Eu", "Sm", "Tb", "Ho", "Dy", "Tm", "Pr", "Nd",
    }
    atoms = {atom.GetSymbol() for atom in mol.GetAtoms()}
    if atoms & METALS:
        # 金属配合物：保留原始 SMILES 作为结构记录，不拆片段
        return {
            "rdkit_parse_ok": True,
            "parent_smiles": None,  # 显式不产生"母体"，避免破坏配位结构
            "n_fragments": len(Chem.GetMolFrags(mol)),
            "salt_policy_applied": "metal_complex_untouched",
        }

    fragments = Chem.GetMolFrags(mol, asMols=True)
    if len(fragments) == 1:
        parent = mol
    else:
        # 最大片段作为母体（盐型策略：最大有机片段），策略显式记录在 salt_policy 列
        parent = max(fragments, key=lambda m: m.GetNumAtoms())
    try:
        parent = Chem.RemoveHs(parent)
        parent_smiles = Chem.MolToSmiles(parent, isomericSmiles=True)
    except Exception:  # noqa: BLE001
        parent_smiles = Chem.MolToSmiles(parent, isomericSmiles=True)

    return {
        "rdkit_parse_ok": True,
        "parent_smiles": parent_smiles,
        "n_fragments": len(fragments),
        "salt_policy_applied": (
            "single_fragment" if len(fragments) == 1 else "largest_fragment_as_parent"
        ),
    }


def main() -> int:
    CANONICAL_DIR.mkdir(parents=True, exist_ok=True)

    drugs = get_gdsc2_drugs()
    print(f"GDSC2 药物数: {len(drugs)}")

    seeds = load_seed_cids()
    print(f"种子 CID 覆盖药物名: {len(seeds)}")

    matched_rows: list[dict] = []
    unmatched: list[dict] = []
    stats = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "n_drugs": len(drugs),
        "n_seed_matched": 0,
        "n_seed_cid_fetch_failed": 0,
        "n_name_searched": 0,
        "n_name_search_matched": 0,
        "n_unmatched": 0,
        "n_rdkit_parse_failed": 0,
    }

    for i, drug in enumerate(drugs, 1):
        drug_id = drug["DRUG_ID"]
        name = drug["DRUG_NAME"]
        target = drug["PUTATIVE_TARGET"]
        print(f"[{i}/{len(drugs)}] DRUG_ID={drug_id} {name} ...")

        candidates = seeds.get(name, [])
        fetched = None
        mapping_method = None
        mapping_confidence = None

        if candidates:
            # 候选 CID 直接核验
            for cand in candidates:
                time.sleep(REQUEST_INTERVAL)
                fetched = fetch_by_cid(cand["cid"])
                if fetched is not None:
                    mapping_method = f"seed_cid:{cand['source']}"
                    mapping_confidence = "medium"  # 种子仅候选，需后续 InChIKey 级审计
                    break
            if fetched is None:
                stats["n_seed_cid_fetch_failed"] += 1
                print("    [warn] 种子 CID 均拉取失败，回退名称检索")
            else:
                stats["n_seed_matched"] += 1

        if fetched is None:
            # 名称检索（仅产生候选；内部代号先查人工别名表）
            stats["n_name_searched"] += 1
            search_name = ALIAS_HINTS.get(name, name)
            if search_name != name:
                print(f"    [alias] {name!r} -> {search_name!r}")
                mapping_method = "manual_alias_then_pubchem_name_top1"
            else:
                mapping_method = "pubchem_name_search_top1"
            time.sleep(REQUEST_INTERVAL)
            results = search_by_name(search_name)
            if results:
                fetched = results[0]  # 取首个命中，标记低置信度
                # 名称检索不能直接采信第一个命中（计划 4.2 规则 1）
                mapping_confidence = "low"
                stats["n_name_search_matched"] += 1

        if fetched is None:
            stats["n_unmatched"] += 1
            unmatched.append(
                {
                    "source_drug_id": drug_id,
                    "source_name": name,
                    "putative_target": target,
                    "reason": "no_pubchem_match_or_fetch_failed",
                    "seed_cid_candidates": ";".join(c["cid"] for c in candidates) or "none",
                }
            )
            print("    [unmatched]")
            continue

        rdkit = rdkit_check(fetched["isomeric_smiles"])
        if not rdkit["rdkit_parse_ok"]:
            stats["n_rdkit_parse_failed"] += 1

        row = {
            "source_drug_id": drug_id,
            "source_name": name,
            "putative_target": target,
            "pubchem_cid": fetched["pubchem_cid"],
            "original_smiles": fetched["isomeric_smiles"],
            "standardized_parent_smiles": rdkit["parent_smiles"],
            "inchikey": fetched["inchikey"],
            "inchi": fetched["inchi"],
            "molecular_formula": fetched["molecular_formula"],
            "rdkit_parse_ok": rdkit["rdkit_parse_ok"],
            "n_fragments": rdkit["n_fragments"],
            "salt_policy": rdkit["salt_policy_applied"],
            "stereochemistry_status": "preserved_isomeric",
            "mapping_method": mapping_method,
            "mapping_confidence": mapping_confidence,
            "review_note": "",
        }
        matched_rows.append(row)

    stats["finished_at"] = datetime.now(timezone.utc).isoformat()
    stats["n_matched"] = len(matched_rows)

    if matched_rows:
        with (CANONICAL_DIR / "compound_master_raw.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=list(matched_rows[0].keys()))
            writer.writeheader()
            writer.writerows(matched_rows)

    if unmatched:
        with (CANONICAL_DIR / "compound_unmatched.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=list(unmatched[0].keys()))
            writer.writeheader()
            writer.writerows(unmatched)

    with (CANONICAL_DIR / "pubchem_fetch_report.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(json.dumps(stats, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())