"""阶段 C1：基因主表（program/harmonize/build_gene_master.py）。

计划 §4.3：
- 稳定键 = NCBI Entrez Gene ID（26Q1 表达矩阵列名内嵌；Hallmark entrez GMT 直连）。
- HGNC（版本未知）只作对照列，不进 Core 依赖链；冲突时以 26Q1 冻结列为准。
- 突变/CNV/GMT 基因全部对照主表登记覆盖情况。

输出：
- data/processed/entities/gene_master.csv（gene_id 唯一）
- data/processed/entities/gene_mapping_audit.csv
- data/manifests/runs/C1/config_resolved.json + 运行记录
"""

from __future__ import annotations

import csv
import sys
import time
from collections import Counter

import pandas as pd

from program.common import paths as P
from program.common.runlog import (
    resolve_config,
    save_config_resolved,
    save_run_record,
    sha256_file,
)


def parse_expr_gene_columns() -> tuple[list[tuple[str, str]], list[str]]:
    """解析表达矩阵表头 gene 列 → [(symbol, entrez)]；返回解析失败列。"""
    with open(P.DEPMAP_EXPRESSION_CSV, encoding="utf-8") as f:
        header = next(csv.reader(f))
    gene_cols = header[6:]
    pairs, bad = [], []
    for c in gene_cols:
        if c.endswith(")") and " (" in c:
            sym, eid = c[:-1].rsplit(" (", 1)
            if eid.isdigit():
                pairs.append((sym, eid))
            else:
                bad.append(c)
        else:
            bad.append(c)
    return pairs, bad


def collect_matrix_entrez_spaces() -> dict[str, set[str]]:
    """CNV（gene (entrez) 列）与突变（EntrezGeneID 列）的 Entrez 空间。"""
    spaces: dict[str, set[str]] = {}
    with open(P.DEPMAP_CNV_CSV, encoding="utf-8") as f:
        header = next(csv.reader(f))
    cnv = set()
    for c in header:
        if c.endswith(")") and " (" in c:
            eid = c[:-1].rsplit(" (", 1)[1]
            if eid.isdigit():
                cnv.add(eid)
    spaces["cnv_wgs"] = cnv

    mut = pd.read_csv(P.DEPMAP_MUTATIONS_CSV, usecols=["EntrezGeneID"])
    e = mut["EntrezGeneID"].dropna()
    spaces["somatic_mutations"] = {str(int(float(v))) for v in e}
    return spaces


def read_hallmark_entrez() -> dict[str, set[str]]:
    gmt: dict[str, set[str]] = {}
    with open(P.HALLMARK_ENTREZ_GMT, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                gmt[parts[0]] = {p for p in parts[2:] if p.isdigit()}
    return gmt


def main() -> int:
    t0 = time.time()
    stage = "C1_build_gene_master"

    # ---- 1. 表达矩阵列解析（Core 稳定空间） ----
    pairs, bad_cols = parse_expr_gene_columns()
    sym_counts = Counter(s for s, _ in pairs)
    eid_counts = Counter(e for _, e in pairs)
    dup_symbols = {s: n for s, n in sym_counts.items() if n > 1}
    dup_entrez = {e: n for e, n in eid_counts.items() if n > 1}
    expr_eids = set(eid_counts)
    sym_by_eid = {e: s for s, e in pairs}
    print(f"[C1] 表达矩阵 gene 列: {len(pairs)}；解析失败: {len(bad_cols)}")
    print(f"[C1] 重复 symbol: {len(dup_symbols)}；重复 entrez: {len(dup_entrez)}")

    # ---- 2. HGNC 对照（仅对照列，版本未知不进 Core 依赖） ----
    hgnc = pd.read_csv(P.HGNC_GENE_CSV, dtype=str, low_memory=False)
    hgnc["entrez_norm"] = hgnc["entrez_id"].str.replace(r"\.0$", "", regex=True).str.strip()
    hgnc = hgnc[hgnc["entrez_norm"].notna() & (hgnc["entrez_norm"] != "")]
    hgnc_map = (
        hgnc.groupby("entrez_norm")
        .agg(
            hgnc_symbol=("symbol", lambda x: "|".join(sorted(set(x)))),
            hgnc_id=("hgnc_id", lambda x: "|".join(sorted(set(x)))),
            ensembl_gene_id=(
                "ensembl_gene_id",
                lambda x: "|".join(sorted(set(x.dropna()))),
            ),
        )
        .reset_index()
    )
    hgnc_by_entrez = dict(zip(hgnc_map["entrez_norm"], hgnc_map["hgnc_symbol"]))
    hgnc_eids = set(hgnc_map["entrez_norm"])

    # ---- 3. 主表 ----
    master = pd.DataFrame({"gene_id": sorted(expr_eids, key=int)})
    master["symbol_26q1"] = master["gene_id"].map(sym_by_eid)
    master = master.merge(hgnc_map, how="left", left_on="gene_id", right_on="entrez_norm")
    master = master.drop(columns=["entrez_norm"])
    master["source"] = "26q1_expression_column"
    master["status"] = "core"
    master["notes"] = ""

    # ---- 4. 审计表 ----
    audit_rows: list[dict] = []

    # 4a. HGNC 与 26Q1 symbol 冲突 → 26Q1 冻结列为准
    for _, row in master.iterrows():
        hs = row["hgnc_symbol"]
        if isinstance(hs, str) and hs != row["symbol_26q1"]:
            audit_rows.append(
                {
                    "audit_class": "symbol_conflict_hgnc_vs_26q1",
                    "gene_id": row["gene_id"],
                    "symbol_26q1": row["symbol_26q1"],
                    "hgnc_symbol": hs,
                    "resolution": "26Q1 frozen column wins (plan C1 rule 2)",
                }
            )

    # 4b. 26Q1 entrez 不在 HGNC（版本未知差异）
    for eid in sorted(expr_eids - hgnc_eids, key=int):
        audit_rows.append(
            {
                "audit_class": "entrez_absent_in_hgnc_unverified",
                "gene_id": eid,
                "symbol_26q1": sym_by_eid.get(eid, ""),
                "hgnc_symbol": None,
                "resolution": "HGNC version unknown; 26Q1 entry retained, HGNC columns blank",
            }
        )

    # 4c. 同 symbol/同 entrez 多列（26Q1 理论为零，须证实）
    for s in sorted(dup_symbols):
        cols = [f"{x} ({e})" for x, e in pairs if x == s]
        audit_rows.append(
            {
                "audit_class": "duplicate_symbol_columns_in_26q1",
                "gene_id": ";".join(sorted({e for x, e in pairs if x == s}, key=int)),
                "symbol_26q1": s,
                "hgnc_symbol": None,
                "resolution": f"columns kept separate: {cols}",
            }
        )
    for e in sorted(dup_entrez):
        cols = [f"{x} ({y})" for x, y in pairs if y == e]
        audit_rows.append(
            {
                "audit_class": "duplicate_entrez_columns_in_26q1",
                "gene_id": e,
                "symbol_26q1": ";".join(sorted({x for x, y in pairs if y == e})),
                "hgnc_symbol": None,
                "resolution": f"columns kept separate: {cols}",
            }
        )

    # ---- 5. CNV/突变/GMT 覆盖登记 ----
    spaces = collect_matrix_entrez_spaces()
    gmt = read_hallmark_entrez()

    def coverage(space: set[str], name: str) -> None:
        inside = space & expr_eids
        outside = space - expr_eids
        for eid in sorted(outside, key=int):
            audit_rows.append(
                {
                    "audit_class": f"not_measurable_in_expression_from_{name}",
                    "gene_id": eid,
                    "symbol_26q1": None,
                    "hgnc_symbol": hgnc_by_entrez.get(eid),
                    "resolution": "present in source space but not in 26Q1 expression matrix",
                }
            )
        print(f"[C1] {name}: {len(space)} genes；{len(inside)} 在表达矩阵内，{len(outside)} 不在")

    coverage(spaces["cnv_wgs"], "cnv_wgs")
    coverage(spaces["somatic_mutations"], "somatic_mutations")

    gmt_all = set().union(*gmt.values()) if gmt else set()
    coverage(gmt_all, "hallmark_entrez_gmt")
    master["in_hallmark"] = master["gene_id"].isin(gmt_all)

    audit = pd.DataFrame(
        audit_rows,
        columns=["audit_class", "gene_id", "symbol_26q1", "hgnc_symbol", "resolution"],
    )

    # ---- 6. 验收 ----
    assert master["gene_id"].is_unique, "gene_id 不唯一"
    assert len(master) == len(expr_eids), "主表行数与表达空间不一致"
    assert len(pairs) == 19215 and not bad_cols, "表达列未全部解析"

    out_master = P.ENTITIES_DIR / "gene_master.csv"
    out_audit = P.ENTITIES_DIR / "gene_mapping_audit.csv"
    master.to_csv(out_master, index=False)
    audit.to_csv(out_audit, index=False)

    base_cfg = {
        "stage": stage,
        "stable_key": "ncbi_entrez_gene_id",
        "hgnc_role": "control_columns_only_version_unknown",
        "inputs": {
            "expression": str(P.DEPMAP_EXPRESSION_CSV),
            "cnv": str(P.DEPMAP_CNV_CSV),
            "mutations": str(P.DEPMAP_MUTATIONS_CSV),
            "hgnc": str(P.HGNC_GENE_CSV),
            "hallmark_gmt": str(P.HALLMARK_ENTREZ_GMT),
        },
        "outputs": {"gene_master": str(out_master), "audit": str(out_audit)},
        "counts": {
            "expression_gene_columns": len(pairs),
            "master_rows": len(master),
            "audit_rows": len(audit),
            "audit_class_counts": audit["audit_class"].value_counts().to_dict(),
            "cnv_space": len(spaces["cnv_wgs"]),
            "mutation_entrez": len(spaces["somatic_mutations"]),
            "hallmark_sets": len(gmt),
            "hallmark_members_total": len(gmt_all),
        },
    }
    cfg = resolve_config(base_cfg)
    save_config_resolved(stage, cfg)
    inputs = {
        str(p): sha256_file(p)
        for p in [
            P.DEPMAP_EXPRESSION_CSV,
            P.DEPMAP_CNV_CSV,
            P.DEPMAP_MUTATIONS_CSV,
            P.HGNC_GENE_CSV,
            P.HALLMARK_ENTREZ_GMT,
        ]
    }
    save_run_record(
        stage,
        cfg,
        input_hashes=inputs,
        outputs={str(out_master): sha256_file(out_master), str(out_audit): sha256_file(out_audit)},
        status="succeeded",
        started_at=t0,
    )
    print(f"[C1] 完成：主表 {len(master)} 行 → {out_master}")
    print(f"[C1] 审计 {len(audit)} 行 → {out_audit}")
    return 0


if __name__ == "__main__":
    sys.exit(main())