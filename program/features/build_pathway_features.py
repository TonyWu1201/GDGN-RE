"""阶段 E3：Hallmark 通路特征（program/features/build_pathway_features.py）。

§7.4：50 集 entrez GMT 直连 C1 主表；固定成员关系的均值聚合；
通路分数按训练折标准化（逐折，只拟合训练子集）；
分数标注为表达的派生表示，干预表达时同步重算（§16.5-6）。

输出：data/features/<data_hash>/<split_hash>/<preprocess_hash>/pathway/
      fold_*/pathway_train.npz + pathway_test.npz + pathway_state.npz + members_by_set.json
      + coverage_audit.json
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

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
from program.features.cache import (
    data_hash_from_files,
    feature_dir_for,
    hash_inputs,
    split_hash_from_fold_files,
)
from program.features.pathway_features import (
    compute_pathway_scores,
    pathway_coverage_mask,
    read_hallmark,
    set_overlap_jaccard,
)

MIN_MEMBERS = 10  # cohort.json


def main() -> int:
    t0 = time.time()
    stage = "E3_build_pathway_features"

    cohort_hash = (P.COHORTS_DIR / "cohort_hash.txt").read_text().strip()
    split_dir = P.SPLITS_DIR / cohort_hash
    core = pd.read_parquet(P.COHORTS_DIR / "core_samples.parquet")
    cell = pd.read_csv(P.ENTITIES_DIR / "cell_master.csv", dtype=str)
    cell2mid = dict(zip(cell["cell_id"], cell["profile_rna_model_id"].fillna("")))
    mid2cell = {v: k for k, v in cell2mid.items()}

    gene_master = pd.read_csv(P.ENTITIES_DIR / "gene_master.csv", dtype=str)
    measurable = set(gene_master["gene_id"])

    gmt = read_hallmark()
    mask, effective = pathway_coverage_mask(gmt, measurable, min_members=MIN_MEMBERS)
    valid_sets = {k: v for k, v in gmt.items() if mask[k]}

    # 表达矩阵全量读取一次（1,719 行 × 4,376 可测 Hallmark 成员列）
    import csv

    with open(P.DEPMAP_EXPRESSION_CSV, encoding="utf-8") as f:
        header = next(csv.reader(f))
    gene_cols = header[6:]
    gene_pairs = [c[:-1].rsplit(" (", 1) for c in gene_cols]
    member_positions: dict[str, list[int]] = {}
    for name, members in valid_sets.items():
        pos = [i for i, (s, e) in enumerate(gene_pairs) if e in set(members)]
        member_positions[name] = pos
    member_cols = sorted({p for ps in member_positions.values() for p in ps})

    mid_i = header.index("ModelID")
    rows_mid: list[str] = []
    rows_val: list[list[float]] = []
    with open(P.DEPMAP_EXPRESSION_CSV, encoding="utf-8") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            rows_val.append([float(x) if x not in ("", "NA") else np.nan for x in row[6:]])
            rows_mid.append(row[mid_i])
    expr_all = np.asarray(rows_val, dtype=float)
    # 构造每集合成员列索引（相对 member_cols）
    all_member_cols = sorted({p for ps in member_positions.values() for p in ps})
    col_newidx = {c: i for i, c in enumerate(all_member_cols)}
    members_rel = {name: [col_newidx[p] for p in ps] for name, ps in member_positions.items()}
    sub = expr_all[:, all_member_cols]

    # 全量通路分数（原始尺度），随后按折标准化
    names = list(members_rel)
    raw_scores = np.column_stack([sub[:, members_rel[n]].mean(axis=1) for n in names])

    # 哈希链
    data_hash = data_hash_from_files([P.COHORTS_DIR / "core_samples.parquet", P.ENTITIES_DIR / "gene_master.csv", P.HALLMARK_ENTREZ_GMT, P.DEPMAP_EXPRESSION_CSV])
    fold_files = sorted((split_dir / "lco").glob("fold_candidate_*.csv")) + [split_dir / "lco" / "fold_quick_screening_0.csv"]
    split_hash = split_hash_from_fold_files(fold_files)
    prep_hash = hash_inputs({"min_members": MIN_MEMBERS, "aggregation": "mean_of_scaled_members", "n_sets": len(names), "implementation_sha256": sha256_file(Path(__file__))})

    out_dir = feature_dir_for(P.FEATURES_DIR, data_hash, split_hash, prep_hash) / "pathway"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 逐折：成员基因按训练折标准化 → 均值聚合 → 分数按训练折标准化
    output_hashes = {}
    for fold_file in fold_files:
        fold = pd.read_csv(fold_file, dtype=str)
        train_ids = set(fold[fold["role"] == "train"]["sample_id"])
        test_ids = set(fold[fold["role"] == "test"]["sample_id"])
        train_cells = sorted({cell2mid[c] for c in core[core["sample_id"].isin(train_ids)]["cell_id"] if cell2mid[c]})
        test_cells = sorted({cell2mid[c] for c in core[core["sample_id"].isin(test_ids)]["cell_id"] if cell2mid[c]})
        mid2row = {m: i for i, m in enumerate(rows_mid)}
        tr_rows = [mid2row[m] for m in train_cells if m in mid2row]
        te_rows = [mid2row[m] for m in test_cells if m in mid2row]

        # 成员基因按训练折标准化（唯一实体统计）
        tr_expr = sub[tr_rows]
        gene_mu = np.nanmean(tr_expr, axis=0)
        gene_sd = np.nanstd(tr_expr, axis=0, ddof=1)
        gene_sd = np.where(np.isfinite(gene_sd) & (gene_sd > 1e-8), gene_sd, 1.0)
        tr_scaled = (tr_expr - gene_mu) / gene_sd
        te_scaled = (sub[te_rows] - gene_mu) / gene_sd

        # 通路分数（标准化成员均值）
        tr_scores = np.column_stack([tr_scaled[:, members_rel[n]].mean(axis=1) for n in names])
        te_scores = np.column_stack([te_scaled[:, members_rel[n]].mean(axis=1) for n in names])
        # 分数再按训练折标准化
        s_mu = tr_scores.mean(axis=0)
        s_sd = tr_scores.std(axis=0, ddof=1)
        s_sd = np.where(np.isfinite(s_sd) & (s_sd > 1e-8), s_sd, 1.0)
        tr_scores = (tr_scores - s_mu) / s_sd
        te_scores = (te_scores - s_mu) / s_sd

        fold_out = out_dir / Path(fold_file).stem
        fold_out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(fold_out / "pathway_train.npz", data=tr_scores, rows=np.array([m for m in train_cells if m in mid2row]), sets=np.array(names))
        np.savez_compressed(fold_out / "pathway_test.npz", data=te_scores, rows=np.array([m for m in test_cells if m in mid2row]), sets=np.array(names))
        np.savez_compressed(
            fold_out / "pathway_state.npz",
            gene_columns=np.array(all_member_cols), gene_mean=gene_mu, gene_std=gene_sd,
            score_mean=s_mu, score_std=s_sd, set_names=np.array(names),
        )
        (fold_out / "members_by_set.json").write_bytes(deterministic_json_bytes(members_rel))
        for name in ("pathway_train.npz", "pathway_test.npz", "pathway_state.npz", "members_by_set.json"):
            output_hashes[str(fold_out / name)] = sha256_file(fold_out / name)

    # 覆盖审计 + Jaccard
    coverage = {
        "n_sets_total": len(gmt),
        "n_sets_valid": len(valid_sets),
        "invalid_sets": {k: effective[k] for k in gmt if not mask[k]},
        "effective_members": effective,
        "min_members": MIN_MEMBERS,
        "note": "scores are derived representation of expression, not independent omics (§7.4)",
    }
    jac = set_overlap_jaccard(valid_sets)
    jac.to_csv(out_dir / "set_overlap_jaccard.csv")

    cfg = resolve_config(
        {
            "stage": stage,
            "collection": "MSigDB Human Hallmark 50 (entrez GMT v2026.1.Hs)",
            "min_effective_members": MIN_MEMBERS,
            "data_hash": data_hash,
            "split_hash": split_hash,
            "preprocess_hash": prep_hash,
            "n_sets_valid": len(valid_sets),
        }
    )
    save_config_resolved(stage, cfg)
    (out_dir / "coverage_audit.json").write_bytes(deterministic_json_bytes(coverage))
    output_hashes[str(out_dir / "coverage_audit.json")] = sha256_file(out_dir / "coverage_audit.json")
    save_run_record(
        stage,
        cfg,
        input_hashes={str(p): sha256_file(p) for p in (P.HALLMARK_ENTREZ_GMT, P.DEPMAP_EXPRESSION_CSV)},
        outputs=output_hashes,
        status="succeeded",
        started_at=t0,
    )
    print(f"[E3] {len(valid_sets)}/{len(gmt)} 集有效（min {MIN_MEMBERS} 成员）→ {out_dir}")
    return 0


def member_positions_list(member_positions):
    return sorted({p for ps in member_positions.values() for p in ps})


if __name__ == "__main__":
    sys.exit(main())
