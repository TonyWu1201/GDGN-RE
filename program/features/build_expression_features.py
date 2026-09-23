"""阶段 E1：训练折表达特征（program/features/build_expression_features.py）。

§7.1：逐折流程——确定实体 → 划分（读取 D3 折文件）→ 训练子集拟合
（去高缺失/无变异 → 中位数插补 → 唯一实体标准化统计 → HVG 1,000）
→ 原样变换验证/测试 → 保存 gene_order + preprocessing_state。

量纲：26Q1 表达 = log2(TPM+1)（C0 证实），不再取 log；缺失与真实零分别编码
（NaN=缺失，0=真实零）。

输出：data/features/<data_hash>/<split_hash>/<preprocess_hash>/lco/fold_*/
      expression_train.npz + expression_test.npz + gene_order.json + preprocessing_state.json
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
from program.features.expression_preprocess import fit_preprocessing, transform_with_state

HVG_MAIN = 1000  # models.json 预注册；sensitivity 500/2000 另行注册


def load_expression_subset(model_ids: list[str]) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """读取表达矩阵并按 ModelID 子集化（默认 profile 已在 C2 锚定）。"""
    import csv

    with open(P.DEPMAP_EXPRESSION_CSV, encoding="utf-8") as f:
        header = next(csv.reader(f))
    gene_cols = header[6:]
    gene_pairs = [c[:-1].rsplit(" (", 1) for c in gene_cols]
    mid_i = header.index("ModelID")
    want = set(model_ids)
    rows: list[tuple[str, list[float]]] = []
    with open(P.DEPMAP_EXPRESSION_CSV, encoding="utf-8") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if row[mid_i] in want:
                rows.append((row[mid_i], [float(x) if x not in ("", "NA") else np.nan for x in row[6:]]))
    df = pd.DataFrame({mid: vals for mid, vals in rows}).T
    df.index.name = "ModelID"
    return df, gene_pairs


def main() -> int:
    t0 = time.time()
    stage = "E1_build_expression_features"

    cohort_hash = (P.COHORTS_DIR / "cohort_hash.txt").read_text().strip()
    split_dir = P.SPLITS_DIR / cohort_hash
    core = pd.read_parquet(P.COHORTS_DIR / "core_samples.parquet")
    cell = pd.read_csv(P.ENTITIES_DIR / "cell_master.csv", dtype=str)
    cell2mid = dict(zip(cell["cell_id"], cell["profile_rna_model_id"].fillna("")))

    # data/split/preprocess 哈希链
    data_hash = data_hash_from_files(
        [P.COHORTS_DIR / "core_samples.parquet", P.DEPMAP_EXPRESSION_CSV, P.ENTITIES_DIR / "gene_master.csv"]
    )
    fold_files = sorted((split_dir / "lco").glob("fold_candidate_*.csv")) + [split_dir / "lco" / "fold_quick_screening_0.csv"]
    split_hash = split_hash_from_fold_files(fold_files)
    prep_hash = hash_inputs({"hvg": HVG_MAIN, "impute": "median", "scale": "entity_z", "min_variance": 1e-8, "fit_sha256": sha256_file(Path(__file__).with_name("expression_preprocess.py")), "build_sha256": sha256_file(Path(__file__))})

    out_dir = feature_dir_for(P.FEATURES_DIR, data_hash, split_hash, prep_hash) / "lco"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_hashes = {}

    for fold_file in fold_files:
        fold = pd.read_csv(fold_file, dtype=str)
        train_samples = fold[fold["role"] == "train"]["sample_id"]
        test_samples = fold[fold["role"] == "test"]["sample_id"]
        train_cells = sorted({cell2mid[c] for c in core[core["sample_id"].isin(train_samples)]["cell_id"] if cell2mid[c]})
        test_cells = sorted({cell2mid[c] for c in core[core["sample_id"].isin(test_samples)]["cell_id"] if cell2mid[c]})

        # 读取表达（一次全量太慢 → 逐折只读需要的行）
        train_expr, gene_pairs = load_expression_subset(train_cells)
        test_expr, _ = load_expression_subset(test_cells)
        # 列对齐（同一 gene 顺序）
        test_expr = test_expr[train_expr.columns]

        state = fit_preprocessing(train_expr.to_numpy(), impute="median", scale=True, hvg_n=HVG_MAIN, entity_ids=None)
        train_out = transform_with_state(train_expr.to_numpy(), state)
        test_out = transform_with_state(test_expr.to_numpy(), state)
        train_missing = (~np.isfinite(train_expr.to_numpy()[:, state["selected_columns"]])).astype(np.uint8)
        test_missing = (~np.isfinite(test_expr.to_numpy()[:, state["selected_columns"]])).astype(np.uint8)

        fold_out = out_dir / Path(fold_file).stem
        fold_out.mkdir(parents=True, exist_ok=True)
        column_ids = np.asarray(train_expr.columns[state["selected_columns"]], dtype=str)
        np.savez_compressed(fold_out / "expression_train.npz", data=train_out, missing_mask=train_missing, rows=np.asarray(train_expr.index, dtype=str), cols=column_ids)
        np.savez_compressed(fold_out / "expression_test.npz", data=test_out, missing_mask=test_missing, rows=np.asarray(test_expr.index, dtype=str), cols=column_ids)
        (fold_out / "gene_order.json").write_bytes(
            deterministic_json_bytes({"selected_columns": [str(c) for c in state["selected_columns"]], "gene_pairs_in_order": [list(g) for g in [gene_pairs[c] for c in state["selected_columns"]]]})
        )
        (fold_out / "preprocessing_state.json").write_bytes(deterministic_json_bytes(state))
        for name in ("expression_train.npz", "expression_test.npz", "gene_order.json", "preprocessing_state.json"):
            output_hashes[str(fold_out / name)] = sha256_file(fold_out / name)

    cfg = resolve_config(
        {
            "stage": stage,
            "hvg_main": HVG_MAIN,
            "unit": "log2(TPM+1); no re-log (C0 dictionary note)",
            "data_hash": data_hash,
            "split_hash": split_hash,
            "preprocess_hash": prep_hash,
            "output_dir": str(out_dir),
        }
    )
    save_config_resolved(stage, cfg)
    save_run_record(
        stage,
        cfg,
        input_hashes={str(f): sha256_file(f) for f in fold_files},
        outputs=output_hashes,
        status="succeeded",
        started_at=t0,
    )
    print(f"[E1] 特征缓存 → {out_dir}")
    print(f"[E1] 哈希链: data={data_hash} split={split_hash} prep={prep_hash}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
