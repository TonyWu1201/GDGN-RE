"""阶段 E2：Morgan 指纹 + Tanimoto 近邻表（program/features/build_fingerprints.py）。

§7.3：输入 = C3 standardized_isomeric_smiles；Morgan radius=2、2,048-bit 二值
（models.json 预注册）；RDKit 版本写入 config_resolved。
产出：指纹矩阵 + Tanimoto 近邻表（近邻基线与相似性分层共用）。

输出：data/features/<data_hash>/<split_hash>/<preprocess_hash>/fingerprints.npz
      + tanimoto_neighbors.csv + compound_order.json
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, __version__ as rdkit_version
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs import ExplicitBitVect

from program.common import paths as P
from program.common.runlog import (
    deterministic_json_bytes,
    resolve_config,
    save_config_resolved,
    save_run_record,
    sha256_file,
)
from program.features.cache import data_hash_from_files, feature_dir_for, hash_inputs, split_hash_from_fold_files

RADIUS = 2
N_BITS = 2048


def smiles_to_fp(smi: str, gen) -> ExplicitBitVect | None:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return gen.GetFingerprint(mol)


def main() -> int:
    t0 = time.time()
    stage = "E2_build_fingerprints"

    cohort_hash = (P.COHORTS_DIR / "cohort_hash.txt").read_text().strip()
    split_dir = P.SPLITS_DIR / cohort_hash
    comp = pd.read_csv(P.ENTITIES_DIR / "compound_master.csv", dtype=str)
    comp = comp.sort_values("compound_id").reset_index(drop=True)

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    fps: list[ExplicitBitVect] = []
    order: list[str] = []
    for _, r in comp.iterrows():
        fp = smiles_to_fp(r["standardized_isomeric_smiles"], gen)
        if fp is None:
            continue
        fps.append(fp)
        order.append(r["compound_id"])

    mat = np.zeros((len(fps), N_BITS), dtype=np.uint8)
    for i, fp in enumerate(fps):
        DataStructs.ConvertToNumpyArray(fp, mat[i])

    # Tanimoto 近邻表（全 Core 药物对；同 normalized_parent_id 记 neighbor_same_parent）
    n = len(fps)
    tan = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps)
        tan[i] = np.asarray(sims, dtype=np.float32)
    cid2parent = dict(zip(comp["compound_id"], comp["normalized_parent_id"]))

    neighbors = []
    for i, cid in enumerate(order):
        row = tan[i].copy()
        row[i] = -1  # 排除自身
        j = int(np.argmax(row))
        neighbors.append(
            {
                "compound_id": cid,
                "nearest_compound_id": order[j],
                "tanimoto_nearest": float(row[j]),
                "nearest_same_parent": cid2parent[order[j]] == cid2parent[cid],
            }
        )
    neighbor_df = pd.DataFrame(neighbors)

    # 哈希链
    data_hash = data_hash_from_files([P.COHORTS_DIR / "core_samples.parquet", P.ENTITIES_DIR / "compound_master.csv"])
    fold_files = sorted((split_dir / "lco").glob("fold_candidate_*.csv"))
    split_hash = split_hash_from_fold_files(fold_files)
    prep_hash = hash_inputs({"rdkit": rdkit_version, "morgan_radius": RADIUS, "morgan_bits": N_BITS, "chiral": True})

    out_dir = feature_dir_for(P.FEATURES_DIR, data_hash, split_hash, prep_hash)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "fingerprints.npz", data=mat, compound_ids=np.array(order))
    neighbor_df.to_csv(out_dir / "tanimoto_neighbors.csv", index=False)
    (out_dir / "compound_order.json").write_bytes(deterministic_json_bytes({"compound_order": order}))
    (out_dir / "config_resolved.json").write_bytes(deterministic_json_bytes({"rdkit": rdkit_version, "radius": RADIUS, "bits": N_BITS, "chiral": True}))

    cfg = resolve_config(
        {
            "stage": stage,
            "morgan": {"radius": RADIUS, "bits": N_BITS, "binary": True},
            "rdkit_version": rdkit_version,
            "data_hash": data_hash,
            "split_hash": split_hash,
            "preprocess_hash": prep_hash,
            "n_compounds": len(order),
            "n_unique_scaffold_groups": int(neighbor_df["nearest_same_parent"].sum()),
        }
    )
    save_config_resolved(stage, cfg)
    save_run_record(
        stage,
        cfg,
        input_hashes={str(P.ENTITIES_DIR / "compound_master.csv"): sha256_file(P.ENTITIES_DIR / "compound_master.csv")},
        outputs={str(out_dir / "fingerprints.npz"): sha256_file(out_dir / "fingerprints.npz")},
        status="succeeded",
        started_at=t0,
    )
    print(f"[E2] 指纹 {mat.shape} → {out_dir}")
    print(f"[E2] 最近邻 Tanimoto 均值 {neighbor_df['tanimoto_nearest'].mean():.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())