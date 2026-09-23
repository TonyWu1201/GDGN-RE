"""GDGN-RE 阶段 C–F 共享路径常量（rebuild-2026）。

目录纪律（AGENTS.md / DECISION_LOG 2026-09-23）：
- 输入 raw 只读；既有审定表在 data/canonical/ 原位引用；
- 新派生产物写 data/processed/；splits 入 git；features 哈希链目录；
- data/holdout/ 仅容器级操作，不解析响应标签。
"""

from __future__ import annotations

from pathlib import Path

from program.acquire.paths import DATA_DIR, REPO_ROOT  # noqa: F401  (re-export)

# ---- 输入（只读） ----
RAW_DIR = DATA_DIR / "raw"
CANONICAL_DIR = DATA_DIR / "canonical"

GDSC2_RESPONSE_CSV = CANONICAL_DIR / "GDSC" / "release8.5" / "GDSC2_fitted_dose_response_27Oct23.csv"
GDSC_SCREENED_COMPOUNDS_CSV = RAW_DIR / "GDSC" / "release8.5" / "screened_compounds_rel_8.5.csv"
GDSC_SCREENED_COMPOUNDS_INCHI_CSV = (
    RAW_DIR / "GDSC" / "release8.5" / "screened_compounds_rel_8.5_inchi.csv"
)
GDSC2_RAW_ZIP = RAW_DIR / "GDSC" / "release8.5" / "GDSC2_public_raw_data_27Oct23.zip"

DEPMAP_26Q1_DIR = RAW_DIR / "DepMap" / "26Q1"
DEPMAP_MODEL_CSV = DEPMAP_26Q1_DIR / "Model.csv"
DEPMAP_PROFILES_CSV = DEPMAP_26Q1_DIR / "OmicsProfiles.csv"
DEPMAP_MODEL_CONDITION_CSV = DEPMAP_26Q1_DIR / "ModelCondition.csv"
DEPMAP_EXPRESSION_CSV = DEPMAP_26Q1_DIR / "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"
DEPMAP_CNV_CSV = DEPMAP_26Q1_DIR / "OmicsCNGeneWGS.csv"
DEPMAP_MUTATIONS_CSV = DEPMAP_26Q1_DIR / "OmicsSomaticMutations.csv"

CELLOSAURUS_TXT = RAW_DIR / "Cellosaurus" / "56.0" / "cellosaurus.txt"
CMP_MODEL_LIST_CSV = RAW_DIR / "CellModelPassports" / "20260921" / "model_list_20260921.csv"
HGNC_GENE_CSV = RAW_DIR / "HGNC" / "unverified" / "Gene.csv"
HALLMARK_ENTREZ_GMT = (
    RAW_DIR
    / "MSigDB"
    / "2026.1.Hs"
    / "msigdb_v2026.1.Hs_files_to_download_locally"
    / "msigdb_v2026.1.Hs_GMTs"
    / "h.all.v2026.1.Hs.entrez.gmt"
)
LEGACY_SEED_DIR = RAW_DIR / "legacy_structure_seed" / "unverified"

COMPOUND_ADJUDICATION_CSV = CANONICAL_DIR / "compound_structure_adjudication.csv"
COMPOUND_MASTER_RAW_CSV = CANONICAL_DIR / "compound_master_raw.csv"
COMPOUND_UNMATCHED_CSV = CANONICAL_DIR / "compound_unmatched.csv"
COMPOUND_OFFICIAL_KEY_AUDIT_CSV = CANONICAL_DIR / "compound_official_key_audit.csv"
CELL_MAPPING_CONFLICTS_CSV = CANONICAL_DIR / "cell_mapping_conflicts.csv"
PUBCHEM_SNAPSHOT_DIR = RAW_DIR / "PubChem" / "snapshot_20260923"

# ---- 输出（本管线新建） ----
PROCESSED_DIR = DATA_DIR / "processed"
ENTITIES_DIR = PROCESSED_DIR / "entities"
RESPONSE_DIR = PROCESSED_DIR / "response"
COHORTS_DIR = PROCESSED_DIR / "cohorts"
SPLITS_DIR = DATA_DIR / "splits"
FEATURES_DIR = DATA_DIR / "features"
RUNS_DIR = DATA_DIR / "manifests" / "runs"

GUIDANCE_DIR = REPO_ROOT / "guidance"

for _d in (ENTITIES_DIR, RESPONSE_DIR, COHORTS_DIR, SPLITS_DIR, FEATURE_DIR := FEATURES_DIR, RUNS_DIR):
    _d.mkdir(parents=True, exist_ok=True)