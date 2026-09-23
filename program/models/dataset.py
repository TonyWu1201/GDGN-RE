"""Development-only sample loader and inner-train-only feature preparation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from program.common import paths as P
from program.common.runlog import sha256_file
from program.features.build_expression_features import load_expression_subset
from program.features.expression_preprocess import fit_preprocessing, transform_with_state


@dataclass
class PreparedData:
    train: pd.DataFrame
    validation: pd.DataFrame
    expression: np.ndarray
    fingerprints: np.ndarray
    model_ids: list[str]
    compound_ids: list[str]
    state: dict
    gene_order: list[list[str]]

    def arrays(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mid_index = {mid: i for i, mid in enumerate(self.model_ids)}
        drug_index = {drug: i for i, drug in enumerate(self.compound_ids)}
        cells = frame["profile_rna_model_id"].map(mid_index)
        drugs = frame["compound_id"].map(drug_index)
        if cells.isna().any() or drugs.isna().any():
            raise ValueError("failed feature row join")
        return cells.to_numpy(dtype=np.int32), drugs.to_numpy(dtype=np.int32), frame["y"].to_numpy(dtype=np.float32)

    def pair_matrix(self, frame: pd.DataFrame, side: str = "both") -> np.ndarray:
        ci, di, _ = self.arrays(frame)
        if side == "expression":
            return self.expression[ci]
        if side == "structure":
            return self.fingerprints[di]
        if side == "both":
            return np.concatenate((self.expression[ci], self.fingerprints[di]), axis=1)
        raise ValueError(f"unknown feature side: {side}")


class ProjectData:
    """Never opens sealed response files; reads only IDs named by an active dev split."""

    def __init__(self, protocol: str = "lco", fold: str | None = None):
        if protocol not in {"lco", "lpo"}:
            raise ValueError("M1–M3 permit only lco/lpo development protocols")
        self.protocol = protocol
        self.fold = fold or ("fold_quick_screening_0" if protocol == "lco" else "fold_0")
        self.cohort_hash = (P.COHORTS_DIR / "cohort_hash.txt").read_text(encoding="utf-8").strip()
        self.split_path = P.SPLITS_DIR / self.cohort_hash / protocol / f"{self.fold}.csv"
        if not self.split_path.is_file():
            raise FileNotFoundError(self.split_path)
        self.split_hash = sha256_file(self.split_path)
        self.cell_master = pd.read_csv(P.ENTITIES_DIR / "cell_master.csv", dtype=str)
        self.compound_master = pd.read_csv(P.ENTITIES_DIR / "compound_master.csv", dtype=str)
        if not self.cell_master["cell_id"].is_unique or not self.compound_master["compound_id"].is_unique:
            raise ValueError("duplicate entity master key")
        self._load_fingerprints()
        self.frame = self._load_samples()
        self._raw_expr: pd.DataFrame | None = None
        self._gene_pairs: list[list[str]] | None = None

    def _load_fingerprints(self) -> None:
        cfg_path = P.RUNS_DIR / "E2_build_fingerprints" / "config_resolved.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        directory = P.FEATURES_DIR / cfg["data_hash"] / cfg["split_hash"] / cfg["preprocess_hash"]
        order = json.loads((directory / "compound_order.json").read_text(encoding="utf-8"))["compound_order"]
        with np.load(directory / "fingerprints.npz", allow_pickle=False) as npz:
            self.fingerprints = npz["data"].astype(np.float32)
            self.compound_ids = npz["compound_ids"].astype(str).tolist()
        if self.compound_ids != order or len(set(order)) != len(order):
            raise ValueError("fingerprint compound order mismatch")
        if not np.isfinite(self.fingerprints).all() or set(np.unique(self.fingerprints)) - {0.0, 1.0}:
            raise ValueError("invalid binary fingerprints")
        self.fingerprint_hash = sha256_file(directory / "fingerprints.npz")
        self.fingerprint_config = cfg

    def _load_samples(self) -> pd.DataFrame:
        split = pd.read_csv(self.split_path, dtype=str)
        required = {"sample_id", "cell_id", "compound_id", "role"}
        if not required <= set(split) or not split["sample_id"].is_unique:
            raise ValueError("malformed or duplicate split sample ID")
        if set(split["role"]) != {"train", "test"}:
            raise ValueError("split must contain train and test roles")
        cell_assign = pd.read_csv(P.SPLITS_DIR / self.cohort_hash / "holdout" / "cell_group_assignment.csv", dtype=str)
        drug_assign = pd.read_csv(P.SPLITS_DIR / self.cohort_hash / "holdout" / "drug_group_assignment.csv", dtype=str)
        cdev = set(cell_assign.loc[cell_assign["region"] == "dev", "cell_id"])
        ddev = set(drug_assign.loc[drug_assign["region"] == "dev", "compound_id"])
        if not set(split["cell_id"]) <= cdev or not set(split["compound_id"]) <= ddev:
            raise ValueError("development split references sealed cell or drug")
        ids = split["sample_id"].tolist()
        columns = ["sample_id", "cell_id", "compound_id", "y", "any_extrapolation", "oncotree_lineage", "normalized_parent_id", "scaffold_id"]
        core = pd.read_parquet(P.COHORTS_DIR / "core_samples.parquet", columns=columns, filters=[("sample_id", "in", ids)])
        if len(core) != len(split) or not core["sample_id"].is_unique:
            raise ValueError("split/core sample coverage or uniqueness failure")
        frame = split[["sample_id", "cell_id", "compound_id", "role"]].merge(
            core, on="sample_id", validate="one_to_one", suffixes=("_split", ""), sort=False
        )
        if not np.array_equal(frame["cell_id_split"], frame["cell_id"]) or not np.array_equal(frame["compound_id_split"], frame["compound_id"]):
            raise ValueError("split and core entity IDs differ")
        frame = frame.drop(columns=["cell_id_split", "compound_id_split"])
        frame = frame.merge(self.cell_master[["cell_id", "profile_rna_model_id"]], on="cell_id", validate="many_to_one", sort=False)
        frame = frame.merge(cell_assign[["cell_id", "group_id"]].rename(columns={"group_id": "cell_group"}), on="cell_id", validate="many_to_one", sort=False)
        frame = frame.merge(drug_assign[["compound_id", "chemical_group_id"]].rename(columns={"chemical_group_id": "chemical_group"}), on="compound_id", validate="many_to_one", sort=False)
        if frame[["profile_rna_model_id", "cell_group", "chemical_group"]].isna().any().any():
            raise ValueError("missing entity-to-feature or cluster mapping")
        if not set(frame["compound_id"]) <= set(self.compound_ids):
            raise ValueError("fingerprints missing a split compound")
        if not np.isfinite(frame["y"].to_numpy(dtype=float)).all():
            raise ValueError("non-finite development label")
        if self.protocol == "lco":
            a = set(frame.loc[frame["role"] == "train", "cell_group"])
            b = set(frame.loc[frame["role"] == "test", "cell_group"])
            if a & b:
                raise ValueError("LCO cell group leaks across roles")
        else:
            pair = frame["cell_id"] + "|" + frame["compound_id"]
            if set(pair[frame["role"] == "train"]) & set(pair[frame["role"] == "test"]):
                raise ValueError("LPO pair leaks across roles")
        return frame

    def subset(self, sample_ids: list[str]) -> pd.DataFrame:
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("duplicate requested sample ID")
        chosen = self.frame.set_index("sample_id").reindex(sample_ids)
        if chosen["role"].isna().any():
            raise ValueError("requested sample ID absent from active split")
        return chosen.reset_index()

    def raw_expression(self) -> tuple[pd.DataFrame, list[list[str]]]:
        if self._raw_expr is None:
            wanted = sorted(set(self.frame["profile_rna_model_id"]))
            expr, pairs = load_expression_subset(wanted)
            if not expr.index.is_unique or set(expr.index) != set(wanted):
                raise ValueError("expression ModelID join incomplete or duplicated")
            self._raw_expr = expr
            self._gene_pairs = [list(pair) for pair in pairs]
        return self._raw_expr, self._gene_pairs  # type: ignore[return-value]

    def prepare(self, train: pd.DataFrame, validation: pd.DataFrame, hvg_n: int = 1000) -> PreparedData:
        if train.empty or validation.empty or set(train["sample_id"]) & set(validation["sample_id"]):
            raise ValueError("train/validation must be nonempty and disjoint")
        if self.protocol == "lco" and set(train["cell_group"]) & set(validation["cell_group"]):
            raise ValueError("inner LCO cell group leakage")
        expr, pairs = self.raw_expression()
        train_mids = sorted(set(train["profile_rna_model_id"]))
        all_mids = sorted(set(train_mids) | set(validation["profile_rna_model_id"]))
        state = fit_preprocessing(expr.loc[train_mids].to_numpy(), hvg_n=hvg_n)
        transformed = transform_with_state(expr.loc[all_mids].to_numpy(), state).astype(np.float32)
        if not np.isfinite(transformed).all():
            raise ValueError("non-finite transformed expression")
        genes = [pairs[i] for i in state["selected_columns"]]
        return PreparedData(train, validation, transformed, self.fingerprints, all_mids,
                            self.compound_ids, state, genes)


def inner_split(frame: pd.DataFrame, protocol: str, seed: int = 20260934, fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty or not 0 < fraction < 1:
        raise ValueError("invalid inner split")
    rng = np.random.default_rng(seed)
    if protocol == "lco":
        if "cell_group" not in frame:
            raise ValueError("LCO inner split needs cell_group")
        groups = np.sort(frame["cell_group"].unique())
        rng.shuffle(groups)
        selected = set(groups[:max(1, round(len(groups) * fraction))])
        mask = frame["cell_group"].isin(selected)
    elif protocol == "lpo":
        pairs = np.sort((frame["cell_id"] + "|" + frame["compound_id"]).unique())
        rng.shuffle(pairs)
        selected = set(pairs[:max(1, round(len(pairs) * fraction))])
        mask = (frame["cell_id"] + "|" + frame["compound_id"]).isin(selected)
    else:
        raise ValueError(protocol)
    if mask.all() or not mask.any():
        raise ValueError("inner split has empty side")
    return frame.loc[~mask].copy(), frame.loc[mask].copy()
