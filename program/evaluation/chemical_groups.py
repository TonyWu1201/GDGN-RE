"""Chemical identity and scaffold groups used by sealed and development splits."""

from __future__ import annotations

import pandas as pd


def chemical_clusters(compounds: pd.DataFrame) -> dict[str, str]:
    """Join compounds sharing either a standardized parent or a scaffold."""
    required = {"compound_id", "normalized_parent_id", "scaffold_id"}
    if not required <= set(compounds.columns):
        raise ValueError(f"compound master missing {required - set(compounds.columns)}")
    if compounds[list(required)].isna().any().any():
        raise ValueError("chemical grouping keys must be complete")
    if not compounds["compound_id"].is_unique:
        raise ValueError("compound_id must be unique")

    ids = sorted(compounds["compound_id"].astype(str))
    parent = {cid: cid for cid in ids}

    def root(cid: str) -> str:
        while parent[cid] != cid:
            parent[cid] = parent[parent[cid]]
            cid = parent[cid]
        return cid

    def join(left: str, right: str) -> None:
        a, b = root(left), root(right)
        parent[max(a, b)] = min(a, b)

    for key in ("normalized_parent_id", "scaffold_id"):
        for _, group in compounds.groupby(key, sort=True):
            members = sorted(group["compound_id"].astype(str))
            for cid in members[1:]:
                join(members[0], cid)

    return {cid: f"CHEM_{root(cid)}" for cid in ids}
