"""Train-fold-only Hallmark and matched-control features for A2."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from program.features.pathway_features import read_hallmark
from program.models.dataset import PreparedData, ProjectData


@dataclass
class PathwayFeatures:
    values: np.ndarray
    names: list[str]
    state: dict


def rewire_memberships(members: list[list[int]], seed: int, swaps: int | None = None) -> list[list[int]]:
    """Bipartite double-edge swaps preserve every set size and gene degree."""
    groups = [set(item) for item in members]
    edges = [(group, gene) for group, genes in enumerate(groups) for gene in genes]
    if not edges:
        raise ValueError("empty pathway membership")
    rng = np.random.default_rng(seed)
    target = swaps if swaps is not None else 20 * len(edges)
    completed = 0
    for _ in range(target * 10):
        if completed >= target:
            break
        a, b = rng.integers(0, len(edges), size=2)
        p, g = edges[a]
        q, h = edges[b]
        if p == q or g == h or h in groups[p] or g in groups[q]:
            continue
        groups[p].remove(g)
        groups[q].remove(h)
        groups[p].add(h)
        groups[q].add(g)
        edges[a], edges[b] = (p, h), (q, g)
        completed += 1
    if completed == 0:
        raise ValueError("random pathway control could not change membership")
    result = [sorted(group) for group in groups]
    if result == [sorted(set(group)) for group in members]:
        raise ValueError("random pathway control has unchanged membership")
    return result


def transform_pathways(raw_expression: np.ndarray, state: dict) -> np.ndarray:
    """Recompute derived scores after any expression intervention."""
    raw = np.asarray(raw_expression, dtype=float)
    union = np.asarray(state["raw_columns"], dtype=int)
    values = raw[:, union]
    mu = np.asarray(state["gene_mean"], dtype=float)
    sd = np.asarray(state["gene_std"], dtype=float)
    scaled = np.where(np.isfinite(values), values, mu)
    scaled = (scaled - mu) / sd
    if state["mode"] == "projection":
        scores = scaled @ np.asarray(state["projection"], dtype=float)
    else:
        scores = np.column_stack([scaled[:, positions].mean(axis=1)
                                  for positions in state["members_relative"]])
    scores = (scores - np.asarray(state["score_mean"])) / np.asarray(state["score_std"])
    if not np.isfinite(scores).all():
        raise ValueError("non-finite pathway features")
    return scores.astype(np.float32)


def prepare_pathways(
    data: ProjectData, prepared: PreparedData, train: pd.DataFrame,
    mode: str = "hallmark", seed: int = 42, min_members: int = 10,
) -> PathwayFeatures:
    if mode not in {"hallmark", "random", "projection"}:
        raise ValueError(f"unknown pathway mode: {mode}")
    raw, gene_pairs = data.raw_expression()
    if set(prepared.model_ids) - set(raw.index):
        raise ValueError("pathway expression rows missing")
    all_mids = prepared.model_ids
    train_mids = sorted(set(train["profile_rna_model_id"]))
    if not set(train_mids) <= set(all_mids):
        raise ValueError("train model IDs missing from prepared features")
    entrez = [str(pair[1]) for pair in gene_pairs]
    available = set(entrez)
    gmt = read_hallmark()
    names, members = [], []
    for name, genes in gmt.items():
        gene_set = set(genes)
        matches = [i for i, gene in enumerate(entrez) if gene in gene_set]
        if len(matches) >= min_members:
            names.append(name)
            members.append(matches)
    if not names:
        raise ValueError("no Hallmark sets meet fixed coverage rule")
    union = sorted(set().union(*map(set, members)))
    position = {column: i for i, column in enumerate(union)}
    relative = [[position[column] for column in group] for group in members]
    original_relative = [set(group) for group in relative]
    if mode == "random":
        relative = rewire_memberships(relative, seed)
    values = raw.loc[all_mids].to_numpy(dtype=float)[:, union]
    train_rows = np.asarray([all_mids.index(mid) for mid in train_mids], dtype=int)
    training = values[train_rows]
    mu = np.nanmean(training, axis=0)
    sd = np.nanstd(training, axis=0, ddof=1)
    if not np.isfinite(mu).all():
        raise ValueError("training pathway gene entirely missing")
    sd = np.where(np.isfinite(sd) & (sd > 1e-8), sd, 1.0)
    scaled = (np.where(np.isfinite(values), values, mu) - mu) / sd
    state = {
        "mode": mode, "seed": seed, "min_members": min_members,
        "set_names": names, "raw_columns": union,
        "members_relative": relative if mode != "projection" else [],
        "gene_mean": mu.tolist(), "gene_std": sd.tolist(),
        "train_model_ids": train_mids,
        "gene_entrez_ids": [entrez[column] for column in union],
        "measurable_entrez_count": len(available),
    }
    if mode == "random":
        shared = sum(len(a & set(b)) for a, b in zip(original_relative, relative))
        state["membership_edge_overlap_fraction"] = shared / sum(map(len, original_relative))
    if mode == "projection":
        generator = np.random.default_rng(seed)
        projection = generator.standard_normal((len(union), len(names))) / np.sqrt(len(union))
        state["projection"] = projection.tolist()
        scores = scaled @ projection
    else:
        scores = np.column_stack([scaled[:, group].mean(axis=1) for group in relative])
    score_mu = scores[train_rows].mean(axis=0)
    score_sd = scores[train_rows].std(axis=0, ddof=1)
    score_sd = np.where(np.isfinite(score_sd) & (score_sd > 1e-8), score_sd, 1.0)
    state["score_mean"], state["score_std"] = score_mu.tolist(), score_sd.tolist()
    transformed = transform_pathways(raw.loc[all_mids].to_numpy(dtype=float), state)
    return PathwayFeatures(transformed, names, state)
