"""Leakage-safe split: rows sharing a claim or a substantive evidence passage stay in one partition."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

MIN_EVIDENCE_CHARS = 40  # paper: shorter strings may recur across partitions by design


def components(claims: pd.Series, evidence: pd.Series, min_chars: int = MIN_EVIDENCE_CHARS) -> np.ndarray:
    """Component id per row; rows are linked by equal claim or equal evidence of >= min_chars."""
    n = len(claims)
    ev = evidence.fillna("").astype(str)
    keys = [claims.astype(str).to_numpy(), np.where(ev.str.len() >= min_chars, ev, None)]
    src, dst = [], []
    for key in keys:
        first: dict[str, int] = {}
        for i, k in enumerate(key):
            if k is None:
                continue
            j = first.setdefault(k, i)
            if j != i:
                src.append(i)
                dst.append(j)
    graph = coo_matrix((np.ones(len(src)), (src, dst)), shape=(n, n))
    return connected_components(graph, directed=False)[1]


def assign(comp: np.ndarray, sizes: dict[str, int], seed: int = 42) -> np.ndarray:
    """Greedily fill partitions (in dict order) with whole shuffled components; leftovers get ''."""
    ids, counts = np.unique(comp, return_counts=True)
    order = np.random.default_rng(seed).permutation(len(ids))
    out = {}
    filled = dict.fromkeys(sizes, 0)
    for c, size in zip(ids[order], counts[order]):
        for part, target in sizes.items():
            if filled[part] + size <= target:
                out[c] = part
                filled[part] += size
                break
        else:
            out[c] = ""
    return np.array([out[c] for c in comp])
