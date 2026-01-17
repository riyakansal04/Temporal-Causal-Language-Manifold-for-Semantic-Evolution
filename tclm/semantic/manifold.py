from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize


def _mean_pool(vectors: np.ndarray) -> np.ndarray:
    if len(vectors) == 0:
        return np.zeros((vectors.shape[1],), dtype=np.float32)
    return vectors.mean(axis=0)


def build_concept_trajectory(df: pd.DataFrame, concept: str) -> pd.DataFrame:
    """Compute time series of cosine similarity of bins to concept embedding proxy.

    Proxy concept vector: average embedding of docs whose text contains the term.
    """
    out = df.copy()
    out["contains"] = out["text"].str.lower().str.contains(concept.lower())
    if out["contains"].any():
        concept_vec = _mean_pool(np.stack(out.loc[out["contains"], "embedding"].values))
    else:
        # fallback to average of all
        concept_vec = _mean_pool(np.stack(out["embedding"].values))

    concept_vec = normalize(concept_vec.reshape(1, -1))[0]

    def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    series = []
    for time_bin, g in out.groupby("time_bin"):
        mat = np.stack(g["embedding"].values)
        center = _mean_pool(mat)
        center = normalize(center.reshape(1, -1))[0]
        series.append({"time_bin": time_bin, "value": cos_sim(center, concept_vec)})

    traj = pd.DataFrame(series).sort_values("time_bin").reset_index(drop=True)
    traj["source"] = "consensus-free"
    return traj


