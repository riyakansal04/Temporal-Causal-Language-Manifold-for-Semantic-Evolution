from __future__ import annotations

import numpy as np
import pandas as pd


def fuse_consensus(traj: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Simple consensus: average per-source bin centers, then align to concept trajectory scale.

    This is a placeholder for a Kalman fusion; here we compute per-source bin centroids,
    average them per bin, and z-score to align with the concept trajectory.
    """
    centers = (
        df.groupby(["time_bin", "source"])  # average embeddings per source/bin
        ["embedding"].apply(lambda v: np.stack(v.values).mean(axis=0))
        .reset_index()
    )
    # scalar proxy per bin: norm of centroid
    centers["scalar"] = centers["embedding"].map(lambda x: float(np.linalg.norm(x)))
    fused = centers.groupby("time_bin")["scalar"].mean().reset_index()
    fused = fused.rename(columns={"scalar": "consensus"})

    out = traj.merge(fused, on="time_bin", how="left")
    for col in ("value", "consensus"):
        x = out[col].values
        x = (x - np.nanmean(x)) / (np.nanstd(x) + 1e-9)
        out[col] = x
    return out


