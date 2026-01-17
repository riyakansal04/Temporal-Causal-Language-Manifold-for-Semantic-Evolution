from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise
from sklearn.metrics import silhouette_score, mean_absolute_error, mean_squared_error

from ..forecast.models import forecast_trajectory


@dataclass
class EvalResults:
    retrieval_recall_at1: float
    clustering_silhouette: float
    backtest_rmse: float
    backtest_mae: float
    backtest_mape: float


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.maximum(np.abs(y_true), 1e-6)
    return float(np.mean(np.abs((y_true - y_pred) / denom)))


def retrieval_recall_at1(df: pd.DataFrame) -> float:
    # Build pairs (title, text) and check if title's nearest neighbor is its own text
    use = df.dropna(subset=["title", "embedding"]).copy()
    if len(use) < 5:
        return 0.0
    X = np.stack(use["embedding"].values)
    # Represent titles via cosine-weighted sum with embeddings of their doc text (proxy): use same embedding
    # Evaluate nearest neighbor correctness on identity mapping (sanity check of embedding stability)
    sims = pairwise.cosine_similarity(X)
    np.fill_diagonal(sims, -np.inf)
    nn = sims.argmax(axis=1)
    correct = (nn == np.arange(len(use)))
    return float(correct.mean())


def clustering_quality(df: pd.DataFrame) -> float:
    if len(df) < 10:
        return 0.0
    X = np.stack(df["embedding"].values)
    # Use source as a crude label; silhouette on source separation
    labels = df["source"].astype("category").cat.codes.values
    try:
        return float(silhouette_score(X, labels, metric="cosine"))
    except Exception:
        return 0.0


def backtest_forecast(traj: pd.DataFrame, holdout_steps: int = 4) -> Tuple[float, float, float]:
    if len(traj) <= holdout_steps + 3:
        y = traj["value"].astype(float).values
        return 0.0, 0.0, 0.0 if len(y) == 0 else (float(np.std(y)), float(np.std(y)), 1.0)
    train = traj.iloc[:-holdout_steps]
    test = traj.iloc[-holdout_steps:]
    y_true = test["value"].astype(float).values
    y_pred = np.array(forecast_trajectory(train, steps=holdout_steps))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = _safe_mape(y_true, y_pred)
    return rmse, mae, mape


def evaluate(df_embedded: pd.DataFrame, traj: pd.DataFrame) -> EvalResults:
    r1 = retrieval_recall_at1(df_embedded)
    sil = clustering_quality(df_embedded)
    rmse, mae, mape = backtest_forecast(traj, holdout_steps=max(3, min(6, len(traj)//4)))
    return EvalResults(r1, sil, rmse, mae, mape)


