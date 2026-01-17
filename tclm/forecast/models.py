from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from pmdarima import auto_arima

try:
    from prophet import Prophet
except Exception:  # pragma: no cover - optional dependency
    Prophet = None


def _forecast_prophet(traj: pd.DataFrame, steps: int) -> List[float]:
    """Forecast with Prophet. Falls back to last value if Prophet unavailable."""
    if Prophet is None:
        raise ImportError("prophet not installed")

    df = pd.DataFrame({"ds": traj["time_bin"], "y": traj["value"].astype(float)})

    # Infer frequency; Prophet needs a regular frequency
    try:
        freq = pd.infer_freq(df["ds"])
    except Exception:
        freq = None
    if freq is None:
        if len(df["ds"]) >= 2:
            # Use the median delta as an offset
            deltas = np.diff(df["ds"].values).astype("timedelta64[ns]")
            median_delta = pd.to_timedelta(np.median(deltas))
            freq = median_delta
        else:
            freq = "D"

    m = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=False)
    m.fit(df)
    future = m.make_future_dataframe(periods=steps, freq=freq, include_history=False)
    forecast = m.predict(future)
    return forecast["yhat"].astype(float).tolist()


def forecast_trajectory(traj: pd.DataFrame, steps: int = 6, model_type: str = "auto_arima") -> List[float]:
    """
    Forecast a trajectory using the chosen model.

    Args:
        traj: DataFrame with columns ["time_bin", "value"]
        steps: number of future periods to predict
        model_type: "auto_arima" (default) or "prophet"
    """
    y = traj["value"].astype(float).values
    if len(y) < 5:
        last = float(y[-1]) if len(y) else 0.0
        return [last] * steps

    model_type = (model_type or "auto_arima").lower()

    if model_type == "prophet":
        try:
            return _forecast_prophet(traj, steps)
        except Exception:
            # Fallback to auto_arima if Prophet not available or fails
            pass

    # Default: Auto-ARIMA
    model = auto_arima(y, seasonal=False, error_action="ignore", suppress_warnings=True)
    fcst = model.predict(n_periods=steps)
    return [float(v) for v in fcst]


