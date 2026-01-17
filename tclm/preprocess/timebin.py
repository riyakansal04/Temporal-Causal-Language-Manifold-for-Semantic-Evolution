from __future__ import annotations

import pandas as pd


def timebin_corpus(df: pd.DataFrame, freq: str = "M") -> pd.DataFrame:
    df = df.copy()
    df["time_bin"] = df["published"].dt.to_period(freq).dt.to_timestamp()
    return df


