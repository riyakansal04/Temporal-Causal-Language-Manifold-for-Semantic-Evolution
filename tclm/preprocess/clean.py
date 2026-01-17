from __future__ import annotations

import re
from typing import Iterable, List

import pandas as pd


_WS = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    s = s or ""
    s = s.replace("\xa0", " ")
    s = re.sub(r"http[s]?://\S+", "", s)
    s = re.sub(r"[^\x00-\x7F]+", " ", s)
    s = _WS.sub(" ", s).strip()
    return s


def clean_corpus(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    # Filter out empty dataframes before concatenating
    non_empty_frames = [f for f in frames if not f.empty]
    if not non_empty_frames:
        return pd.DataFrame()
    
    df = pd.concat(non_empty_frames, ignore_index=True)
    
    # Check if required columns exist
    if df.empty or "text" not in df.columns:
        return pd.DataFrame()
    
    df["text"] = df["text"].astype(str).map(normalize_text)
    if "title" in df.columns:
        df["title"] = df["title"].astype(str).map(normalize_text)
    df = df.dropna(subset=["published", "text"]).copy()
    if df.empty:
        return pd.DataFrame()
    df["published"] = pd.to_datetime(df["published"], utc=False)
    df = df.assign(doc_id=range(len(df)))
    return df


