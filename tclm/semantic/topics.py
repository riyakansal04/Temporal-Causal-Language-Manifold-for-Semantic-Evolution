from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def extract_topics(df: pd.DataFrame, n_topics: int = 8) -> List[dict]:
    if df.empty:
        return []
    X = np.stack(df["embedding"].values)
    k = min(n_topics, max(1, len(df) // 20))
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X)
    topics = []
    for i in range(k):
        idx = np.where(labels == i)[0]
        if len(idx) == 0:
            continue
        topics.append({"topic_id": i, "size": int(len(idx))})
    return topics


