from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from joblib import Memory


_memory = Memory(location=None, verbose=0)


@_memory.cache
def _load_model(name: str) -> SentenceTransformer:
    return SentenceTransformer(name)


def embed_corpus(df: pd.DataFrame, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> pd.DataFrame:
    model = _load_model(model_name)
    texts: List[str] = df["text"].fillna("").astype(str).tolist()
    embs = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    emb_arr = np.asarray(embs, dtype=np.float32)
    out = df.copy()
    out["embedding"] = list(emb_arr)
    return out


