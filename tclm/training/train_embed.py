from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

from ..config import Paths


@dataclass
class TrainConfig:
    base_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    epochs: int = 1
    train_batch_size: int = 64
    lr: float = 2e-5
    output_dir: Path = Paths.artifacts / "models" / "domain-mini"


def build_weak_pairs(df: pd.DataFrame, max_pairs: int = 20000) -> list[InputExample]:
    df = df.dropna(subset=["title", "text"]).copy()
    df = df[df["title"].str.len() > 10]
    df = df[df["text"].str.len() > 40]
    if len(df) == 0:
        return []
    df = df.sample(n=min(max_pairs, len(df)), random_state=42)
    examples = []
    for _, r in df.iterrows():
        examples.append(InputExample(texts=[str(r["title"]), str(r["text"])], label=1.0))
    return examples


def train_embedding(df: pd.DataFrame, cfg: Optional[TrainConfig] = None) -> str:
    cfg = cfg or TrainConfig()
    pairs = build_weak_pairs(df)
    if not pairs:
        raise ValueError("Not enough data to build training pairs")

    model = SentenceTransformer(cfg.base_model)
    train_loader = DataLoader(pairs, shuffle=True, batch_size=cfg.train_batch_size, drop_last=True)
    loss_fn = losses.MultipleNegativesRankingLoss(model)

    warmup_steps = max(10, int(0.1 * len(train_loader) * cfg.epochs))
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    model.fit(
        train_objectives=[(train_loader, loss_fn)],
        epochs=cfg.epochs,
        warmup_steps=warmup_steps,
        scheduler="warmuplinear",
        use_amp=True,
        output_path=str(cfg.output_dir),
        optimizer_params={"lr": cfg.lr},
        show_progress_bar=True,
    )
    return str(cfg.output_dir)


