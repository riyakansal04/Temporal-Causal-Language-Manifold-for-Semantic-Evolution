from __future__ import annotations

from datetime import datetime
from typing import List, Optional

import feedparser
import pandas as pd


RSS_FEEDS: List[str] = [
    "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml",
    "https://feeds.arstechnica.com/arstechnica/technology-lab",
    "https://www.theverge.com/rss/index.xml",
    "https://www.wired.com/feed/rss",
]


def _parse_entry(e) -> dict:
    published = None
    for key in ("published", "updated", "created"):
        val = e.get(key)
        if val:
            try:
                published = datetime(*e.get(key + "_parsed")[:6])
                break
            except Exception:
                try:
                    published = datetime.fromisoformat(val)  # best-effort
                    break
                except Exception:
                    pass
    return {
        "source": "rss",
        "title": e.get("title", ""),
        "summary": e.get("summary", ""),
        "text": f"{e.get('title','')}\n{e.get('summary','')}",
        "published": published,
        "url": e.get("link"),
    }


def fetch_rss_corpus(
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    max_docs: int = 1000,
) -> pd.DataFrame:
    records = []
    for url in RSS_FEEDS:
        feed = feedparser.parse(url)
        for e in feed.entries:
            row = _parse_entry(e)
            if row["published"] is None:
                continue
            if start and row["published"] < start:
                continue
            if end and row["published"] > end:
                continue
            records.append(row)
            if len(records) >= max_docs:
                break
        if len(records) >= max_docs:
            break

    df = pd.DataFrame.from_records(records)
    if not df.empty:
        df = df.sort_values("published").reset_index(drop=True)
    return df


