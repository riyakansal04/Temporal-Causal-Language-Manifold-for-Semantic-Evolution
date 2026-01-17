from __future__ import annotations

from datetime import datetime
from typing import Optional
from urllib.parse import quote

import feedparser
import pandas as pd


def fetch_arxiv_corpus(
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    max_docs: int = 1000,
    query: str = "cs.AI OR cs.CL",
) -> pd.DataFrame:
    # Use arXiv RSS search feed to avoid API keys
    # URL encode the query to handle spaces and special characters
    encoded_query = quote(query, safe='')
    url = f"https://export.arxiv.org/api/query?search_query=all:{encoded_query}&sortBy=submittedDate&sortOrder=descending&max_results={max_docs}"
    feed = feedparser.parse(url)
    records = []
    for e in feed.entries:
        published = None
        if getattr(e, "published_parsed", None):
            published = datetime(*e.published_parsed[:6])
        if not published:
            continue
        if start and published < start:
            continue
        if end and published > end:
            continue
        records.append(
            {
                "source": "arxiv",
                "title": e.get("title", ""),
                "summary": e.get("summary", ""),
                "text": f"{e.get('title','')}\n{e.get('summary','')}",
                "published": published,
                "url": e.get("link"),
            }
        )

    df = pd.DataFrame.from_records(records)
    if not df.empty:
        df = df.sort_values("published").reset_index(drop=True)
    return df


