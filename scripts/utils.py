"""
Shared utilities for the intent classifier pipeline.
"""

import re


def normalize_query(q: str) -> str:
    """
    Normalize a query string for deduplication comparison.

    Strips, lowercases, collapses whitespace, and removes punctuation so that
    minor surface variants of the same query are treated as identical.
    """
    q = q.strip().lower()
    q = re.sub(r'\s+', ' ', q)       # collapse whitespace
    q = re.sub(r'[^\w\s]', '', q)    # remove punctuation
    return q
