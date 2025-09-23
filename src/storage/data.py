from __future__ import annotations
from typing import List, Dict
from src.core.utils import read_jsonl

def load_seeds(path: str, limit: int | None = None) -> List[Dict]:
    """Load seeds from JSONL; optionally cap with limit."""
    data = read_jsonl(path)
    if limit is not None:
        data = data[:limit]
    return data
