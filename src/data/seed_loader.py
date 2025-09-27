# src/data/seed_loader.py
"""
Seed loading and preparation utilities.

What this module guarantees:
- Every time you click "Start", we rewrite cfg.paths.seeds_path (JSONL) with the
  EXACT seeds this run will use. That makes runs reproducible and clear.
- Supports "local" (your own seeds.json / seeds.jsonl) and "dna" (Hugging Face dataset).
- Deterministic sampling: given the same (count, seed), you get the same sample.
"""

from __future__ import annotations
from typing import Tuple, List, Dict, Any, Optional
import json, pathlib, random

# For DNA/HF datasets:
try:
    from datasets import load_dataset
except Exception:
    load_dataset = None  # We'll raise a clear error at runtime if needed.


# ---------- low-level loaders ----------

def read_local_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read a .jsonl file and return a list of {'text': ...} dicts."""
    rows: List[Dict[str, Any]] = []
    p = pathlib.Path(path)
    if not p.exists():
        return rows
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                # Normalize: if {'text': "..."} keep it; if string, wrap it.
                if isinstance(obj, dict) and isinstance(obj.get("text"), str):
                    rows.append({"text": obj["text"]})
                elif isinstance(obj, str):
                    rows.append({"text": obj})
            except Exception:
                pass
    return rows

def read_local_json(path: str) -> List[Dict[str, Any]]:
    """Read a .json file and return a list of {'text': ...} dicts."""
    p = pathlib.Path(path)
    if not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    rows: List[Dict[str, Any]] = []
    if isinstance(data, list):
        # Can be ['txt', ...] OR [{'text': '...'}, ...]
        for item in data:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                rows.append({"text": item["text"]})
            elif isinstance(item, str):
                rows.append({"text": item})
    elif isinstance(data, dict):
        # Try common keys
        for key in ("seeds", "items", "rows", "data"):
            v = data.get(key)
            if isinstance(v, list):
                for item in v:
                    if isinstance(item, dict) and isinstance(item.get("text"), str):
                        rows.append({"text": item["text"]})
                    elif isinstance(item, str):
                        rows.append({"text": item})
    return rows

def read_local_auto(candidates: list[str]) -> Tuple[Optional[str], Optional[str]]:
    """Return (format, path) for the first existing candidate, else (None, None)."""
    for c in candidates:
        p = pathlib.Path(c)
        if p.exists():
            if c.endswith(".jsonl"):
                return ("jsonl", c)
            elif c.endswith(".json"):
                return ("json", c)
    return (None, None)


# ---------- sampling & writing ----------

def _sample_rows(rows: List[Dict[str, Any]], count: Optional[int], seed: int) -> List[Dict[str, Any]]:
    """Shuffle deterministically and take 'count' items (or all if count is None/too big)."""
    rows = list(rows)
    rnd = random.Random(int(seed))
    rnd.shuffle(rows)
    if not isinstance(count, int) or count <= 0 or count >= len(rows):
        return rows
    return rows[:count]

def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    """Write rows to JSONL with {'text': ...} per line."""
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"text": r.get("text","")}, ensure_ascii=False) + "\n")


# ---------- main entry point used by server ----------

def prepare_seeds(
    *,
    source: str,              # "local" or "dna"
    out_path: str,            # where to write the prepared JSONL (engine will read this)
    local_path: str,          # your local seeds file to read from (json or jsonl)
    hf_dataset_id: str,       # HF dataset id (for DNA mode)
    hf_split: str,            # split name ("train")
    hf_text_column: str,      # which column has the question/prompt text
    count: Optional[int],     # how many to keep (None = ALL available)
    seed: int,                # for deterministic shuffling
) -> Dict[str, Any]:
    """
    Build the exact seeds list for THIS run and write it to out_path (JSONL).
    Returns a small info dict with stats.
    """
    info: Dict[str, Any] = {"source": source, "out_path": out_path}

    if source.lower() == "local":
        # Load local seeds from the provided local_path.
        if local_path.endswith(".jsonl"):
            rows = read_local_jsonl(local_path)
        else:
            rows = read_local_json(local_path)
        total = len(rows)
        picked = _sample_rows(rows, count, seed)
        _write_jsonl(out_path, picked)
        info.update({
            "local_path": local_path,
            "available": total,
            "kept": len(picked),
        })
        return info

    # DNA / HF path
    if load_dataset is None:
        raise RuntimeError("datasets not installed; run: pip install datasets")

    ds = load_dataset(hf_dataset_id, split=hf_split)
    # Extract text column into the normalized [{'text': ...}, ...] format
    raw: List[str] = [str(x) for x in ds[hf_text_column]]
    rows = [{"text": t} for t in raw if t and t.strip()]
    total = len(rows)
    picked = _sample_rows(rows, count, seed)
    _write_jsonl(out_path, picked)
    info.update({
        "hf_dataset_id": hf_dataset_id,
        "hf_split": hf_split,
        "hf_text_column": hf_text_column,
        "available": total,
        "kept": len(picked),
    })
    return info
