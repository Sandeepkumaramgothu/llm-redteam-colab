# src/data/seed_loader.py
import os, json, random
from typing import List, Dict, Optional

def read_local_jsonl(jsonl_path: str) -> List[Dict]:
    rows: List[Dict] = []
    if not os.path.exists(jsonl_path):
        return rows
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows

def _rng(seed: Optional[int]) -> random.Random:
    """seed=None → SystemRandom (very random). seed=int → reproducible."""
    if seed is None:
        return random.SystemRandom()
    return random.Random(int(seed))

def load_hf_dna_prompts(
    dataset_id: str = "LibrAI/do-not-answer",
    split: str = "train",
    text_column: str = "question",
    max_items: Optional[int] = None,
    seed: Optional[int] = None,
) -> List[Dict]:
    """Load DNA and return a truly random subset of size=max_items."""
    try:
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError("Please `pip install datasets` to use the DNA loader") from e

    ds = load_dataset(dataset_id, split=split)

    pool: List[Dict] = []
    for i, row in enumerate(ds):
        txt = row.get(text_column)
        if not isinstance(txt, str) or not txt.strip():
            continue
        rid = row.get("id", i)
        pool.append({"id": f"dna-{rid}", "text": txt.strip()})

    if isinstance(max_items, int) and 0 < max_items < len(pool):
        rng = _rng(seed)
        idxs = rng.sample(range(len(pool)), k=max_items)
        return [pool[i] for i in idxs]

    return pool if max_items is None else pool[:max_items]

def write_jsonl(rows: List[Dict], out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def prepare_seeds(
    source: str,
    out_path: str,
    local_path: str = "src/storage/seeds.jsonl",
    hf_dataset_id: str = "LibrAI/do-not-answer",
    hf_split: str = "train",
    hf_text_column: str = "question",
    count: int = 200,
    seed: Optional[int] = None,
) -> Dict:
    """
    Make out_path as JSONL with exactly `count` rows when possible, selected randomly.
    """
    if source == "local":
        rows = read_local_jsonl(local_path)
        if len(rows) > count:
            rng = _rng(seed)
            rows = [rows[i] for i in rng.sample(range(len(rows)), k=count)]
        else:
            rows = rows[:count]
        write_jsonl(rows, out_path)
        return {"source": "local", "kept": len(rows), "out_path": out_path}

    if source == "dna":
        rows = load_hf_dna_prompts(
            dataset_id=hf_dataset_id,
            split=hf_split,
            text_column=hf_text_column,
            max_items=count,
            seed=seed,
        )
        write_jsonl(rows, out_path)
        return {"source": "dna", "dataset_id": hf_dataset_id, "kept": len(rows), "out_path": out_path}

    raise ValueError(f"Unknown source: {source}. Use 'local' or 'dna'.")
