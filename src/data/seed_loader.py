# src/data/seed_loader.py
import os, json, random
from typing import List, Dict, Optional, Tuple, Any

def _normalize_rows(raw: List[Any]) -> List[Dict]:
    out: List[Dict] = []
    for i, item in enumerate(raw):
        if isinstance(item, dict):
            txt = item.get("text") or item.get("prompt")
            if isinstance(txt, str) and txt.strip():
                rid = item.get("id", i)
                out.append({"id": str(rid), "text": txt.strip()})
        elif isinstance(item, str) and item.strip():
            out.append({"id": str(i), "text": item.strip()})
    return out

def read_local_jsonl(jsonl_path: str) -> List[Dict]:
    rows: List[Dict] = []
    if not os.path.exists(jsonl_path):
        return rows
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                txt = obj.get("text") or obj.get("prompt")
                if isinstance(txt, str) and txt.strip():
                    rid = obj.get("id", i)
                    rows.append({"id": str(rid), "text": txt.strip()})
    return rows

def read_local_json(json_path: str) -> List[Dict]:
    if not os.path.exists(json_path):
        return []
    try:
        data = json.load(open(json_path, "r", encoding="utf-8"))
    except Exception:
        return []
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                return _normalize_rows(v)
        return []
    if isinstance(data, list):
        return _normalize_rows(data)
    return []

def read_local_auto(candidates: List[str]) -> Tuple[List[Dict], Optional[str]]:
    for p in candidates:
        if p.endswith(".jsonl"):
            rows = read_local_jsonl(p)
        elif p.endswith(".json"):
            rows = read_local_json(p)
        else:
            continue
        if rows:
            return rows, p
        if os.path.exists(p):
            return [], p
    return [], None

def _rng(seed: Optional[int]) -> random.Random:
    return random.SystemRandom() if seed is None else random.Random(int(seed))

def load_hf_dna_prompts(
    dataset_id: str = "LibrAI/do-not-answer",
    split: str = "train",
    text_column: str = "question",
    max_items: Optional[int] = None,
    seed: Optional[int] = None,
) -> List[Dict]:
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
    count: Optional[int] = None,
    seed: Optional[int] = None,
) -> Dict:
    if source == "local":
        rows = read_local_jsonl(local_path) if local_path.endswith(".jsonl") else read_local_json(local_path)
        total = len(rows)
        if count is None or int(count) <= 0:
            kept = rows
        else:
            c = int(count)
            if total > c:
                rng = _rng(seed)
                kept = [rows[i] for i in rng.sample(range(total), k=c)]
            else:
                kept = rows[:c]
        write_jsonl(kept, out_path)
        return {"source": "local", "kept": len(kept), "local_path": local_path, "out_path": out_path, "total_local": total}

    if source == "dna":
        c = None if (count is None or int(count) <= 0) else int(count)
        kept = load_hf_dna_prompts(
            dataset_id=hf_dataset_id,
            split=hf_split,
            text_column=hf_text_column,
            max_items=c,
            seed=seed,
        )
        write_jsonl(kept, out_path)
        return {"source": "dna", "dataset_id": hf_dataset_id, "kept": len(kept), "out_path": out_path}

    raise ValueError(f"Unknown source: {source}. Use 'local' or 'dna'.")
