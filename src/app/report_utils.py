from __future__ import annotations
import os, json, glob
from typing import List, Dict

def list_run_dirs(runs_root: str = "runs") -> List[str]:
    """
    Find all folders inside 'runs/' and return them sorted (old -> new).
    A "run" is a folder we created on Day 3 (like runs/20250922-201530).
    """
    if not os.path.isdir(runs_root):
        return []
    dirs = [d for d in glob.glob(os.path.join(runs_root, "*")) if os.path.isdir(d)]
    dirs.sort()  # oldest first
    return dirs

def latest_run_dir(runs_root: str = "runs") -> str | None:
    """
    Return the newest run folder, or None if no runs exist yet.
    """
    dirs = list_run_dirs(runs_root)
    return dirs[-1] if dirs else None

def load_metrics(run_dir: str) -> Dict:
    """
    Read metrics.json from a run folder.
    If it's missing, return a safe default.
    """
    path = os.path.join(run_dir, "metrics.json")
    if os.path.exists(path):
        return json.load(open(path, "r", encoding="utf-8"))
    return {"asr": None, "count": 0}

def read_first_interactions(run_dir: str, k: int = 5) -> List[Dict]:
    """
    Read the first k lines from interactions.jsonl (each line is a JSON object).
    """
    path = os.path.join(run_dir, "interactions.jsonl")
    rows: List[Dict] = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
            if len(rows) >= k:
                break
    return rows
