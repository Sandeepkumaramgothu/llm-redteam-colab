from __future__ import annotations
import json, time
from pathlib import Path
from typing import Any, Dict, List

def now_ts() -> str:
    """Return a timestamp like 20250922-123456 for naming folders."""
    return time.strftime("%Y%m%d-%H%M%S")

def ensure_dir(path: str | Path) -> None:
    """Create a directory (and parents) if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)

def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    """Read JSON Lines (one JSON object per line) into a list of dicts."""
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def write_json(path: str | Path, obj: Dict[str, Any]) -> None:
    """Write pretty JSON to a file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def append_jsonl(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    """Append multiple dicts to a JSONL file (one per line)."""
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
