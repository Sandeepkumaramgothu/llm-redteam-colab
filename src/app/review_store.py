# src/app/review_store.py
from __future__ import annotations
import json, pathlib, statistics
from typing import Dict, Any, List, Optional

def _labels_path(run_dir: pathlib.Path) -> pathlib.Path:
    """labels.jsonl inside runs/<id>"""
    return run_dir / "labels.jsonl"

def load_interactions(run_dir: pathlib.Path) -> List[Dict[str, Any]]:
    """
    Return list of dicts from interactions.jsonl.
    Guarantees each row has an 'index' (0-based) even if file didn’t have it.
    """
    ip = run_dir / "interactions.jsonl"
    out: List[Dict[str, Any]] = []
    if not ip.exists():
        return out
    with ip.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            # Ensure an index (engine normally writes one; we add if missing)
            if "index" not in obj or not isinstance(obj["index"], int):
                obj["index"] = i
            out.append(obj)
    return out

def load_labels(run_dir: pathlib.Path) -> Dict[int, Dict[str, Any]]:
    """
    Read labels.jsonl (append-only log).
    Return: { index -> latest_label_dict }
    Each label row looks like:
      {"index": 12, "label": "safe"|"violation", "severity": int|None, "notes": str|None}
    """
    p = _labels_path(run_dir)
    latest: Dict[int, Dict[str, Any]] = {}
    if not p.exists():
        return latest
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            idx = obj.get("index", None)
            if isinstance(idx, int):
                latest[idx] = {
                    "label": str(obj.get("label","")).lower(),
                    "severity": obj.get("severity", None),
                    "notes": obj.get("notes", None),
                }
    return latest

def save_label(run_dir: pathlib.Path, *, index: int, label: str, severity: Optional[int], notes: Optional[str]) -> Dict[str, Any]:
    """
    Append one label to labels.jsonl; latest entry wins.
    Returns the snapshot for that index after write.
    """
    row = {"index": int(index), "label": str(label).lower(), "severity": severity, "notes": notes}
    p = _labels_path(run_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    snap = load_labels(run_dir).get(int(index))
    return {"ok": True, "index": int(index), "label": snap}

def recompute_and_write_metrics(run_dir: pathlib.Path) -> Dict[str, Any]:
    """
    Compute manual metrics from labels and merge them into runs/<id>/metrics.json under 'manual'.
    Also ensures metrics.json exists even if engine didn’t write it yet.
    """
    items = load_interactions(run_dir)
    labels = load_labels(run_dir)

    total_items = len(items)
    labeled_idxs = list(labels.keys())
    total_labeled = len(labeled_idxs)
    violations = 0
    severities: List[int] = []

    for idx in labeled_idxs:
        lab = labels.get(idx) or {}
        if str(lab.get("label","")).lower() in ("violation", "unsafe", "fail"):
            violations += 1
        sev = lab.get("severity", None)
        if isinstance(sev, int):
            severities.append(sev)

    violation_rate = (violations / total_labeled) if total_labeled > 0 else 0.0
    avg_severity = statistics.mean(severities) if severities else None

    mp = run_dir / "metrics.json"
    metrics = {}
    if mp.exists():
        try:
            metrics = json.load(mp.open("r", encoding="utf-8"))
        except Exception:
            metrics = {}
    metrics["manual"] = {
        "total_items": total_items,
        "total_labeled": total_labeled,
        "violations": violations,
        "violation_rate": round(violation_rate, 4),
        "avg_severity": avg_severity,
    }
    with mp.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return {"ok": True, "manual": metrics["manual"]}
