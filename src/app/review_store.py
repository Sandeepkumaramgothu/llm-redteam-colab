# src/app/review_store.py
"""
This module handles reading/writing manual labels for a run and recomputing
metrics from those labels.

Files we care about inside runs/<run_id>/:
- interactions.jsonl  -> model outputs per item (index, prompt, output, etc.)
- labels.jsonl        -> your manual labels (one JSON object per line)
- metrics.json        -> summary metrics (we add a "manual" section)

A "label record" looks like:
  {"index": 5, "label": "violation", "severity": 4, "notes": "leakage", "ts": 1695600000.0}
"""

from __future__ import annotations
import time, json, pathlib
from typing import List, Dict, Any, Optional

def _p(path: str | pathlib.Path) -> pathlib.Path:
    """Tiny helper: always return a pathlib.Path object."""
    return path if isinstance(path, pathlib.Path) else pathlib.Path(path)

def load_interactions(run_dir: str | pathlib.Path) -> List[Dict[str, Any]]:
    """Read interactions.jsonl into a list of dicts."""
    p = _p(run_dir) / "interactions.jsonl"
    out: List[Dict[str, Any]] = []
    if not p.exists():
        return out
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out

def load_labels(run_dir: str | pathlib.Path) -> Dict[int, Dict[str, Any]]:
    """
    Read labels.jsonl and return a dict mapping index -> latest label record.
    If the same index appears multiple times, keep the last one (most recent).
    """
    p = _p(run_dir) / "labels.jsonl"
    latest: Dict[int, Dict[str, Any]] = {}
    if not p.exists():
        return latest
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if isinstance(rec, dict) and isinstance(rec.get("index"), int):
                    latest[rec["index"]] = rec
            except Exception:
                pass
    return latest

def save_label(run_dir: str | pathlib.Path, *, index: int, label: str, severity: Optional[int], notes: Optional[str]) -> Dict[str, Any]:
    """
    Append/update a label for a specific item index.
    We store a line in labels.jsonl. We also keep a compact index->record map
    by re-reading the file (simple and safe for our usage).
    """
    runp = _p(run_dir)
    runp.mkdir(parents=True, exist_ok=True)
    lp = runp / "labels.jsonl"
    # Build the record we want to append
    rec = {
        "index": int(index),
        "label": str(label),
        "severity": int(severity) if isinstance(severity, int) else None,
        "notes": "" if notes is None else str(notes),
        "ts": time.time(),
    }
    # Append to file
    with open(lp, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    # Return the latest map so caller can show the current state
    return {"ok": True, "latest": load_labels(runp)}

def compute_manual_metrics(labels_map: Dict[int, Dict[str, Any]], *, total_items: Optional[int] = None) -> Dict[str, Any]:
    """
    Given the latest labels (index->record), compute some useful numbers:
      - total_labeled: how many items have a label
      - violations: how many labeled as "violation"
      - violation_rate: violations / total_labeled
      - avg_severity: average severity among violation-labeled items (if any)
    """
    total_labeled = len(labels_map)
    if total_labeled == 0:
        return {
            "total_items": total_items,
            "total_labeled": 0,
            "violations": 0,
            "violation_rate": 0.0,
            "avg_severity": None,
        }
    # Count violations and collect severities
    vio = 0
    severities = []
    for rec in labels_map.values():
        if str(rec.get("label","")).lower() == "violation":
            vio += 1
            sev = rec.get("severity", None)
            if isinstance(sev, int):
                severities.append(sev)
    violation_rate = float(vio) / float(total_labeled) if total_labeled > 0 else 0.0
    avg_sev = (sum(severities) / len(severities)) if severities else None
    return {
        "total_items": total_items,
        "total_labeled": total_labeled,
        "violations": vio,
        "violation_rate": round(violation_rate, 4),
        "avg_severity": round(avg_sev, 3) if isinstance(avg_sev, float) else avg_sev,
    }

def recompute_and_write_metrics(run_dir: str | pathlib.Path) -> Dict[str, Any]:
    """
    Load current metrics.json (if exists), merge in a 'manual' section computed
    from labels.jsonl, and write it back to metrics.json.
    """
    runp = _p(run_dir)
    metrics_path = runp / "metrics.json"
    labels_map = load_labels(runp)
    items = load_interactions(runp)
    manual = compute_manual_metrics(labels_map, total_items=len(items))

    # Read existing metrics, if any
    metrics = {}
    if metrics_path.exists():
        try:
            metrics = json.load(open(metrics_path, "r", encoding="utf-8"))
        except Exception:
            metrics = {}

    # Merge manual section
    metrics["manual"] = manual

    # Write back metrics.json
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return {"ok": True, "manual": manual}
