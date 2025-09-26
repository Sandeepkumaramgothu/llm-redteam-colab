# src/core/utils.py
import json, pathlib, time

def ensure_dir(path: str):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

def write_json(path: str, obj):
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def ts_run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S")
