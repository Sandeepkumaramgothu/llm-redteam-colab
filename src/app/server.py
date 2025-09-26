# src/app/server.py
from __future__ import annotations
import os, sys, json, glob, threading, time, pathlib
from zipfile import ZipFile, ZIP_DEFLATED
from flask import Flask, jsonify, request, render_template, send_from_directory

# --- absolute paths based on this file location ---
ROOT = pathlib.Path(__file__).resolve().parents[2]   # project root
CFG_PATH = ROOT / "configs" / "baseline.yaml"        # <-- absolute config
RUNS_DIR = ROOT / "runs"
TEMPLATES_DIR = ROOT / "src" / "app" / "templates"

# make src importable
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.core.config import load_config
from src.core.engine import IterationEngine
from src.core.utils import write_json
from src.data.seed_loader import prepare_seeds, read_local_auto, read_local_json, read_local_jsonl

app = Flask(__name__, template_folder=str(TEMPLATES_DIR))
_current = {"engine": None, "thread": None}

def _now_seed() -> int:
    return time.time_ns() % 1_000_000

def _latest_run_id() -> str | None:
    RUNS_DIR.mkdir(exist_ok=True)
    dirs = sorted([p.name for p in RUNS_DIR.glob("*") if p.is_dir()])
    return dirs[-1] if dirs else None

def _resolve_run(run_identifier: str | None):
    if not run_identifier:
        rid = _latest_run_id()
        if not rid:
            return None
        return rid, RUNS_DIR / rid
    rid = run_identifier
    if rid.startswith("runs/"):
        rid = rid.split("/", 1)[1]
    return rid, RUNS_DIR / rid

def _read_preview(run_dir: pathlib.Path, k: int | None = None) -> list[dict]:
    out = []
    ip = run_dir / "interactions.jsonl"
    if not ip.exists():
        return out
    with open(ip, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
            if isinstance(k, int) and i >= k:
                break
    return out

def _clamp(val, lo, hi, default):
    try:
        x = float(val)
    except Exception:
        return default
    return max(lo, min(hi, x))

def _make_report_html_safe(run_dir: pathlib.Path, max_samples: int | None = None) -> str:
    try:
        from scripts.mk_report import make_html as _fancy
        return _fancy(str(run_dir), max_samples=(10**9 if max_samples is None else max_samples))
    except Exception:
        pass
    def esc(x):
        import html
        return html.escape("" if x is None else str(x))
    meta = {}
    mp = run_dir / "run_meta.json"
    if mp.exists():
        try:
            meta = json.load(open(mp, "r", encoding="utf-8"))
        except Exception:
            meta = {}
    rows = []
    for i, rec in enumerate(_read_preview(run_dir, k=max_samples), start=1):
        prompt = esc(rec.get("prompt", ""))
        out = esc(rec.get("output", ""))[:200000]
        rw = rec.get("reward") or {}
        score = esc(rw.get("score", ""))
        level = esc(rw.get("level", ""))
        rows.append(f"<tr><td>{i}</td><td class='mono'>{prompt}</td><td class='mono'>{out}</td><td>{score}/{level}</td></tr>")
    return f"""<!doctype html><html><head><meta charset='utf-8'/>
<style>
 body{{font-family:system-ui,Arial;margin:2rem}}
 .mono{{font-family:ui-monospace,Menlo,Consolas;white-space:pre-wrap}}
 table{{border-collapse:collapse;width:100%}}
 th,td{{border:1px solid #ddd;padding:6px;vertical-align:top}}
 th{{background:#f6f6f6}}
 .box{{border:1px solid #ddd;padding:10px;border-radius:8px;margin:10px 0;background:#fafafa}}
</style>
<title>Run Report</title></head><body>
<h2>Run Report</h2>
<div class='box'>
  <h3>Run Settings</h3>
  <ul>
    <li><b>Target model:</b> {esc(meta.get('target_model_name',''))}</li>
    <li><b>Temperature:</b> {esc(meta.get('temperature',''))}</li>
    <li><b>Max new tokens:</b> {esc(meta.get('max_new_tokens',''))}</li>
    <li><b>Seed source:</b> {esc(meta.get('seed_source',''))}</li>
    <li><b>Prepared count:</b> {esc(meta.get('prepared_count',''))}</li>
  </ul>
</div>
<table><tr><th>#</th><th>Prompt</th><th>Output</th><th>Reward</th></tr>
{''.join(rows)}
</table></body></html>"""

@app.get("/")
def home():
    return render_template("index.html")

@app.post("/start")
def start():
    if _current["thread"] and _current["thread"].is_alive():
        return jsonify({"ok": False, "error": "run_in_progress"}), 400

    # ABSOLUTE PATH HERE:
    try:
        cfg = load_config(str(CFG_PATH))
    except Exception as e:
        return jsonify({"ok": False, "error": f"config_load_failed: {e}"}), 500

    payload = request.get_json(silent=True) or {}

    default_local = cfg.paths.seeds_path
    candidates = [default_local, "src/storage/seeds.json", "src/storage/seeds.jsonl"]
    _, chosen_local = read_local_auto(candidates)
    local_src = chosen_local or default_local

    seed_source = payload.get("seed_source", "local")
    raw_count   = payload.get("seed_count", None)
    seed_count  = None if (raw_count is None or (isinstance(raw_count, int) and raw_count <= 0)) else int(raw_count)
    if seed_source == "local" and (seed_count is None):
        if local_src.endswith(".jsonl"):
            seed_count = len(read_local_jsonl(local_src))
        else:
            seed_count = len(read_local_json(local_src))

    temperature = _clamp(payload.get("temperature", cfg.run.temperature), 0.0, 1.0, cfg.run.temperature)
    max_tokens  = int(_clamp(payload.get("max_new_tokens", cfg.run.max_new_tokens), 1, 512, cfg.run.max_new_tokens))

    shuffle_seed  = int(payload.get("seed", _now_seed()))
    seeds_out = cfg.paths.seeds_path
    try:
        info = prepare_seeds(
            source=seed_source,
            out_path=seeds_out,
            local_path=local_src,
            hf_dataset_id=payload.get("hf_dataset_id", "LibrAI/do-not-answer"),
            hf_split=payload.get("hf_split", "train"),
            hf_text_column=payload.get("hf_text_column", "question"),
            count=seed_count,
            seed=shuffle_seed,
        )
    except Exception as e:
        return jsonify({"ok": False, "error": f"seed_prep_failed: {e}"}), 500

    try:
        kept = int(info.get("kept", 0))
        if kept > 0:
            cfg.run.seeds_per_iter = kept
            cfg.run.iterations = 1
    except Exception:
        pass

    cfg.run.temperature = temperature
    cfg.run.max_new_tokens = max_tokens

    engine = IterationEngine(cfg)
    _current["engine"] = engine
    try:
        meta = {
            "target_model_name": cfg.models.target_model_name,
            "temperature": cfg.run.temperature,
            "max_new_tokens": cfg.run.max_new_tokens,
            "seed_source": seed_source,
            "prepared_count": kept if 'kept' in locals() else None,
        }
        write_json(str(engine.run_dir / "run_meta.json"), meta)
    except Exception:
        pass

    def work():
        try:
            engine.run()
        except Exception as e:
            write_json(str(engine.run_dir / "error.json"), {"error": str(e)})
            engine._set_status("error", engine.status.get("iter", 0), engine.status.get("percent", 0))

    t = threading.Thread(target=work, daemon=True)
    _current["thread"] = t
    t.start()

    return jsonify({"ok": True, "run_id": engine.run_id, "seed_info": info})

@app.get("/status")
def status():
    eng = _current.get("engine")
    if eng is None:
        return jsonify({"phase": "idle", "iter": 0, "percent": 0})
    return jsonify(eng.status)

@app.get("/result")
def result():
    eng = _current.get("engine")
    if eng is None:
        return jsonify({"error": "no_run"}), 400
    rid = eng.run_id
    return jsonify({
        "run_id": rid,
        "artifacts": {
            "config":        f"runs/{rid}/config_resolved.json",
            "metrics":       f"runs/{rid}/metrics.json",
            "interactions":  f"runs/{rid}/interactions.jsonl",
            "status":        f"runs/{rid}/status.json",
            "run_meta":      f"runs/{rid}/run_meta.json",
        }
    })

@app.get("/report")
def report():
    rid = request.args.get("run_id")
    resolved = _resolve_run(rid)
    if not resolved:
        return render_template("report.html", found=False, run_id=None, settings={}, samples=[])
    run_id, run_dir = resolved
    settings = {}
    rp = run_dir / "run_meta.json"
    if rp.exists():
        try:
            settings = json.load(open(rp, "r", encoding="utf-8"))
        except Exception:
            settings = {}
    samples = _read_preview(run_dir, k=None)
    return render_template("report.html", found=True, run_id=f"runs/{run_id}", settings=settings, samples=samples)

@app.get("/runs/<path:subpath>")
def serve_runs(subpath: str):
    return send_from_directory(str(RUNS_DIR), subpath, as_attachment=False)

@app.get("/dl/<path:subpath>")
def download_runs(subpath: str):
    return send_from_directory(str(RUNS_DIR), subpath, as_attachment=True)

@app.post("/export_report")
def export_report():
    try:
        payload = request.get_json(silent=True) or {}
        rid, run_dir = _resolve_run(payload.get("run_id")) or (None, None)
        if not run_dir:
            return jsonify({"ok": False, "error": "no_runs"}), 400
        html_out = _make_report_html_safe(run_dir, max_samples=None)
        report_path = run_dir / "report.html"
        index_path  = run_dir / "index.html"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(html_out, encoding="utf-8")
        index_path.write_text(html_out, encoding="utf-8")
        return jsonify({"ok": True, "report_path": f"runs/{rid}/report.html", "report_dl": f"dl/{rid}/report.html"}), 200
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.post("/zip_run")
def zip_run():
    try:
        payload = request.get_json(silent=True) or {}
        rid, run_dir = _resolve_run(payload.get("run_id")) or (None, None)
        if not run_dir:
            return jsonify({"ok": False, "error": "no_runs"}), 400
        if not run_dir.exists() or not run_dir.is_dir():
            return jsonify({"ok": False, "error": f"missing run dir: {run_dir}"}), 400
        zip_abs = RUNS_DIR / f"{rid}.zip"
        with ZipFile(zip_abs, "w", ZIP_DEFLATED) as zf:
            for path in glob.glob(str(run_dir / "**"), recursive=True):
                if os.path.isfile(path):
                    arcname = os.path.relpath(path, str(RUNS_DIR))
                    zf.write(path, arcname=arcname)
        return jsonify({"ok": True, "zip_path": f"runs/{rid}.zip", "zip_dl": f"dl/{rid}.zip"}), 200
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.post("/shutdown")
def shutdown():
    try:
        def _exit_soon():
            time.sleep(0.3)
            os._exit(0)
        threading.Thread(target=_exit_soon, daemon=True).start()
        return jsonify({"ok": True, "msg": "server will stop"}), 200
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

def run_app(port: int = 8000):
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
