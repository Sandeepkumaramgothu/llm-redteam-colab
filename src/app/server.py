# src/app/server.py
from __future__ import annotations
import os, sys, json, glob, threading, time, pathlib
from zipfile import ZipFile, ZIP_DEFLATED
from flask import Flask, jsonify, request, render_template, send_from_directory

ROOT = pathlib.Path(__file__).resolve().parents[2]
CFG_PATH = ROOT / "configs" / "baseline.yaml"
RUNS_DIR = ROOT / "runs"
TEMPLATES_DIR = ROOT / "src" / "app" / "templates"

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

@app.get("/")
def home():
    return render_template("index.html")

@app.post("/start")
def start():
    if _current["thread"] and _current["thread"].is_alive():
        return jsonify({"ok": False, "error": "run_in_progress"}), 400

    try:
        cfg = load_config(str(CFG_PATH))
    except Exception as e:
        return jsonify({"ok": False, "error": f"config_load_failed: {e}"}), 500

    payload = request.get_json(silent=True) or {}

    # Optional preferred model override from UI
    preferred_model = payload.get("preferred_model")
    if isinstance(preferred_model, str) and preferred_model.strip():
        cfg.models.target_model_name = preferred_model.strip()

    # Decide local source path
    default_local = cfg.paths.seeds_path
    candidates = [default_local, "src/storage/seeds.json", "src/storage/seeds.jsonl"]
    _, chosen_local = read_local_auto(candidates)
    local_src = chosen_local or default_local

    # Seed source and counts
    seed_source = payload.get("seed_source", "local")
    raw_count   = payload.get("seed_count", None)
    seed_count  = None if (raw_count is None or (isinstance(raw_count, int) and raw_count <= 0)) else int(raw_count)
    if seed_source == "local" and (seed_count is None):
        # Blank count for local = use ALL rows
        if local_src.endswith(".jsonl"):
            seed_count = len(read_local_jsonl(local_src))
        else:
            seed_count = len(read_local_json(local_src))

    # Generation knobs
    temperature = _clamp(payload.get("temperature", cfg.run.temperature), 0.0, 1.0, cfg.run.temperature)
    max_tokens  = int(_clamp(payload.get("max_new_tokens", cfg.run.max_new_tokens), 1, 512, cfg.run.max_new_tokens))

    # Prepare (rewrite) seeds file for this run
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

    # Ensure we process ALL prepared seeds in one pass
    try:
        kept = int(info.get("kept", 0))
        if kept > 0:
            cfg.run.seeds_per_iter = kept
            cfg.run.iterations = 1
    except Exception:
        pass

    # Apply generation knobs
    cfg.run.temperature = temperature
    cfg.run.max_new_tokens = max_tokens

    # Start engine (loads model; may fall back)
    engine = IterationEngine(cfg)
    _current["engine"] = engine

    # Record run metadata for the report
    try:
        actual_model = getattr(engine.textgen, "model_name", cfg.models.target_model_name)
        meta = {
            "preferred_model": cfg.models.target_model_name,
            "actual_model": actual_model,
            "fallback_used": (actual_model != cfg.models.target_model_name),
            "temperature": cfg.run.temperature,
            "max_new_tokens": cfg.run.max_new_tokens,
            "seed_source": seed_source,
            "prepared_count": kept if 'kept' in locals() else None,
        }
        write_json(str(engine.run_dir / "run_meta.json"), meta)
    except Exception:
        pass

    # Run in background
    def work():
        try:
            engine.run()
        except Exception as e:
            write_json(str(engine.run_dir / "error.json"), {"error": str(e)})
            engine._set_status("error", engine.status.get("current", 0), engine.status.get("percent", 0))

    t = threading.Thread(target=work, daemon=True)
    _current["thread"] = t
    t.start()

    return jsonify({"ok": True, "run_id": engine.run_id, "seed_info": info})

@app.get("/status")
def status():
    eng = _current.get("engine")
    if eng is None:
        return jsonify({"phase": "idle", "iter": 0, "current": 0, "total": 0, "percent": 0})
    return jsonify(eng.status)

@app.get("/result")
def result():
    eng = _current.get("engine")
    if eng is None:
        return jsonify({"error": "no_run"}), 400
    rid = eng.run_id
    actual_model = getattr(eng.textgen, "model_name", None)
    return jsonify({
        "run_id": rid,
        "actual_model": actual_model,
        "artifacts": {
            "config":        f"runs/{rid}/config_resolved.json",
            "metrics":       f"runs/{rid}/metrics.json",
            "interactions":  f"runs/{rid}/interactions.jsonl",
            "status":        f"runs/{rid}/status.json",
            "run_meta":      f"runs/{rid}/run_meta.json",
            "seeds_used":    f"runs/{rid}/seeds_used.jsonl",
        }
    })

@app.get("/report")
def report():
    rid = request.args.get("run_id")
    resolved = _resolve_run(rid)
    if not resolved:
        return render_template("report.html", found=False, run_id=None, settings={}, metrics={}, samples=[])
    run_id, run_dir = resolved

    settings, metrics = {}, {}
    rp = run_dir / "run_meta.json"
    mt = run_dir / "metrics.json"
    if rp.exists():
        try:
            settings = json.load(open(rp, "r", encoding="utf-8"))
        except Exception:
            settings = {}
    if mt.exists():
        try:
            metrics = json.load(open(mt, "r", encoding="utf-8"))
        except Exception:
            metrics = {}

    samples = _read_preview(run_dir, k=None)
    return render_template("report.html", found=True, run_id=f"runs/{run_id}", settings=settings, metrics=metrics, samples=samples)

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

        # Build a lightweight inline report if fancy exporter not present
        try:
            from scripts.mk_report import make_html as _fancy
            html_out = _fancy(str(run_dir), max_samples=None)
        except Exception:
            # Minimal inline builder (mirrors report.html content)
            def esc(x):
                import html
                return html.escape("" if x is None else str(x))
            # read meta/metrics
            settings, metrics = {}, {}
            rp = run_dir / "run_meta.json"
            mt = run_dir / "metrics.json"
            if rp.exists():
                try: settings = json.load(open(rp,"r",encoding="utf-8"))
                except: settings = {}
            if mt.exists():
                try: metrics = json.load(open(mt,"r",encoding="utf-8"))
                except: metrics = {}
            # table rows
            rows=[]
            ip = run_dir / "interactions.jsonl"
            if ip.exists():
                with open(ip,"r",encoding="utf-8") as f:
                    for i,line in enumerate(f, start=1):
                        try:
                            rec=json.loads(line)
                        except: 
                            continue
                        rows.append(f"<tr><td>{i}</td><td class='mono'>{esc(rec.get('prompt',''))}</td><td class='mono'>{esc(rec.get('output',''))}</td><td>{esc(rec.get('new_tokens',''))} tok / {esc(rec.get('gen_time_sec',''))} s</td></tr>")
            html_out=f"""<!doctype html><html><meta charset='utf-8'><style>
            body{{font-family:system-ui,Arial;margin:2rem}}
            .mono{{font-family:ui-monospace,Menlo,Consolas;white-space:pre-wrap}}
            table{{border-collapse:collapse;width:100%}}
            th,td{{border:1px solid #ddd;padding:6px;vertical-align:top}}
            th{{background:#f6f6f6}}
            .box{{border:1px solid #ddd;padding:10px;border-radius:8px;margin:10px 0;background:#fafafa}}
            </style><body>
            <h2>Run Report</h2>
            <div class='box'><h3>Run Settings</h3><ul>
              <li><b>Preferred model:</b> {esc(settings.get('preferred_model',''))}</li>
              <li><b>Actual model:</b> {esc(settings.get('actual_model',''))}{' (fallback used)' if settings.get('fallback_used') else ''}</li>
              <li><b>Temperature:</b> {esc(settings.get('temperature',''))}</li>
              <li><b>Max new tokens:</b> {esc(settings.get('max_new_tokens',''))}</li>
              <li><b>Seed source:</b> {esc(settings.get('seed_source',''))}</li>
              <li><b>Prepared count:</b> {esc(settings.get('prepared_count',''))}</li>
            </ul></div>
            <div class='box'><h3>Metrics</h3><ul>
              <li><b>Items processed:</b> {esc(metrics.get('count',''))}</li>
              <li><b>Total new tokens:</b> {esc(metrics.get('total_new_tokens',''))}</li>
              <li><b>Generation time (sum):</b> {esc(metrics.get('gen_time_sec',''))} s</li>
              <li><b>Wall time:</b> {esc(metrics.get('wall_time_sec',''))} s</li>
              <li><b>Tokens/sec (avg):</b> {esc(metrics.get('tokens_per_sec',''))}</li>
            </ul></div>
            <table><tr><th>#</th><th>Prompt</th><th>Output</th><th>Per-item</th></tr>
            {''.join(rows)}</table></body></html>"""

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
                    import os as _os
                    arcname = _os.path.relpath(path, str(RUNS_DIR))
                    zf.write(path, arcname=arcname)
        return jsonify({"ok": True, "zip_path": f"runs/{rid}.zip", "zip_dl": f"dl/{rid}.zip"}), 200
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.post("/shutdown")
def shutdown():
    try:
        def _exit_soon():
            time.sleep(0.3); os._exit(0)
        threading.Thread(target=_exit_soon, daemon=True).start()
        return jsonify({"ok": True, "msg": "server will stop"}), 200
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

def run_app(port: int = 8000):
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
