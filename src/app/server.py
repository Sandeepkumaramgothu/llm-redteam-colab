# src/app/server.py
from __future__ import annotations
import os, sys, json, glob, threading, time, pathlib
from zipfile import ZipFile, ZIP_DEFLATED
from flask import Flask, jsonify, request, render_template, send_from_directory

# Absolute paths so spawned processes always know where files live
ROOT = pathlib.Path(__file__).resolve().parents[2]   # .../redteam_app
RUNS_DIR = ROOT / "runs"
TEMPLATES_DIR = ROOT / "src" / "app" / "templates"

# Make project root importable
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.core.config import load_config
from src.core.engine import IterationEngine
from src.core.utils import write_json
from src.data.seed_loader import prepare_seeds

app = Flask(__name__, template_folder=str(TEMPLATES_DIR))
_current = {"engine": None, "thread": None}

# ---------------- helpers ----------------

def _now_seed() -> int:
    """Fresh random seed: use current time (ns) so each run is different."""
    return time.time_ns() % 1_000_000

def _latest_run_id() -> str | None:
    """Newest run directory name under runs/ (or None)."""
    RUNS_DIR.mkdir(exist_ok=True)
    dirs = sorted([p.name for p in RUNS_DIR.glob("*") if p.is_dir()])
    return dirs[-1] if dirs else None

def _resolve_run(run_identifier: str | None) -> tuple[str, pathlib.Path] | None:
    """
    Accept '2025...' OR 'runs/2025...' OR None.
    Return (run_id, absolute_dir) OR None.
    """
    if not run_identifier:
        rid = _latest_run_id()
        if not rid:
            return None
        return rid, RUNS_DIR / rid
    rid = run_identifier
    if rid.startswith("runs/"):
        rid = rid.split("/", 1)[1]
    return rid, RUNS_DIR / rid

def _read_samples_for_preview(run_dir: pathlib.Path, k: int = 20) -> list[dict]:
    """Read up to k interaction rows for the preview page."""
    out: list[dict] = []
    ip = run_dir / "interactions.jsonl"
    if not ip.exists():
        return out
    with open(ip, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            if i > k: break
            line = line.strip()
            if not line: continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out

def _make_report_html_safe(run_dir: pathlib.Path, max_samples: int = 20) -> str:
    """Try fancy exporter; if it fails, write a simple, safe HTML."""
    try:
        from scripts.mk_report import make_html as _fancy
        return _fancy(str(run_dir), max_samples=max_samples)
    except Exception:
        pass

    def esc(x):
        import html
        return html.escape("" if x is None else str(x))

    metrics = {}
    mp = run_dir / "metrics.json"
    if mp.exists():
        try:
            metrics = json.load(open(mp, "r", encoding="utf-8"))
        except Exception:
            metrics = {}
    count = metrics.get("count", "")
    asr = metrics.get("asr", "")

    rows = []
    for i, rec in enumerate(_read_samples_for_preview(run_dir, k=max_samples), start=1):
        prompt = esc(rec.get("prompt", ""))
        out = esc(rec.get("output", ""))[:2000]
        rw = rec.get("reward") or {}
        score = esc(rw.get("score", ""))
        level = esc(rw.get("level", ""))
        rows.append(
            f"<tr><td>{i}</td><td class='mono'>{prompt}</td>"
            f"<td class='mono'>{out}</td><td>{score}/{level}</td></tr>"
        )

    return f"""<!doctype html><html><head><meta charset='utf-8'/>
<style>
 body{{font-family:system-ui,Arial;margin:2rem}}
 .mono{{font-family:ui-monospace,Menlo,Consolas;white-space:pre-wrap}}
 table{{border-collapse:collapse;width:100%}}
 th,td{{border:1px solid #ddd;padding:6px;vertical-align:top}}
 th{{background:#f6f6f6}}
</style>
<title>Run Report</title></head><body>
<h2>Run Report</h2>
<p><b>Run:</b> {esc(str(run_dir))}</p>
<ul><li><b>count</b>: {esc(count)}</li><li><b>asr</b>: {esc(asr)}</li></ul>
<table><tr><th>#</th><th>Prompt</th><th>Output</th><th>Reward</th></tr>
{''.join(rows)}
</table></body></html>"""

# ---------------- routes ----------------

@app.get("/")
def home():
    return render_template("index.html")

@app.post("/start")
def start():
    """
    Start a run:
    - prepare seeds.jsonl (dna/local)
    - run the engine in a background thread
    """
    if _current["thread"] and _current["thread"].is_alive():
        return jsonify({"ok": False, "error": "run_in_progress"}), 400

    try:
        cfg = load_config("configs/baseline.yaml")
    except Exception as e:
        return jsonify({"ok": False, "error": f"config_load_failed: {e}"}), 500

    payload = request.get_json(silent=True) or {}
    seed_source   = payload.get("seed_source", "dna")
    seed_count    = int(payload.get("seed_count", 20))
    hf_dataset_id = payload.get("hf_dataset_id", "LibrAI/do-not-answer")
    hf_split      = payload.get("hf_split", "train")
    hf_text_col   = payload.get("hf_text_column", "question")
    shuffle_seed  = int(payload.get("seed", _now_seed()))   # fresh random each run

    seeds_out = cfg.paths.seeds_path  # "src/storage/seeds.jsonl"

    try:
        info = prepare_seeds(
            source=seed_source,
            out_path=seeds_out,
            local_path=seeds_out,
            hf_dataset_id=hf_dataset_id,
            hf_split=hf_split,
            hf_text_column=hf_text_col,
            count=seed_count,
            seed=shuffle_seed,
        )
    except Exception as e:
        return jsonify({"ok": False, "error": f"seed_prep_failed: {e}"}), 500

    engine = IterationEngine(cfg)
    _current["engine"] = engine

    def work():
        try:
            engine.run()
        except Exception as e:
            write_json(str(engine.run_dir / "error.json"), {"error": str(e)})
            engine._set_status("error",
                               engine.status.get("iter", 0),
                               engine.status.get("percent", 0))

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
        }
    })

@app.get("/report")
def report():
    # Show latest (or chosen) run with up to 20 preview samples
    rid = request.args.get("run_id")
    resolved = _resolve_run(rid)
    if not resolved:
        return render_template("report.html", found=False, run_id=None, metrics={}, samples=[])
    run_id, run_dir = resolved

    metrics = {}
    mp = run_dir / "metrics.json"
    if mp.exists():
        try:
            metrics = json.load(open(mp, "r", encoding="utf-8"))
        except Exception:
            metrics = {}

    samples = _read_samples_for_preview(run_dir, k=20)
    return render_template("report.html", found=True, run_id=f"runs/{run_id}", metrics=metrics, samples=samples)

# Serve files from absolute runs/ folder
@app.get("/runs/<path:subpath>")
def serve_runs(subpath: str):
    return send_from_directory(str(RUNS_DIR), subpath, as_attachment=False)

# Force-download from runs/ folder
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

        html_out = _make_report_html_safe(run_dir, max_samples=20)
        out_path = run_dir / "report.html"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html_out)

        return jsonify({
            "ok": True,
            "report_path": f"runs/{rid}/report.html",
            "report_dl":   f"dl/{rid}/report.html"
        }), 200
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

        return jsonify({
            "ok": True,
            "zip_path": f"runs/{rid}.zip",
            "zip_dl":   f"dl/{rid}.zip"
        }), 200
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
