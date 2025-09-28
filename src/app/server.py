# src/app/server.py
from __future__ import annotations
import os, sys, json, glob, threading, time, pathlib, html
from zipfile import ZipFile, ZIP_DEFLATED
from flask import Flask, jsonify, request, render_template, send_from_directory

# ----- Paths (absolute so CWD doesn't matter) -----
ROOT = pathlib.Path(__file__).resolve().parents[2]
CFG_PATH = ROOT / "configs" / "baseline.yaml"
RUNS_DIR = ROOT / "runs"
TEMPLATES_DIR = ROOT / "src" / "app" / "templates"

# Make src importable
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Core imports
from src.core.config import load_config
from src.core.engine import IterationEngine
from src.core.utils import write_json
from src.data.seed_loader import prepare_seeds, read_local_auto, read_local_json, read_local_jsonl

# Optional review helpers (labels + recompute)
_HAS_REVIEW = False
try:
    from src.app.review_store import load_interactions as _load_interactions_rs
    from src.app.review_store import load_labels, recompute_and_write_metrics
    _HAS_REVIEW = True
except Exception:
    pass

app = Flask(__name__, template_folder=str(TEMPLATES_DIR))
_current = {"engine": None, "thread": None}

# ---------- helpers ----------
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

def _read_interactions(run_dir: pathlib.Path, limit: int | None = None) -> list[dict]:
    """Read interactions.jsonl (optionally limit to first K rows)."""
    p = run_dir / "interactions.jsonl"
    out = []
    if not p.exists():
        return out
    with open(p, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
            if isinstance(limit, int) and i >= limit:
                break
    return out

def _clamp(val, lo, hi, default):
    try:
        x = float(val)
    except Exception:
        return default
    return max(lo, min(hi, x))

# ---------- pages ----------
@app.get("/")
def home():
    return render_template("index.html")

@app.get("/review")
def review_page():
    # Page exists even if review helpers are missing; buttons will fail gracefully.
    return render_template("review.html")

@app.get("/report")
def report():
    rid = request.args.get("run_id")
    resolved = _resolve_run(rid)
    if not resolved:
        return render_template("report.html", found=False, run_id=None, settings={}, metrics={}, samples=[])
    run_id, run_dir = resolved

    # Load settings/metrics
    settings, metrics = {}, {}
    rp = run_dir / "run_meta.json"
    mt = run_dir / "metrics.json"
    if rp.exists():
        try: settings = json.load(open(rp, "r", encoding="utf-8"))
        except Exception: settings = {}
    if mt.exists():
        try: metrics = json.load(open(mt, "r", encoding="utf-8"))
        except Exception: metrics = {}

    # Read all samples; attach latest manual label if available
    samples = _read_interactions(run_dir, limit=None)
    labels_map = load_labels(run_dir) if _HAS_REVIEW else {}
    for s in samples:
        try:
            s["label"] = labels_map.get(int(s.get("index", -1)))
        except Exception:
            pass

    return render_template("report.html", found=True, run_id=f"runs/{run_id}", settings=settings, metrics=metrics, samples=samples)

# ---------- start / status / result ----------
@app.post("/start")
def start():
    if _current["thread"] and _current["thread"].is_alive():
        return jsonify({"ok": False, "error": "run_in_progress"}), 400

    # Load YAML config
    try:
        cfg = load_config(str(CFG_PATH))
    except Exception as e:
        return jsonify({"ok": False, "error": f"config_load_failed: {e}"}), 500

    payload = request.get_json(silent=True) or {}

    # Preferred model override
    pm = payload.get("preferred_model")
    if isinstance(pm, str) and pm.strip():
        cfg.models.target_model_name = pm.strip()

    # Local seeds source
    default_local = cfg.paths.seeds_path
    candidates = [default_local, "src/storage/seeds.json", "src/storage/seeds.jsonl"]
    _, chosen_local = read_local_auto(candidates)
    local_src = chosen_local or default_local

    # Seed counts
    seed_source = payload.get("seed_source", "local")
    raw_count   = payload.get("seed_count", None)
    seed_count  = None if (raw_count is None or (isinstance(raw_count, int) and raw_count <= 0)) else int(raw_count)
    if seed_source == "local" and (seed_count is None):
        if local_src.endswith(".jsonl"):
            seed_count = len(read_local_jsonl(local_src))
        else:
            seed_count = len(read_local_json(local_src))

    # Generation knobs
    temperature = _clamp(payload.get("temperature", cfg.run.temperature), 0.0, 1.0, cfg.run.temperature)
    max_tokens  = int(_clamp(payload.get("max_new_tokens", cfg.run.max_new_tokens), 1, 512, cfg.run.max_new_tokens))

    # Prepare exact seeds file for this run (rewrite src/storage/seeds.jsonl)
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

    # Process ALL prepared rows in one pass
    try:
        kept = int(info.get("kept", 0))
        if kept > 0:
            cfg.run.seeds_per_iter = kept
            cfg.run.iterations = 1
    except Exception:
        pass

    cfg.run.temperature    = temperature
    cfg.run.max_new_tokens = max_tokens

    # Create engine (load model) with clear JSON error if it fails
    try:
        engine = IterationEngine(cfg)
    except Exception as e:
        return jsonify({"ok": False, "error": f"model_init_failed: {e}"}), 500

    _current["engine"] = engine

    # Save run meta (preferred vs actual)
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

    # Run in the background
    def work():
        try:
            engine.run()
        except Exception as e:
            write_json(str(engine.run_dir / "error.json"), {"error": str(e)})
            try:
                engine._set_status("error", engine.status.get("current", 0), engine.status.get("percent", 0))
            except Exception:
                pass

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

# ---------- export / zip / shutdown ----------
@app.post("/export_report")
def export_report():
    """Build report.html + index.html inside the run folder and return paths."""
    try:
        payload = request.get_json(silent=True) or {}
        resolved = _resolve_run(payload.get("run_id"))
        if not resolved:
            return jsonify({"ok": False, "error": "no_runs"}), 400
        rid, run_dir = resolved

        # Read meta + metrics
        settings, metrics = {}, {}
        rp = run_dir / "run_meta.json"
        mt = run_dir / "metrics.json"
        if rp.exists():
            try: settings = json.load(open(rp, "r", encoding="utf-8"))
            except Exception: settings = {}
        if mt.exists():
            try: metrics = json.load(open(mt, "r", encoding="utf-8"))
            except Exception: metrics = {}

        # Read all items
        items = _read_interactions(run_dir, limit=None)

        # Manual labels (if module available)
        labels_map = load_labels(run_dir) if _HAS_REVIEW else {}

        # Build table rows (with escaping)
        rows_html = []
        for i, rec in enumerate(items, start=1):
            prompt = html.escape(rec.get("prompt",""))
            output = html.escape(rec.get("output",""))
            auto   = rec.get("auto") or {}
            auto_label = html.escape(str(auto.get("auto_label","")))
            auto_hits  = ",".join(auto.get("hits", [])) if isinstance(auto.get("hits"), list) else ""
            auto_hits  = html.escape(auto_hits)

            man = labels_map.get(int(rec.get("index", -1)), {})
            man_label = html.escape(str(man.get("label",""))) if isinstance(man, dict) else ""
            man_sev   = man.get("severity", None) if isinstance(man, dict) else None
            man_notes = html.escape(str(man.get("notes",""))) if isinstance(man, dict) and man.get("notes") else ""

            per = f"{rec.get('new_tokens','')} tok / {rec.get('gen_time_sec','')} s"
            per = html.escape(per)

            row = (
                f"<tr>"
                f"<td>{i}</td>"
                f"<td class='mono'>{prompt}</td>"
                f"<td class='mono'>{output}</td>"
                f"<td>{auto_label}{(' — ' + auto_hits) if auto_hits else ''}</td>"
                f"<td>{man_label}{(' (sev ' + str(man_sev) + ')') if (isinstance(man_sev,int)) else ''}"
                f"{(' — ' + man_notes) if man_notes else ''}</td>"
                f"<td>{per}</td>"
                f"</tr>"
            )
            rows_html.append(row)

        # Compose the HTML
        def esc(x): return html.escape("" if x is None else str(x))
        manual = (metrics.get("manual") or {})
        auto   = (metrics.get("auto") or {})
        html_out = f"""<!doctype html><html><meta charset='utf-8'><style>
        body{{font-family:system-ui,Arial;margin:2rem}}
        .mono{{font-family:ui-monospace,Menlo,Consolas;white-space:pre-wrap}}
        table{{border-collapse:collapse;width:100%}}
        th,td{{border:1px solid #ddd;padding:6px;vertical-align:top}}
        th{{background:#f6f6f6}}
        .box{{border:1px solid #ddd;padding:10px;border-radius:8px;margin:10px 0;background:#fafafa}}
        a{{color:#1a73e8;text-decoration:none}}
        a:hover{{text-decoration:underline}}
        </style><body>
        <p><a href="/">← Back to Home</a></p>
        <h2>Run Report</h2>
        <div class='box'><h3>Run Settings</h3><ul>
          <li><b>Preferred model:</b> {esc(settings.get('preferred_model',''))}</li>
          <li><b>Actual model:</b> {esc(settings.get('actual_model',''))}{' (fallback used)' if settings.get('fallback_used') else ''}</li>
          <li><b>Temperature:</b> {esc(settings.get('temperature',''))}</li>
          <li><b>Max new tokens:</b> {esc(settings.get('max_new_tokens',''))}</li>
          <li><b>Seed source:</b> {esc(settings.get('seed_source',''))}</li>
          <li><b>Prepared count:</b> {esc(settings.get('prepared_count',''))}</li>
          <li><b>Run folder:</b> runs/{esc(rid)}</li>
        </ul></div>
        <div class='box'><h3>Generation Metrics</h3><ul>
          <li><b>Items processed:</b> {esc(metrics.get('count',''))}</li>
          <li><b>Total new tokens:</b> {esc(metrics.get('total_new_tokens',''))}</li>
          <li><b>Generation time (sum):</b> {esc(metrics.get('gen_time_sec',''))} s</li>
          <li><b>Wall time:</b> {esc(metrics.get('wall_time_sec',''))} s</li>
          <li><b>Tokens/sec (avg):</b> {esc(metrics.get('tokens_per_sec',''))}</li>
        </ul></div>
        <div class='box'><h3>Auto Evaluation</h3><ul>
          <li><b>Total items:</b> {esc(auto.get('total_items',''))}</li>
          <li><b>Auto violations:</b> {esc(auto.get('violations',''))}</li>
          <li><b>Auto violation rate:</b> {esc(auto.get('violation_rate',''))}</li>
        </ul></div>
        <div class='box'><h3>Manual Labels</h3><ul>
          <li><b>Total labeled:</b> {esc(manual.get('total_labeled',''))} / {esc(manual.get('total_items',''))}</li>
          <li><b>Violations:</b> {esc(manual.get('violations',''))}</li>
          <li><b>Violation rate:</b> {esc(manual.get('violation_rate',''))}</li>
          <li><b>Avg severity:</b> {esc(manual.get('avg_severity',''))}</li>
        </ul></div>
        <h3>Samples</h3>
        <table>
          <tr><th>#</th><th>Prompt</th><th>Output</th><th>Auto Eval</th><th>Manual Label</th><th>Per-item</th></tr>
          {''.join(rows_html)}
        </table>
        </body></html>"""

        # Write out
        (run_dir / "report.html").write_text(html_out, encoding="utf-8")
        (run_dir / "index.html").write_text(html_out, encoding="utf-8")

        return jsonify({"ok": True, "report_path": f"runs/{rid}/report.html", "report_dl": f"dl/{rid}/report.html"}), 200
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.post("/zip_run")
def zip_run():
    try:
        payload = request.get_json(silent=True) or {}
        resolved = _resolve_run(payload.get("run_id"))
        if not resolved:
            return jsonify({"ok": False, "error": "no_runs"}), 400
        rid, run_dir = resolved
        if not run_dir.exists():
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
    """Stop the Flask server process (used by the Stop button)."""
    try:
        def _exit_soon():
            time.sleep(0.3); os._exit(0)
        threading.Thread(target=_exit_soon, daemon=True).start()
        return jsonify({"ok": True, "msg": "server will stop"}), 200
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

# ---------- review APIs (only if helper module is available) ----------
if _HAS_REVIEW:
    @app.get("/api/review_data")
    def api_review_data():
        run_id = request.args.get("run_id")
        start  = int(request.args.get("start", 0))
        limit  = int(request.args.get("limit", 1))
        resolved = _resolve_run(run_id)
        if not resolved:
            return jsonify({"ok": False, "error": "no_runs"}), 400
        rid, run_dir = resolved
        items = _load_interactions_rs(run_dir)
        labels = load_labels(run_dir)
        total = len(items)
        start = max(0, min(start, max(0, total-1)))
        end = min(total, start + max(1, limit))
        sub = []
        for i in range(start, end):
            row = dict(items[i])
            row["label"] = labels.get(i)
            sub.append(row)
        return jsonify({"ok": True, "run_id": rid, "total": total, "items": sub})

    @app.post("/api/review_save")
    def api_review_save():
        payload = request.get_json(silent=True) or {}
        run_id = payload.get("run_id")
        idx = payload.get("index")
        if not isinstance(idx, int):
            return jsonify({"ok": False, "error": "index required"}), 400
        resolved = _resolve_run(run_id)
        if not resolved:
            return jsonify({"ok": False, "error": "no_runs"}), 400
        rid, run_dir = resolved
        from src.app.review_store import save_label
        label = str(payload.get("label", "safe")).lower()
        severity = payload.get("severity", None)
        notes = payload.get("notes", None)
        out = save_label(run_dir, index=idx, label=label, severity=severity, notes=notes)
        return jsonify(out)

    @app.post("/api/recompute_metrics")
    def api_recompute_metrics():
        payload = request.get_json(silent=True) or {}
        run_id = payload.get("run_id")
        resolved = _resolve_run(run_id)
        if not resolved:
            return jsonify({"ok": False, "error": "no_runs"}), 400
        rid, run_dir = resolved
        out = recompute_and_write_metrics(run_dir)
        return jsonify(out)

def run_app(port: int = 8000):
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
