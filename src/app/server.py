# src/app/server.py
from __future__ import annotations

import os, sys, json, csv, glob, time, html, pathlib, threading
from zipfile import ZipFile, ZIP_DEFLATED
from typing import Optional, Tuple, List, Dict, Any

from flask import Flask, jsonify, request, render_template, send_from_directory

ROOT = pathlib.Path(__file__).resolve().parents[2]
CFG_PATH = ROOT / "configs" / "baseline.yaml"
RUNS_DIR = ROOT / "runs"
TEMPLATES_DIR = ROOT / "src" / "app" / "templates"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.config import load_config
from src.core.engine import IterationEngine
from src.core.utils import write_json
from src.data.seed_loader import prepare_seeds, read_local_auto, read_local_json, read_local_jsonl

_HAS_REVIEW = False
try:
    from src.app.review_store import (
        load_interactions as _load_interactions_rs,
        load_labels, save_label, recompute_and_write_metrics,
    )
    _HAS_REVIEW = True
except Exception:
    def _load_interactions_rs(run_dir: pathlib.Path): return []
    def load_labels(run_dir: pathlib.Path): return {}
    def save_label(run_dir: pathlib.Path, **kwargs): return {"ok": False, "error": "review helpers missing"}
    def recompute_and_write_metrics(run_dir: pathlib.Path): return {"ok": False, "error": "review helpers missing"}

# ---- Simple on-demand text generation cache (so /api/rerun_item doesn't reload each time)
_GEN_CACHE = {"name": None, "pipe": None, "task": None}

def _ensure_gen(model_name: str):
    """
    Build or reuse a transformers pipeline for 'text-generation' or 'text2text-generation'.
    Very small convenience so per-item re-runs are snappy.
    """
    try:
        import torch
        from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, pipeline
    except Exception as e:
        raise RuntimeError(f"transformers/torch missing: {e}")

    if _GEN_CACHE["name"] == model_name and _GEN_CACHE["pipe"] is not None:
        return _GEN_CACHE["pipe"], _GEN_CACHE["task"]

    # Decide the task (causal vs. seq2seq) by model config name
    cfg = AutoConfig.from_pretrained(model_name, token=os.environ.get("HF_TOKEN", None))
    archs = [a.lower() for a in (cfg.architectures or [])]
    is_t5 = any("t5" in a for a in archs)
    task = "text2text-generation" if is_t5 else "text-generation"

    tok = AutoTokenizer.from_pretrained(model_name, token=os.environ.get("HF_TOKEN", None), use_fast=True)
    if task == "text2text-generation":
        mdl = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, token=os.environ.get("HF_TOKEN", None), device_map="auto"
        )
    else:
        mdl = AutoModelForCausalLM.from_pretrained(
            model_name, token=os.environ.get("HF_TOKEN", None), device_map="auto"
        )

    pipe = pipeline(task, model=mdl, tokenizer=tok, device_map="auto")
    _GEN_CACHE["name"] = model_name
    _GEN_CACHE["pipe"] = pipe
    _GEN_CACHE["task"] = task
    return pipe, task

app = Flask(__name__, template_folder=str(TEMPLATES_DIR))
_current = {"engine": None, "thread": None}

def _latest_run_id() -> Optional[str]:
    RUNS_DIR.mkdir(exist_ok=True)
    dirs = sorted([p.name for p in RUNS_DIR.glob("*") if p.is_dir()])
    return dirs[-1] if dirs else None

def _resolve_run(run_identifier: Optional[str]):
    if not run_identifier:
        rid = _latest_run_id()
        if not rid: return None
        return rid, RUNS_DIR / rid
    rid = run_identifier
    if rid.startswith("runs/"): rid = rid.split("/", 1)[1]
    p = RUNS_DIR / rid
    return (rid, p) if p.exists() and p.is_dir() else None

def _read_interactions(run_dir: pathlib.Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    p = run_dir / "interactions.jsonl"
    out = []
    if not p.exists(): return out
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line: continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
            if isinstance(limit, int) and i >= limit:
                break
    return out

def _clamp(val, lo, hi, default):
    try: x = float(val)
    except Exception: return default
    return max(lo, min(hi, x))

@app.get("/")
def home():
    return render_template("index.html")

@app.get("/review")
def review_page():
    return render_template("review.html")

@app.get("/runs")
def runs_page():
    return render_template("runs.html")

@app.get("/dashboard")
def dashboard_page():
    return render_template("dashboard.html")

@app.get("/inspect")
def inspect_page():
    return render_template("inspect.html")

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
        try: settings = json.load(rp.open("r", encoding="utf-8"))
        except Exception: settings = {}
    if mt.exists():
        try: metrics = json.load(mt.open("r", encoding="utf-8"))
        except Exception: metrics = {}

    samples = _read_interactions(run_dir, limit=None)
    labels_map = load_labels(run_dir) if _HAS_REVIEW else {}
    for s in samples:
        try:
            s["label"] = labels_map.get(int(s.get("index", -1)))
        except Exception:
            pass

    return render_template("report.html", found=True, run_id=f"runs/{run_id}", settings=settings, metrics=metrics, samples=samples)

@app.post("/start")
def start():
    if _current["thread"] and _current["thread"].is_alive():
        return jsonify({"ok": False, "error": "run_in_progress"}), 400
    try:
        cfg = load_config(str(CFG_PATH))
    except Exception as e:
        return jsonify({"ok": False, "error": f"config_load_failed: {e}"}), 500

    payload = request.get_json(silent=True) or {}

    pm = payload.get("preferred_model")
    if isinstance(pm, str) and pm.strip():
        cfg.models.target_model_name = pm.strip()

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
    max_tokens  = int(_clamp(payload.get("max_new_tokens", cfg.run.max_new_tokens), 1, 1024, cfg.run.max_new_tokens))

    shuffle_seed  = int(payload.get("seed", time.time_ns() % 1_000_000))
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

    cfg.run.temperature    = temperature
    cfg.run.max_new_tokens = max_tokens

    try:
        engine = IterationEngine(cfg)
    except Exception as e:
        return jsonify({"ok": False, "error": f"model_init_failed: {e}"}), 500

    _current["engine"] = engine

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

# -------- Review APIs --------
@app.get("/api/review_data")
def api_review_data():
    run_id = request.args.get("run_id")
    start  = int(request.args.get("start", 0))
    limit  = int(request.args.get("limit", 5))
    resolved = _resolve_run(run_id)
    if not resolved: return jsonify({"ok": False, "error": "no_runs"}), 400
    rid, run_dir = resolved
    items = _load_interactions_rs(run_dir)
    labels = load_labels(run_dir) if _HAS_REVIEW else {}
    total = len(items)
    start = max(0, min(start, max(0, total-1)))
    end = min(total, start + max(1, limit))
    sub = []
    for i in range(start, end):
        row = dict(items[i])
        row["label"] = labels.get(int(row.get("index", i)))
        sub.append(row)
    return jsonify({"ok": True, "run_id": rid, "total": total, "items": sub})

@app.post("/api/review_save")
def api_review_save():
    payload = request.get_json(silent=True) or {}
    run_id = payload.get("run_id")
    idx = payload.get("index")
    if not isinstance(idx, int): return jsonify({"ok": False, "error": "index required"}), 400
    resolved = _resolve_run(run_id)
    if not resolved: return jsonify({"ok": False, "error": "no_runs"}), 400
    rid, run_dir = resolved
    label = str(payload.get("label", "")).lower()
    sev = payload.get("severity", None)
    if isinstance(sev, str) and sev.lower() == "none": sev = None
    notes = payload.get("notes", None)
    out = save_label(run_dir, index=idx, label=label, severity=sev, notes=notes)
    return jsonify(out)

@app.post("/api/recompute_metrics")
def api_recompute_metrics():
    payload = request.get_json(silent=True) or {}
    run_id = payload.get("run_id")
    resolved = _resolve_run(run_id)
    if not resolved: return jsonify({"ok": False, "error": "no_runs"}), 400
    rid, run_dir = resolved
    out = recompute_and_write_metrics(run_dir)
    return jsonify(out)

# -------- Inspect APIs --------
def _revisions_path(run_dir: pathlib.Path, index: int) -> pathlib.Path:
    d = run_dir / "revisions"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"index_{index}.jsonl"

def _read_revisions(run_dir: pathlib.Path, index: int) -> List[Dict[str, Any]]:
    p = _revisions_path(run_dir, index)
    out = []
    if not p.exists(): return out
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: out.append(json.loads(line))
            except Exception: pass
    return out

@app.get("/api/item")
def api_item():
    rid_q = request.args.get("run_id")
    idx = int(request.args.get("index", 0))
    resolved = _resolve_run(rid_q)
    if not resolved: return jsonify({"ok": False, "error": "no_runs"}), 400
    rid, run_dir = resolved

    items = _load_interactions_rs(run_dir)
    if not items or idx < 0 or idx >= len(items):
        return jsonify({"ok": False, "error": "index_out_of_range", "total": len(items)}), 400

    row = dict(items[idx])
    labels = load_labels(run_dir) if _HAS_REVIEW else {}
    row["label"] = labels.get(int(row.get("index", idx)))
    hist = _read_revisions(run_dir, idx)

    return jsonify({"ok": True, "run_id": rid, "item": row, "history": hist})

@app.post("/api/rerun_item")
def api_rerun_item():
    payload = request.get_json(silent=True) or {}
    rid_q = payload.get("run_id")
    idx = int(payload.get("index", 0))
    model_override = payload.get("preferred_model")
    temperature = float(payload.get("temperature", 0.7))
    max_new = int(payload.get("max_new_tokens", 128))

    resolved = _resolve_run(rid_q)
    if not resolved: return jsonify({"ok": False, "error": "no_runs"}), 400
    rid, run_dir = resolved

    items = _load_interactions_rs(run_dir)
    if not items or idx < 0 or idx >= len(items):
        return jsonify({"ok": False, "error": "index_out_of_range", "total": len(items)}), 400

    prompt = str(items[idx].get("prompt", ""))

    # Decide model: override if provided else config default
    try:
        cfg = load_config(str(CFG_PATH))
    except Exception as e:
        return jsonify({"ok": False, "error": f"config_load_failed: {e}"}), 500
    model_name = (model_override.strip() if isinstance(model_override, str) and model_override.strip() else cfg.models.target_model_name)

    # Build/reuse pipeline
    try:
        pipe, task = _ensure_gen(model_name)
    except Exception as e:
        return jsonify({"ok": False, "error": f"gen_init_failed: {e}"}), 500

    # Run generation
    try:
        if task == "text2text-generation":
            results = pipe(prompt, max_new_tokens=max_new, temperature=temperature, do_sample=(temperature>0))
            text = results[0]["generated_text"]
        else:
            results = pipe(prompt, max_new_tokens=max_new, temperature=temperature, do_sample=(temperature>0))
            # For causal LMs, HF may return full prompt+output; take 'generated_text' and strip leading prompt if duplicated
            text = results[0]["generated_text"]
            if text.startswith(prompt):
                text = text[len(prompt):]
    except Exception as e:
        return jsonify({"ok": False, "error": f"generation_failed: {e}"}), 500

    # Record revision
    rec = {
        "when": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
        "temperature": round(float(temperature), 3),
        "max_new_tokens": int(max_new),
        "output": text,
    }
    rp = _revisions_path(run_dir, idx)
    with rp.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return jsonify({"ok": True, "index": idx, "revision": rec})

# -------- Exports / downloads --------
@app.post("/export_csv")
def export_csv():
    try:
        payload = request.get_json(silent=True) or {}
        resolved = _resolve_run(payload.get("run_id"))
        if not resolved: return jsonify({"ok": False, "error": "no_runs"}), 400
        rid, run_dir = resolved
        items = _read_interactions(run_dir, limit=None)
        labels_map = load_labels(run_dir) if _HAS_REVIEW else {}
        csv_path = run_dir / "interactions_labeled.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["index","prompt","output","new_tokens","gen_time_sec","auto_label","auto_hits","manual_label","manual_severity","manual_notes"])
            for rec in items:
                auto = rec.get("auto") or {}
                man  = labels_map.get(int(rec.get("index",-1))) or {}
                w.writerow([
                    rec.get("index",""),
                    rec.get("prompt",""),
                    rec.get("output",""),
                    rec.get("new_tokens",""),
                    rec.get("gen_time_sec",""),
                    (auto.get("auto_label","") if isinstance(auto,dict) else ""),
                    ",".join(auto.get("hits",[])) if isinstance(auto,dict) and isinstance(auto.get("hits",[]), list) else "",
                    (man.get("label","") if isinstance(man,dict) else ""),
                    (man.get("severity","") if isinstance(man,dict) else ""),
                    (man.get("notes","") if isinstance(man,dict) else ""),
                ])
        return jsonify({"ok": True, "csv_path": f"runs/{rid}/interactions_labeled.csv", "csv_dl": f"dl/{rid}/interactions_labeled.csv"}), 200
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.post("/export_report")
def export_report():
    try:
        payload = request.get_json(silent=True) or {}
        resolved = _resolve_run(payload.get("run_id"))
        if not resolved: return jsonify({"ok": False, "error": "no_runs"}), 400
        rid, run_dir = resolved

        settings, metrics = {}, {}
        rp = run_dir / "run_meta.json"
        mt = run_dir / "metrics.json"
        if rp.exists():
            try: settings = json.load(rp.open("r", encoding="utf-8"))
            except Exception: settings = {}
        if mt.exists():
            try: metrics = json.load(mt.open("r", encoding="utf-8"))
            except Exception: metrics = {}

        items = _read_interactions(run_dir, limit=None)
        labels_map = load_labels(run_dir) if _HAS_REVIEW else {}

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
            rows_html.append(
                f"<tr><td>{i}</td><td class='mono'>{prompt}</td><td class='mono'>{output}</td>"
                f"<td>{auto_label}{(' — ' + auto_hits) if auto_hits else ''}</td>"
                f"<td>{man_label}{(' (sev ' + str(man_sev) + ')') if (isinstance(man_sev,int)) else ''}{(' — ' + man_notes) if man_notes else ''}</td>"
                f"<td>{per}</td></tr>"
            )

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
        a{{color:#1a73e8;text-decoration:none}} a:hover{{text-decoration:underline}}
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
        (run_dir / "report.html").write_text(html_out, encoding="utf-8")
        (run_dir / "index.html").write_text(html_out, encoding="utf-8")
        return jsonify({"ok": True, "report_path": f"runs/{rid}/report.html", "report_dl": f"dl/{rid}/report.html"}), 200
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.post("/export_all_csv")
def export_all_csv():
    try:
        RUNS_DIR.mkdir(exist_ok=True)
        out_path = RUNS_DIR / "_summary_all_runs.csv"
        rows = []
        for d in sorted([p for p in RUNS_DIR.glob("*") if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True):
            rid = d.name
            when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(d.stat().st_mtime))
            metrics, meta = {}, {}
            mp = d / "metrics.json"
            rp = d / "run_meta.json"
            if mp.exists():
                try: metrics = json.load(mp.open("r", encoding="utf-8"))
                except Exception: metrics = {}
            if rp.exists():
                try: meta = json.load(rp.open("r", encoding="utf-8"))
                except Exception: meta = {}
            auto   = metrics.get("auto") or {}
            manual = metrics.get("manual") or {}
            rows.append({
                "run_id": rid, "when": when,
                "model": meta.get("actual_model") or meta.get("preferred_model") or "",
                "count": metrics.get("count"),
                "total_new_tokens": metrics.get("total_new_tokens"),
                "auto_violations": auto.get("violations"),
                "auto_violation_rate": auto.get("violation_rate"),
                "manual_total_labeled": manual.get("total_labeled"),
                "manual_violations": manual.get("violations"),
                "manual_violation_rate": manual.get("violation_rate"),
                "manual_avg_severity": manual.get("avg_severity"),
            })
        with out_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "run_id","when","model","count","total_new_tokens",
                "auto_violations","auto_violation_rate",
                "manual_total_labeled","manual_violations","manual_violation_rate","manual_avg_severity"
            ])
            for r in rows:
                w.writerow([
                    r["run_id"], r["when"], r["model"], r["count"], r["total_new_tokens"],
                    r["auto_violations"], r["auto_violation_rate"],
                    r["manual_total_labeled"], r["manual_violations"], r["manual_violation_rate"], r["manual_avg_severity"]
                ])
        return jsonify({"ok": True, "csv_path": "runs/_summary_all_runs.csv", "csv_dl": "dl/_summary_all_runs.csv"}), 200
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.post("/zip_run")
def zip_run():
    try:
        payload = request.get_json(silent=True) or {}
        resolved = _resolve_run(payload.get("run_id"))
        if not resolved: return jsonify({"ok": False, "error": "no_runs"}), 400
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
    try:
        def _exit(): time.sleep(0.3); os._exit(0)
        threading.Thread(target=_exit, daemon=True).start()
        return jsonify({"ok": True, "msg": "server will stop"}), 200
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.get("/runs/<path:subpath>")
def serve_runs(subpath: str):
    return send_from_directory(str(RUNS_DIR), subpath, as_attachment=False)

@app.get("/dl/<path:subpath>")
def download_runs(subpath: str):
    return send_from_directory(str(RUNS_DIR), subpath, as_attachment=True)

@app.get("/api/runs")
def api_runs():
    RUNS_DIR.mkdir(exist_ok=True)
    rows = []
    for d in sorted([p for p in RUNS_DIR.glob("*") if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True):
        rid = d.name
        when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(d.stat().st_mtime))
        metrics, meta = {}, {}
        mp = d / "metrics.json"
        rp = d / "run_meta.json"
        if mp.exists():
            try: metrics = json.load(mp.open("r", encoding="utf-8"))
            except Exception: metrics = {}
        if rp.exists():
            try: meta = json.load(rp.open("r", encoding="utf-8"))
            except Exception: meta = {}
        rows.append({
            "run_id": rid,
            "when": when,
            "model": meta.get("actual_model") or meta.get("preferred_model") or "",
            "count": metrics.get("count"),
            "total_new_tokens": metrics.get("total_new_tokens"),
            "auto": metrics.get("auto") or {},
            "manual": metrics.get("manual") or {},
        })
    return jsonify({"ok": True, "runs": rows})

def run_app(port: int = 8000):
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
