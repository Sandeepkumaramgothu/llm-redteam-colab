from __future__ import annotations
import threading, os, sys
from flask import Flask, jsonify, request, render_template
from src.core.config import load_config
from src.core.engine import IterationEngine
from src.core.utils import write_json
from src.app.report_utils import latest_run_dir, list_run_dirs, load_metrics, read_first_interactions

app = Flask(__name__, template_folder="templates")
_current = {"engine": None, "thread": None}

@app.get("/")
def home():
    return render_template("index.html")

@app.post("/start")
def start():
    if _current["thread"] and _current["thread"].is_alive():
        return jsonify({"ok": False, "error": "run_in_progress"}), 400

    cfg = load_config("configs/baseline.yaml")
    engine = IterationEngine(cfg)
    _current["engine"] = engine

    def work():
        try:
            engine.run()
        except Exception as e:
            write_json(f"{engine.run_dir}/error.json", {"error": str(e)})
            engine._set_status("error", engine.status.get("iter", 0), engine.status.get("percent", 0))

    t = threading.Thread(target=work, daemon=True)
    _current["thread"] = t
    t.start()
    return jsonify({"ok": True, "run_id": engine.run_id})

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
            "config": f"runs/{rid}/config_resolved.json",
            "metrics": f"runs/{rid}/metrics.json",
            "interactions": f"runs/{rid}/interactions.jsonl",
            "status": f"runs/{rid}/status.json"
        }
    })

@app.get("/runs")
def runs():
    dirs = list_run_dirs("runs")
    items = []
    for d in dirs:
        m = load_metrics(d)
        items.append({"run_dir": d, "asr": m.get("asr"), "count": m.get("count")})
    return jsonify({"runs": items})

@app.get("/report")
def report():
    run_id = request.args.get("run_id") or latest_run_dir("runs")
    if not run_id:
        return render_template("report.html", found=False, run_id=None, metrics={}, samples=[])
    metrics = load_metrics(run_id)
    samples = read_first_interactions(run_id, k=5)
    return render_template("report.html", found=True, run_id=run_id, metrics=metrics, samples=samples)

# ---- Robust shutdown (works even if Werkzeug hook is missing) ----
def _shutdown_werkzeug_if_present() -> bool:
    func = request.environ.get("werkzeug.server.shutdown")
    if func:
        func()
        return True
    return False

@app.post("/shutdown")
def shutdown():
    """
    Stop the Flask server PROCESS.
    Strategy:
      1) Try Werkzeug's own shutdown hook (if running under it)
      2) Otherwise, exit the process immediately.
    """
    ok = _shutdown_werkzeug_if_present()
    if ok:
        return jsonify({"ok": True, "msg": "werkzeug shutdown"})
    # Fallback: hard-exit the process
    os._exit(0)  # exits only the server process when launched via multiprocessing
    return jsonify({"ok": True, "msg": "process exit requested"})

def run_app(port: int = 8000):
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
