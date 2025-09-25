from __future__ import annotations
import os, json, html, sys
from typing import List, Dict, Any

def read_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def read_jsonl(path: str, k: int | None = None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if isinstance(rec, dict):
                    out.append(rec)
            except Exception:
                # ignore malformed lines
                pass
            if k is not None and len(out) >= k:
                break
    return out

def _esc(x: Any) -> str:
    if x is None:
        return ""
    if not isinstance(x, str):
        x = str(x)
    return html.escape(x)

def make_html(run_dir: str, max_samples: int = 20) -> str:
    # tolerate missing metrics.json
    metrics = read_json(os.path.join(run_dir, "metrics.json"))
    if not isinstance(metrics, dict):
        metrics = {}
    count = metrics.get("count", 0)
    asr = metrics.get("asr", None)

    # read a handful of interactions, tolerate missing keys
    samples = read_jsonl(os.path.join(run_dir, "interactions.jsonl"), k=max_samples)

    rows = []
    for i, s in enumerate(samples, start=1):
        prompt = _esc(s.get("prompt", ""))
        output = _esc(s.get("output", ""))[:2000]
        # reward might be missing or null
        reward = s.get("reward") or {}
        score = reward.get("score", "")
        level = reward.get("level", "")
        rows.append(f"""
        <tr>
          <td>{i}</td>
          <td class='mono'>{prompt}</td>
          <td class='mono'>{output}{'…' if len(output) >= 2000 else ''}</td>
          <td><span class='pill'>score: {_esc(score)}</span> <span class='pill'>level: {_esc(level)}</span></td>
        </tr>
        """)

    html_doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Run Report — {_esc(run_dir)}</title>
  <style>
    body {{ font-family: system-ui, Arial, sans-serif; margin: 2rem; }}
    h2 {{ margin-bottom: 0.25rem; }}
    .muted {{ color: #666; font-size: 0.95rem; }}
    .mono {{ font-family: ui-monospace, Menlo, Consolas, monospace; white-space: pre-wrap; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 1rem; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
    th {{ background: #f6f6f6; }}
    .pill {{ display: inline-block; padding: 2px 8px; border-radius: 999px; background: #eef; }}
    a {{ color: #1a73e8; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .files a {{ margin-right: 10px; }}
  </style>
</head>
<body>
  <h2>Run Report</h2>
  <div class="muted">Run folder: <span class="mono">{_esc(run_dir)}</span></div>

  <h3>Summary</h3>
  <ul>
    <li><b>Count</b>: {_esc(count)}</li>
    <li><b>ASR</b>: {_esc(asr)}</li>
  </ul>

  <h3>Artifacts</h3>
  <div class="files">
    <a href="config_resolved.json">config_resolved.json</a>
    <a href="metrics.json">metrics.json</a>
    <a href="interactions.jsonl">interactions.jsonl</a>
    <a href="status.json">status.json</a>
  </div>

  <h3>First samples (up to {max_samples})</h3>
  <table>
    <tr><th>#</th><th>Prompt</th><th>Model Output</th><th>Reward</th></tr>
    {''.join(rows)}
  </table>
</body>
</html>"""
    return html_doc

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python scripts/mk_report.py runs/2025xxxx-xxxxxx [max_samples]")
        raise SystemExit(2)
    run_dir = sys.argv[1]
    max_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    out = make_html(run_dir, max_samples=max_samples)
    out_path = os.path.join(run_dir, "report.html")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(out)
    print("Wrote:", out_path)
