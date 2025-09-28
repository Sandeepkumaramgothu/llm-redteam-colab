# src/core/engine.py
from __future__ import annotations
import json, pathlib, time, shutil
from typing import List, Dict

from .config import AppConfig
from .utils import write_json, ensure_dir, ts_run_id
from .hf_textgen import TextGenerator
from .eval import evaluate_output  # ← NEW: auto evaluator

class IterationEngine:
    """
    Processes a batch of seeds and writes artifacts into runs/<run_id>/.
    Now also performs a lightweight auto evaluation on each output.
    """
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.run_id  = ts_run_id()
        self.run_dir = pathlib.Path(self.cfg.paths.runs_dir) / self.run_id
        ensure_dir(str(self.run_dir))

        # Save resolved config
        write_json(str(self.run_dir / "config_resolved.json"), {
            "run": {
                "seeds_per_iter": self.cfg.run.seeds_per_iter,
                "iterations":     self.cfg.run.iterations,
                "max_new_tokens": self.cfg.run.max_new_tokens,
                "temperature":    self.cfg.run.temperature,
            },
            "models": {"target_model_name": self.cfg.models.target_model_name},
            "paths":  {"runs_dir": self.cfg.paths.runs_dir, "seeds_path": self.cfg.paths.seeds_path}
        })

        # Status defaults
        self.total = 0
        self.status = {"phase": "idle", "iter": 0, "current": 0, "total": 0, "percent": 0}
        write_json(str(self.run_dir / "status.json"), self.status)

        # Text generator (may fallback)
        self.textgen = TextGenerator(self.cfg.models.target_model_name, load_4bit=True)

        # Snapshot exact seeds used
        try:
            src = pathlib.Path(self.cfg.paths.seeds_path)
            dst = self.run_dir / "seeds_used.jsonl"
            if src.exists():
                shutil.copyfile(str(src), str(dst))
        except Exception:
            pass

    def _set_status(self, phase: str, current: int, pct: int):
        self.status.update({
            "phase": phase,
            "iter": current,
            "current": current,
            "total": self.total,
            "percent": pct,
        })
        write_json(str(self.run_dir / "status.json"), self.status)

    def _read_seeds(self) -> List[Dict]:
        seeds_path = pathlib.Path(self.cfg.paths.seeds_path)
        out: List[Dict] = []
        if not seeds_path.exists():
            return out
        with open(seeds_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict) and isinstance(obj.get("text"), str):
                        out.append(obj)
                except Exception:
                    pass
        return out

    def run(self):
        seeds = self._read_seeds()
        self.total = min(len(seeds), int(self.cfg.run.seeds_per_iter))
        self._set_status("running", 0, 0)

        if self.total == 0:
            write_json(str(self.run_dir / "metrics.json"), {"count": 0, "auto": {"total_items": 0, "violations": 0, "violation_rate": 0.0}})
            self._set_status("done", 0, 100)
            return

        interactions_path = self.run_dir / "interactions.jsonl"
        total_new_tokens = 0.0
        total_gen_time   = 0.0
        auto_violations  = 0
        run_start = time.perf_counter()

        for i in range(self.total):
            prompt = seeds[i].get("text", "")

            # Generate
            stats = self.textgen.generate_stats(
                prompt=prompt,
                max_new_tokens=int(self.cfg.run.max_new_tokens),
                temperature=float(self.cfg.run.temperature),
            )
            output     = stats["text"]
            new_toks   = float(stats["new_tokens"])
            time_taken = float(stats["time_sec"])

            total_new_tokens += new_toks
            total_gen_time   += time_taken

            # NEW: auto evaluation (simple rule-based)
            auto = evaluate_output(prompt, output)
            if auto.get("auto_label") == "violation":
                auto_violations += 1

            # Append record
            rec = {
                "index": i,
                "prompt": prompt,
                "output": output,
                "new_tokens": int(new_toks),
                "gen_time_sec": round(time_taken, 6),
                "reward": {"score": "", "level": ""},  # reserved for future
                "auto": auto,  # ← store auto eval alongside
            }
            with open(interactions_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            pct = int((i + 1) * 100 / self.total)
            self._set_status("running", i + 1, pct)

        # Summary metrics
        run_wall = time.perf_counter() - run_start
        tok_per_sec = (total_new_tokens / total_gen_time) if total_gen_time > 0 else None
        auto_rate = (auto_violations / self.total) if self.total > 0 else 0.0

        write_json(str(self.run_dir / "metrics.json"), {
            "count": int(self.total),
            "total_new_tokens": int(total_new_tokens),
            "gen_time_sec": round(total_gen_time, 6),
            "wall_time_sec": round(run_wall, 6),
            "tokens_per_sec": round(tok_per_sec, 4) if tok_per_sec else None,
            "auto": {  # ← NEW: auto metrics summary
                "total_items": int(self.total),
                "violations": int(auto_violations),
                "violation_rate": round(auto_rate, 4),
            }
        })

        self._set_status("done", self.total, 100)
