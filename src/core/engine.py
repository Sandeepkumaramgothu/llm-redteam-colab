# src/core/engine.py
from __future__ import annotations
import json, pathlib, time, shutil
from typing import List, Dict

from .config import AppConfig
from .utils import write_json, ensure_dir, ts_run_id
from .hf_textgen import TextGenerator

class IterationEngine:
    """
    Processes a batch of seeds and writes artifacts into runs/<run_id>/.
    Status now includes: phase, current, total, percent.
    We also snapshot the exact seeds used into the run folder:
      - seeds_used.jsonl  (copied from cfg.paths.seeds_path at run start)
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

        # Default status
        self.total = 0
        self.status = {"phase": "idle", "iter": 0, "current": 0, "total": 0, "percent": 0}
        write_json(str(self.run_dir / "status.json"), self.status)

        # Build generator (may fall back to open model if gated)
        self.textgen = TextGenerator(self.cfg.models.target_model_name, load_4bit=True)

        # Snapshot the seeds file we are about to use (for transparency)
        try:
            src = pathlib.Path(self.cfg.paths.seeds_path)
            dst = self.run_dir / "seeds_used.jsonl"
            if src.exists():
                shutil.copyfile(str(src), str(dst))
        except Exception:
            pass

    def _set_status(self, phase: str, current: int, pct: int):
        """Update status with current item and percent; keep total and iter (alias)."""
        self.status.update({
            "phase": phase,
            "iter": current,     # legacy name (kept so old UI still shows numbers)
            "current": current,  # clearer name
            "total": self.total, # how many items in this run
            "percent": pct,
        })
        write_json(str(self.run_dir / "status.json"), self.status)

    def _read_seeds(self) -> List[Dict]:
        """Read prepared seeds (JSONL) into a list of dicts with 'text'."""
        seeds_path = pathlib.Path(self.cfg.paths.seeds_path)
        out: List[Dict] = []
        if not seeds_path.exists():
            return out
        with open(seeds_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict) and isinstance(obj.get("text"), str):
                        out.append(obj)
                except Exception:
                    pass
        return out

    def run(self):
        """Generate outputs for each seed; record per-item time/tokens and summary metrics."""
        # Load seeds and decide how many to process
        seeds = self._read_seeds()
        self.total = min(len(seeds), int(self.cfg.run.seeds_per_iter))

        # Start status
        self._set_status("running", 0, 0)

        # If no work, finish gracefully
        if self.total == 0:
            write_json(str(self.run_dir / "metrics.json"), {"count": 0})
            self._set_status("done", 0, 100)
            return

        interactions_path = self.run_dir / "interactions.jsonl"
        total_new_tokens = 0.0
        total_gen_time   = 0.0
        run_start = time.perf_counter()

        # Main loop
        for i in range(self.total):
            prompt = seeds[i].get("text", "")

            # Generate + stats (time and token count)
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

            # Append one line to interactions.jsonl
            rec = {
                "index": i,
                "prompt": prompt,
                "output": output,
                "new_tokens": int(new_toks),
                "gen_time_sec": round(time_taken, 6),
                "reward": {"score": "", "level": ""},
            }
            with open(interactions_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            # Update progress (current item = i+1)
            pct = int((i + 1) * 100 / self.total)
            self._set_status("running", i + 1, pct)

        # Summary metrics
        run_wall = time.perf_counter() - run_start
        tok_per_sec = (total_new_tokens / total_gen_time) if total_gen_time > 0 else None

        write_json(str(self.run_dir / "metrics.json"), {
            "count": int(self.total),
            "total_new_tokens": int(total_new_tokens),
            "gen_time_sec": round(total_gen_time, 6),
            "wall_time_sec": round(run_wall, 6),
            "tokens_per_sec": round(tok_per_sec, 4) if tok_per_sec else None
        })

        # Done
        self._set_status("done", self.total, 100)
