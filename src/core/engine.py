# src/core/engine.py
from __future__ import annotations
import json, pathlib
from typing import List, Dict
from .config import AppConfig
from .utils import write_json, ensure_dir, ts_run_id
from .hf_textgen import TextGenerator

class IterationEngine:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.run_id  = ts_run_id()
        self.run_dir = pathlib.Path(self.cfg.paths.runs_dir) / self.run_id
        ensure_dir(str(self.run_dir))

        # Save resolved config
        write_json(str(self.run_dir / "config_resolved.json"), {
            "run": {
                "seeds_per_iter": self.cfg.run.seeds_per_iter,
                "iterations": self.cfg.run.iterations,
                "max_new_tokens": self.cfg.run.max_new_tokens,
                "temperature": self.cfg.run.temperature,
            },
            "models": {"target_model_name": self.cfg.models.target_model_name},
            "paths":  {"runs_dir": self.cfg.paths.runs_dir, "seeds_path": self.cfg.paths.seeds_path}
        })

        self.status = {"phase": "idle", "iter": 0, "percent": 0}
        write_json(str(self.run_dir / "status.json"), self.status)

        # Text generator with 4-bit + fallbacks
        self.textgen = TextGenerator(self.cfg.models.target_model_name, load_4bit=True)

    def _set_status(self, phase: str, it: int, pct: int):
        self.status.update({"phase": phase, "iter": it, "percent": pct})
        write_json(str(self.run_dir / "status.json"), self.status)

    def _read_seeds(self) -> List[Dict]:
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
        self._set_status("running", 0, 0)
        seeds = self._read_seeds()
        total = min(len(seeds), int(self.cfg.run.seeds_per_iter))
        if total == 0:
            write_json(str(self.run_dir / "metrics.json"), {"count": 0})
            self._set_status("done", 0, 100)
            return

        interactions_path = self.run_dir / "interactions.jsonl"
        for i in range(total):
            prompt = seeds[i].get("text", "")
            output = self.textgen.generate(prompt, max_new_tokens=int(self.cfg.run.max_new_tokens), temperature=float(self.cfg.run.temperature))
            rec = {"index": i, "prompt": prompt, "output": output, "reward": {"score": "", "level": ""}}
            with open(interactions_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            pct = int((i + 1) * 100 / total)
            self._set_status("running", i + 1, pct)

        write_json(str(self.run_dir / "metrics.json"), {"count": total})
        self._set_status("done", total, 100)
