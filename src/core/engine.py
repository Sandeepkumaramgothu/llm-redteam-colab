from __future__ import annotations
from typing import List, Dict
import time
from src.core.utils import ensure_dir, now_ts, append_jsonl, write_json
from src.storage.data import load_seeds
from src.models.target import TargetModel
from src.metrics.reward import score_response

class IterationEngine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.run_id = now_ts()
        self.run_dir = f"{cfg.paths.runs_dir}/{self.run_id}"
        ensure_dir(self.run_dir)
        self.status = {"phase": "idle", "iter": 0, "percent": 0}

    def _set_status(self, phase: str, iter_n: int, percent: int):
        self.status = {"phase": phase, "iter": iter_n, "percent": percent}
        write_json(f"{self.run_dir}/status.json", self.status)

    def _save_config(self):
        write_json(
            f"{self.run_dir}/config_resolved.json",
            {
                "run": self.cfg.run.__dict__,
                "models": self.cfg.models.__dict__,
                "paths": self.cfg.paths.__dict__
            },
        )

    def run(self) -> str:
        self._save_config()

        # Load seeds
        self._set_status("loading_seeds", 0, 5)
        seeds = load_seeds(self.cfg.paths.seeds_path, limit=self.cfg.run.seeds_per_iter)

        # Init target model
        self._set_status("loading_model", 0, 10)
        model = TargetModel(
            self.cfg.models.target_model_name,
            max_new_tokens=self.cfg.run.max_new_tokens,
            temperature=self.cfg.run.temperature
        )

        # Week 1: single iteration
        interactions: List[Dict] = []
        n = max(1, len(seeds))
        for i, item in enumerate(seeds, start=1):
            self._set_status("generating", 1, int(10 + 80 * (i / n)))
            prompt = item["text"]
            output = model.generate(prompt)
            reward = score_response(output)
            interactions.append({
                "seed_id": item.get("id", i),
                "prompt": prompt,
                "output": output,
                "reward": reward
            })
            time.sleep(0.02)  # tiny pause for smoother progress updates

        # Save interactions + basic metrics
        append_jsonl(f"{self.run_dir}/interactions.jsonl", interactions)
        asr = sum(1 for it in interactions if it["reward"]["level"] >= 2) / len(interactions)
        metrics = {"asr": asr, "count": len(interactions)}
        write_json(f"{self.run_dir}/metrics.json", metrics)

        self._set_status("done", 1, 100)
        return self.run_id
