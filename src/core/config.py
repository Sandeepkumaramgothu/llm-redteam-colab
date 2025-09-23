from __future__ import annotations
from dataclasses import dataclass
import yaml

@dataclass
class RunConfig:
    seeds_per_iter: int
    iterations: int
    max_new_tokens: int
    temperature: float

@dataclass
class ModelConfig:
    target_model_name: str

@dataclass
class PathsConfig:
    runs_dir: str
    seeds_path: str

@dataclass
class AppConfig:
    run: RunConfig
    models: ModelConfig
    paths: PathsConfig

def load_config(path: str) -> AppConfig:
    """Read configs/baseline.yaml and return a typed AppConfig."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    run = RunConfig(**raw["run"])
    models = ModelConfig(**raw["models"])
    paths = PathsConfig(**raw["paths"])
    return AppConfig(run=run, models=models, paths=paths)
