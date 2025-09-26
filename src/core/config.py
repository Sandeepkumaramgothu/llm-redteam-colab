# src/core/config.py
from dataclasses import dataclass
import yaml, pathlib

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
class PathConfig:
    runs_dir: str
    seeds_path: str

@dataclass
class AppConfig:
    run: RunConfig
    models: ModelConfig
    paths: PathConfig

def load_config(path: str) -> 'AppConfig':
    p = pathlib.Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    run = RunConfig(
        seeds_per_iter=int(data["run"]["seeds_per_iter"]),
        iterations=int(data["run"]["iterations"]),
        max_new_tokens=int(data["run"]["max_new_tokens"]),
        temperature=float(data["run"]["temperature"]),
    )
    models = ModelConfig(target_model_name=str(data["models"]["target_model_name"]))
    paths  = PathConfig(runs_dir=str(data["paths"]["runs_dir"]), seeds_path=str(data["paths"]["seeds_path"]))
    return AppConfig(run=run, models=models, paths=paths)
