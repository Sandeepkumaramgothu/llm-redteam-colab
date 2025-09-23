from __future__ import annotations
from typing import Optional
import os          # 👈 we need this (was missing before)
import torch
from transformers import pipeline

def _pick_task(model_name: str) -> str:
    name = model_name.lower()
    if "t5" in name or "flan" in name:
        return "text2text-generation"
    return "text-generation"

class TargetModel:
    """
    Small wrapper around Hugging Face `pipeline`.
    - model_name: e.g., "google/flan-t5-small"
    - max_new_tokens: how many tokens the model may add to the output
    - temperature: how random the model is (0.0 = deterministic, 0.7 = a bit creative)
    """
    def __init__(self, model_name: str, max_new_tokens: int = 64, temperature: float = 0.7):
        self.model_name = model_name
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)

        task = _pick_task(model_name)
        device = -1 if os.environ.get('FORCE_CPU') == '1' else (0 if torch.cuda.is_available() else -1)  # 0 = use GPU if available; -1 = CPU

        # These prints help you SEE what is happening.
        print("[TargetModel] Initializing HF pipeline")
        print("  task           :", task)
        print("  model (Hub id) :", model_name)
        print("  device         :", "GPU:0" if device == 0 else "CPU")
        print("  HF cache dir   :", os.environ.get("HF_HOME", "/root/.cache/huggingface"))

        # This is the Hugging Face call:
        self.pipe = pipeline(task, model=model_name, device=device)

        print("[TargetModel] Loaded")
        try:
            print("  model path     :", getattr(self.pipe.model, "name_or_path", "(unknown)"))
            print("  tokenizer path :", getattr(self.pipe.tokenizer, "name_or_path", "(unknown)"))
        except Exception:
            pass

    def generate(self, prompt: str) -> str:
        """
        Send the prompt to the model and get text back.
        """
        outputs = self.pipe(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            truncation=True
        )
        # pipeline returns a list of dicts, e.g. [{"generated_text": "..."}]
        if isinstance(outputs, list) and outputs:
            first = outputs[0]
            for key in ("generated_text", "summary_text", "translation_text"):
                if key in first and isinstance(first[key], str):
                    return first[key]
        return str(outputs)
