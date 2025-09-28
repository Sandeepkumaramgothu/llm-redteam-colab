# src/core/hf_textgen.py
"""
Robust text generation helper for Colab:
- Tries your preferred model, then fallbacks:
    1) microsoft/Phi-3-mini-4k-instruct  (small, fast, open)
    2) Qwen/Qwen2.5-7B-Instruct          (bigger, open)
    3) TinyLlama/TinyLlama-1.1B-Chat-v1.0 (very small)
- Uses 4-bit on GPU when possible. Falls back to fp16 on GPU or CPU.
- generate_stats(prompt, max_new_tokens, temperature) returns:
    {"text": str, "new_tokens": int, "time_sec": float}
"""

from typing import Optional, List, Dict, Any
import time, warnings
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)

_FALLBACKS = [
    "microsoft/Phi-3-mini-4k-instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
]

def _is_t5(name: str) -> bool:
    return "t5" in (name or "").lower()

class TextGenerator:
    def __init__(self, model_name: str, load_4bit: bool = True, fallbacks: Optional[List[str]] = None):
        self.device = 0 if torch.cuda.is_available() else -1
        self.load_4bit = bool(load_4bit)
        self.fallbacks = list(fallbacks) if fallbacks else list(_FALLBACKS)

        last_err = None
        for candidate in [model_name] + self.fallbacks:
            try:
                self._try_load(candidate)
                self.model_name = candidate
                if candidate != model_name:
                    warnings.warn(f"Primary model '{model_name}' unavailable; using fallback '{candidate}'.")
                return
            except Exception as e:
                last_err = e
                continue
        raise RuntimeError(f"Could not load any model (tried { [model_name] + self.fallbacks }): {last_err}")

    def _try_load(self, model_name: str):
        # T5-style models: use text2text pipeline (kept for completeness)
        if _is_t5(model_name):
            self.pipe = pipeline(
                "text2text-generation",
                model=model_name,
                tokenizer=model_name,
                device=self.device,
            )
            self.tok = self.pipe.tokenizer
            self.model = self.pipe.model
            self.is_causal = False
            return

        # Causal chat models: build tokenizer, then try several load strategies.
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        # Strategy A: 4-bit on GPU (best VRAM savings)
        if torch.cuda.is_available() and self.load_4bit:
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    device_map="auto",
                    torch_dtype=torch.float16,   # fp16 works broadly on Colab GPUs
                )
                self._post_init()
                return
            except Exception as e:
                last_a = e  # keep and try next

        # Strategy B: fp16 on GPU (no 4-bit)
        if torch.cuda.is_available():
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",
                    torch_dtype=torch.float16,
                )
                self._post_init()
                return
            except Exception as e:
                last_b = e

        # Strategy C: CPU (slow, but always works)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self._post_init()

    def _post_init(self):
        if self.tok.pad_token is None and self.tok.eos_token is not None:
            self.tok.pad_token = self.tok.eos_token
        self.is_causal = True

    def generate_stats(self, prompt: str, max_new_tokens: int = 128, temperature: float = 0.7) -> Dict[str, Any]:
        # T5 path
        if not self.is_causal:
            t0 = time.perf_counter()
            out = self.pipe(prompt, max_new_tokens=int(max_new_tokens), temperature=float(temperature))
            dt = time.perf_counter() - t0
            text = out[0]["generated_text"]
            new_tokens = len(self.tok.encode(text))
            return {"text": text, "new_tokens": new_tokens, "time_sec": dt}

        # Chat path
        messages = [
            {"role": "system", "content": "You are a helpful, safe assistant."},
            {"role": "user",   "content": prompt},
        ]
        if hasattr(self.tok, "apply_chat_template"):
            inputs = self.tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        else:
            inputs = self.tok.encode(self.tok.bos_token + prompt, return_tensors="pt")

        inputs = inputs.to(self.model.device)
        input_len = inputs.shape[1]

        t0 = time.perf_counter()
        with torch.no_grad():
            gen = self.model.generate(
                inputs,
                max_new_tokens=int(max_new_tokens),
                do_sample=(float(temperature) > 0),
                temperature=float(temperature),
                pad_token_id=self.tok.eos_token_id,
                eos_token_id=self.tok.eos_token_id,
            )
        dt = time.perf_counter() - t0

        new_tokens_tensor = gen[0, input_len:]
        new_tokens = int(new_tokens_tensor.shape[0])
        text = self.tok.decode(new_tokens_tensor, skip_special_tokens=True)
        return {"text": text, "new_tokens": new_tokens, "time_sec": dt}
