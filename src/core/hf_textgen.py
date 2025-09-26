# src/core/hf_textgen.py
"""
TextGenerator with model fallback:
1) Try requested model_name (e.g., meta-llama/Llama-2-7b-chat-hf).
2) If gated or load fails → try Qwen/Qwen2.5-7B-Instruct (open).
3) If that fails → try microsoft/Phi-3-mini-4k-instruct (open, ~3.8B).

- Uses 4-bit when possible (saves VRAM on Colab).
- For causal chat models, uses tokenizer.apply_chat_template if available.
"""
from typing import Optional, List
import torch, warnings

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    pipeline,
)

_OPEN_FALLBACKS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "microsoft/Phi-3-mini-4k-instruct",
]

def _is_t5(name: str) -> bool:
    return "t5" in (name or "").lower()

class TextGenerator:
    def __init__(self, model_name: str, load_4bit: bool = True, fallbacks: Optional[List[str]] = None):
        self.device = 0 if torch.cuda.is_available() else -1
        self.load_4bit = bool(load_4bit)
        self.fallbacks = list(fallbacks) if fallbacks else list(_OPEN_FALLBACKS)

        # Try primary, else fallbacks.
        last_err = None
        for candidate in [model_name] + self.fallbacks:
            try:
                self._load(candidate)
                self.model_name = candidate
                if candidate != model_name:
                    warnings.warn(f"Primary model '{model_name}' unavailable; using fallback '{candidate}'.")
                return
            except Exception as e:
                last_err = e
                continue
        # If we get here, all candidates failed.
        raise RuntimeError(f"Could not load any model (tried { [model_name] + self.fallbacks }): {last_err}")

    def _load(self, model_name: str):
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

        # Causal LM (chat-style)
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        kwargs = {}
        if torch.cuda.is_available() and self.load_4bit:
            kwargs.update(dict(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                device_map="auto",
                torch_dtype=torch.bfloat16,
            ))
        else:
            kwargs.update(dict(
                device_map="auto" if torch.cuda.is_available() else None,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
            ))
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        if self.tok.pad_token is None and self.tok.eos_token is not None:
            self.tok.pad_token = self.tok.eos_token
        self.is_causal = True

    def generate(self, prompt: str, max_new_tokens: int = 128, temperature: float = 0.7) -> str:
        if not self.is_causal:
            out = self.pipe(prompt, max_new_tokens=int(max_new_tokens), temperature=float(temperature))[0]["generated_text"]
            return out

        # Build inputs using chat template if available; else plain encode.
        messages = [
            {"role": "system", "content": "You are a helpful, safe assistant."},
            {"role": "user", "content": prompt},
        ]
        if hasattr(self.tok, "apply_chat_template"):
            inputs = self.tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        else:
            # Fallback: BOS + prompt
            inputs = self.tok.encode(self.tok.bos_token + prompt, return_tensors="pt")
        inputs = inputs.to(self.model.device)

        input_len = inputs.shape[1]
        with torch.no_grad():
            gen = self.model.generate(
                inputs,
                max_new_tokens=int(max_new_tokens),
                do_sample=(float(temperature) > 0),
                temperature=float(temperature),
                pad_token_id=self.tok.eos_token_id,
                eos_token_id=self.tok.eos_token_id,
            )
        new_tokens = gen[0, input_len:]
        return self.tok.decode(new_tokens, skip_special_tokens=True)
