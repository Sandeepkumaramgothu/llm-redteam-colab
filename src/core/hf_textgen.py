# src/core/hf_textgen.py
"""
TextGenerator with fallbacks and timing.

- We try to load your requested model name first (e.g., meta-llama/Llama-2-7b-chat-hf).
- If that fails (gated or unavailable), we try open fallbacks:
    1) Qwen/Qwen2.5-7B-Instruct
    2) microsoft/Phi-3-mini-4k-instruct
- We load in 4-bit on GPU if available (saves VRAM on Colab).
- We provide .generate_stats(...) which returns:
    {"text": str, "new_tokens": int, "time_sec": float}
"""

from typing import Optional, List, Dict, Any
import time
import torch, warnings

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    pipeline,
)

# List of open models we can use if the requested one is blocked/gated.
_OPEN_FALLBACKS = [
    "Qwen/Qwen2.5-7B-Instruct",          # Open 7B instruct model
    "microsoft/Phi-3-mini-4k-instruct",  # Open ~3.8B instruct model
]

def _is_t5(name: str) -> bool:
    """Return True if the model name looks like a T5-family model."""
    return "t5" in (name or "").lower()

class TextGenerator:
    """
    A tiny class that hides model differences (T5 vs Llama-style chat).
    It tries to load your preferred model, else falls back to an open model.
    """
    def __init__(self, model_name: str, load_4bit: bool = True, fallbacks: Optional[List[str]] = None):
        # device index for GPU if available, else -1 (CPU)
        self.device = 0 if torch.cuda.is_available() else -1
        # whether to try 4-bit loading on GPU
        self.load_4bit = bool(load_4bit)
        # use provided fallbacks or the defaults above
        self.fallbacks = list(fallbacks) if fallbacks else list(_OPEN_FALLBACKS)

        # We'll try primary model first, then each fallback in order.
        last_err = None
        for candidate in [model_name] + self.fallbacks:
            try:
                self._load(candidate)               # try to load the model/tokenizer
                self.model_name = candidate         # remember which one actually loaded
                if candidate != model_name:
                    # If we used a fallback, let the user know (warning only).
                    warnings.warn(f"Primary model '{model_name}' unavailable; using fallback '{candidate}'.")
                return                              # stop after first success
            except Exception as e:
                last_err = e                        # remember the error and try the next candidate
                continue

        # If we reach here, all candidates failed → raise a clear error.
        raise RuntimeError(f"Could not load any model (tried { [model_name] + self.fallbacks }): {last_err}")

    def _load(self, model_name: str):
        """
        Internal: actually load the model and tokenizer for a given name.
        We handle T5-family (text2text) and causal chat models differently.
        """
        if _is_t5(model_name):
            # T5 uses an encoder-decoder "text2text" pipeline.
            self.pipe = pipeline(
                "text2text-generation",
                model=model_name,
                tokenizer=model_name,
                device=self.device,          # GPU if available, else CPU
            )
            self.tok = self.pipe.tokenizer   # save tokenizer for later
            self.model = self.pipe.model     # save model for completeness
            self.is_causal = False           # not a causal chat model
            return

        # For Llama/Qwen/Phi (causal chat models):
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        # Build kwargs to load in 4-bit if we have GPU; else use bfloat16 or CPU.
        kwargs = {}
        if torch.cuda.is_available() and self.load_4bit:
            # 4-bit path with bitsandbytes; saves VRAM on Colab GPUs.
            kwargs.update(dict(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                device_map="auto",             # place model on available GPU(s)
                torch_dtype=torch.bfloat16,    # good default for modern GPUs
            ))
        else:
            # Non-4-bit path (still works; may use more VRAM).
            kwargs.update(dict(
                device_map="auto" if torch.cuda.is_available() else None,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
            ))

        # Actually load the Causal LM weights with our chosen kwargs.
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

        # Some tokenizers don't have a pad token; we set it to EOS for generation.
        if self.tok.pad_token is None and self.tok.eos_token is not None:
            self.tok.pad_token = self.tok.eos_token

        # Mark that this is a causal chat model (not T5).
        self.is_causal = True

    def generate_stats(self, prompt: str, max_new_tokens: int = 128, temperature: float = 0.7) -> Dict[str, Any]:
        """
        Generate text and also report how many tokens we produced and how long it took.

        Returns a dictionary like:
          {"text": "...", "new_tokens": 57, "time_sec": 0.84}
        """
        if not self.is_causal:
            # T5 path: call the pipeline and measure the time.
            t0 = time.perf_counter()                              # start a timer
            result = self.pipe(                                   # run the model
                prompt,
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
            )
            dt = time.perf_counter() - t0                         # elapsed seconds
            text = result[0]["generated_text"]                    # get the text string
            # We tokenize ONLY the generated text to estimate token count.
            new_tokens = len(self.tok.encode(text))
            return {"text": text, "new_tokens": new_tokens, "time_sec": dt}

        # Causal chat path: build a conversation and generate a reply.
        messages = [
            {"role": "system", "content": "You are a helpful, safe assistant."},
            {"role": "user",   "content": prompt},
        ]

        # If the tokenizer supports chat templates, we use it; else we fallback.
        if hasattr(self.tok, "apply_chat_template"):
            inputs = self.tok.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            )
        else:
            # Simpler fallback: just encode a BOS token + prompt text.
            inputs = self.tok.encode(self.tok.bos_token + prompt, return_tensors="pt")

        # Move inputs to the model's device (GPU or CPU).
        inputs = inputs.to(self.model.device)

        # We want to know how many tokens are "prompt" vs "generated".
        input_len = inputs.shape[1]

        # Start the timer, run generation, stop the timer.
        t0 = time.perf_counter()
        with torch.no_grad():     # no gradients needed for inference
            gen = self.model.generate(
                inputs,
                max_new_tokens=int(max_new_tokens),
                do_sample=(float(temperature) > 0),
                temperature=float(temperature),
                pad_token_id=self.tok.eos_token_id,
                eos_token_id=self.tok.eos_token_id,
            )
        dt = time.perf_counter() - t0

        # Slice out ONLY the newly generated tokens (after the prompt length).
        new_tokens_tensor = gen[0, input_len:]

        # Count how many new tokens the model produced.
        new_tokens = int(new_tokens_tensor.shape[0])

        # Turn those new tokens back into readable text.
        text = self.tok.decode(new_tokens_tensor, skip_special_tokens=True)

        # Return everything we measured.
        return {"text": text, "new_tokens": new_tokens, "time_sec": dt}
