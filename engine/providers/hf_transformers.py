from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Tuple

def hf_is_available() -> Tuple[bool, str]:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return True, "ok"
    except Exception as e:
        return False, str(e)


@lru_cache(maxsize=2)
def _load_model(model_path: str):
    """
    Load model/tokenizer from a local folder path.
    Uses CPU by default for Windows-friendly runs.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.float32,
        device_map={"": "cpu"},
    )
    model.eval()
    return tok, model


@dataclass
class HFGenerator:
    model_path: str
    temperature: float = 0.2
    max_tokens: int = 700

    def generate(self, prompt: str) -> str:
        import torch
        from transformers import TextGenerationPipeline

        tok, model = _load_model(self.model_path)

        pipe = TextGenerationPipeline(
            model=model,
            tokenizer=tok,
            device=-1,  # CPU
        )

        # Some chat models expect special formatting; for simplicity we provide plain prompt.
        out = pipe(
            prompt,
            do_sample=True if self.temperature > 0 else False,
            temperature=float(self.temperature),
            max_new_tokens=int(self.max_tokens),
            return_full_text=False,
        )
        if not out:
            return ""
        text = out[0].get("generated_text") or ""
        return str(text).strip()
