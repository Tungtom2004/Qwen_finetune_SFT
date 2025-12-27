# evaluation/preference/pickscore.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
from PIL import Image
import numpy as np

"""
PickScore is a learned preference model. There are multiple community implementations.
This wrapper supports a common approach:

- Use a PickScore model from Hugging Face / community repo
- Input: (prompt, image)
- Output: scalar score (higher is better)

Because exact model class may differ, this wrapper is built to be easily adapted.
If you have a specific PickScore HF repo you want to use, tell me its name and I’ll
wire the exact code.

For now, this file provides:
- a placeholder that raises if not configured
- a simple fallback using CLIP text-image similarity as a proxy (NOT true PickScore)
"""

from evaluation.clip_utils import ClipEmbedder, ClipConfig


@dataclass
class PickScoreConfig:
    backend: str = "clip_proxy"  # "clip_proxy" or "custom"
    clip_model: str = "openai/clip-vit-base-patch32"
    device: Optional[str] = None
    custom_module: Optional[str] = None
    custom_fn: Optional[str] = None


class PickScore:
    def __init__(self, cfg: PickScoreConfig = PickScoreConfig()):
        self.cfg = cfg
        if cfg.backend == "custom":
            if not (cfg.custom_module and cfg.custom_fn):
                raise ValueError("custom backend requires custom_module and custom_fn.")
            import importlib
            mod = importlib.import_module(cfg.custom_module)
            self.fn = getattr(mod, cfg.custom_fn)
        else:
            # Proxy: CLIP text-image similarity
            self.clip = ClipEmbedder(ClipConfig(model_name=cfg.clip_model, device=cfg.device))

    def compute(self, prompts: List[str], images: List[Image.Image]) -> np.ndarray:
        if self.cfg.backend == "custom":
            out = self.fn(prompts, images)
            return np.asarray(out, dtype=np.float32)
        # Proxy
        return self.clip.clip_text_image(prompts, images).astype(np.float32)
