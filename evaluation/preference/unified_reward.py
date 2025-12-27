# evaluation/preference/unified_reward.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
from PIL import Image
import numpy as np

"""
UnifiedReward (UniRwd) is also usually a learned reward model.
Same pattern: custom backend preferred, else fallback proxy.

Tell me which UniReward repo you're using and I can wire the exact code.
"""

from evaluation.clip_utils import ClipEmbedder, ClipConfig


@dataclass
class UnifiedRewardConfig:
    backend: str = "clip_proxy"  # "clip_proxy" or "custom"
    clip_model: str = "openai/clip-vit-base-patch32"
    device: Optional[str] = None
    custom_module: Optional[str] = None
    custom_fn: Optional[str] = None


class UnifiedRewardScore:
    def __init__(self, cfg: UnifiedRewardConfig = UnifiedRewardConfig()):
        self.cfg = cfg
        if cfg.backend == "custom":
            if not (cfg.custom_module and cfg.custom_fn):
                raise ValueError("custom backend requires custom_module and custom_fn.")
            import importlib
            mod = importlib.import_module(cfg.custom_module)
            self.fn = getattr(mod, cfg.custom_fn)
        else:
            self.clip = ClipEmbedder(ClipConfig(model_name=cfg.clip_model, device=cfg.device))

    def compute(self, prompts: List[str], images: List[Image.Image]) -> np.ndarray:
        if self.cfg.backend == "custom":
            out = self.fn(prompts, images)
            return np.asarray(out, dtype=np.float32)
        return self.clip.clip_text_image(prompts, images).astype(np.float32)
