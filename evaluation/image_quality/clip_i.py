# evaluation/image_quality/clip_i.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List
from PIL import Image
import numpy as np

from evaluation.clip_utils import ClipEmbedder, ClipConfig


@dataclass
class ClipIMetricConfig:
    clip_model: str = "openai/clip-vit-base-patch32"
    device: Optional[str] = None


class ClipI:
    """
    CLIP-I: similarity between two images in CLIP image embedding space.
    Typical usage:
      - compare predicted image vs ground-truth image (editing benchmark)
      - compare predicted vs source to measure preservation (optional)
    """

    def __init__(self, cfg: ClipIMetricConfig = ClipIMetricConfig()):
        self.embedder = ClipEmbedder(
            ClipConfig(model_name=cfg.clip_model, device=cfg.device, dtype="float16", normalize=True)
        )

    def compute(self, preds: List[Image.Image], refs: List[Image.Image]) -> np.ndarray:
        return self.embedder.clip_image_image(preds, refs)
