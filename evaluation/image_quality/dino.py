# evaluation/image_quality/dino.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List
from PIL import Image
import numpy as np

from evaluation.dino_utils import DinoEmbedder, DinoConfig


@dataclass
class DinoMetricConfig:
    dino_model: str = "vit_base_patch16_224.dino"
    device: Optional[str] = None


class DinoSim:
    """
    DINO similarity: image-image similarity in self-supervised DINO feature space.
    Often correlates with structure/layout/detail preservation.
    """

    def __init__(self, cfg: DinoMetricConfig = DinoMetricConfig()):
        self.embedder = DinoEmbedder(
            DinoConfig(model_name=cfg.dino_model, device=cfg.device, dtype="float16", normalize=True)
        )

    def compute(self, preds: List[Image.Image], refs: List[Image.Image]) -> np.ndarray:
        return self.embedder.dino_image_image(preds, refs)
