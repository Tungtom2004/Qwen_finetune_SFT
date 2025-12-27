# evaluation/dino_utils.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

try:
    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
except Exception:  # pragma: no cover
    timm = None


@dataclass
class DinoConfig:
    model_name: str = "vit_base_patch16_224.dino"
    device: Optional[str] = None
    dtype: str = "float16"
    normalize: bool = True


class DinoEmbedder:
    """
    DINO features are self-supervised visual representations.
    Typically used for image-image similarity (cosine) focusing on structure/layout/details.
    """

    def __init__(self, cfg: DinoConfig = DinoConfig()):
        if timm is None:
            raise ImportError("timm is required for DINO. Install: pip install timm")

        self.cfg = cfg
        self.device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if cfg.dtype == "float16" else torch.float32

        self.model = timm.create_model(cfg.model_name, pretrained=True, num_classes=0).to(self.device)
        self.model.eval()
        if self.device.startswith("cuda") and self.dtype == torch.float16:
            self.model = self.model.half()

        data_cfg = resolve_data_config({}, model=self.model)
        self.transform = create_transform(**data_cfg)

    @torch.inference_mode()
    def embed_images(self, images: List[Image.Image]) -> torch.Tensor:
        xs = []
        for im in images:
            if im.mode != "RGB":
                im = im.convert("RGB")
            x = self.transform(im)  # (C,H,W) float
            xs.append(x)
        x = torch.stack(xs, dim=0).to(self.device)
        if self.device.startswith("cuda") and self.dtype == torch.float16:
            x = x.half()
        feats = self.model(x)  # (B, D)
        if self.cfg.normalize:
            feats = torch.nn.functional.normalize(feats, dim=-1)
        return feats

    @staticmethod
    def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if b.shape[0] == 1:
            b = b.expand(a.shape[0], -1)
        return torch.sum(a * b, dim=-1)

    def dino_image_image(self, img_a: List[Image.Image], img_b: List[Image.Image]) -> np.ndarray:
        ea = self.embed_images(img_a)
        eb = self.embed_images(img_b)
        sims = self.cosine_sim(ea, eb)
        return sims.detach().cpu().numpy()
