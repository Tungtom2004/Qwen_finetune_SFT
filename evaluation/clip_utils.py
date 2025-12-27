# evaluation/clip_utils.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

try:
    from transformers import CLIPModel, CLIPProcessor
except Exception as e:  # pragma: no cover
    CLIPModel = None
    CLIPProcessor = None


@dataclass
class ClipConfig:
    model_name: str = "openai/clip-vit-base-patch32"
    device: Optional[str] = None  # "cuda" or "cpu"
    dtype: str = "float16"  # "float16" or "float32"
    normalize: bool = True


class ClipEmbedder:
    """
    CLIP embedder that can produce:
      - image embeddings: f_img(I) in R^D
      - text embeddings: f_txt(T) in R^D
    Similarity is typically cosine between normalized embeddings.
    """

    def __init__(self, cfg: ClipConfig = ClipConfig()):
        if CLIPModel is None or CLIPProcessor is None:
            raise ImportError(
                "transformers is required for CLIP. Install: pip install transformers"
            )

        self.cfg = cfg
        self.device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if cfg.dtype == "float16" else torch.float32

        self.model = CLIPModel.from_pretrained(cfg.model_name).to(self.device)
        self.model.eval()

        # NOTE: CLIPModel uses fp32 weights by default; we cast for speed if desired.
        if self.device.startswith("cuda") and self.dtype == torch.float16:
            self.model = self.model.half()

        self.processor = CLIPProcessor.from_pretrained(cfg.model_name)

    @torch.inference_mode()
    def embed_images(self, images: List[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if self.device.startswith("cuda") and self.dtype == torch.float16:
            # pixel_values should be float16
            inputs["pixel_values"] = inputs["pixel_values"].half()

        feats = self.model.get_image_features(**inputs)  # (B, D)
        if self.cfg.normalize:
            feats = torch.nn.functional.normalize(feats, dim=-1)
        return feats

    @torch.inference_mode()
    def embed_texts(self, texts: List[str]) -> torch.Tensor:
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        feats = self.model.get_text_features(**inputs)  # (B, D)
        if self.cfg.normalize:
            feats = torch.nn.functional.normalize(feats, dim=-1)
        return feats

    @staticmethod
    def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        a: (B, D), b: (B, D) or (1, D)
        returns: (B,)
        """
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError("cosine_sim expects 2D tensors (B,D).")
        if b.shape[0] == 1:
            b = b.expand(a.shape[0], -1)
        return torch.sum(a * b, dim=-1)

    def clip_image_image(self, img_a: List[Image.Image], img_b: List[Image.Image]) -> np.ndarray:
        ea = self.embed_images(img_a)
        eb = self.embed_images(img_b)
        sims = self.cosine_sim(ea, eb)
        return sims.detach().cpu().numpy()

    def clip_text_image(self, texts: List[str], images: List[Image.Image]) -> np.ndarray:
        et = self.embed_texts(texts)
        ei = self.embed_images(images)
        sims = self.cosine_sim(et, ei)
        return sims.detach().cpu().numpy()
