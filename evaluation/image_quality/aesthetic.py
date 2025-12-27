# evaluation/image_quality/aesthetic.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List
from PIL import Image
import numpy as np
import torch

"""
Aesthetic score has multiple variants in the wild.
This wrapper supports a common pattern:

- Compute CLIP image embedding (e.g., ViT-L/14 or ViT-B/32)
- Apply a linear/regression head (weights provided separately) to produce an aesthetic scalar

You must provide a path to a .pt or .pth containing:
  {"weight": Tensor[D], "bias": Tensor[1]} or a torch.nn.Module state_dict for a linear layer.

If you don't have weights, this metric will raise with instructions.
"""

from evaluation.clip_utils import ClipEmbedder, ClipConfig


@dataclass
class AestheticConfig:
    clip_model: str = "openai/clip-vit-base-patch32"
    weights_path: Optional[str] = None  # required
    device: Optional[str] = None


class AestheticScore:
    def __init__(self, cfg: AestheticConfig = AestheticConfig()):
        if not cfg.weights_path:
            raise ValueError(
                "AestheticScore requires weights_path for the aesthetic head. "
                "Provide a linear head weights file (pt/pth)."
            )
        self.cfg = cfg
        self.embedder = ClipEmbedder(
            ClipConfig(model_name=cfg.clip_model, device=cfg.device, dtype="float16", normalize=True)
        )
        self.device = self.embedder.device

        ckpt = torch.load(cfg.weights_path, map_location=self.device)
        # Accept (weight,bias) dict or state_dict of nn.Linear
        if isinstance(ckpt, dict) and "weight" in ckpt and "bias" in ckpt:
            w = ckpt["weight"].to(self.device)
            b = ckpt["bias"].to(self.device)
            self.linear = torch.nn.Linear(w.numel(), 1, bias=True).to(self.device)
            with torch.no_grad():
                self.linear.weight.copy_(w.view(1, -1))
                self.linear.bias.copy_(b.view(-1))
        else:
            # Assume it's a state_dict for Linear(D,1)
            # We'll infer D from keys.
            # Expected keys: "weight", "bias" or "linear.weight"/"linear.bias"
            state = ckpt
            if "linear.weight" in state and "linear.bias" in state:
                w = state["linear.weight"]
                D = w.shape[1]
                self.linear = torch.nn.Linear(D, 1, bias=True).to(self.device)
                self.linear.load_state_dict({"weight": state["linear.weight"], "bias": state["linear.bias"]})
            elif "weight" in state and "bias" in state:
                D = state["weight"].shape[1]
                self.linear = torch.nn.Linear(D, 1, bias=True).to(self.device)
                self.linear.load_state_dict({"weight": state["weight"], "bias": state["bias"]})
            else:
                raise ValueError(
                    "Unsupported aesthetic weights format. Provide dict with weight/bias or state_dict."
                )

        self.linear.eval()
        if self.device.startswith("cuda"):
            self.linear = self.linear.half()

    @torch.inference_mode()
    def compute(self, images: List[Image.Image]) -> np.ndarray:
        feats = self.embedder.embed_images(images)  # (B,D) normalized
        # Some aesthetic heads expect unnormalized; if you need that, set normalize=False in ClipConfig.
        out = self.linear(feats).squeeze(-1)  # (B,)
        return out.detach().float().cpu().numpy()
