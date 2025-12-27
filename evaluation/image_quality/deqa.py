# evaluation/image_quality/deqa.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
from PIL import Image
import numpy as np

"""
DeQA in some papers refers to a learned Image Quality Assessment model.
There are multiple implementations. This wrapper tries:

1) pyiqa (recommended): pip install pyiqa
   - if "deqa" metric exists: use it
   - else fallback to "brisque" (no-reference IQA proxy)

If neither available, raises with clear install instructions.

Note: output scale depends on the chosen metric.
We return "higher is better" by default. Some IQA metrics are "lower is better" (e.g., BRISQUE).
We convert to higher-better when needed.
"""

@dataclass
class DeQAConfig:
    metric_name: str = "deqa"  # try "deqa" else "brisque"
    device: Optional[str] = None


class DeQA:
    def __init__(self, cfg: DeQAConfig = DeQAConfig()):
        self.cfg = cfg
        self.device = cfg.device

        try:
            import torch
            import pyiqa
        except Exception as e:
            raise ImportError(
                "DeQA wrapper requires pyiqa. Install: pip install pyiqa"
            ) from e

        self.torch = torch
        self.pyiqa = pyiqa

        self.dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Build metric (will throw if not found)
        try:
            self.metric = pyiqa.create_metric(cfg.metric_name, device=self.dev)
            self.metric_name = cfg.metric_name
        except Exception:
            # Fallback
            self.metric = pyiqa.create_metric("brisque", device=self.dev)
            self.metric_name = "brisque"

        # Some metrics are lower-better; BRISQUE is lower-better
        self.lower_better = (self.metric_name.lower() in {"brisque", "niqe"})

    def compute(self, images: List[Image.Image]) -> np.ndarray:
        # pyiqa expects torch tensors in [0,1], shape (B,3,H,W)
        ts = []
        for im in images:
            if im.mode != "RGB":
                im = im.convert("RGB")
            arr = np.asarray(im).astype(np.float32) / 255.0  # (H,W,3)
            t = self.torch.from_numpy(arr).permute(2, 0, 1)  # (3,H,W)
            ts.append(t)
        x = self.torch.stack(ts, dim=0).to(self.dev)

        with self.torch.inference_mode():
            s = self.metric(x)  # (B,) or scalar
            if not isinstance(s, self.torch.Tensor):
                s = self.torch.tensor(s, device=self.dev)

        s = s.detach().float().cpu().numpy().reshape(-1)

        # Convert to higher-better if needed
        if self.lower_better:
            s = -s
        return s
