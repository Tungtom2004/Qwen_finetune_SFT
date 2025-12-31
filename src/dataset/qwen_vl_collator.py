import torch
from typing import List, Dict, Any


class QwenVLDataCollator:
    """
    Collator cho Qwen-VL:
    - Stack tensor
    - Giữ metadata dạng list (KHÔNG stack)
    """

    def __call__(self, features):
        batch = {}

        # ===== tensor fields =====
        for k in ["input_ids", "labels", "attention_mask"]:
            if k in features[0]:
                batch[k] = torch.stack([f[k] for f in features], dim=0)

        if "pixel_values" in features[0]:
            batch["pixel_values"] = torch.stack(
                [f["pixel_values"] for f in features], dim=0
            )

        if "image_grid_thw" in features[0]:
            grid = []
            for f in features:
                g = f["image_grid_thw"]
                if g.ndim == 2 and g.shape[0] == 1:
                    g = g.squeeze(0)
                grid.append(g)
            batch["image_grid_thw"] = torch.stack(grid, dim=0)

        # ===== metadata fields (DO NOT stack) =====
        for k in ["sample_id", "post_id", "reviewer_id", "image_url"]:
            if k in features[0]:
                batch[k] = [f.get(k) for f in features]

        return batch

