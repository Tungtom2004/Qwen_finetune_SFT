import torch
from typing import List, Dict, Any


class QwenVLDataCollator:
    """
    Collator bắt buộc cho Qwen-VL:
    - image_grid_thw: (B, 3)
    - KHÔNG squeeze
    """

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = {}

        # ===== input_ids / labels / attention_mask =====
        for k in ["input_ids", "labels", "attention_mask"]:
            if k in features[0]:
                batch[k] = torch.stack([f[k] for f in features], dim=0)

        # ===== pixel_values =====
        if "pixel_values" in features[0]:
            # pixel_values: list[tensor(num_patches, dim)]
            batch["pixel_values"] = torch.stack(
                [f["pixel_values"] for f in features], dim=0
            )

        # ===== image_grid_thw =====
        if "image_grid_thw" in features[0]:
            grid = [f["image_grid_thw"] for f in features]

            # đảm bảo mỗi cái là (3,)
            fixed = []
            for g in grid:
                if g.ndim == 1:
                    fixed.append(g)
                elif g.ndim == 2 and g.shape[0] == 1:
                    fixed.append(g.squeeze(0))
                else:
                    raise ValueError(f"Bad image_grid_thw shape: {g.shape}")

            # stack → (B, 3)
            batch["image_grid_thw"] = torch.stack(fixed, dim=0)

        # ===== optional id =====
        if "id" in features[0]:
            batch["id"] = [f.get("id") for f in features]

        return batch
