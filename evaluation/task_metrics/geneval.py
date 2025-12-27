# evaluation/task_metrics/geneval.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from PIL import Image
import numpy as np

"""
GenEval is a compositional generation evaluator. In many implementations it relies on:
- object detection / segmentation / attribute classifiers
- checking relations (left/right/on top of), counts, colors, etc.

Because GenEval is not a single universal Python package, this file provides a wrapper that:
- Can call an external GenEval evaluator if you have it installed
- Or raises a clear error with instructions to integrate your local GenEval scripts.

You can adapt `external_eval_fn` to your GenEval pipeline.
"""

@dataclass
class GenEvalConfig:
    # If you have a custom function or module path, you can wire it here.
    # Example: evaluator_module="geneval.evaluator", evaluator_fn="evaluate_batch"
    evaluator_module: Optional[str] = None
    evaluator_fn: Optional[str] = None


class GenEvalAccuracy:
    def __init__(self, cfg: GenEvalConfig = GenEvalConfig()):
        self.cfg = cfg
        if cfg.evaluator_module and cfg.evaluator_fn:
            import importlib
            mod = importlib.import_module(cfg.evaluator_module)
            self.external_eval_fn = getattr(mod, cfg.evaluator_fn)
        else:
            self.external_eval_fn = None

    def compute(self, images: List[Image.Image], prompts: List[str]) -> np.ndarray:
        """
        Returns per-sample accuracy in [0,1] (1 = prompt satisfied).
        """
        if self.external_eval_fn is None:
            raise RuntimeError(
                "GenEval evaluator is not wired. Provide GenEvalConfig(evaluator_module=..., evaluator_fn=...). "
                "Your evaluator should accept (images, prompts) and return list/np array of accuracies."
            )
        out = self.external_eval_fn(images, prompts)
        return np.asarray(out, dtype=np.float32)
