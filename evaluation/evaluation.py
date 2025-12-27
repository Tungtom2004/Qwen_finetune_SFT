# evaluation/evaluate.py
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from PIL import Image

# Task metrics
from evaluation.task_metrics.geneval import GenEvalAccuracy, GenEvalConfig
from evaluation.task_metrics.ocr_acc import OCRAccuracy, OCRConfig

# Image quality
from evaluation.image_quality.clip_i import ClipI, ClipIMetricConfig
from evaluation.image_quality.dino import DinoSim, DinoMetricConfig
from evaluation.image_quality.deqa import DeQA, DeQAConfig

# Preference
from evaluation.preference.pickscore import PickScore, PickScoreConfig
from evaluation.preference.image_reward import ImageRewardScore, ImageRewardConfig
from evaluation.preference.unified_reward import UnifiedRewardScore, UnifiedRewardConfig

# Aesthetic (optional, needs weights)
# from evaluation.image_quality.aesthetic import AestheticScore, AestheticConfig


def _load_image(path: str) -> Image.Image:
    im = Image.open(path)
    if im.mode != "RGB":
        im = im.convert("RGB")
    return im


def l1_l2(preds: List[Image.Image], gts: List[Image.Image]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pixel L1/L2 between preds and gts. Assumes same size; if not, resize GT to pred size.
    Returns per-sample (L1, L2) averaged over HWC in [0,1] space.
    """
    l1s, l2s = [], []
    for p, g in zip(preds, gts):
        if p.size != g.size:
            g = g.resize(p.size, resample=Image.BICUBIC)
        pa = np.asarray(p).astype(np.float32) / 255.0
        ga = np.asarray(g).astype(np.float32) / 255.0
        diff = pa - ga
        l1s.append(np.mean(np.abs(diff)))
        l2s.append(np.mean(diff ** 2))
    return np.asarray(l1s, dtype=np.float32), np.asarray(l2s, dtype=np.float32)


def read_manifest(jsonl_path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def summarize(scores: Dict[str, np.ndarray]) -> Dict[str, float]:
    out = {}
    for k, v in scores.items():
        v = np.asarray(v).reshape(-1)
        if len(v) == 0:
            continue
        out[k] = float(np.mean(v))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, required=True, help="Path to eval_manifest.jsonl")
    ap.add_argument("--out", type=str, default="eval_results.json", help="Output JSON path")
    ap.add_argument("--batch", type=int, default=8)

    # toggles
    ap.add_argument("--do_l1l2", action="store_true")
    ap.add_argument("--do_clip_i", action="store_true")
    ap.add_argument("--do_dino", action="store_true")
    ap.add_argument("--do_clip_t", action="store_true")
    ap.add_argument("--do_ocr", action="store_true")
    ap.add_argument("--do_deqa", action="store_true")
    ap.add_argument("--do_geneval", action="store_true")
    ap.add_argument("--do_pickscore", action="store_true")
    ap.add_argument("--do_imgrwd", action="store_true")
    ap.add_argument("--do_unirwd", action="store_true")

    args = ap.parse_args()

    rows = read_manifest(args.manifest)

    # Prepare metric objects (lazy create when needed)
    clipi = ClipI(ClipIMetricConfig())
    dino = DinoSim(DinoMetricConfig())
    deqa = None

    # CLIP-T uses clip_utils directly to compare instruction(text) vs pred(image)
    from evaluation.clip_utils import ClipEmbedder, ClipConfig
    clip = ClipEmbedder(ClipConfig())

    ocr = OCRAccuracy(OCRConfig()) if args.do_ocr else None

    geneval = None
    if args.do_geneval:
        geneval = GenEvalAccuracy(GenEvalConfig())  # wire external evaluator in geneval.py

    pick = PickScore(PickScoreConfig()) if args.do_pickscore else None
    imgrwd = ImageRewardScore(ImageRewardConfig()) if args.do_imgrwd else None
    unirwd = UnifiedRewardScore(UnifiedRewardConfig()) if args.do_unirwd else None

    scores_all: Dict[str, List[float]] = {}

    def add_scores(name: str, arr: np.ndarray):
        scores_all.setdefault(name, [])
        scores_all[name].extend([float(x) for x in arr.reshape(-1)])

    # batching
    B = args.batch
    for i in range(0, len(rows), B):
        batch = rows[i:i+B]

        preds = [_load_image(r["image_pred"]) for r in batch]

        gts = None
        if any("image_gt" in r and r["image_gt"] for r in batch):
            gts = [_load_image(r["image_gt"]) for r in batch]

        # L1/L2
        if args.do_l1l2:
            if gts is None:
                raise ValueError("L1/L2 require image_gt for each sample.")
            l1, l2 = l1_l2(preds, gts)
            add_scores("L1", l1)
            add_scores("L2", l2)

        # CLIP-I (pred vs gt)
        if args.do_clip_i:
            if gts is None:
                raise ValueError("CLIP-I requires image_gt for each sample.")
            s = clipi.compute(preds, gts)
            add_scores("CLIP-I", s)

        # DINO (pred vs gt)
        if args.do_dino:
            if gts is None:
                raise ValueError("DINO requires image_gt for each sample.")
            s = dino.compute(preds, gts)
            add_scores("DINO", s)

        # CLIP-T: text-image similarity. Prefer instruction if exists else prompt.
        if args.do_clip_t:
            texts = []
            for r in batch:
                if "instruction" in r and r["instruction"]:
                    texts.append(r["instruction"])
                elif "prompt" in r and r["prompt"]:
                    texts.append(r["prompt"])
                else:
                    raise ValueError("CLIP-T requires instruction or prompt text per sample.")
            s = clip.clip_text_image(texts, preds)
            add_scores("CLIP-T", s)

        # OCR Acc requires gt_text per sample
        if args.do_ocr:
            gt_texts = []
            for r in batch:
                if "gt_text" not in r:
                    raise ValueError("OCR Acc requires gt_text in manifest for each sample.")
                gt_texts.append(r["gt_text"])
            s = ocr.compute(preds, gt_texts)
            add_scores("OCR-Acc", s)

        # DeQA (no-ref IQA)
        if args.do_deqa:
            if deqa is None:
                deqa = DeQA(DeQAConfig())  # may fallback to brisque
            s = deqa.compute(preds)
            add_scores("DeQA", s)

        # GenEval
        if args.do_geneval:
            prompts = []
            for r in batch:
                if "prompt" not in r:
                    raise ValueError("GenEval requires prompt per sample.")
                prompts.append(r["prompt"])
            s = geneval.compute(preds, prompts)
            add_scores("GenEval", s)

        # Preference scores
        if args.do_pickscore:
            prompts = [r.get("prompt", "") for r in batch]
            if any(not p for p in prompts):
                raise ValueError("PickScore requires prompt per sample.")
            s = pick.compute(prompts, preds)
            add_scores("PickScore", s)

        if args.do_imgrwd:
            prompts = [r.get("prompt", "") for r in batch]
            if any(not p for p in prompts):
                raise ValueError("ImageReward requires prompt per sample.")
            s = imgrwd.compute(prompts, preds)
            add_scores("ImgRwd", s)

        if args.do_unirwd:
            prompts = [r.get("prompt", "") for r in batch]
            if any(not p for p in prompts):
                raise ValueError("UnifiedReward requires prompt per sample.")
            s = unirwd.compute(prompts, preds)
            add_scores("UniRwd", s)

    # Convert to arrays and summarize
    scores_np = {k: np.asarray(v, dtype=np.float32) for k, v in scores_all.items()}
    summary = summarize(scores_np)

    out = {
        "num_samples": len(rows),
        "summary_mean": summary,
        "per_sample": {k: scores_np[k].tolist() for k in scores_np},
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("Saved:", args.out)
    print("Summary (mean):")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
