import os
import ast
import json
import pathlib
from datetime import datetime
from typing import Any, Dict, Optional
import re
import torch
from transformers import (
    AutoProcessor,
    AutoConfig,
    BitsAndBytesConfig,
    HfArgumentParser,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
    TrainerCallback,
)

from peft import LoraConfig, get_peft_model

from src.trainer import QwenSFTTrainer
from src.dataset import make_supervised_data_module
from src.params import ModelArguments, DataArguments, TrainingArguments
from src.train.train_utils import (
    get_peft_state_maybe_zero_3,
    get_peft_state_non_lora_maybe_zero_3,
    safe_save_model_for_hf_trainer,
)

from src.train.monkey_patch_forward import (
    replace_qwen_2_with_mixed_modality_forward,
    replace_qwen2_5_with_mixed_modality_forward,
    replace_qwen3_with_mixed_modality_forward,
    replace_qwen3_vl_moe_with_mixed_modality_forward,
)
from src.train.monkey_patch_vision import replace_qwen2_5_vision


# =========================================================
# Helpers
# =========================================================

def _to_tensor(x: Any, device: Optional[torch.device] = None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        t = x
    else:
        t = torch.as_tensor(x)
    if device is not None:
        t = t.to(device)
    return t


def ensure_ids_2d(ids: Any, device: Optional[torch.device] = None) -> torch.Tensor:
    ids = _to_tensor(ids, device=device)
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    elif ids.ndim == 3:
        ids = ids[:, 0, :]
    elif ids.ndim > 3:
        ids = ids.reshape(ids.shape[0], -1)
    return ids


def ensure_grid_thw(grid: Any, device: Optional[torch.device] = None) -> torch.Tensor:
    g = _to_tensor(grid, device=device).long()
    if g.ndim == 1:
        g = g.unsqueeze(0)
    elif g.ndim == 3 and g.shape[1] == 1:
        g = g.squeeze(1)
    if g.ndim != 2 or g.shape[1] != 3:
        raise ValueError(f"Bad image_grid_thw shape: {g.shape}")
    return g


<<<<<<< HEAD
def batch_to_device_and_fix(batch: Dict[str, Any], device: torch.device):
    out = {}
=======
def ensure_pixel_values(pv: Any, device: Optional[torch.device] = None) -> torch.Tensor:
    return _to_tensor(pv, device=device)


def batch_to_device_and_fix(
    batch: Dict[str, Any],
    model_device: torch.device,
) -> Dict[str, Any]:
    """
    Move tensors to GPU + fix shapes for Qwen-VL forward.
    Keep non-tensor metadata (strings/ints/lists) as-is.
    """
    out: Dict[str, Any] = {}
>>>>>>> d42941fdf55ec6678377e656572e15e4b70780be
    for k, v in batch.items():
        out[k] = v.to(device) if isinstance(v, torch.Tensor) else v

    if "input_ids" in out:
        out["input_ids"] = ensure_ids_2d(out["input_ids"], device)
    if "attention_mask" in out:
        out["attention_mask"] = ensure_ids_2d(out["attention_mask"], device)
    if "image_grid_thw" in out:
        out["image_grid_thw"] = ensure_grid_thw(out["image_grid_thw"], device)
    return out


# =========================================================
<<<<<<< HEAD
# Logging callback (memory-safe)
# =========================================================

class LogSampleCallback(TrainerCallback):
=======
# Debug + Logging callback: print + save jsonl (NO GOLDEN)
# =========================================================

class LogSampleCallback(TrainerCallback):
    """
    SFT logging (no leakage):
      - generate ONLY from prompt
      - decode ONLY newly generated tokens
      - log sample_id/post_id/reviewer_id/image_url for tracing
      - DO NOT log golden/labels
    """

>>>>>>> d42941fdf55ec6678377e656572e15e4b70780be
    def __init__(
        self,
        processor,
        every_n_steps=50,
        max_new_tokens=128,
        save_path="outputs/train_predictions.jsonl",
        print_chars=400,
    ):
        self.processor = processor
        self.every_n_steps = every_n_steps
        self.max_new_tokens = max_new_tokens
        self.save_path = save_path
        self.print_chars = print_chars

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        if not os.path.exists(save_path):
            open(save_path, "w").close()

    @staticmethod
    def _first(x):
        if isinstance(x, list):
            return x[0]
        return x

    @staticmethod
    def _first_of(x):
        # batch metadata thường là list[str] / list[int] / scalar
        if x is None:
            return None
        if isinstance(x, (list, tuple)):
            return x[0] if len(x) > 0 else None
        return x

    @torch.no_grad()
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 0 or state.global_step % self.every_n_steps != 0:
            return

        model = kwargs["model"]
        loader = kwargs["train_dataloader"]

        model.eval()
        batch = next(iter(loader))
        batch = batch_to_device_and_fix(batch, model.device)

<<<<<<< HEAD
=======
        # 1) get batch
        raw_batch = next(iter(train_dataloader))
        batch = batch_to_device_and_fix(raw_batch, model_device=model.device)

        # ---- extract metadata for tracing ----
        sample_id = self._first_of(batch.get("sample_id"))
        post_id = self._first_of(batch.get("post_id"))
        reviewer_id = self._first_of(batch.get("reviewer_id"))
        image_url = self._first_of(batch.get("image_url"))

        # 2) compute prompt_len from labels IGNORE_INDEX
>>>>>>> d42941fdf55ec6678377e656572e15e4b70780be
        labels = batch["labels"]
        prompt_len = int((labels[0] == -100).sum().item())

        prompt_ids = batch["input_ids"][:1, :prompt_len]
        attn = batch["attention_mask"][:1, :prompt_len]

        pixel = batch.get("pixel_values")
        grid = batch.get("image_grid_thw")
        if pixel is not None:
            pixel = pixel[:1]
        if grid is not None:
            grid = grid[:1]

        with torch.cuda.amp.autocast(dtype=torch.float16):
            gen = model.generate(
                input_ids=prompt_ids,
                attention_mask=attn,
                pixel_values=pixel,
                image_grid_thw=grid,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=False,
            )

<<<<<<< HEAD
        gen = ensure_ids_2d(gen)
        pred = self.processor.tokenizer.decode(
            gen[0, prompt_ids.shape[1]:], skip_special_tokens=True
        ).strip()

        record = {
            "step": int(state.global_step),
            "sample_id": self._first(batch.get("sample_id")),
            "post_id": self._first(batch.get("post_id")),
            "reviewer_id": self._first(batch.get("reviewer_id")),
            "image_url": self._first(batch.get("image_url")),
            "prediction": pred,
=======
        # 4) decode generated part ONLY
        gen_only = gen_ids[:, prompt_ids.shape[1]:]
        pred_text = self.processor.tokenizer.batch_decode(
            gen_only, skip_special_tokens=True
        )[0].strip()

        if pred_text == "":
            pred_text = "[EMPTY_GENERATION]"

        record = {
            "time": datetime.now().isoformat(),
            "global_step": int(state.global_step),

            # tracing fields
            "sample_id": sample_id,
            "post_id": post_id,
            "reviewer_id": reviewer_id,
            "image_url": image_url,

            # prediction
            "prediction": pred_text,

            # debug info
            "prompt_len": prompt_len,
            "gen_new_tokens": int(gen_only.shape[1]),
>>>>>>> d42941fdf55ec6678377e656572e15e4b70780be
        }

        with open(self.save_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("\n" + "=" * 90)
<<<<<<< HEAD
        print(f"[STEP {state.global_step}]")
        print("PRED:", pred[: self.print_chars])
        print("=" * 90)
=======
        print(f"[STEP {state.global_step}] saved -> {self.save_path}")
        print(f"sample_id={sample_id} | post_id={post_id} | reviewer_id={reviewer_id}")
        print(f"image_url={image_url}")
        print(f"prompt_len={prompt_len} | gen_new_tokens={int(gen_only.shape[1])}")
        print("PRED:", pred_text[: self.print_chars])
        print("=" * 90 + "\n")
>>>>>>> d42941fdf55ec6678377e656572e15e4b70780be

        del gen, prompt_ids, attn
        torch.cuda.empty_cache()
        model.train()


# =========================================================
# Enable last-layer LoRA only
# =========================================================

def enable_only_last_layer_lora(model, last_layer=-1):
    layer_re = re.compile(r"\.language_model\.layers\.(\d+)\.")
    layers = set()

    for n, p in model.named_parameters():
        if "lora_" in n:
            m = layer_re.search(n)
            if m:
                layers.add(int(m.group(1)))

    if not layers:
        raise RuntimeError("No LoRA layers found")

    if last_layer < 0:
        last_layer = max(layers)

    for p in model.parameters():
        p.requires_grad = False

    for n, p in model.named_parameters():
        if "lora_" in n and f".layers.{last_layer}." in n:
            p.requires_grad = True

    print(f"[LoRA] Training ONLY layer {last_layer}")


# =========================================================
# Train
# =========================================================

def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    dtype = torch.float16 if training_args.fp16 else torch.float32

    config = AutoConfig.from_pretrained(model_args.model_id)
    attn_impl = "sdpa"

    if config.model_type == "qwen2_5_vl":
        replace_qwen2_5_with_mixed_modality_forward()
        replace_qwen2_5_vision()
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_id, dtype=dtype, attn_implementation=attn_impl
        )
    else:
        replace_qwen_2_with_mixed_modality_forward()
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_id, dtype=dtype, attn_implementation=attn_impl
        )

    model.config.use_cache = False
    model.gradient_checkpointing_enable()  # 🔥 critical

<<<<<<< HEAD
    if training_args.lora_enable:
        lora = LoraConfig(
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            lora_dropout=training_args.lora_dropout,
            target_modules=["q_proj", "v_proj"],
        )
        model = get_peft_model(model, lora)
        enable_only_last_layer_lora(
            model,
            int(os.environ.get("LORA_LAST_LAYER", "-1")),
        )
=======
    # Freeze / unfreeze
    set_requires_grad(model.language_model.parameters(), not training_args.freeze_llm)
    set_requires_grad(model.lm_head.parameters(), not training_args.freeze_llm)
    set_requires_grad(model.visual.parameters(), not training_args.freeze_vision_tower)

    # =========================
    # LoRA: ONLY 1 LAYER
    # =========================
    if training_args.lora_enable:
        # FIX: os.environ.get, not os.environ(...)
        LORA_LAYER_INDEX = int(os.environ.get("LORA_LAYER_INDEX", "31"))
        target_modules = ["q_proj", "v_proj"]

        lora_cfg = LoraConfig(
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,

            target_modules=target_modules,
            layers_to_transform=[LORA_LAYER_INDEX],
            layers_pattern="layers",
            exclude_modules=".*visual.*",
        )

        model = get_peft_model(model, lora_cfg)
        rank0_print(f"[LoRA] Enabled on layer={LORA_LAYER_INDEX}, targets={target_modules}")
>>>>>>> d42941fdf55ec6678377e656572e15e4b70780be

    processor = AutoProcessor.from_pretrained(model_args.model_id)

    data_module = make_supervised_data_module(
        model_args.model_id, processor, data_args
    )

    trainer = QwenSFTTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        **data_module,
    )

    trainer.add_callback(
        LogSampleCallback(
            processor,
            every_n_steps=training_args.logging_steps or 50,
            max_new_tokens=128,
            save_path=os.path.join(training_args.output_dir, "train_predictions.jsonl"),
        )
    )

    trainer.train()
    trainer.save_state()

    if training_args.lora_enable:
        model.save_pretrained(training_args.output_dir)
        processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train()
