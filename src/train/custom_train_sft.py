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
# Helpers: make shapes safe for Qwen-VL
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
    elif g.ndim == 3 and g.shape[1] == 1 and g.shape[2] == 3:
        g = g.squeeze(1)
    if g.ndim != 2 or g.shape[1] != 3:
        raise ValueError(f"[FATAL] image_grid_thw must be (B,3). Got {tuple(g.shape)}")
    return g


def ensure_pixel_values(pv: Any, device: Optional[torch.device] = None) -> torch.Tensor:
    return _to_tensor(pv, device=device)


def batch_to_device_and_fix(batch: Dict[str, Any], model_device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        out[k] = v.to(model_device) if isinstance(v, torch.Tensor) else v

    if "image_grid_thw" in out and out["image_grid_thw"] is not None:
        out["image_grid_thw"] = ensure_grid_thw(out["image_grid_thw"], device=model_device)

    if "pixel_values" in out and out["pixel_values"] is not None:
        out["pixel_values"] = ensure_pixel_values(out["pixel_values"], device=model_device)

    if "input_ids" in out and out["input_ids"] is not None:
        out["input_ids"] = ensure_ids_2d(out["input_ids"], device=model_device)

    if "attention_mask" in out and out["attention_mask"] is not None:
        out["attention_mask"] = ensure_ids_2d(out["attention_mask"], device=model_device)

    return out


# =========================================================
# Debug + Logging callback: print + save jsonl (NO GOLDEN)
# =========================================================

class LogSampleCallback(TrainerCallback):
    def __init__(
        self,
        processor,
        every_n_steps: int = 50,
        max_new_tokens: int = 128,
        save_path: str = "outputs/train_predictions.jsonl",
        print_chars: int = 400,
    ):
        self.processor = processor
        self.every_n_steps = max(1, int(every_n_steps))
        self.max_new_tokens = int(max_new_tokens)
        self.save_path = save_path
        self.print_chars = int(print_chars)

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        if not os.path.exists(save_path):
            open(save_path, "w", encoding="utf-8").close()

    @staticmethod
    def _first_of(x):
        if x is None:
            return None
        if isinstance(x, (list, tuple)):
            return x[0] if len(x) > 0 else None
        return x

    @torch.no_grad()
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 0 or state.global_step % self.every_n_steps != 0:
            return

        model = kwargs.get("model")
        train_dataloader = kwargs.get("train_dataloader")
        if model is None or train_dataloader is None:
            return

        model.eval()
        raw_batch = next(iter(train_dataloader))
        batch = batch_to_device_and_fix(raw_batch, model_device=model.device)

        sample_id = self._first_of(batch.get("sample_id"))
        post_id = self._first_of(batch.get("post_id"))
        reviewer_id = self._first_of(batch.get("reviewer_id"))
        image_url = self._first_of(batch.get("image_url"))

        labels = batch["labels"]
        IGNORE_INDEX = -100
        prompt_len = int((labels[0] == IGNORE_INDEX).sum().item())

        prompt_ids = batch["input_ids"][:, :prompt_len]
        prompt_attention = batch["attention_mask"][:, :prompt_len]

        gen_ids = model.generate(
            input_ids=prompt_ids,
            attention_mask=prompt_attention,
            pixel_values=batch.get("pixel_values"),
            image_grid_thw=batch.get("image_grid_thw"),
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            num_return_sequences=1,
        )

        gen_ids = ensure_ids_2d(gen_ids, device=None)
        gen_only = gen_ids[:, prompt_ids.shape[1]:]

        pred_text = self.processor.tokenizer.batch_decode(
            gen_only, skip_special_tokens=True
        )[0].strip() or "[EMPTY_GENERATION]"

        record = {
            "time": datetime.now().isoformat(),
            "global_step": int(state.global_step),
            "sample_id": sample_id,
            "post_id": post_id,
            "reviewer_id": reviewer_id,
            "image_url": image_url,
            "prediction": pred_text,
            "prompt_len": prompt_len,
            "gen_new_tokens": int(gen_only.shape[1]),
        }

        with open(self.save_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("\n" + "=" * 90)
        print(f"[STEP {state.global_step}] saved -> {self.save_path}")
        print(f"sample_id={sample_id} | post_id={post_id} | reviewer_id={reviewer_id}")
        print(f"image_url={image_url}")
        print(f"prompt_len={prompt_len} | gen_new_tokens={int(gen_only.shape[1])}")
        print("PRED:", pred_text[: self.print_chars])
        print("=" * 90 + "\n")

        model.train()


# =========================================================
# Utils
# =========================================================

local_rank = None


def rank0_print(*args):
    if local_rank in (None, 0, "0", -1):
        print(*args)


def set_requires_grad(params, flag: bool):
    for p in params:
        p.requires_grad = flag


def freeze_all_params(model: torch.nn.Module):
    for p in model.parameters():
        p.requires_grad = False


def enable_only_last_layer_lora(model: torch.nn.Module, last_layer: int = -1):
    """
    Enable gradients ONLY for LoRA params belonging to the last transformer layer.
    Auto-detect last layer index if last_layer == -1.
    Compatible with param names like:
      base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A...
    """

    # 1) collect all layer indices that appear in LoRA param names
    layer_re = re.compile(r"\.language_model\.layers\.(\d+)\.")
    lora_names = []
    layer_ids = set()

    for name, p in model.named_parameters():
        if "lora_" not in name:
            continue
        lora_names.append(name)
        m = layer_re.search(name)
        if m:
            layer_ids.add(int(m.group(1)))

    if len(lora_names) == 0:
        raise RuntimeError("[FATAL] No LoRA params found at all. get_peft_model() may have failed.")

    if len(layer_ids) == 0:
        print("[DEBUG] Example LoRA param name:", lora_names[0])
        raise RuntimeError(
            "[FATAL] Found LoRA params but cannot parse layer indices.\n"
            "Your naming does not match .language_model.layers.<i>.\n"
        )

    detected_last = max(layer_ids)
    if last_layer is None or int(last_layer) < 0:
        last_layer = detected_last

    print(f"[LoRA] Detected LoRA layers: min={min(layer_ids)} max={detected_last} (count={len(layer_ids)})")
    print(f"[LoRA] Using last_layer={last_layer}")

    # 2) Freeze everything first
    for p in model.parameters():
        p.requires_grad = False

    # 3) Enable only LoRA params in that layer
    target_re = re.compile(rf"\.language_model\.layers\.{int(last_layer)}\.")
    trainable = []

    for name, p in model.named_parameters():
        if "lora_" not in name:
            continue
        if target_re.search(name):
            p.requires_grad = True
            trainable.append(name)

    print(f"[LoRA] Trainable LoRA params in layer {last_layer}: {len(trainable)}")
    for n in trainable[:20]:
        print("  ", n)

    if len(trainable) == 0:
        # show some nearby layer examples for debugging
        print("\n[DEBUG] No trainable params matched. Showing a few LoRA names:")
        for n in lora_names[:30]:
            print(" ", n)
        raise RuntimeError(
            "[FATAL] No trainable params after enabling last-layer LoRA.\n"
            "→ Either last_layer index is wrong, or model doesn't have that layer.\n"
        )
# =========================================================
# Main train
# =========================================================

def train():
    global local_rank

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    if data_args.fps is not None or data_args.nframes is not None:
        raise ValueError("Video is not supported for this SFT pipeline.")

    compute_dtype = (
        torch.float16 if training_args.fp16
        else torch.bfloat16 if training_args.bf16
        else torch.float32
    )

    bnb_args = {}
    if training_args.bits in (4, 8):
        bnb_args["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=training_args.double_quant,
            bnb_4bit_quant_type=training_args.quant_type,
            llm_int8_skip_modules=["visual", "lm_head"],
        )
        bnb_args["device_map"] = {"": training_args.device}

    config = AutoConfig.from_pretrained(model_args.model_id)
    attn_impl = "sdpa"

    if config.model_type == "qwen3_vl_moe":
        replace_qwen3_vl_moe_with_mixed_modality_forward()
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_args.model_id, dtype=compute_dtype, attn_implementation=attn_impl, **bnb_args
        )
    elif config.model_type == "qwen3_vl":
        replace_qwen3_with_mixed_modality_forward()
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_args.model_id, dtype=compute_dtype, attn_implementation=attn_impl, **bnb_args
        )
    elif config.model_type == "qwen2_5_vl":
        replace_qwen2_5_with_mixed_modality_forward()
        replace_qwen2_5_vision()
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_id, dtype=compute_dtype, attn_implementation=attn_impl, **bnb_args
        )
    else:
        replace_qwen_2_with_mixed_modality_forward()
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_id, dtype=compute_dtype, attn_implementation=attn_impl, **bnb_args
        )

    model.config.use_cache = False

    # =========================
    # LoRA: ONLY LAST LAYER (Qwen2-VL-7B -> 31)
    # =========================
    LAST_LAYER = int(os.environ.get("LORA_LAST_LAYER", "27"))

    if training_args.lora_enable:
        lora_cfg = LoraConfig(
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            target_modules=["q_proj", "v_proj"],
        )

        model = get_peft_model(model, lora_cfg)

        # auto-detect last layer unless you override by env
        # export LORA_LAST_LAYER=31 (if you want force)
        last_layer_env = int(os.environ.get("LORA_LAST_LAYER", "-1"))
        enable_only_last_layer_lora(model, last_layer=last_layer_env)

        rank0_print(f"[LoRA] Enabled ONLY last layer (auto or env override)")


    processor = AutoProcessor.from_pretrained(model_args.model_id)

    data_module = make_supervised_data_module(
        model_id=model_args.model_id, processor=processor, data_args=data_args
    )

    trainer = QwenSFTTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        **data_module,
    )

    trainer.add_callback(
        LogSampleCallback(
            processor=processor,
            every_n_steps=(training_args.logging_steps or 50),
            max_new_tokens=128,
            save_path=os.path.join(training_args.output_dir, "train_predictions.jsonl"),
            print_chars=400,
        )
    )

    ckpts = list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
    trainer.train(resume_from_checkpoint=bool(ckpts))

    trainer.save_state()
    model.config.use_cache = True

    if training_args.lora_enable:
        lora_state = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters(), require_grad_only=True
        )

        if local_rank in (0, -1, None, "0"):
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=lora_state)
            processor.save_pretrained(training_args.output_dir)
            torch.save(
                non_lora_state,
                os.path.join(training_args.output_dir, "non_lora_state_dict.bin"),
            )
    else:
        safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
