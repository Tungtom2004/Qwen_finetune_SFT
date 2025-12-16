import os
import ast
import pathlib
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
from train.train_utils import (
    get_peft_state_maybe_zero_3,
    get_peft_state_non_lora_maybe_zero_3,
    safe_save_model_for_hf_trainer,
)

from monkey_patch_forward import (
    replace_qwen_2_with_mixed_modality_forward,
    replace_qwen2_5_with_mixed_modality_forward,
    replace_qwen3_with_mixed_modality_forward,
    replace_qwen3_vl_moe_with_mixed_modality_forward,
)
from monkey_patch_vision import replace_qwen2_5_vision


# =========================================================
# Debug callback: print prediction vs golden
# =========================================================

class PrintSampleCallback(TrainerCallback):
    def __init__(self, processor, every_n_steps=1, max_new_tokens=128):
        self.processor = processor
        self.every_n_steps = every_n_steps
        self.max_new_tokens = max_new_tokens

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.every_n_steps != 0:
            return

        model = kwargs["model"]
        train_dataloader = kwargs["train_dataloader"]

        model.eval()

        batch = next(iter(train_dataloader))
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(model.device)

        with torch.no_grad():
            gen_ids = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch.get("pixel_values"),
                image_grid_thw=batch.get("image_grid_thw"),
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        pred_text = self.processor.batch_decode(
            gen_ids, skip_special_tokens=True
        )[0]

        label_ids = batch["labels"].clone()
        label_ids[label_ids < 0] = self.processor.tokenizer.pad_token_id
        label_text = self.processor.batch_decode(
            label_ids.unsqueeze(0), skip_special_tokens=True
        )[0]

        print("\n" + "=" * 100)
        print(f"[STEP {state.global_step}] MODEL PREDICTION:")
        print(pred_text)
        print("-" * 100)
        print("GOLDEN (LABEL):")
        print(label_text)
        print("=" * 100 + "\n")

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


def find_lora_targets(model, exclude=None, max_modules=-1):
    exclude = exclude or []
    targets = []

    for name, module in model.named_modules():
        if any(x in name for x in exclude):
            continue
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            targets.append(name)

    if max_modules > 0:
        targets = targets[-max_modules:]

    rank0_print(f"[LoRA] target modules = {len(targets)}")
    return targets


# =========================================================
# Main train
# =========================================================

def train():
    global local_rank

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    # Dataset của bạn KHÔNG dùng video
    if data_args.fps is not None or data_args.nframes is not None:
        raise ValueError("Video is not supported for this SFT pipeline.")

    compute_dtype = (
        torch.float16 if training_args.fp16
        else torch.bfloat16 if training_args.bf16
        else torch.float32
    )

    # Quantization
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

    # Load model
    config = AutoConfig.from_pretrained(model_args.model_id)

    if config.model_type == "qwen3_vl_moe":
        replace_qwen3_vl_moe_with_mixed_modality_forward()
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_args.model_id,
            dtype=compute_dtype,
            attn_implementation="flash_attention_2",
            **bnb_args,
        )
    elif config.model_type == "qwen3_vl":
        replace_qwen3_with_mixed_modality_forward()
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_args.model_id,
            dtype=compute_dtype,
            attn_implementation="flash_attention_2",
            **bnb_args,
        )
    elif config.model_type == "qwen2_5_vl":
        replace_qwen2_5_with_mixed_modality_forward()
        replace_qwen2_5_vision()
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_id,
            dtype=compute_dtype,
            attn_implementation="flash_attention_2",
            **bnb_args,
        )
    else:
        replace_qwen_2_with_mixed_modality_forward()
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_id,
            dtype=compute_dtype,
            attn_implementation="flash_attention_2",
            **bnb_args,
        )

    model.config.use_cache = False

    # Freeze / unfreeze
    set_requires_grad(model.language_model.parameters(), not training_args.freeze_llm)
    set_requires_grad(model.lm_head.parameters(), not training_args.freeze_llm)
    set_requires_grad(model.visual.parameters(), not training_args.freeze_vision_tower)

    # LoRA
    if training_args.lora_enable:
        exclude = ast.literal_eval(training_args.lora_namespan_exclude) \
            if training_args.lora_namespan_exclude else []
        exclude += ["visual"]

        lora_cfg = LoraConfig(
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            target_modules=find_lora_targets(
                model,
                exclude=exclude,
                max_modules=training_args.num_lora_modules,
            ),
        )
        model = get_peft_model(model, lora_cfg)
        rank0_print("LoRA enabled")

    processor = AutoProcessor.from_pretrained(model_args.model_id)

    data_module = make_supervised_data_module(
        model_id=model_args.model_id,
        processor=processor,
        data_args=data_args,
    )

    trainer = QwenSFTTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        **data_module,
    )

    # 🔥 ADD DEBUG CALLBACK
    trainer.add_callback(
        PrintSampleCallback(
            processor=processor,
            every_n_steps=training_args.logging_steps or 1,
            max_new_tokens=128,
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

        if local_rank in (0, -1):
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=lora_state)
            processor.save_pretrained(training_args.output_dir)
            torch.save(
                non_lora_state,
                os.path.join(training_args.output_dir, "non_lora_state_dict.bin"),
            )
    else:
        safe_save_model_for_hf_trainer(
            trainer, output_dir=training_args.output_dir
        )


if __name__ == "__main__":
    train()
