from transformers import TrainerCallback
import torch

class PrintSampleCallback(TrainerCallback):
    def __init__(self, processor, every_n_steps=1, max_new_tokens=128):
        self.processor = processor
        self.every_n_steps = every_n_steps
        self.max_new_tokens = max_new_tokens

    def on_step_end(self, args, state, control, **kwargs):
        # chỉ in ở rank 0
        if state.global_step % self.every_n_steps != 0:
            return

        model = kwargs["model"]
        train_dataloader = kwargs["train_dataloader"]

        model.eval()

        # Lấy 1 batch đầu tiên (debug purpose)
        batch = next(iter(train_dataloader))

        # Move to device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch.get("pixel_values", None),
                image_grid_thw=batch.get("image_grid_thw", None),
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        pred_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        label_text = self.processor.batch_decode(
            batch["labels"].masked_fill(batch["labels"] < 0, self.processor.tokenizer.pad_token_id),
            skip_special_tokens=True
        )[0]

        print("\n" + "=" * 80)
        print(f"[STEP {state.global_step}] MODEL OUTPUT:")
        print(pred_text)
        print("-" * 80)
        print("GOLDEN (LABEL):")
        print(label_text)
        print("=" * 80 + "\n")

        model.train()
