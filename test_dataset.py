import torch
from transformers import AutoProcessor
from src.params import DataArguments
from src.dataset.custom_dataset_sft import SupervisedDataset

model_id = "Qwen/Qwen2.5-VL-3B-Instruct"

processor = AutoProcessor.from_pretrained(model_id)

data_args = DataArguments(
    data_path="/disk/yuu/Qwen_finetune_SFT/demo.jsonl",
    image_folder=".",   # hoặc path nếu ảnh local
    image_min_pixels=224*224,
    image_max_pixels=448*448,
    image_resized_width=448,
    image_resized_height=448,
)

dataset = SupervisedDataset(
    data_path=data_args.data_path,
    processor=processor,
    data_args=data_args,
    model_id=model_id,
)

sample = dataset[0]
batch = {k: v.unsqueeze(0) if v.ndim == 1 else v for k, v in sample.items()}
print("Batch keys:", batch.keys())

print("input_ids:", sample["input_ids"].shape)
print("labels:", sample["labels"].shape)
print("pixel_values:", sample["pixel_values"].shape)
print("image_grid_thw:", sample["image_grid_thw"])
