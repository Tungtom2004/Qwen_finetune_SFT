import copy
import os
from typing import Dict
import torch
import transformers
import ujson as json
from torch.utils.data import Dataset

from src.params import DataArguments
from src.constants import (
    IGNORE_INDEX,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    SYSTEM_MESSAGE,
)

from src.utils import get_image_info, llava_to_openai, pad_sequence


class SupervisedDataset(Dataset):
    """
    Dataset for image editing SFT:
    Input  = image + critique
    Output = list of editing actions
    """

    def __init__(
        self,
        data_path: str,
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        model_id,
        padding=True,
    ):
        super().__init__()

        # ===== Load JSONL dataset =====
        self.list_data_dict = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:

                line = line.strip()

                if not line:
                    continue 

                try:
                    obj = json.loads(line)
                except Exception as e:
                    print("JSON parse error, skipping line:")
                    print(line[:200])
                    print(e)
                    continue

                # ---- Build LLaVA-style sample ----
                human_prompt = (
                    f"{DEFAULT_IMAGE_TOKEN}\n"
                    f"Critique: {obj['Critique']}\n"
                    "Task: Based on the critique, write clear step-by-step "
                    "image editing actions to improve the image."
                )

                assistant_answer = obj["List of action"]

                self.list_data_dict.append({
                    "image": obj["Image_URL"],   # ONLY original image
                    "conversations": [
                        {"from": "human", "value": human_prompt},
                        {"from": "gpt", "value": assistant_answer},
                    ],
                    "id": obj.get("ID", None),
                })

        self.processor = processor
        self.data_args = data_args
        self.model_id = model_id
        self.padding = padding

        self.image_min_pixel = data_args.image_min_pixels
        self.image_max_pixel = data_args.image_max_pixels
        self.image_resized_w = data_args.image_resized_width
        self.image_resized_h = data_args.image_resized_height

        # Qwen-VL patch size
        self.image_patch_size = 16 if "Qwen3" in model_id else 14

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]

        # ===== Load image =====
        image_file = sources["image"]
        if not os.path.exists(image_file) and not image_file.startswith("http"):
            image_file = os.path.join(self.data_args.image_folder, image_file)

        image_input = get_image_info(
            image_file,
            self.image_min_pixel,
            self.image_max_pixel,
            self.image_resized_w,
            self.image_resized_h,
            self.image_patch_size,
        )

        # ===== Convert conversation format =====
        sources = llava_to_openai(sources["conversations"], is_video=False)

        all_input_ids = []
        all_labels = []
        all_pixel_values = []
        all_image_grid_thw = []

        # ===== Optional system message =====
        if len(SYSTEM_MESSAGE) > 0 and "Qwen3" not in self.model_id:
            system_message = (
                f"{DEFAULT_IM_START_TOKEN}system\n"
                f"{SYSTEM_MESSAGE}"
                f"{DEFAULT_IM_END_TOKEN}\n"
            )
            system_ids = self.processor.tokenizer(
                system_message, add_special_tokens=False, return_tensors="pt"
            )["input_ids"]

            all_input_ids.append(system_ids.squeeze(0))
            all_labels.append(torch.full_like(system_ids.squeeze(0), IGNORE_INDEX))

        # ===== Single-turn conversation =====
        user_input = sources[0]
        gpt_response = sources[1]

        user_text = (
            f"{DEFAULT_IM_START_TOKEN}{user_input['role']}\n"
            f"{user_input['content']}"
            f"{DEFAULT_IM_END_TOKEN}\n"
            f"{DEFAULT_IM_START_TOKEN}{gpt_response['role']}\n"
        )

        assistant_text = f"{gpt_response['content']}{DEFAULT_IM_END_TOKEN}\n"

        # Encode image + prompt
        inputs = self.processor(
            text=[user_text],
            images=[image_input],
            padding=False,
            do_resize=False,
            return_tensors="pt",
        )

        prompt_ids = inputs["input_ids"]
        response_ids = self.processor.tokenizer(
            assistant_text, add_special_tokens=False, return_tensors="pt"
        )["input_ids"]

        input_ids = torch.cat([prompt_ids, response_ids], dim=1).squeeze(0)
        labels = torch.cat(
            [
                torch.full((prompt_ids.size(1),), IGNORE_INDEX),
                response_ids.squeeze(0),
            ],
            dim=0,
        )

        all_input_ids.append(input_ids)
        all_labels.append(labels)

        input_ids = torch.cat(all_input_ids)
        labels = torch.cat(all_labels)

        attention_mask = input_ids != self.processor.tokenizer.pad_token_id

        data_dict = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "pixel_values": inputs["pixel_values"],
            "image_grid_thw": inputs["image_grid_thw"],
        }

        return data_dict
