import os

from typing import Dict

from io import BytesIO



import torch

import transformers

import ujson as json

from torch.utils.data import Dataset

from PIL import Image

import requests



from src.params import DataArguments

from src.constants import (

    IGNORE_INDEX,

    DEFAULT_IM_START_TOKEN,

    DEFAULT_IM_END_TOKEN,

    DEFAULT_IMAGE_TOKEN,

)



from src.dataset.qwen_vl_collator import QwenVLDataCollator

from src.utils import llava_to_openai





# =========================================================

# Helper: load image (local or URL)

# =========================================================

def load_image(path_or_url: str) -> Image.Image:

    if path_or_url.startswith("http"):

        r = requests.get(path_or_url, timeout=10)

        r.raise_for_status()

        return Image.open(BytesIO(r.content)).convert("RGB")

    else:

        return Image.open(path_or_url).convert("RGB")





# =========================================================

# Dataset

# =========================================================

class SupervisedDataset(Dataset):

    """

    Image Editing SFT Dataset (Qwen-VL)



    Input  : image + critique

    Output : list of editing actions (text)



    NOTE:

      - input_ids / labels: dùng cho training

      - metadata fields   : CHỈ dùng cho logging / tracing

    """



    def __init__(

        self,

        data_path: str,

        processor: transformers.ProcessorMixin,

        data_args: DataArguments,

        model_id: str,

    ):

        super().__init__()



        self.processor = processor

        self.data_args = data_args

        self.model_id = model_id



        self.samples = []



        # ===== Load JSONL =====

        with open(data_path, "r", encoding="utf-8") as f:

            for line_idx, line in enumerate(f):

                line = line.strip()

                if not line:

                    continue



                try:

                    obj = json.loads(line)

                except Exception as e:

                    print(f"[WARN] JSON parse error at line {line_idx}, skip")

                    print(e)

                    continue



                human_prompt = (

                    f"{DEFAULT_IMAGE_TOKEN}\n"

                    f"Critique: {obj['Critique']}\n"

                    "Task: Based on the critique, write clear step-by-step "

                    "image editing actions to improve the image."

                )



                assistant_answer = obj["List of action"]



                self.samples.append({

                    # ===== identifiers =====

                    "id": obj.get("ID", f"line_{line_idx}"),

                    "Post_ID": obj.get("Post_ID"),

                    "Reviewer_ID": obj.get("ReviewerID"),



                    # ===== data =====

                    "image": obj["Image_URL"],

                    "conversations": [

                        {"from": "human", "value": human_prompt},

                        {"from": "gpt", "value": assistant_answer},

                    ],

                })



    def __len__(self):

        return len(self.samples)



    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:

        sample = self.samples[idx]



        # ===== Load image =====

        image_path = sample["image"]

        if not image_path.startswith("http") and not os.path.exists(image_path):

            image_path = os.path.join(self.data_args.image_folder, image_path)



        image = load_image(image_path)



        # ===== Convert conversation to OpenAI-style =====

        conv = llava_to_openai(sample["conversations"], is_video=False)

        user_msg, gpt_msg = conv



        user_text = (

            f"{DEFAULT_IM_START_TOKEN}{user_msg['role']}\n"

            f"{user_msg['content']}"

            f"{DEFAULT_IM_END_TOKEN}\n"

            f"{DEFAULT_IM_START_TOKEN}{gpt_msg['role']}\n"

        )



        assistant_text = f"{gpt_msg['content']}{DEFAULT_IM_END_TOKEN}"



        # ===== Encode prompt + image =====

        inputs = self.processor(

            text=user_text,

            images=image,

            return_tensors="pt",

        )



        prompt_ids = inputs["input_ids"]              # (1, Lp)

        pixel_values = inputs["pixel_values"]         # (N, C)

        image_grid_thw = inputs["image_grid_thw"]     # (1, 3)



        # ===== Encode response =====

        response_ids = self.processor.tokenizer(

            assistant_text,

            add_special_tokens=False,

            return_tensors="pt",

        )["input_ids"]                                # (1, Lr)



        # ===== Build final tensors (BEFORE truncate) =====

        input_ids = torch.cat([prompt_ids, response_ids], dim=1).squeeze(0)

        labels = torch.cat(

            [

                torch.full_like(prompt_ids, IGNORE_INDEX),

                response_ids,

            ],

            dim=1,

        ).squeeze(0)



        attention_mask = input_ids != self.processor.tokenizer.pad_token_id



        # =========================================================

        # 🔥 HARD TRUNCATE (CRITICAL – PREVENT OOM)

        # =========================================================

        max_len = self.data_args.max_seq_length



        if input_ids.shape[0] > max_len:

            # lengths

            Lp = prompt_ids.shape[1]

            Lr = response_ids.shape[1]



            # always keep full response

            if Lr >= max_len:

                # extreme case: response itself too long

                input_ids = response_ids.squeeze(0)[-max_len:]

                labels = response_ids.squeeze(0)[-max_len:]

                attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

            else:

                # truncate prompt

                keep_prompt_len = max_len - Lr



                input_ids = torch.cat(

                    [

                        prompt_ids.squeeze(0)[-keep_prompt_len:],

                        response_ids.squeeze(0),

                    ],

                    dim=0,

                )



                labels = torch.cat(

                    [

                        torch.full((keep_prompt_len,), IGNORE_INDEX, dtype=torch.long),

                        response_ids.squeeze(0),

                    ],

                    dim=0,

                )



                attention_mask = torch.ones_like(input_ids, dtype=torch.bool)



        # ===== RETURN =====

        return {

            # --- training ---

            "input_ids": input_ids,

            "labels": labels,

            "attention_mask": attention_mask,

            "pixel_values": pixel_values,

            "image_grid_thw": image_grid_thw,



            # --- metadata (for logging / tracing only) ---

            "sample_id": sample["id"],

            "post_id": sample["Post_ID"],

            "reviewer_id": sample["Reviewer_ID"],

            "image_url": sample["image"],

        }





# =========================================================

# Data module

# =========================================================

def make_supervised_data_module(model_id, processor, data_args):

    train_dataset = SupervisedDataset(

        data_path=data_args.data_path,

        processor=processor,

        data_args=data_args,

        model_id=model_id,

    )



    eval_dataset = None

    if data_args.eval_path is not None:

        eval_dataset = SupervisedDataset(

            data_path=data_args.eval_path,

            processor=processor,

            data_args=data_args,

            model_id=model_id,

        )



    return {

        "train_dataset": train_dataset,

        "eval_dataset": eval_dataset,

        "data_collator": QwenVLDataCollator(),  # unchanged

    }

