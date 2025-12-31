import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from transformers import CLIPProcessor, CLIPModel
import timm

# =========================
# Utils
# =========================
def cosine_similarity(a, b):
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return (a * b).sum(dim=-1)


def l1(i_pred, i_gt):
    return torch.abs(i_pred - i_gt).flatten(1).mean(dim=1)


def l2(i_pred, i_gt):
    return ((i_pred - i_gt) ** 2).flatten(1).mean(dim=1)


def resize_to_gt(I_pred, I_gt):
    _, _, H, W = I_gt.shape
    return F.interpolate(I_pred, size=(H, W), mode="bilinear", align_corners=False)


# =========================
# CLIP scorer
# =========================
class CLIPScorer:
    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def encode_images(self, pil_images):
        inputs = self.processor(
            images=pil_images,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return self.model.get_image_features(**inputs)

    @torch.no_grad()
    def encode_texts(self, texts):
        inputs = self.processor(
            text=texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        assert "input_ids" in inputs, "CLIP text inputs missing input_ids"
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return self.model.get_text_features(**inputs)

    def clip_i(self, pred_images, gt_images):
        f_pred = self.encode_images(pred_images)
        f_gt   = self.encode_images(gt_images)
        return cosine_similarity(f_pred, f_gt)

    def clip_t(self, instructions, pred_images):
        f_txt  = self.encode_texts(instructions)
        f_pred = self.encode_images(pred_images)
        return cosine_similarity(f_txt, f_pred)



# =========================
# DINO scorer (FIXED)
# =========================
# -------------------------
# DINO scorer (OFFICIAL)
# -------------------------
class DINOScorer:
    """
    DINO similarity ↑ : cosine( dino(I_pred), dino(I_gt) )
    Uses official DINO v1 (facebookresearch).
    """

    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # ✅ OFFICIAL DINO
        self.model = torch.hub.load(
            "facebookresearch/dino:main",
            "dino_vitb16"
        ).to(self.device).eval()

        self.transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
        ])

    @torch.no_grad()
    def encode_images(self, pil_images):
        x = torch.stack(
            [self.transform(im.convert("RGB")) for im in pil_images],
            dim=0
        ).to(self.device)
        feats = self.model(x)  # (B, 768)
        return feats

    @torch.no_grad()
    def dino_sim(self, pred_images, gt_images):
        f_pred = self.encode_images(pred_images)
        f_gt   = self.encode_images(gt_images)
        return cosine_similarity(f_pred, f_gt)


# =========================
# Reward Models
# =========================
class RewardScorer:
    """
    ImgRwd, PickScore, UniRwd
    """
    def __init__(self):
        self.available = {}

        # ImageReward
        try:
            import ImageReward
            self.img_reward = ImageReward.load("ImageReward-v1.0")
            self.available["ImgRwd"] = True
        except Exception:
            self.available["ImgRwd"] = False

        # PickScore
        try:
            from pickscore import PickScore
            self.pickscore = PickScore()
            self.available["PickScore"] = True
        except Exception:
            self.available["PickScore"] = False

        # UnifiedReward (optional)
        try:
            from unified_reward import UnifiedReward
            self.unirwd = UnifiedReward()
            self.available["UniRwd"] = True
        except Exception:
            self.available["UniRwd"] = False

    def score(self, prompt, image):
        scores = {}

        if self.available.get("ImgRwd"):
            scores["ImgRwd"] = self.img_reward.score(prompt, image)

        if self.available.get("PickScore"):
            scores["PickScore"] = self.pickscore.score(prompt, image)

        if self.available.get("UniRwd"):
            scores["UniRwd"] = self.unirwd.score(prompt, image)

        return scores


# =========================
# Main Evaluator
# =========================
class ImageEditEvaluator:
    def __init__(self):
        self.clip = CLIPScorer()
        self.dino = DINOScorer()
        self.reward = RewardScorer()

    @torch.no_grad()
    def evaluate(
        self,
        I_pred_tensor,
        I_gt_tensor,
        pred_pil,
        gt_pil,
        instructions,
    ):
        if I_pred_tensor.shape != I_gt_tensor.shape:
            I_pred_tensor = resize_to_gt(I_pred_tensor, I_gt_tensor)

        out = {}

        # Pixel metrics
        out["L1"] = l1(I_pred_tensor, I_gt_tensor).mean().item()
        out["L2"] = l2(I_pred_tensor, I_gt_tensor).mean().item()

        # Feature metrics
        out["CLIP-I"] = self.clip.clip_i(pred_pil, gt_pil).mean().item()
        out["DINO"] = self.dino.dino_sim(pred_pil, gt_pil).mean().item()
        out["CLIP-T"] = self.clip.clip_t(instructions, pred_pil).mean().item()

        # Rewards
        rewards = self.reward.score(instructions[0], pred_pil[0])
        out.update(rewards)

        return out


# =========================
# Example
# =========================
if __name__ == "__main__":
    pred_img = Image.open("/disk/yuu/Qwen_finetune_SFT/8a4115ba98bcd8b2__odq5gt_sim_0.8009999990463257_2.jpg").convert("RGB")
    gt_img = Image.open("/disk/yuu/Qwen_finetune_SFT/odq5gt_output_google_nano_banana_reviewer2.png").convert("RGB")

    instruction = "Action 1: Place the road and the bridge in the center of the frame. Action 2: Step back to include the signpost on the left side."

    to_tensor = transforms.ToTensor()
    I_pred = to_tensor(pred_img).unsqueeze(0)
    I_gt = to_tensor(gt_img).unsqueeze(0)

    evaluator = ImageEditEvaluator()
    scores = evaluator.evaluate(
        I_pred, I_gt,
        [pred_img], [gt_img],
        [instruction]
    )

    for k, v in scores.items():
        print(f"{k}: {v}")
