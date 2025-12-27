import torch 
import torch.nn.functional as F
from PIL import Image 
from torchvision import transforms 

from transformers import CLIPProcessor, CLIPModel
import timm


def cosine_similarity(a,b):
    a = F.normalize(a,dim = -1)
    b = F.normalize(b,dim = -1)
    return (a*b).sum(dim = -1)


def l1(i_pred,i_gt,reduction = "mean"):
    diff = torch.abs(i_pred - i_gt)
    if reduction == "mean":
        return diff.flatten(1).mean(dim = 1)
    return diff.mean()


def l2(i_pred,i_gt,reduction = "mean"):
    diff = (i_pred - i_gt)**2
    if reduction == "mean":
        return diff.flatten(1).mean(dim = 1)
    return diff.mean()

class CLIPScorer:
    """
    - CLIP-I: cosine( clip_image(I_pred), clip_image(I_gt) )
    - CLIP-T: cosine( clip_text(instruction), clip_image(I_pred) )
    """
    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def encode_images(self, pil_images: list[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=pil_images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        feats = self.model.get_image_features(**inputs)  # (B, D)
        return feats

    @torch.no_grad()
    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        feats = self.model.get_text_features(**inputs)  # (B, D)
        return feats

    @torch.no_grad()
    def clip_i(self, pred_images: list[Image.Image], gt_images: list[Image.Image]) -> torch.Tensor:
        """
        CLIP-I ↑ : image-image similarity
        return (B,)
        """
        f_pred = self.encode_images(pred_images)
        f_gt   = self.encode_images(gt_images)
        return cosine_similarity(f_pred, f_gt)

    @torch.no_grad()
    def clip_t(self, instructions: list[str], pred_images: list[Image.Image]) -> torch.Tensor:
        """
        CLIP-T ↑ : text-image similarity
        return (B,)
        """
        f_txt  = self.encode_texts(instructions)
        f_pred = self.encode_images(pred_images)
        return cosine_similarity(f_txt, f_pred)

class DINOScorer:
    """
    DINO similarity ↑ : cosine( dino(I_pred), dino(I_gt) )

    Mặc định dùng DINOv2 ViT (timm). Bạn có thể đổi model_name nếu muốn.
    """
    def __init__(self, model_name="vit_base_patch14_dinov2.lvd142m", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0).to(self.device).eval()

        # Preprocess chuẩn ImageNet / ViT
        self.transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
        ])

    @torch.no_grad()
    def encode_images(self, pil_images: list[Image.Image]) -> torch.Tensor:
        x = torch.stack([self.transform(im.convert("RGB")) for im in pil_images], dim=0).to(self.device)  # (B,3,224,224)
        feats = self.model(x)  # (B, D) because num_classes=0 returns global pooled features
        return feats

    @torch.no_grad()
    def dino_sim(self, pred_images: list[Image.Image], gt_images: list[Image.Image]) -> torch.Tensor:
        """
        DINO ↑ : image-image similarity (B,)
        """
        f_pred = self.encode_images(pred_images)
        f_gt   = self.encode_images(gt_images)
        return cosine_similarity(f_pred, f_gt)


class ImageEditEvaluator:
    """
    Compute:
      L1↓, L2↓  (pixel metrics, require tensors)
      CLIP-I↑, DINO↑, CLIP-T↑ (feature similarities, use PIL images + instruction)
    """
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.clip = CLIPScorer(device=self.device)
        self.dino = DINOScorer(device=self.device)

    @torch.no_grad()
    def evaluate(
        self,
        I_pred_tensor: torch.Tensor,  # (B,3,H,W) in [0,1]
        I_gt_tensor: torch.Tensor,    # (B,3,H,W) in [0,1]
        pred_pil: list[Image.Image],
        gt_pil: list[Image.Image],
        instructions: list[str],
    ) -> dict:
        assert I_pred_tensor.shape == I_gt_tensor.shape, "I_pred_tensor and I_gt_tensor must have same shape"
        B = I_pred_tensor.shape[0]
        assert len(pred_pil) == len(gt_pil) == len(instructions) == B

        # Pixel metrics (lower better)
        L1 = l1(I_pred_tensor, I_gt_tensor, reduction="none")  # (B,)
        L2 = l2(I_pred_tensor, I_gt_tensor, reduction="none")  # (B,)

        # Feature metrics (higher better)
        clip_i = self.clip.clip_i(pred_pil, gt_pil)        # (B,)
        dino   = self.dino.dino_sim(pred_pil, gt_pil)      # (B,)
        clip_t = self.clip.clip_t(instructions, pred_pil)  # (B,)

        return {
            "L1": L1.detach().cpu(),
            "L2": L2.detach().cpu(),
            "CLIP-I": clip_i.detach().cpu(),
            "DINO": dino.detach().cpu(),
            "CLIP-T": clip_t.detach().cpu(),
        }


if __name__ == "__main__":
    pred_img = Image.open("D:/Qwen_finetune_SFT/8a4115ba98bcd8b2__odq5gt_sim_0.8009999990463257_2.jpg").convert("RGB")
    gt_img   = Image.open("D:/Qwen_finetune_SFT/odq5gt_output_google_nano_banana_reviewer2.png").convert("RGB")
    instruction = "The photo has a good idea working for it. The open, vast landscape with the road pointing towards it is pushing the viewer's attention to go out and experience the great outdoors. However, placing the bridge and the road off-center somehow defeats the idea. Placing the road and the bridge in the center of the frame would solve the issue. Stepping back a bit to still include the signpost on the left side is ideal and would complete the image."

    to_tensor01 = transforms.ToTensor()
    I_pred = to_tensor01(pred_img).unsqueeze(0)  
    I_gt   = to_tensor01(gt_img).unsqueeze(0)   

    evaluator = ImageEditEvaluator()
    scores = evaluator.evaluate(
        I_pred_tensor=I_pred,
        I_gt_tensor=I_gt,
        pred_pil=[pred_img],
        gt_pil=[gt_img],
        instructions=[instruction],
    )

    for k, v in scores.items():
        print(k, float(v.mean()))