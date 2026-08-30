"""
Change Detection Model Wrapper
Uses a lightweight difference-based approach as baseline.
Swap self.model with your ChangeFormer checkpoint when ready.
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ChangeDetectionModel:
    """
    Wraps change detection inference.
    Currently uses a ResNet-based Siamese diff network as baseline.
    Replace load_checkpoint() with your fine-tuned ChangeFormer weights.
    """

    def __init__(self):
        self.transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
        self.model = self._build_model()
        logger.info("ChangeDetection model initialised (baseline diff-net)")

    def _build_model(self):
        """
        Lightweight Siamese difference network.
        Replace this entire method body with:
            model = ChangeFormer(...)
            model.load_state_dict(torch.load('your_checkpoint.pth'))
        """
        import torchvision.models as tvm
        backbone = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT)
        backbone.fc = nn.Identity()
        return backbone.to(DEVICE).eval()

    def predict(self, img1: Image.Image, img2: Image.Image) -> dict:
        """
        Run change detection on an image pair.
        Returns change map array + statistics.
        """
        t1 = self.transform(img1.convert("RGB")).unsqueeze(0).to(DEVICE)
        t2 = self.transform(img2.convert("RGB")).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            f1 = self.model(t1)
            f2 = self.model(t2)
            diff = torch.abs(f1 - f2)
            change_score = float(diff.mean().cpu())

        # Generate pseudo change map using pixel-level diff
        arr1 = np.array(img1.convert("RGB").resize((256, 256))).astype(float)
        arr2 = np.array(img2.convert("RGB").resize((256, 256))).astype(float)
        pixel_diff = np.abs(arr1 - arr2).mean(axis=2)
        threshold  = np.percentile(pixel_diff, 80)
        change_map = (pixel_diff > threshold).astype(np.uint8) * 255

        changed_pixels   = int(np.sum(change_map > 0))
        total_pixels     = change_map.size
        change_pct       = round(changed_pixels / total_pixels * 100, 2)
        confidence       = min(97.0, round(50 + change_score * 100, 1))

        return {
            "change_map":     change_map,        # 256×256 numpy array
            "change_percent": change_pct,
            "change_score":   round(change_score, 4),
            "confidence":     confidence,
            "changed_pixels": changed_pixels,
            "total_pixels":   total_pixels,
        }
