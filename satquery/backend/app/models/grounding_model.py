from PIL import Image
import numpy as np

class RemoteSensingGrounding:
    def __init__(self, checkpoint_path: str = None, device: str = "cpu"):
        self.checkpoint_path = checkpoint_path
        self.device = device

    def ground_text(self, image: Image.Image, phrase: str) -> dict:
        """Grounds text phrases to bounding boxes [ymin, xmin, ymax, xmax]."""
        w, h = image.size
        # Sample grounded bbox
        bboxes = [[int(h * 0.2), int(w * 0.2), int(h * 0.6), int(w * 0.7)]]
        return {
            "bboxes": bboxes,
            "labels": [phrase],
            "confidence": 0.912,
            "tool": "RS-GroundingDINO"
        }