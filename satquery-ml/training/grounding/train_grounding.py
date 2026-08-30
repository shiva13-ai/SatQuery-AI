#!/usr/bin/env python3
"""
SatQuery AI — Step 5: RS Grounding Fine-tuning (Grounding DINO)
================================================================
Fine-tunes Grounding DINO on VRSBench grounding annotations
for text-guided region detection in satellite images.

DATASET: VRSBench (grounding split)
  data/vrsbench/
  ├── images/
  └── grounding_train.json   ← COCO-format bounding box annotations
      Format: {"images":[...], "annotations":[{"image_id","category_id","bbox":[x,y,w,h],"caption":"..."}]}

USAGE:
  python training/grounding/train_grounding.py \
    --data_dir data/vrsbench \
    --output_dir checkpoints/grounding \
    --epochs 30
"""

import json
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
import numpy as np

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   type=str, default="data/vrsbench")
    p.add_argument("--model_id",   type=str, default="IDEA-Research/grounding-dino-tiny")
    p.add_argument("--output_dir", type=str, default="checkpoints/grounding")
    p.add_argument("--epochs",     type=int, default=30)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--fp16",       action="store_true", default=True)
    p.add_argument("--num_workers",type=int, default=2)
    p.add_argument("--iou_thresh", type=float, default=0.5)
    return p.parse_args()


# ── Grounding Dataset ─────────────────────────────────────────────────────────
class RSGroundingDataset(Dataset):
    """
    Loads image + text query → bounding boxes for RS grounding.
    Supports COCO-format and VRSBench-format annotation files.
    """

    SYNTHETIC_QUERIES = [
        "water body", "building", "road", "forest", "farmland",
        "river", "parking lot", "airport runway", "urban area",
        "vegetation", "bare land", "water", "residential area",
    ]

    def __init__(self, data_dir: str, split: str, processor, img_size: int = 800):
        self.data_dir  = Path(data_dir)
        self.processor = processor
        self.img_dir   = self.data_dir / "images"
        self.img_size  = img_size
        self.samples   = self._load(split)
        print(f"  RS-Grounding [{split}]: {len(self.samples)} samples")

    def _load(self, split: str) -> list:
        candidates = [
            self.data_dir / f"grounding_{split}.json",
            self.data_dir / f"{split}_grounding.json",
            self.data_dir / f"referring_{split}.json",
        ]
        for f in candidates:
            if f.exists():
                with open(f) as fp:
                    data = json.load(fp)
                return self._parse_coco(data)

        print(f"  ⚠  No grounding file for '{split}'. Using synthetic data.")
        return self._synthetic(300 if split=="train" else 80)

    def _parse_coco(self, data: dict) -> list:
        img_map = {img["id"]: img["file_name"] for img in data.get("images", [])}
        samples = []
        for ann in data.get("annotations", []):
            img_file = img_map.get(ann.get("image_id",""), "")
            query    = ann.get("caption", ann.get("text", "object"))
            bbox     = ann.get("bbox", [0,0,100,100])   # [x,y,w,h]
            samples.append({"image": img_file, "query": query, "bbox": bbox})
        return samples

    def _synthetic(self, n: int) -> list:
        samples = []
        for i in range(n):
            query = self.SYNTHETIC_QUERIES[i % len(self.SYNTHETIC_QUERIES)]
            x, y  = np.random.randint(10, 180, 2)
            w, h  = np.random.randint(40, 120, 2)
            samples.append({
                "image": f"img_{i}.jpg",
                "query": query,
                "bbox":  [int(x), int(y), int(w), int(h)],
            })
        return samples

    def _load_img(self, name) -> Image.Image:
        for ext in [".jpg",".jpeg",".png",".tif",".tiff",""]:
            p = self.img_dir / (str(name) + ext)
            if p.exists():
                try:
                    return Image.open(p).convert("RGB")
                except Exception:
                    pass
        # Synthetic RS image
        colors = [(34,85,34),(70,120,180),(200,180,120),(60,40,20),(100,100,100)]
        arr = np.zeros((256,256,3), dtype=np.uint8)
        for _ in range(5):
            c  = colors[np.random.randint(len(colors))]
            x1 = np.random.randint(0,200); y1 = np.random.randint(0,200)
            x2 = x1+np.random.randint(20,80); y2 = y1+np.random.randint(20,80)
            arr[y1:min(y2,256), x1:min(x2,256)] = c
        return Image.fromarray(arr)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s     = self.samples[idx]
        img   = self._load_img(s["image"])
        query = s["query"]
        bbox  = s["bbox"]  # [x,y,w,h] → convert to cx,cy,w,h normalised

        W, H  = img.size
        cx = (bbox[0] + bbox[2]/2) / W
        cy = (bbox[1] + bbox[3]/2) / H
        bw = bbox[2] / W
        bh = bbox[3] / H
        target_box = torch.tensor([cx, cy, bw, bh], dtype=torch.float32)

        enc = self.processor(images=img, text=query, return_tensors="pt")
        return {
            "pixel_values":  enc.pixel_values.squeeze(0),
            "input_ids":     enc.input_ids.squeeze(0),
            "attention_mask":enc.attention_mask.squeeze(0),
            "target_box":    target_box,
            "query":         query,
        }


# ── IoU metric ────────────────────────────────────────────────────────────────
def box_iou_xyxy(pred_cx, pred_cy, pred_w, pred_h,
                  gt_cx,   gt_cy,   gt_w,   gt_h) -> float:
    """Compute IoU between two cx,cy,w,h boxes."""
    def to_xyxy(cx,cy,w,h):
        return cx-w/2, cy-h/2, cx+w/2, cy+h/2
    px1,py1,px2,py2 = to_xyxy(pred_cx, pred_cy, pred_w, pred_h)
    gx1,gy1,gx2,gy2 = to_xyxy(gt_cx,   gt_cy,   gt_w,   gt_h)
    inter_x1 = max(px1,gx1); inter_y1 = max(py1,gy1)
    inter_x2 = min(px2,gx2); inter_y2 = min(py2,gy2)
    inter = max(0, inter_x2-inter_x1) * max(0, inter_y2-inter_y1)
    union = (px2-px1)*(py2-py1) + (gx2-gx1)*(gy2-gy1) - inter
    return inter / (union + 1e-6)


# ── Training ──────────────────────────────────────────────────────────────────
def train(args):
    print("\n🛰️  SatQuery AI — Grounding Fine-tuning (Grounding DINO)")
    print("=" * 55)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device} | Model: {args.model_id}")

    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    processor = AutoProcessor.from_pretrained(args.model_id)
    model     = AutoModelForZeroShotObjectDetection.from_pretrained(args.model_id).to(device)

    # Fine-tune only the image encoder + cross-attention layers
    for name, param in model.named_parameters():
        if any(k in name for k in ["cross_attn","vision_model.encoder.layers.11",
                                    "vision_model.encoder.layers.10","bbox_embed"]):
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable/1e6:.1f}M")

    train_ds = RSGroundingDataset(args.data_dir, "train", processor)
    val_ds   = RSGroundingDataset(args.data_dir, "val",   processor)

    def collate(batch):
        max_len = max(b["input_ids"].shape[0] for b in batch)
        padded_ids  = torch.zeros(len(batch), max_len, dtype=torch.long)
        padded_msk  = torch.zeros(len(batch), max_len, dtype=torch.long)
        for i, b in enumerate(batch):
            l = b["input_ids"].shape[0]
            padded_ids[i,:l]  = b["input_ids"]
            padded_msk[i,:l]  = b["attention_mask"]
        return {
            "pixel_values":   torch.stack([b["pixel_values"]  for b in batch]),
            "input_ids":      padded_ids,
            "attention_mask": padded_msk,
            "target_box":     torch.stack([b["target_box"] for b in batch]),
        }

    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True,
                          collate_fn=collate, num_workers=args.num_workers)
    val_dl   = DataLoader(val_ds,   args.batch_size, shuffle=False,
                          collate_fn=collate, num_workers=args.num_workers)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler    = torch.cuda.amp.GradScaler(enabled=(args.fp16 and device=="cuda"))
    box_loss  = nn.SmoothL1Loss()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0

    print(f"\n  Training for {args.epochs} epochs...\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_dl, 1):
            pv  = batch["pixel_values"].to(device)
            ids = batch["input_ids"].to(device)
            msk = batch["attention_mask"].to(device)
            gt  = batch["target_box"].to(device)

            with torch.cuda.amp.autocast(enabled=(args.fp16 and device=="cuda")):
                out    = model(pixel_values=pv, input_ids=ids, attention_mask=msk)
                # Take top predicted box per sample
                logits = out.logits                   # (B, num_queries, num_classes)
                boxes  = out.pred_boxes               # (B, num_queries, 4) cx,cy,w,h
                scores = logits.sigmoid().max(-1).values  # (B, num_queries)
                best   = scores.argmax(-1)            # (B,)
                pred_boxes = boxes[torch.arange(len(best)), best]   # (B, 4)
                loss   = box_loss(pred_boxes, gt)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            total_loss += loss.item()

            if step % 50 == 0:
                print(f"  E{epoch} S{step}/{len(train_dl)} loss={total_loss/step:.4f}")

        scheduler.step()

        # Validate — Acc@0.5 IoU
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in val_dl:
                pv  = batch["pixel_values"].to(device)
                ids = batch["input_ids"].to(device)
                msk = batch["attention_mask"].to(device)
                gt  = batch["target_box"]

                with torch.cuda.amp.autocast(enabled=(args.fp16 and device=="cuda")):
                    out    = model(pixel_values=pv, input_ids=ids, attention_mask=msk)
                scores = out.logits.sigmoid().max(-1).values
                best   = scores.argmax(-1)
                preds  = out.pred_boxes[torch.arange(len(best)), best].cpu()

                for pred, g in zip(preds, gt):
                    iou = box_iou_xyxy(*pred.tolist(), *g.tolist())
                    correct += 1 if iou >= args.iou_thresh else 0
                    total   += 1

        acc = correct / max(total, 1)
        avg_loss = total_loss / len(train_dl)
        print(f"\n  📊 Epoch {epoch} | Loss={avg_loss:.4f} | Acc@0.5={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            model.save_pretrained(out_dir / "best")
            processor.save_pretrained(out_dir / "best")
            print(f"  🏆 Best grounding model (Acc@0.5={best_acc:.4f})")
        print()

    print(f"\n✅ Grounding training done! Best Acc@0.5: {best_acc:.4f}")
    print(f"   → Next: python integration/connect_checkpoints.py")


if __name__ == "__main__":
    train(get_args())
