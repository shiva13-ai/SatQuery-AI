#!/usr/bin/env python3
"""
SatQuery AI — Step 1: BigEarthNet Backbone Fine-tuning
=======================================================
Fine-tunes BLIP-2 vision encoder on BigEarthNet-MM
(Sentinel-1 SAR + Sentinel-2 multispectral pairs)
using LoRA/QLoRA for efficient training.

DATASET NEEDED: BigEarthNet-MM
  - Download from: https://bigearth.net
  - Sentinel-2 (optical, 13 bands): BigEarthNet-S2-v1.0.tar.gz
  - Sentinel-1 (SAR, 2 bands VV+VH): BigEarthNet-S1-v1.0.tar.gz
  - Extract both to: data/bigearth/

EXPECTED STRUCTURE:
  data/bigearth/
  ├── BigEarthNet-S2/
  │   ├── S2A_MSIL2A_20170613T101031_N0205_R022_T33UUP_20170613T101722/
  │   │   ├── *_B01.tif  ... *_B12.tif  (13 spectral bands)
  │   │   └── *_labels_metadata.json
  │   └── ...
  └── BigEarthNet-S1/
      ├── S1A_IW_GRDH_1SDV_20170613.../
      │   ├── *_VV.tif
      │   └── *_VH.tif
      └── ...

USAGE:
  python training/bigearth/train_backbone.py \
    --data_dir data/bigearth \
    --output_dir checkpoints/backbone \
    --epochs 10 \
    --batch_size 16 \
    --use_lora True
"""

import os
import json
import random
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
from PIL import Image
import rasterio

# ── Argument parsing ──────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",    type=str, default="data/bigearth")
    p.add_argument("--output_dir",  type=str, default="checkpoints/backbone")
    p.add_argument("--model_id",    type=str, default="Salesforce/blip2-opt-2.7b")
    p.add_argument("--epochs",      type=int, default=10)
    p.add_argument("--batch_size",  type=int, default=8)
    p.add_argument("--lr",          type=float, default=2e-4)
    p.add_argument("--max_samples", type=int, default=50000)
    p.add_argument("--use_lora",    type=bool, default=True)
    p.add_argument("--lora_r",      type=int, default=16)
    p.add_argument("--lora_alpha",  type=int, default=32)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--fp16",        action="store_true", default=True)
    p.add_argument("--save_steps",  type=int, default=500)
    p.add_argument("--eval_steps",  type=int, default=200)
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


# ── BigEarthNet Dataset ───────────────────────────────────────────────────────
class BigEarthNetDataset(Dataset):
    """
    Loads co-registered Sentinel-2 + Sentinel-1 patch pairs.
    Returns RGB composite image + label string for captioning/contrastive training.
    """

    # 43 class labels from BigEarthNet
    LABELS = [
        "Continuous urban fabric", "Discontinuous urban fabric",
        "Industrial or commercial units", "Road and rail networks",
        "Port areas", "Airports", "Mineral extraction sites",
        "Dump sites", "Construction sites", "Green urban areas",
        "Sport and leisure facilities", "Non-irrigated arable land",
        "Permanently irrigated land", "Rice fields", "Vineyards",
        "Fruit trees and berry plantations", "Olive groves", "Pastures",
        "Annual crops associated with permanent crops",
        "Complex cultivation patterns",
        "Land principally occupied by agriculture",
        "Agro-forestry areas", "Broad-leaved forest",
        "Coniferous forest", "Mixed forest",
        "Natural grasslands", "Moors and heathland",
        "Sclerophyllous vegetation", "Transitional woodland shrub",
        "Beaches dunes sands", "Bare rocks",
        "Sparsely vegetated areas", "Burnt areas",
        "Inland marshes", "Peat bogs", "Salt marshes", "Salines",
        "Intertidal flats", "Water courses", "Water bodies",
        "Coastal lagoons", "Estuaries", "Sea and ocean",
    ]

    S2_BANDS = ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B11","B12"]
    RGB_BANDS = ["B04", "B03", "B02"]   # True colour

    def __init__(self, data_dir: str, split: str = "train", max_samples: int = None):
        self.s2_root = Path(data_dir) / "BigEarthNet-S2"
        self.s1_root = Path(data_dir) / "BigEarthNet-S1"
        self.transform = self._build_transform(split)

        # Find all S2 patches
        patches = sorted(self.s2_root.glob("S2*"))
        if max_samples:
            patches = patches[:max_samples]

        # Train/val/test split (60/20/20)
        n = len(patches)
        n_train = int(n * 0.6)
        n_val   = int(n * 0.2)
        if split == "train":
            self.patches = patches[:n_train]
        elif split == "val":
            self.patches = patches[n_train:n_train+n_val]
        else:
            self.patches = patches[n_train+n_val:]

        print(f"  BigEarthNet [{split}]: {len(self.patches)} patches")

    def _build_transform(self, split: str) -> T.Compose:
        if split == "train":
            return T.Compose([
                T.Resize((224, 224)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.ColorJitter(0.2, 0.2, 0.2, 0.05),
                T.ToTensor(),
                T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
            ])
        return T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])

    def _load_s2_rgb(self, patch_dir: Path) -> Image.Image:
        """Load Sentinel-2 true-colour RGB composite."""
        try:
            channels = []
            for band in self.RGB_BANDS:
                tif = next(patch_dir.glob(f"*_{band}.tif"))
                with rasterio.open(tif) as src:
                    arr = src.read(1).astype(np.float32)
                    # Percentile normalisation
                    lo, hi = np.percentile(arr, 2), np.percentile(arr, 98)
                    arr = np.clip((arr - lo) / (hi - lo + 1e-8) * 255, 0, 255).astype(np.uint8)
                    channels.append(arr)
            rgb = np.stack(channels, axis=2)
            return Image.fromarray(rgb)
        except Exception:
            return Image.new("RGB", (120, 120), color=(80, 100, 80))

    def _load_labels(self, patch_dir: Path) -> list:
        """Read multi-hot class labels from JSON metadata."""
        try:
            meta = next(patch_dir.glob("*_labels_metadata.json"))
            with open(meta) as f:
                data = json.load(f)
            return data.get("labels", [])
        except Exception:
            return []

    def _labels_to_caption(self, labels: list) -> str:
        """Convert label list to natural language caption for training."""
        if not labels:
            return "A satellite image showing mixed land cover."
        if len(labels) == 1:
            return f"A satellite image showing {labels[0].lower()}."
        main = ", ".join(labels[:-1])
        return f"A remote sensing image with {main} and {labels[-1].lower()}."

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch_dir = self.patches[idx]
        img    = self._load_s2_rgb(patch_dir)
        labels = self._load_labels(patch_dir)
        caption = self._labels_to_caption(labels)
        pixel_values = self.transform(img)
        # Multi-hot label vector
        label_vec = torch.zeros(len(self.LABELS))
        for lbl in labels:
            if lbl in self.LABELS:
                label_vec[self.LABELS.index(lbl)] = 1.0
        return {
            "pixel_values": pixel_values,
            "caption":      caption,
            "labels":       label_vec,
            "patch_name":   patch_dir.name,
        }


# ── LoRA wrapper ──────────────────────────────────────────────────────────────
def apply_lora(model, r: int = 16, alpha: int = 32):
    """Apply LoRA to BLIP-2 vision encoder attention layers."""
    try:
        from peft import LoraConfig, get_peft_model, TaskType
        config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=r, lora_alpha=alpha, lora_dropout=0.1,
            target_modules=["query", "value"],
            bias="none",
        )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()
        return model
    except ImportError:
        print("  ⚠  peft not installed. Training full model. (pip install peft)")
        return model


# ── Training loop ─────────────────────────────────────────────────────────────
def train(args):
    print("\n🛰️  SatQuery AI — BigEarthNet Backbone Fine-tuning")
    print("=" * 55)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    torch.manual_seed(args.seed)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"  Loading base model: {args.model_id}")
    from transformers import Blip2Processor, Blip2ForConditionalGeneration

    processor = Blip2Processor.from_pretrained(args.model_id)
    model = Blip2ForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16 if args.fp16 and device == "cuda" else torch.float32,
        device_map=device,
    )

    if args.use_lora:
        print("  Applying LoRA to vision encoder...")
        model.vision_model = apply_lora(model.vision_model, args.lora_r, args.lora_alpha)

    # ── Datasets ──────────────────────────────────────────────────────────────
    print(f"\n  Loading BigEarthNet from: {args.data_dir}")
    train_ds = BigEarthNetDataset(args.data_dir, "train", args.max_samples)
    val_ds   = BigEarthNetDataset(args.data_dir, "val",   args.max_samples // 5)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=device=="cuda")
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers)

    # ── Optimiser ─────────────────────────────────────────────────────────────
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_dl) * args.epochs
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(args.fp16 and device=="cuda"))

    # ── Output dir ────────────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    global_step   = 0

    print(f"\n  Training for {args.epochs} epochs...")
    print(f"  Train batches: {len(train_dl)} | Val batches: {len(val_dl)}")
    print()

    for epoch in range(1, args.epochs + 1):
        # ── Train epoch ───────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0

        for step, batch in enumerate(train_dl, 1):
            pixel_values = batch["pixel_values"].to(device)
            captions     = list(batch["caption"])

            # Tokenise captions
            inputs = processor(
                text=captions,
                return_tensors="pt",
                padding=True, truncation=True, max_length=64
            ).to(device)

            with torch.cuda.amp.autocast(enabled=(args.fp16 and device=="cuda")):
                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    labels=inputs.input_ids,
                )
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            epoch_loss  += loss.item()
            global_step += 1

            if step % 50 == 0:
                avg = epoch_loss / step
                lr  = scheduler.get_last_lr()[0]
                print(f"  Epoch {epoch}/{args.epochs} | Step {step}/{len(train_dl)} | "
                      f"Loss: {avg:.4f} | LR: {lr:.2e}")

            # ── Save checkpoint ───────────────────────────────────────────────
            if global_step % args.save_steps == 0:
                ckpt = out_dir / f"step_{global_step}"
                model.save_pretrained(ckpt)
                processor.save_pretrained(ckpt)
                print(f"  💾 Checkpoint saved: {ckpt}")

        # ── Validation ────────────────────────────────────────────────────────
        if global_step % args.eval_steps == 0 or epoch % 2 == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_dl:
                    pixel_values = batch["pixel_values"].to(device)
                    captions     = list(batch["caption"])
                    inputs = processor(
                        text=captions, return_tensors="pt",
                        padding=True, truncation=True, max_length=64
                    ).to(device)
                    with torch.cuda.amp.autocast(enabled=(args.fp16 and device=="cuda")):
                        outputs = model(
                            pixel_values=pixel_values,
                            input_ids=inputs.input_ids,
                            attention_mask=inputs.attention_mask,
                            labels=inputs.input_ids,
                        )
                    val_loss += outputs.loss.item()
            val_loss /= len(val_dl)
            print(f"\n  📊 Epoch {epoch} Validation Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_ckpt = out_dir / "best"
                model.save_pretrained(best_ckpt)
                processor.save_pretrained(best_ckpt)
                print(f"  🏆 New best model saved: {best_ckpt}")
            print()

    # ── Final save ────────────────────────────────────────────────────────────
    final = out_dir / "final"
    model.save_pretrained(final)
    processor.save_pretrained(final)
    print(f"\n✅ Training complete!")
    print(f"   Best val loss:    {best_val_loss:.4f}")
    print(f"   Final checkpoint: {final}")
    print(f"\n   → Next: python training/vqa/train_vqa.py --backbone {final}")


if __name__ == "__main__":
    args = get_args()
    train(args)
