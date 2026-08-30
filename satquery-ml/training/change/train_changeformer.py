#!/usr/bin/env python3
"""
SatQuery AI — Step 3: ChangeFormer Training on LEVIR-CD + WHU-CD
=================================================================
Trains a Siamese Transformer-based change detection model.

DATASET NEEDED: LEVIR-CD and/or WHU-CD
  Auto-download via HuggingFace:
    python datasets/setup_datasets.py --dataset levir
    python datasets/setup_datasets.py --dataset whu

EXPECTED STRUCTURE:
  data/levir_cd/
  ├── train/
  │   ├── A/          ← T1 images (before)
  │   ├── B/          ← T2 images (after)
  │   └── label/      ← Binary change masks (0=no change, 255=change)
  ├── val/
  │   ├── A/, B/, label/
  └── test/
      ├── A/, B/, label/

USAGE:
  python training/change/train_changeformer.py \
    --data_dir data/levir_cd \
    --output_dir checkpoints/change \
    --epochs 200 \
    --batch_size 8 \
    --img_size 256
"""

import argparse
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   type=str,  default="data/levir_cd")
    p.add_argument("--output_dir", type=str,  default="checkpoints/change")
    p.add_argument("--epochs",     type=int,  default=200)
    p.add_argument("--batch_size", type=int,  default=8)
    p.add_argument("--lr",         type=float,default=6e-5)
    p.add_argument("--img_size",   type=int,  default=256)
    p.add_argument("--embed_dim",  type=int,  default=256)
    p.add_argument("--num_heads",  type=int,  default=8)
    p.add_argument("--depth",      type=int,  default=4)
    p.add_argument("--fp16",       action="store_true", default=True)
    p.add_argument("--num_workers",type=int,  default=4)
    p.add_argument("--seed",       type=int,  default=42)
    p.add_argument("--save_every", type=int,  default=10)
    return p.parse_args()


# ── Change Detection Dataset ──────────────────────────────────────────────────
class ChangeDetectionDataset(Dataset):
    """
    Loads bi-temporal image pairs + binary change masks.
    Works for LEVIR-CD, WHU-CD, and CDVQA spatial labels.
    """

    def __init__(self, data_dir: str, split: str = "train", img_size: int = 256):
        self.root     = Path(data_dir) / split
        self.img_size = img_size
        self.split    = split

        self.a_dir   = self.root / "A"
        self.b_dir   = self.root / "B"
        self.lbl_dir = self.root / "label"

        if not self.a_dir.exists():
            print(f"  ⚠  {self.a_dir} not found. Using synthetic data for testing.")
            self.files = [f"synthetic_{i}" for i in range(200 if split=="train" else 50)]
            self.synthetic = True
        else:
            self.files    = [f.stem for f in sorted(self.a_dir.glob("*.png"))]
            self.files   += [f.stem for f in sorted(self.a_dir.glob("*.jpg"))]
            self.files   += [f.stem for f in sorted(self.a_dir.glob("*.tif"))]
            self.synthetic = False

        self.img_tf  = self._build_img_tf(split)
        self.lbl_tf  = T.Compose([T.Resize((img_size, img_size), T.InterpolationMode.NEAREST), T.ToTensor()])
        print(f"  ChangeDetection [{split}]: {len(self.files)} pairs")

    def _build_img_tf(self, split):
        base = [T.Resize((self.img_size, self.img_size)), T.ToTensor(),
                T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]
        if split == "train":
            aug = [T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                   T.RandomRotation(90), T.ColorJitter(0.2,0.2,0.2)]
            return T.Compose(aug + base)
        return T.Compose(base)

    def _synth_pair(self):
        """Generate a synthetic image pair with a random change region."""
        h = w = self.img_size
        base  = np.random.randint(30, 200, (h, w, 3), dtype=np.uint8)
        t1    = base + np.random.randint(0, 20, (h, w, 3), dtype=np.uint8)
        t2    = t1.copy()
        mask  = np.zeros((h, w), dtype=np.uint8)
        # Random rectangular change
        x1,y1 = np.random.randint(0, h//2, 2)
        x2,y2 = x1 + np.random.randint(20, h//2), y1 + np.random.randint(20, w//2)
        x2,y2 = min(x2,h), min(y2,w)
        t2[x1:x2, y1:y2] = np.random.randint(100, 255, (x2-x1, y2-y1, 3), dtype=np.uint8)
        mask[x1:x2, y1:y2] = 255
        return (Image.fromarray(t1.clip(0,255)), Image.fromarray(t2.clip(0,255)),
                Image.fromarray(mask))

    def _load_img(self, path: Path) -> Image.Image:
        for ext in [".png",".jpg",".jpeg",".tif",".tiff"]:
            fp = path.parent / (path.name + ext)
            if fp.exists():
                return Image.open(fp).convert("RGB")
        return Image.fromarray(np.random.randint(50,200,(256,256,3),dtype=np.uint8))

    def _load_mask(self, name: str) -> Image.Image:
        for ext in [".png",".jpg",".tif"]:
            fp = self.lbl_dir / (name + ext)
            if fp.exists():
                img = Image.open(fp).convert("L")
                # Binarise: anything > 128 = change
                arr = np.array(img)
                arr = (arr > 128).astype(np.uint8) * 255
                return Image.fromarray(arr)
        return Image.fromarray(np.zeros((256,256), dtype=np.uint8))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        if self.synthetic:
            img_a, img_b, mask = self._synth_pair()
        else:
            img_a = self._load_img(self.a_dir   / name)
            img_b = self._load_img(self.b_dir   / name)
            mask  = self._load_mask(name)

        # Consistent random flips for pair
        if self.split == "train" and np.random.rand() > 0.5:
            img_a = img_a.transpose(Image.FLIP_LEFT_RIGHT)
            img_b = img_b.transpose(Image.FLIP_LEFT_RIGHT)
            mask  = mask.transpose(Image.FLIP_LEFT_RIGHT)

        t_a  = self.img_tf(img_a)
        t_b  = self.img_tf(img_b)
        t_m  = self.lbl_tf(mask).squeeze(0)
        t_m  = (t_m > 0.5).long()    # binary: 0=unchanged, 1=changed

        return {"img_a": t_a, "img_b": t_b, "mask": t_m, "name": name}


# ── ChangeFormer Architecture ─────────────────────────────────────────────────
class PatchEmbed(nn.Module):
    """Image → patch tokens."""
    def __init__(self, img_size=256, patch_size=8, in_ch=3, embed_dim=256):
        super().__init__()
        self.proj   = nn.Conv2d(in_ch, embed_dim, patch_size, patch_size)
        self.norm   = nn.LayerNorm(embed_dim)
        self.n_pats = (img_size // patch_size) ** 2

    def forward(self, x):
        x = self.proj(x)                        # (B, E, H', W')
        x = x.flatten(2).transpose(1, 2)        # (B, N, E)
        return self.norm(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0, drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        x2, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x  = x + x2
        x  = x + self.mlp(self.norm2(x))
        return x


class ChangeFormerNet(nn.Module):
    """
    Siamese Transformer encoder + MLP decoder for binary change detection.
    Architecture follows ChangeFormer (Bandara & Patel, 2022).
    """
    def __init__(self, img_size=256, patch_size=8, embed_dim=256,
                 depth=4, num_heads=8):
        super().__init__()
        n = (img_size // patch_size) ** 2

        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        self.pos_embed   = nn.Parameter(torch.zeros(1, n, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.encoder = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(depth)
        ])

        # Difference → change map
        self.change_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, 2),   # [unchanged, changed]
        )

        self.img_size   = img_size
        self.patch_size = patch_size

    def encode(self, x):
        t = self.patch_embed(x) + self.pos_embed
        for blk in self.encoder:
            t = blk(t)
        return t                             # (B, N, E)

    def forward(self, img_a, img_b):
        fa = self.encode(img_a)              # (B, N, E)
        fb = self.encode(img_b)

        diff  = torch.abs(fa - fb)           # (B, N, E)
        logit = self.change_head(diff)       # (B, N, 2)

        # Reshape to spatial map
        hp = self.img_size // self.patch_size
        logit = logit.transpose(1, 2).reshape(
            logit.size(0), 2, hp, hp
        )
        # Upsample to original resolution
        logit = F.interpolate(logit, size=(self.img_size, self.img_size),
                              mode="bilinear", align_corners=False)
        return logit                          # (B, 2, H, W)


# ── Loss: BCE + Dice ──────────────────────────────────────────────────────────
class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce_w = bce_weight
        self.bce   = nn.CrossEntropyLoss()

    def dice_loss(self, pred, target):
        pred   = torch.softmax(pred, dim=1)[:, 1]    # change prob
        target = target.float()
        inter  = (pred * target).sum()
        denom  = pred.sum() + target.sum() + 1e-6
        return 1 - 2 * inter / denom

    def forward(self, pred, target):
        return self.bce_w * self.bce(pred, target) + (1-self.bce_w) * self.dice_loss(pred, target)


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(pred_logits, gt_masks):
    """IoU and F1 for change class."""
    pred = pred_logits.argmax(dim=1)          # (B, H, W)
    tp   = ((pred == 1) & (gt_masks == 1)).sum().float()
    fp   = ((pred == 1) & (gt_masks == 0)).sum().float()
    fn   = ((pred == 0) & (gt_masks == 1)).sum().float()
    tn   = ((pred == 0) & (gt_masks == 0)).sum().float()
    iou  = tp / (tp + fp + fn + 1e-6)
    f1   = 2*tp / (2*tp + fp + fn + 1e-6)
    acc  = (tp + tn) / (tp + tn + fp + fn + 1e-6)
    return {"iou": iou.item(), "f1": f1.item(), "acc": acc.item()}


# ── Training ──────────────────────────────────────────────────────────────────
def train(args):
    print("\n🛰️  SatQuery AI — ChangeFormer Training (LEVIR-CD)")
    print("=" * 55)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}  |  Image size: {args.img_size}")
    torch.manual_seed(args.seed)

    model = ChangeFormerNet(
        img_size=args.img_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters: {total_params:.1f}M")

    train_ds = ChangeDetectionDataset(args.data_dir, "train", args.img_size)
    val_ds   = ChangeDetectionDataset(args.data_dir, "val",   args.img_size)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers)

    criterion = BCEDiceLoss(bce_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(args.fp16 and device=="cuda"))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = 0.0
    print(f"\n  Training for {args.epochs} epochs...\n")

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_dl, 1):
            a    = batch["img_a"].to(device)
            b    = batch["img_b"].to(device)
            mask = batch["mask"].to(device)

            with torch.cuda.amp.autocast(enabled=(args.fp16 and device=="cuda")):
                logits = model(a, b)
                loss   = criterion(logits, mask)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            total_loss += loss.item()
            if step % 50 == 0:
                print(f"  E{epoch}/{args.epochs} S{step}/{len(train_dl)} "
                      f"loss={total_loss/step:.4f}")

        scheduler.step()

        # Validate
        model.eval()
        all_iou = all_f1 = all_acc = 0.0
        with torch.no_grad():
            for batch in val_dl:
                a    = batch["img_a"].to(device)
                b    = batch["img_b"].to(device)
                mask = batch["mask"].to(device)
                with torch.cuda.amp.autocast(enabled=(args.fp16 and device=="cuda")):
                    logits = model(a, b)
                m = compute_metrics(logits.cpu(), mask.cpu())
                all_iou += m["iou"]; all_f1 += m["f1"]; all_acc += m["acc"]

        n = len(val_dl)
        iou, f1, acc = all_iou/n, all_f1/n, all_acc/n
        avg_loss = total_loss / len(train_dl)
        print(f"\n  📊 Epoch {epoch:3d} | Loss={avg_loss:.4f} "
              f"| IoU={iou:.4f} | F1={f1:.4f} | Acc={acc:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "f1": f1, "iou": iou, "args": vars(args),
            }, out_dir / "best_changeformer.pth")
            print(f"  🏆 Best saved (F1={best_f1:.4f})")

        if epoch % args.save_every == 0:
            torch.save(model.state_dict(), out_dir / f"epoch_{epoch}.pth")
        print()

    print(f"\n✅ ChangeFormer training done! Best F1: {best_f1:.4f}")
    print(f"   Checkpoint: {out_dir/'best_changeformer.pth'}")
    print(f"\n   → Next: python training/grounding/train_grounding.py")


if __name__ == "__main__":
    train(get_args())
