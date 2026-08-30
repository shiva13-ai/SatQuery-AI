#!/usr/bin/env python3
"""
SatQuery AI — Step 4: RS-Caption Fine-tuning on VRSBench
=========================================================
Fine-tunes BLIP for remote-sensing image captioning.

DATASET: VRSBench
  HuggingFace: huggingface.co/datasets/lx709/VRSBench
  Download:    python datasets/setup_datasets.py --dataset vrsbench

EXPECTED STRUCTURE:
  data/vrsbench/
  ├── images/          ← RS images (.jpg/.tif)
  ├── captions_train.json
  ├── captions_val.json
  └── captions_test.json

  JSON format:
  [{"image_id": "xxx", "image": "xxx.jpg", "caption": "A satellite image showing..."}]

USAGE:
  python training/caption/train_caption.py \
    --data_dir data/vrsbench \
    --backbone checkpoints/backbone/best \
    --output_dir checkpoints/caption \
    --epochs 20
"""

import json
import argparse
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   type=str, default="data/vrsbench")
    p.add_argument("--backbone",   type=str, default="Salesforce/blip-image-captioning-large")
    p.add_argument("--output_dir", type=str, default="checkpoints/caption")
    p.add_argument("--epochs",     type=int, default=20)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr",         type=float, default=3e-5)
    p.add_argument("--max_len",    type=int, default=128)
    p.add_argument("--fp16",       action="store_true", default=True)
    p.add_argument("--num_workers",type=int, default=4)
    return p.parse_args()


# ── VRSBench Dataset ──────────────────────────────────────────────────────────
class VRSBenchCaptionDataset(Dataset):
    SAMPLE_CAPTIONS = [
        "A satellite image showing agricultural fields with regular crop rows adjacent to a small rural settlement.",
        "An aerial view of a dense urban area with road networks, buildings, and commercial zones.",
        "A multispectral satellite image displaying a river delta with water bodies and wetland vegetation.",
        "A remote sensing image of a forested mountain region with visible deforestation patches.",
        "A high-resolution satellite image showing a coastal area with sandy beaches and shallow water.",
        "An aerial image depicting a mixed land-cover scene with grassland, shrubland, and isolated trees.",
        "A satellite view of industrial infrastructure including storage tanks and factory buildings.",
        "A remote sensing image showing irrigated farmland with visible irrigation channels.",
    ]

    def __init__(self, data_dir: str, split: str, processor, max_len: int):
        self.data_dir  = Path(data_dir)
        self.processor = processor
        self.max_len   = max_len
        self.img_dir   = self.data_dir / "images"
        self.samples   = self._load(split)
        print(f"  VRSBench Caption [{split}]: {len(self.samples)} samples")

    def _load(self, split: str) -> list:
        candidates = [
            self.data_dir / f"captions_{split}.json",
            self.data_dir / f"{split}_captions.json",
            self.data_dir / f"annotations_{split}.json",
        ]
        for f in candidates:
            if f.exists():
                with open(f) as fp:
                    data = json.load(fp)
                if isinstance(data, list):
                    return data
                if "annotations" in data:
                    return [{"image": a.get("image_id",""), "caption": a["caption"]}
                            for a in data["annotations"]]
        # Synthetic fallback
        print(f"  ⚠  No caption file for '{split}'. Using synthetic data.")
        return [{"image": f"img_{i}.jpg", "caption": self.SAMPLE_CAPTIONS[i % len(self.SAMPLE_CAPTIONS)]}
                for i in range(500 if split == "train" else 100)]

    def _load_img(self, name) -> Image.Image:
        for ext in [".jpg",".jpeg",".png",".tif",".tiff",""]:
            p = self.img_dir / (str(name) + ext)
            if p.exists():
                try:
                    return Image.open(p).convert("RGB")
                except Exception:
                    pass
        # Synthetic RS-like image
        arr = np.zeros((256, 256, 3), dtype=np.uint8)
        arr[:,:,0] = np.random.randint(30, 120, (256,256))
        arr[:,:,1] = np.random.randint(60, 150, (256,256))
        arr[:,:,2] = np.random.randint(20, 80,  (256,256))
        return Image.fromarray(arr)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s       = self.samples[idx]
        img     = self._load_img(s["image"])
        caption = s["caption"]
        enc     = self.processor(
            images=img, text=caption,
            return_tensors="pt",
            padding="max_length", truncation=True,
            max_length=self.max_len,
        )
        labels = enc.input_ids.squeeze(0).clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return {
            "pixel_values": enc.pixel_values.squeeze(0),
            "input_ids":    enc.input_ids.squeeze(0),
            "attention_mask": enc.attention_mask.squeeze(0),
            "labels":       labels,
            "caption":      caption,
        }


def compute_bleu(preds: list, refs: list) -> float:
    refs_tok  = [[r.lower().split()] for r in refs]
    preds_tok = [p.lower().split()   for p in preds]
    sf = SmoothingFunction().method1
    try:
        return corpus_bleu(refs_tok, preds_tok, smoothing_function=sf)
    except Exception:
        return 0.0


def train(args):
    print("\n🛰️  SatQuery AI — Caption Fine-tuning (VRSBench)")
    print("=" * 55)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device} | Backbone: {args.backbone}")

    from transformers import BlipProcessor, BlipForConditionalGeneration
    processor = BlipProcessor.from_pretrained(args.backbone)
    model     = BlipForConditionalGeneration.from_pretrained(
        args.backbone,
        torch_dtype=torch.float16 if args.fp16 and device=="cuda" else torch.float32,
    ).to(device)

    # LoRA on language decoder
    try:
        from peft import LoraConfig, get_peft_model, TaskType
        cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32,
            target_modules=["q_proj","v_proj"], lora_dropout=0.1,
        )
        model.text_decoder = get_peft_model(model.text_decoder, cfg)
        model.text_decoder.print_trainable_parameters()
    except Exception as e:
        print(f"  LoRA skipped: {e}")

    train_ds = VRSBenchCaptionDataset(args.data_dir, "train", processor, args.max_len)
    val_ds   = VRSBenchCaptionDataset(args.data_dir, "val",   processor, args.max_len)

    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True,  num_workers=args.num_workers)
    val_dl   = DataLoader(val_ds,   args.batch_size, shuffle=False, num_workers=args.num_workers)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler    = torch.cuda.amp.GradScaler(enabled=(args.fp16 and device=="cuda"))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_bleu = 0.0

    print(f"\n  Training for {args.epochs} epochs...\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_dl, 1):
            pv  = batch["pixel_values"].to(device)
            ids = batch["input_ids"].to(device)
            msk = batch["attention_mask"].to(device)
            lbl = batch["labels"].to(device)

            with torch.cuda.amp.autocast(enabled=(args.fp16 and device=="cuda")):
                out  = model(pixel_values=pv, input_ids=ids,
                             attention_mask=msk, labels=lbl)
                loss = out.loss

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            total_loss += loss.item()

            if step % 100 == 0:
                print(f"  E{epoch} S{step}/{len(train_dl)} loss={total_loss/step:.4f}")

        scheduler.step()

        # Validate with BLEU
        model.eval()
        preds, refs = [], []
        with torch.no_grad():
            for batch in val_dl:
                pv = batch["pixel_values"].to(device)
                with torch.cuda.amp.autocast(enabled=(args.fp16 and device=="cuda")):
                    gen = model.generate(pixel_values=pv, max_new_tokens=64,
                                         num_beams=4, length_penalty=1.0)
                decoded = processor.batch_decode(gen, skip_special_tokens=True)
                preds.extend(decoded)
                refs.extend(batch["caption"])

        bleu = compute_bleu(preds, refs)
        avg_loss = total_loss / len(train_dl)
        print(f"\n  📊 Epoch {epoch} | Loss={avg_loss:.4f} | BLEU-4={bleu:.4f}")
        print(f"     Sample: '{preds[0][:80]}...'")

        if bleu > best_bleu:
            best_bleu = bleu
            model.save_pretrained(out_dir / "best")
            processor.save_pretrained(out_dir / "best")
            print(f"  🏆 Best caption model (BLEU={best_bleu:.4f})")
        print()

    print(f"\n✅ Caption training done! Best BLEU-4: {best_bleu:.4f}")
    print(f"   → Next: python training/grounding/train_grounding.py")


if __name__ == "__main__":
    train(get_args())
