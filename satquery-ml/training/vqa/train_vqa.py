#!/usr/bin/env python3
"""
SatQuery AI — Step 2: Remote-Sensing VQA Fine-tuning
=====================================================
Fine-tunes BLIP-2 (with BigEarthNet backbone) on RSVQA-LR + RSVQA-HR
for single-image visual question answering.

DATASET NEEDED: RSVQA-LR and/or RSVQA-HR
  Download: https://zenodo.org/record/6344334  (LR)
            https://zenodo.org/record/6344337  (HR)
  Or run:   python datasets/setup_datasets.py --dataset rsvqa_lr

EXPECTED STRUCTURE:
  data/rsvqa_lr/
  ├── Images/             ← .tif RS images
  ├── Questions_LR_train.json
  ├── Questions_LR_val.json
  └── Questions_LR_test.json

USAGE:
  python training/vqa/train_vqa.py \
    --data_dir data/rsvqa_lr \
    --backbone checkpoints/backbone/best \
    --output_dir checkpoints/vqa \
    --epochs 15
"""

import json
import argparse
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",    type=str, default="data/rsvqa_lr")
    p.add_argument("--backbone",    type=str, default="Salesforce/blip2-opt-2.7b",
                   help="Path to BigEarthNet fine-tuned backbone OR HF model id")
    p.add_argument("--output_dir",  type=str, default="checkpoints/vqa")
    p.add_argument("--epochs",      type=int, default=15)
    p.add_argument("--batch_size",  type=int, default=8)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--max_len",     type=int, default=32)
    p.add_argument("--fp16",        action="store_true", default=True)
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


# ── RSVQA Dataset ─────────────────────────────────────────────────────────────
class RSVQADataset(Dataset):
    """
    RSVQA-LR / RSVQA-HR dataset loader.
    Each item: image + question → answer (open-ended or closed).
    """

    def __init__(self, data_dir: str, split: str, processor, max_len: int = 32):
        self.data_dir  = Path(data_dir)
        self.processor = processor
        self.max_len   = max_len
        self.img_dir   = self.data_dir / "Images"
        self.samples   = self._load_qa(split)
        print(f"  RSVQA [{split}]: {len(self.samples)} QA pairs")

    def _load_qa(self, split: str) -> list:
        """Load Q/A pairs from RSVQA JSON format."""
        # Try both naming conventions
        candidates = [
            self.data_dir / f"Questions_LR_{split}.json",
            self.data_dir / f"Questions_HR_{split}.json",
            self.data_dir / f"{split}_questions.json",
            self.data_dir / f"qa_{split}.json",
        ]
        qa_file = next((f for f in candidates if f.exists()), None)
        if not qa_file:
            print(f"  ⚠  No QA file found for split '{split}' in {self.data_dir}")
            return self._generate_dummy_samples(100)

        with open(qa_file) as f:
            data = json.load(f)

        # Handle different JSON structures
        if isinstance(data, list):
            return data
        if "questions" in data:
            qs = {q["id"]: q for q in data["questions"]}
            ans = {a["question_id"]: a for a in data.get("answers", [])}
            samples = []
            for qid, q in qs.items():
                a = ans.get(qid, {})
                samples.append({
                    "image":    q.get("img_id", q.get("image_id", "")),
                    "question": q["question"],
                    "answer":   a.get("answer", "unknown"),
                    "type":     q.get("type", "other"),
                })
            return samples
        return []

    def _generate_dummy_samples(self, n: int) -> list:
        """Fallback dummy data for testing the pipeline without real data."""
        types = ["presence", "count", "area", "comparison"]
        qs    = [
            "Is there any water body in this image?",
            "How many buildings can you see?",
            "What is the dominant land cover type?",
            "Is the area urban or rural?",
            "What type of vegetation is visible?",
        ]
        ans = ["yes", "no", "3", "urban", "forest", "grassland", "water"]
        return [
            {"image": f"img_{i}.tif", "question": qs[i%len(qs)],
             "answer": ans[i%len(ans)], "type": types[i%len(types)]}
            for i in range(n)
        ]

    def _load_image(self, img_name: str) -> Image.Image:
        """Load image by name, trying multiple extensions."""
        for ext in [".tif", ".tiff", ".jpg", ".png", ".jpeg"]:
            p = self.img_dir / (str(img_name) + ext)
            if p.exists():
                try:
                    return Image.open(p).convert("RGB")
                except Exception:
                    pass
            # Try without extension
            p2 = self.img_dir / str(img_name)
            if p2.exists():
                try:
                    return Image.open(p2).convert("RGB")
                except Exception:
                    pass
        # Return placeholder
        return Image.fromarray(
            (np.random.rand(224, 224, 3) * 200 + 30).astype(np.uint8)
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s   = self.samples[idx]
        img = self._load_image(s["image"])

        # Format as VQA prompt
        prompt  = f"Question: {s['question']} Answer:"
        answer  = str(s["answer"])

        # Process with BLIP-2 processor
        encoding = self.processor(
            images=img,
            text=prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
        )
        # Encode answer
        ans_enc = self.processor.tokenizer(
            answer,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=16,
        )
        return {
            "pixel_values":  encoding.pixel_values.squeeze(0),
            "input_ids":     encoding.input_ids.squeeze(0),
            "attention_mask":encoding.attention_mask.squeeze(0),
            "labels":        ans_enc.input_ids.squeeze(0),
            "answer":        answer,
            "question":      s["question"],
        }


# ── VQA Accuracy metric ───────────────────────────────────────────────────────
def vqa_accuracy(pred: str, gt: str) -> float:
    """Soft accuracy: exact match or GT in prediction."""
    pred, gt = pred.lower().strip(), gt.lower().strip()
    if pred == gt:
        return 1.0
    if gt in pred or pred in gt:
        return 0.5
    return 0.0


# ── Training ──────────────────────────────────────────────────────────────────
def train(args):
    print("\n🛰️  SatQuery AI — VQA Fine-tuning (RSVQA)")
    print("=" * 55)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device} | Backbone: {args.backbone}")
    torch.manual_seed(args.seed)

    # Load model
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    print(f"  Loading: {args.backbone}")
    processor = Blip2Processor.from_pretrained(args.backbone)
    model     = Blip2ForConditionalGeneration.from_pretrained(
        args.backbone,
        torch_dtype=torch.float16 if args.fp16 and device=="cuda" else torch.float32,
        device_map=device,
    )

    # Apply LoRA to language model head
    try:
        from peft import LoraConfig, get_peft_model, TaskType
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8, lora_alpha=16, lora_dropout=0.05,
            target_modules=["q_proj","v_proj","k_proj","out_proj"],
        )
        model.language_model = get_peft_model(model.language_model, lora_cfg)
        model.language_model.print_trainable_parameters()
    except Exception as e:
        print(f"  LoRA skipped: {e}")

    # Datasets
    train_ds = RSVQADataset(args.data_dir, "train", processor, args.max_len)
    val_ds   = RSVQADataset(args.data_dir, "val",   processor, args.max_len)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr,
        steps_per_epoch=len(train_dl), epochs=args.epochs
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(args.fp16 and device=="cuda"))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_acc = 0.0
    print(f"\n  Training VQA for {args.epochs} epochs...\n")

    for epoch in range(1, args.epochs + 1):
        # Train
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
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            total_loss += loss.item()
            if step % 100 == 0:
                print(f"  E{epoch} S{step}/{len(train_dl)} loss={total_loss/step:.4f}")

        # Validate — compute VQA accuracy
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in val_dl:
                pv = batch["pixel_values"].to(device)
                ids = batch["input_ids"].to(device)
                with torch.cuda.amp.autocast(enabled=(args.fp16 and device=="cuda")):
                    generated = model.generate(pixel_values=pv, input_ids=ids,
                                               max_new_tokens=16)
                preds = processor.batch_decode(generated, skip_special_tokens=True)
                for pred, gt in zip(preds, batch["answer"]):
                    correct += vqa_accuracy(pred, gt)
                    total   += 1

        acc = correct / total if total > 0 else 0
        avg_loss = total_loss / len(train_dl)
        print(f"\n  📊 Epoch {epoch} | Train Loss: {avg_loss:.4f} | Val Acc: {acc:.4f}\n")

        if acc > best_acc:
            best_acc = acc
            model.save_pretrained(out_dir / "best")
            processor.save_pretrained(out_dir / "best")
            print(f"  🏆 Best VQA model saved (acc={best_acc:.4f})")

        # Epoch checkpoint
        model.save_pretrained(out_dir / f"epoch_{epoch}")
        processor.save_pretrained(out_dir / f"epoch_{epoch}")

    print(f"\n✅ VQA training done! Best accuracy: {best_acc:.4f}")
    print(f"   Checkpoint: {out_dir/'best'}")
    print(f"\n   → Next: python training/change/train_changeformer.py")


if __name__ == "__main__":
    train(get_args())
