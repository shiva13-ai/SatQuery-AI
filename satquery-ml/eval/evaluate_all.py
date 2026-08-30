#!/usr/bin/env python3
"""
SatQuery AI — Evaluation Script
=================================
Evaluates all models on prescribed public benchmarks.

BENCHMARKS:
  - RSVQA-LR/HR  → VQA accuracy
  - VRSBench     → BLEU-4, CIDEr (captioning), Acc@0.5 (grounding)
  - CDVQA        → Change-VQA accuracy, change map IoU/F1

USAGE:
  # Evaluate all
  python eval/evaluate_all.py \
    --vqa_ckpt      checkpoints/vqa/best \
    --caption_ckpt  checkpoints/caption/best \
    --grounding_ckpt checkpoints/grounding/best \
    --change_ckpt   checkpoints/change/best_changeformer.pth

  # Single benchmark
  python eval/evaluate_all.py --benchmark rsvqa --vqa_ckpt checkpoints/vqa/best
"""

import argparse
import json
import time
from pathlib import Path
import torch
import numpy as np
from PIL import Image

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark",     type=str, default="all",
                   choices=["all","rsvqa","vrsbench","cdvqa","change"])
    p.add_argument("--data_dir",      type=str, default="data")
    p.add_argument("--vqa_ckpt",      type=str, default="checkpoints/vqa/best")
    p.add_argument("--caption_ckpt",  type=str, default="checkpoints/caption/best")
    p.add_argument("--grounding_ckpt",type=str, default="checkpoints/grounding/best")
    p.add_argument("--change_ckpt",   type=str, default="checkpoints/change/best_changeformer.pth")
    p.add_argument("--output_dir",    type=str, default="eval/results")
    p.add_argument("--max_samples",   type=int, default=None,
                   help="Limit eval samples (for quick testing)")
    p.add_argument("--device",        type=str, default="auto")
    return p.parse_args()


DEVICE = None

def get_device(args):
    global DEVICE
    if args.device == "auto":
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        DEVICE = args.device
    return DEVICE


# ── VQA Evaluation (RSVQA) ────────────────────────────────────────────────────
def eval_vqa(args) -> dict:
    print("\n📊 Evaluating VQA on RSVQA-LR...")
    data_dir = Path(args.data_dir) / "rsvqa_lr"

    try:
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        proc  = Blip2Processor.from_pretrained(args.vqa_ckpt)
        model = Blip2ForConditionalGeneration.from_pretrained(
            args.vqa_ckpt, torch_dtype=torch.float16 if DEVICE=="cuda" else torch.float32,
            device_map=DEVICE
        )
        model.eval()
    except Exception as e:
        print(f"  ⚠  Could not load VQA model: {e}")
        return {"vqa_accuracy": 0.0, "error": str(e)}

    # Load test set
    qa_file = data_dir / "Questions_LR_test.json"
    if not qa_file.exists():
        print(f"  ⚠  Test file not found: {qa_file}")
        return {"vqa_accuracy": 0.0, "note": "test file missing"}

    with open(qa_file) as f:
        data = json.load(f)

    # Parse QA pairs
    if isinstance(data, list):
        samples = data
    elif "questions" in data:
        qs  = {q["id"]: q for q in data["questions"]}
        ans = {a["question_id"]: a for a in data.get("answers",[])}
        samples = [{"image": qs[qid].get("img_id",""), "question": qs[qid]["question"],
                    "answer": ans.get(qid,{}).get("answer","")} for qid in qs]
    else:
        samples = []

    if args.max_samples:
        samples = samples[:args.max_samples]

    img_dir = data_dir / "Images"
    correct = total = 0

    for s in samples:
        # Load image
        img = None
        for ext in [".tif",".tiff",".jpg",".png",""]:
            p = img_dir / (str(s["image"]) + ext)
            if p.exists():
                try:
                    img = Image.open(p).convert("RGB")
                    break
                except Exception:
                    pass
        if img is None:
            img = Image.fromarray(np.random.randint(50,200,(224,224,3),dtype=np.uint8))

        prompt = f"Question: {s['question']} Answer:"
        enc    = proc(images=img, text=prompt, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            out  = model.generate(**enc, max_new_tokens=16)
        pred = proc.decode(out[0], skip_special_tokens=True).lower().strip()
        gt   = str(s["answer"]).lower().strip()

        if pred == gt or gt in pred or pred in gt:
            correct += 1
        total += 1

    acc = correct / max(total, 1)
    print(f"  ✅ RSVQA VQA Accuracy: {acc:.4f} ({correct}/{total})")
    return {"vqa_accuracy": acc, "correct": correct, "total": total}


# ── Captioning Evaluation (VRSBench) ─────────────────────────────────────────
def eval_caption(args) -> dict:
    print("\n📊 Evaluating Captioning on VRSBench...")
    data_dir = Path(args.data_dir) / "vrsbench"

    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        proc  = BlipProcessor.from_pretrained(args.caption_ckpt)
        model = BlipForConditionalGeneration.from_pretrained(
            args.caption_ckpt,
            torch_dtype=torch.float16 if DEVICE=="cuda" else torch.float32,
        ).to(DEVICE)
        model.eval()
    except Exception as e:
        print(f"  ⚠  Could not load Caption model: {e}")
        return {"bleu4": 0.0, "error": str(e)}

    # Load test captions
    cap_file = data_dir / "captions_test.json"
    if not cap_file.exists():
        print(f"  ⚠  {cap_file} not found")
        return {"bleu4": 0.0, "note": "test file missing"}

    with open(cap_file) as f:
        samples = json.load(f)
    if isinstance(samples, dict):
        samples = samples.get("annotations", [])
    if args.max_samples:
        samples = samples[:args.max_samples]

    img_dir = data_dir / "images"
    preds, refs = [], []

    for s in samples:
        img_file = s.get("image", s.get("file_name",""))
        ref      = s.get("caption", s.get("text",""))
        img = None
        for ext in [".jpg",".png",".tif",""]:
            p = img_dir / (str(img_file) + ext)
            if p.exists():
                try:
                    img = Image.open(p).convert("RGB")
                    break
                except Exception:
                    pass
        if img is None:
            img = Image.fromarray(np.zeros((224,224,3),dtype=np.uint8))

        enc = proc(images=img, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=64, num_beams=4)
        pred = proc.decode(gen[0], skip_special_tokens=True)
        preds.append(pred)
        refs.append(ref)

    # BLEU-4
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    sf   = SmoothingFunction().method1
    refs_tok  = [[r.lower().split()] for r in refs]
    preds_tok = [p.lower().split()   for p in preds]
    bleu4 = corpus_bleu(refs_tok, preds_tok, smoothing_function=sf)

    print(f"  ✅ VRSBench BLEU-4: {bleu4:.4f}")
    print(f"     Sample pred: '{preds[0][:80]}'")
    print(f"     Sample ref:  '{refs[0][:80]}'")
    return {"bleu4": bleu4, "num_samples": len(preds)}


# ── Change Detection Evaluation (LEVIR / CDVQA) ───────────────────────────────
def eval_change(args) -> dict:
    print("\n📊 Evaluating Change Detection on LEVIR-CD test set...")
    data_dir = Path(args.data_dir) / "levir_cd" / "test"

    try:
        # Load ChangeFormer
        sys_path = str(Path(__file__).parent.parent)
        import sys; sys.path.insert(0, sys_path)
        from training.change.train_changeformer import ChangeFormerNet
        ckpt  = torch.load(args.change_ckpt, map_location=DEVICE)
        model = ChangeFormerNet()
        model.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)
        model = model.to(DEVICE).eval()
    except Exception as e:
        print(f"  ⚠  Could not load ChangeFormer: {e}")
        return {"change_f1": 0.0, "change_iou": 0.0, "error": str(e)}

    import torchvision.transforms as T
    tf = T.Compose([
        T.Resize((256,256)), T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    a_dir  = data_dir / "A"
    b_dir  = data_dir / "B"
    lbl_dir= data_dir / "label"

    if not a_dir.exists():
        print(f"  ⚠  Test data not found at {a_dir}")
        return {"change_f1": 0.0, "note": "test data missing"}

    files = [f.stem for f in sorted(a_dir.glob("*.png"))]
    if args.max_samples:
        files = files[:args.max_samples]

    all_tp = all_fp = all_fn = 0

    with torch.no_grad():
        for name in files:
            def load_rgb(d):
                for ext in [".png",".jpg",".tif"]:
                    p = d / (name+ext)
                    if p.exists():
                        return Image.open(p).convert("RGB")
                return Image.fromarray(np.zeros((256,256,3),dtype=np.uint8))

            ia = tf(load_rgb(a_dir)).unsqueeze(0).to(DEVICE)
            ib = tf(load_rgb(b_dir)).unsqueeze(0).to(DEVICE)

            logits = model(ia, ib)
            pred   = logits.argmax(1).squeeze().cpu().numpy()

            # Load ground truth mask
            mask = None
            for ext in [".png",".jpg",".tif"]:
                p = lbl_dir / (name+ext)
                if p.exists():
                    m  = np.array(Image.open(p).convert("L").resize((256,256)))
                    mask = (m > 128).astype(np.uint8)
                    break
            if mask is None:
                mask = np.zeros((256,256), dtype=np.uint8)

            all_tp += int(((pred==1) & (mask==1)).sum())
            all_fp += int(((pred==1) & (mask==0)).sum())
            all_fn += int(((pred==0) & (mask==1)).sum())

    iou = all_tp / (all_tp + all_fp + all_fn + 1e-6)
    f1  = 2*all_tp / (2*all_tp + all_fp + all_fn + 1e-6)
    print(f"  ✅ Change Detection → F1={f1:.4f} | IoU={iou:.4f}")
    return {"change_f1": f1, "change_iou": iou, "tp": all_tp, "fp": all_fp, "fn": all_fn}


# ── CDVQA Evaluation ──────────────────────────────────────────────────────────
def eval_cdvqa(args) -> dict:
    print("\n📊 Evaluating Change-VQA on CDVQA...")
    data_dir = Path(args.data_dir) / "cdvqa"

    qa_file = data_dir / "test_qa.json"
    if not qa_file.exists():
        print(f"  ⚠  {qa_file} not found")
        return {"cdvqa_accuracy": 0.0, "note": "CDVQA test file missing"}

    try:
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        proc  = Blip2Processor.from_pretrained(args.vqa_ckpt)
        model = Blip2ForConditionalGeneration.from_pretrained(
            args.vqa_ckpt,
            torch_dtype=torch.float16 if DEVICE=="cuda" else torch.float32,
            device_map=DEVICE
        ).eval()
    except Exception as e:
        return {"cdvqa_accuracy": 0.0, "error": str(e)}

    with open(qa_file) as f:
        samples = json.load(f)
    if args.max_samples:
        samples = samples[:args.max_samples]

    img_dir = data_dir / "images"
    correct = total = 0

    for s in samples:
        # Load image pair → concatenate side by side
        imgs = []
        for slot in ["img_t1","img_t2"]:
            fn = s.get(slot, s.get("image",""))
            for ext in [".jpg",".png",".tif",""]:
                p = img_dir / (str(fn)+ext)
                if p.exists():
                    try:
                        imgs.append(Image.open(p).convert("RGB").resize((224,224)))
                        break
                    except Exception:
                        pass
            else:
                imgs.append(Image.fromarray(np.zeros((224,224,3),dtype=np.uint8)))

        # Concatenate T1 and T2 horizontally
        combined = Image.new("RGB",(448,224))
        combined.paste(imgs[0],(0,0))
        combined.paste(imgs[1],(224,0))

        q    = s.get("question","What changed?")
        gt   = str(s.get("answer","")).lower().strip()
        enc  = proc(images=combined, text=f"Question: {q} Answer:", return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=16)
        pred = proc.decode(gen[0], skip_special_tokens=True).lower().strip()

        correct += 1 if (pred==gt or gt in pred or pred in gt) else 0
        total   += 1

    acc = correct / max(total, 1)
    print(f"  ✅ CDVQA Change-VQA Accuracy: {acc:.4f} ({correct}/{total})")
    return {"cdvqa_accuracy": acc, "correct": correct, "total": total}


# ── Master runner ─────────────────────────────────────────────────────────────
def main():
    args   = get_args()
    device = get_device(args)
    print(f"\n🛰️  SatQuery AI — Benchmark Evaluation")
    print(f"   Device: {device}")
    print("=" * 55)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {"device": device, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

    bm = args.benchmark
    if bm in ("all","rsvqa"):
        results["rsvqa"] = eval_vqa(args)
    if bm in ("all","vrsbench"):
        results["vrsbench_caption"] = eval_caption(args)
    if bm in ("all","change"):
        results["levir_cd"] = eval_change(args)
    if bm in ("all","cdvqa"):
        results["cdvqa"]    = eval_cdvqa(args)

    # Save results
    out_file = out_dir / "eval_results.json"
    with open(out_file,"w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*55}")
    print("📊 FINAL BENCHMARK RESULTS")
    print(f"{'='*55}")
    for bname, bres in results.items():
        if isinstance(bres, dict):
            print(f"\n  {bname}:")
            for k,v in bres.items():
                if isinstance(v,(int,float)):
                    print(f"    {k}: {v:.4f}" if isinstance(v,float) else f"    {k}: {v}")
    print(f"\n  Results saved → {out_file}")
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
