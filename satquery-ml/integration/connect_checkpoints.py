#!/usr/bin/env python3
"""
SatQuery AI — Checkpoint Connector
====================================
After training, run this script to:
1. Validate all checkpoints
2. Update backend model loader with your checkpoint paths
3. Test each model with a sample image
4. Generate a model card summary

USAGE:
  python integration/connect_checkpoints.py \
    --vqa_ckpt      checkpoints/vqa/best \
    --caption_ckpt  checkpoints/caption/best \
    --grounding_ckpt checkpoints/grounding/best \
    --change_ckpt   checkpoints/change/best_changeformer.pth \
    --backend_dir   ../satquery/backend
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vqa_ckpt",       type=str, default="checkpoints/vqa/best")
    p.add_argument("--caption_ckpt",   type=str, default="checkpoints/caption/best")
    p.add_argument("--grounding_ckpt", type=str, default="checkpoints/grounding/best")
    p.add_argument("--change_ckpt",    type=str,
                   default="checkpoints/change/best_changeformer.pth")
    p.add_argument("--backbone_ckpt",  type=str, default="checkpoints/backbone/best")
    p.add_argument("--backend_dir",    type=str, default="../satquery/backend")
    p.add_argument("--test_image",     type=str, default=None,
                   help="Path to a test image (uses synthetic if not given)")
    p.add_argument("--skip_tests",     action="store_true")
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────
def check_ckpt(path: str, label: str) -> tuple[bool, str]:
    p = Path(path)
    if not p.exists():
        return False, f"❌ NOT FOUND: {path}"
    # HF checkpoint: needs config.json
    if p.is_dir():
        has_config = (p / "config.json").exists()
        has_model  = any(p.glob("*.safetensors")) or any(p.glob("pytorch_model*.bin"))
        if has_config and has_model:
            size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e9
            return True, f"✅ {label}: {path} ({size:.2f} GB)"
        return False, f"⚠  {label}: directory exists but missing config/weights"
    # PyTorch .pth checkpoint
    if p.suffix == ".pth":
        size = p.stat().st_size / 1e6
        return True, f"✅ {label}: {path} ({size:.1f} MB)"
    return False, f"⚠  {label}: unrecognised format"


def make_test_image() -> Image.Image:
    """Generate a synthetic satellite-like test image."""
    arr = np.zeros((256, 256, 3), dtype=np.uint8)
    # Green land cover
    arr[50:180, 30:220] = [34, 85, 34]
    # Blue water body
    arr[100:160, 80:160] = [30, 80, 150]
    # Urban grey
    arr[20:70, 150:230] = [110, 110, 120]
    arr += np.random.randint(0, 15, arr.shape, dtype=np.uint8)
    return Image.fromarray(arr)


# ── Model tests ───────────────────────────────────────────────────────────────
def test_vqa(ckpt: str, img: Image.Image) -> dict:
    print(f"\n  Testing VQA: {ckpt}")
    try:
        import torch
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        device = "cuda" if torch.cuda.is_available() else "cpu"
        proc  = Blip2Processor.from_pretrained(ckpt)
        model = Blip2ForConditionalGeneration.from_pretrained(
            ckpt, torch_dtype=torch.float16 if device=="cuda" else torch.float32,
            device_map=device
        ).eval()

        query = "Question: Is there any water body visible in this image? Answer:"
        enc   = proc(images=img, text=query, return_tensors="pt").to(device)
        t0    = time.time()
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=20)
        ms   = int((time.time()-t0)*1000)
        ans  = proc.decode(out[0], skip_special_tokens=True)
        print(f"    Query: Is there any water body?")
        print(f"    Answer: {ans} ({ms}ms)")
        return {"status": "ok", "answer": ans, "latency_ms": ms}
    except Exception as e:
        print(f"    ❌ Error: {e}")
        return {"status": "error", "error": str(e)}


def test_caption(ckpt: str, img: Image.Image) -> dict:
    print(f"\n  Testing Caption: {ckpt}")
    try:
        import torch
        from transformers import BlipProcessor, BlipForConditionalGeneration
        device = "cuda" if torch.cuda.is_available() else "cpu"
        proc  = BlipProcessor.from_pretrained(ckpt)
        model = BlipForConditionalGeneration.from_pretrained(
            ckpt, torch_dtype=torch.float16 if device=="cuda" else torch.float32,
        ).to(device).eval()

        enc = proc(images=img, return_tensors="pt").to(device)
        t0  = time.time()
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=60, num_beams=4)
        ms      = int((time.time()-t0)*1000)
        caption = proc.decode(out[0], skip_special_tokens=True)
        print(f"    Caption: {caption} ({ms}ms)")
        return {"status": "ok", "caption": caption, "latency_ms": ms}
    except Exception as e:
        print(f"    ❌ Error: {e}")
        return {"status": "error", "error": str(e)}


def test_grounding(ckpt: str, img: Image.Image) -> dict:
    print(f"\n  Testing Grounding: {ckpt}")
    try:
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        device = "cuda" if torch.cuda.is_available() else "cpu"
        proc  = AutoProcessor.from_pretrained(ckpt)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(ckpt).to(device).eval()

        query = "water body"
        enc   = proc(images=img, text=query, return_tensors="pt").to(device)
        t0    = time.time()
        with torch.no_grad():
            out = model(**enc)
        ms     = int((time.time()-t0)*1000)
        scores = out.logits.sigmoid().max(-1).values[0]
        top3   = scores.topk(min(3,len(scores))).values.tolist()
        print(f"    Query: 'water body'")
        print(f"    Top scores: {[f'{s:.3f}' for s in top3]} ({ms}ms)")
        return {"status": "ok", "top_scores": top3, "latency_ms": ms}
    except Exception as e:
        print(f"    ❌ Error: {e}")
        return {"status": "error", "error": str(e)}


def test_change(ckpt: str, img: Image.Image) -> dict:
    print(f"\n  Testing ChangeFormer: {ckpt}")
    try:
        import torch
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from training.change.train_changeformer import ChangeFormerNet
        import torchvision.transforms as T

        device = "cuda" if torch.cuda.is_available() else "cpu"
        state  = torch.load(ckpt, map_location=device)
        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]

        model = ChangeFormerNet().to(device).eval()
        model.load_state_dict(state)

        tf  = T.Compose([T.Resize((256,256)), T.ToTensor(),
                         T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        ia  = tf(img).unsqueeze(0).to(device)
        # Make a slightly modified T2
        arr2 = np.array(img).copy()
        arr2[80:130, 80:130] = [200, 160, 80]  # simulate urban growth
        ib  = tf(Image.fromarray(arr2)).unsqueeze(0).to(device)

        t0  = time.time()
        with torch.no_grad():
            logits  = model(ia, ib)
            change  = logits.argmax(1).squeeze().cpu().numpy()
        ms  = int((time.time()-t0)*1000)
        pct = float(change.mean() * 100)
        print(f"    Change area: {pct:.1f}% ({ms}ms)")
        return {"status": "ok", "change_percent": pct, "latency_ms": ms}
    except Exception as e:
        print(f"    ❌ Error: {e}")
        return {"status": "error", "error": str(e)}


# ── Backend loader updater ────────────────────────────────────────────────────
def update_backend_loader(args, results: dict):
    loader_path = Path(args.backend_dir) / "app" / "models" / "loader.py"
    if not loader_path.exists():
        print(f"\n  ⚠  Backend loader not found at {loader_path}")
        print("     Make sure --backend_dir points to your satquery/backend directory.")
        return

    content = loader_path.read_text()

    replacements = {
        'VQA_CKPT        = "Salesforce/blip2-opt-2.7b"':
            f'VQA_CKPT        = "{Path(args.vqa_ckpt).resolve()}"',
        'CAPTION_CKPT    = "Salesforce/blip-image-captioning-large"':
            f'CAPTION_CKPT    = "{Path(args.caption_ckpt).resolve()}"',
        'GROUNDING_CKPT  = "IDEA-Research/grounding-dino-tiny"':
            f'GROUNDING_CKPT  = "{Path(args.grounding_ckpt).resolve()}"',
    }

    updated = content
    for old, new in replacements.items():
        if old in updated:
            updated = updated.replace(old, new)
            print(f"  ✏  Updated: {new[:70]}...")

    # Backup original
    backup = loader_path.with_suffix(".py.bak")
    shutil.copy(loader_path, backup)
    loader_path.write_text(updated)
    print(f"\n  ✅ Backend loader updated: {loader_path}")
    print(f"     Backup saved: {backup}")

    # Also update change model path
    change_model_path = Path(args.backend_dir) / "app" / "models" / "change_model.py"
    if change_model_path.exists() and Path(args.change_ckpt).exists():
        cm = change_model_path.read_text()
        new_line = f'        model.load_state_dict(torch.load("{Path(args.change_ckpt).resolve()}", map_location=DEVICE)["model_state"])'
        if "# Replace this entire method body" in cm:
            insert = cm.replace(
                "        return backbone.to(DEVICE).eval()",
                f"        # ── Fine-tuned ChangeFormer ──\n"
                f"        from app.models.loader import DEVICE as _DEV\n"
                f"        import sys; sys.path.insert(0, '/satquery-ml')\n"
                f"        from training.change.train_changeformer import ChangeFormerNet\n"
                f"        model = ChangeFormerNet()\n"
                f"        ckpt = torch.load('{Path(args.change_ckpt).resolve()}', map_location=DEVICE)\n"
                f"        model.load_state_dict(ckpt.get('model_state', ckpt))\n"
                f"        return model.to(DEVICE).eval()"
            )
            change_model_path.write_text(insert)
            print(f"  ✅ ChangeFormer path updated in change_model.py")


# ── Model card writer ─────────────────────────────────────────────────────────
def write_model_card(args, check_results: dict, test_results: dict):
    card = {
        "name": "SatQuery AI",
        "version": "1.0.0",
        "created": time.strftime("%Y-%m-%d"),
        "description": "Agentic VLM for remote-sensing image analysis — ISRO SIH 2025",
        "models": {
            "vqa":       {"checkpoint": args.vqa_ckpt,       "base": "BLIP-2", "dataset": "BigEarthNet+RSVQA"},
            "caption":   {"checkpoint": args.caption_ckpt,   "base": "BLIP",   "dataset": "BigEarthNet+VRSBench"},
            "grounding": {"checkpoint": args.grounding_ckpt, "base": "GDINO",  "dataset": "VRSBench"},
            "change":    {"checkpoint": args.change_ckpt,    "base": "ChangeFormer","dataset": "LEVIR-CD"},
        },
        "checkpoint_status": check_results,
        "test_results": test_results,
    }
    card_path = Path("checkpoints/model_card.json")
    card_path.parent.mkdir(exist_ok=True)
    card_path.write_text(json.dumps(card, indent=2))
    print(f"\n  📄 Model card written: {card_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = get_args()
    print("\n🛰️  SatQuery AI — Checkpoint Connector")
    print("=" * 55)

    # 1. Validate checkpoints
    print("\n[1/4] Validating checkpoints...")
    ckpts = [
        (args.vqa_ckpt,       "VQA model"),
        (args.caption_ckpt,   "Caption model"),
        (args.grounding_ckpt, "Grounding model"),
        (args.change_ckpt,    "ChangeFormer"),
        (args.backbone_ckpt,  "Backbone"),
    ]
    check_results = {}
    all_ok = True
    for ckpt_path, label in ckpts:
        ok, msg = check_ckpt(ckpt_path, label)
        print(f"  {msg}")
        check_results[label] = {"path": ckpt_path, "valid": ok}
        if not ok:
            all_ok = False

    if not all_ok:
        print("\n  ⚠  Some checkpoints missing. Train them first:")
        print("     python training/bigearth/train_backbone.py")
        print("     python training/vqa/train_vqa.py")
        print("     python training/caption/train_caption.py")
        print("     python training/change/train_changeformer.py")
        print("     python training/grounding/train_grounding.py")

    # 2. Load test image
    print("\n[2/4] Loading test image...")
    if args.test_image and Path(args.test_image).exists():
        test_img = Image.open(args.test_image).convert("RGB")
        print(f"  Using: {args.test_image}")
    else:
        test_img = make_test_image()
        print("  Using: synthetic RS image (256×256)")

    # 3. Run model tests
    test_results = {}
    if not args.skip_tests:
        print("\n[3/4] Testing each model (this may take a few minutes)...")
        if check_results.get("VQA model",{}).get("valid"):
            test_results["vqa"]       = test_vqa(args.vqa_ckpt, test_img)
        if check_results.get("Caption model",{}).get("valid"):
            test_results["caption"]   = test_caption(args.caption_ckpt, test_img)
        if check_results.get("Grounding model",{}).get("valid"):
            test_results["grounding"] = test_grounding(args.grounding_ckpt, test_img)
        if check_results.get("ChangeFormer",{}).get("valid"):
            test_results["change"]    = test_change(args.change_ckpt, test_img)
    else:
        print("\n[3/4] Skipping tests (--skip_tests)")

    # 4. Update backend loader
    print("\n[4/4] Updating backend model loader...")
    update_backend_loader(args, test_results)
    write_model_card(args, check_results, test_results)

    print(f"\n{'='*55}")
    print("✅ Checkpoint connection complete!")
    print("\nNext steps:")
    print("  1. Rebuild Docker: cd ../satquery && docker compose up --build")
    print("  2. Open browser:   http://localhost")
    print("  3. Run evals:      python eval/evaluate_all.py")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
