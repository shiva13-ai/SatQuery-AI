#!/usr/bin/env python3
"""
SatQuery AI — Dataset Setup Master Script
==========================================
Run this FIRST before any training.
Downloads and prepares all required datasets.

DATASETS:
1. BigEarthNet-MM  → Fine-tuning backbone (Sentinel-1 SAR + Sentinel-2 optical)
2. RSVQA-LR/HR    → Single-image VQA training + evaluation
3. VRSBench        → Captioning + Grounding + VQA evaluation
4. CDVQA           → Change-based VQA training + evaluation
5. LEVIR-CD        → Change detection training (ChangeFormer)
6. WHU-CD          → Change detection training (additional)

USAGE:
    python datasets/setup_datasets.py --all
    python datasets/setup_datasets.py --dataset bigearth
    python datasets/setup_datasets.py --dataset rsvqa
    python datasets/setup_datasets.py --dataset levir
"""

import os
import sys
import argparse
import subprocess
import zipfile
import tarfile
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
DATASET_INFO = {

    "bigearth": {
        "name": "BigEarthNet-MM",
        "desc": "Sentinel-1 SAR + Sentinel-2 multispectral pairs with multi-label annotations",
        "size": "~65 GB total (Sentinel-2: 43GB, Sentinel-1: 22GB)",
        "url_s2":  "https://bigearth.net/downloads/BigEarthNet-S2-v1.0.tar.gz",
        "url_s1":  "https://bigearth.net/downloads/BigEarthNet-S1-v1.0.tar.gz",
        "url_meta":"https://git.tu-berlin.de/rsim/BigEarthNet-MM/-/raw/master/splits/train.csv",
        "target":  DATA / "bigearth",
        "note": "Register at bigearth.net first — requires manual download due to license.",
        "manual": True,
    },

    "rsvqa_lr": {
        "name": "RSVQA-LR",
        "desc": "Low-Resolution Remote Sensing Visual Question Answering",
        "size": "~1 GB",
        "url":   "https://zenodo.org/record/6344334/files/RSVQA-LR.zip",
        "target": DATA / "rsvqa_lr",
        "manual": False,
    },

    "rsvqa_hr": {
        "name": "RSVQA-HR",
        "desc": "High-Resolution Remote Sensing Visual Question Answering",
        "size": "~3.5 GB",
        "url":   "https://zenodo.org/record/6344337/files/RSVQA-HR.zip",
        "target": DATA / "rsvqa_hr",
        "manual": False,
    },

    "vrsbench": {
        "name": "VRSBench",
        "desc": "Captioning + Grounding + VQA benchmark for RS images",
        "size": "~2 GB",
        "url":   "https://github.com/lx709/VRSBench",
        "target": DATA / "vrsbench",
        "manual": True,
        "note": "Download from HuggingFace: huggingface.co/datasets/lx709/VRSBench",
    },

    "cdvqa": {
        "name": "CDVQA",
        "desc": "Change Detection + Visual Question Answering (bi-temporal)",
        "size": "~4 GB",
        "url":   "https://github.com/YZHJessica/CDVQA",
        "target": DATA / "cdvqa",
        "manual": True,
        "note": "Request from authors at: github.com/YZHJessica/CDVQA",
    },

    "levir": {
        "name": "LEVIR-CD",
        "desc": "Large-scale Remote Sensing Change Detection dataset (building change)",
        "size": "~1 GB",
        "url":   "https://justchenhao.github.io/LEVIR/",
        "hf":    "lucasgonzalez7/levir-cd-dataset",
        "target": DATA / "levir_cd",
        "manual": False,
        "hf_download": True,
    },

    "whu": {
        "name": "WHU-CD",
        "desc": "WHU Building Change Detection dataset",
        "size": "~1 GB",
        "hf":    "yanghui2000/whu-cd",
        "target": DATA / "whu_cd",
        "manual": False,
        "hf_download": True,
    },
}


def download_file(url: str, dest: Path, desc: str = ""):
    """Download with progress bar."""
    print(f"  ⬇  Downloading {desc or url}...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest,
        reporthook=lambda b, bs, t: print(
            f"\r     {min(100, b*bs/t*100):.0f}% ({b*bs/1e6:.1f}/{t/1e6:.1f} MB)",
            end="", flush=True
        ) if t > 0 else None
    )
    print()


def extract(path: Path, dest: Path):
    """Extract zip or tar.gz."""
    print(f"  📦 Extracting {path.name}...")
    dest.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            z.extractall(dest)
    elif path.name.endswith(".tar.gz") or path.name.endswith(".tgz"):
        with tarfile.open(path) as t:
            t.extractall(dest)


def setup_from_hf(hf_repo: str, dest: Path):
    """Download from HuggingFace datasets."""
    print(f"  🤗 Downloading from HuggingFace: {hf_repo}")
    try:
        subprocess.run([
            sys.executable, "-c",
            f"""
from huggingface_hub import snapshot_download
snapshot_download(repo_id='{hf_repo}', repo_type='dataset', local_dir='{dest}')
print('Done.')
"""
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"  ⚠  HF download failed: {e}. Try: pip install huggingface_hub")


def setup_rsvqa(name: str, url: str, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / f"{name}.zip"
    download_file(url, zip_path, name)
    extract(zip_path, dest)
    zip_path.unlink()
    print(f"  ✅ {name} ready at {dest}")


def print_manual_instructions(key: str, info: dict):
    print(f"\n  📋 MANUAL DOWNLOAD REQUIRED: {info['name']}")
    print(f"     Size: {info['size']}")
    print(f"     URL:  {info['url']}")
    print(f"     Note: {info.get('note','')}")
    print(f"     Save to: {info['target']}")
    print()


def verify_dataset(key: str, path: Path) -> bool:
    """Check if dataset directory has content."""
    if not path.exists():
        return False
    files = list(path.rglob("*"))
    return len(files) > 10


def write_dataset_config():
    """Write paths config for training scripts."""
    config = {}
    for key, info in DATASET_INFO.items():
        config[key] = {
            "path":   str(info["target"]),
            "ready":  verify_dataset(key, info["target"]),
            "name":   info["name"],
        }
    cfg_path = DATA / "dataset_config.json"
    cfg_path.write_text(json.dumps(config, indent=2))
    print(f"\n📄 Dataset config written to {cfg_path}")
    return config


def main():
    parser = argparse.ArgumentParser(description="SatQuery AI — Dataset Setup")
    parser.add_argument("--all",     action="store_true", help="Setup all datasets")
    parser.add_argument("--dataset", type=str, choices=list(DATASET_INFO.keys()),
                        help="Setup a specific dataset")
    parser.add_argument("--check",   action="store_true", help="Check which datasets are ready")
    args = parser.parse_args()

    if args.check:
        print("\n📊 Dataset Status:")
        for key, info in DATASET_INFO.items():
            ready = verify_dataset(key, info["target"])
            status = "✅ Ready" if ready else "❌ Missing"
            print(f"  {status} — {info['name']} ({info['size']})")
        return

    targets = list(DATASET_INFO.keys()) if args.all else ([args.dataset] if args.dataset else [])
    if not targets:
        parser.print_help()
        return

    print("\n🛰️  SatQuery AI — Dataset Setup")
    print("=" * 50)

    for key in targets:
        info = DATASET_INFO[key]
        print(f"\n🔄 Setting up: {info['name']}")

        if info.get("manual"):
            print_manual_instructions(key, info)
            continue

        dest = info["target"]

        if info.get("hf_download"):
            setup_from_hf(info["hf"], dest)
        elif "url" in info:
            setup_rsvqa(key, info["url"], dest)

    config = write_dataset_config()
    print("\n📊 Final Status:")
    for key, cfg in config.items():
        status = "✅" if cfg["ready"] else "⚠ "
        print(f"  {status} {cfg['name']}")

    print("\n✅ Setup complete. Run training scripts next.")
    print("   → python training/bigearth/train_backbone.py")
    print("   → python training/vqa/train_vqa.py")
    print("   → python training/change/train_changeformer.py")


if __name__ == "__main__":
    main()
