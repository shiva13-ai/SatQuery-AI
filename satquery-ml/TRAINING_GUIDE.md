# 🛰️ SatQuery AI — Complete ML Training Guide

---

## ⚡ The Complete Pipeline (Do These In Order)

```
Step 0  →  Install deps + download datasets
Step 1  →  Fine-tune BigEarthNet backbone       (~8–12h on 1× A100)
Step 2  →  Fine-tune VQA on RSVQA              (~4–6h  on 1× A100)
Step 3  →  Train ChangeFormer on LEVIR-CD      (~6–10h on 1× A100)
Step 4  →  Fine-tune Captioning on VRSBench    (~3–5h  on 1× A100)
Step 5  →  Fine-tune Grounding on VRSBench     (~4–6h  on 1× A100)
Step 6  →  Evaluate all models on benchmarks
Step 7  →  Connect checkpoints to backend
Step 8  →  Rebuild Docker + test full system
```

---

## Step 0 — Environment Setup

```bash
# Create conda environment
conda create -n satquery python=3.10 -y
conda activate satquery

# Install PyTorch with CUDA 12.1
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121

# Install all training dependencies
pip install -r requirements_train.txt

# Install NLTK data for BLEU
python -c "import nltk; nltk.download('punkt')"
```

---

## Step 0b — Download Datasets

### Automatic (LEVIR-CD, RSVQA):
```bash
python datasets/setup_datasets.py --dataset rsvqa_lr
python datasets/setup_datasets.py --dataset rsvqa_hr
python datasets/setup_datasets.py --dataset levir
python datasets/setup_datasets.py --dataset whu
```

### Manual Downloads Required:

#### 1. BigEarthNet-MM (REQUIRED — primary fine-tuning dataset)
- Register at: https://bigearth.net
- Download Sentinel-2: `BigEarthNet-S2-v1.0.tar.gz` (~43 GB)
- Download Sentinel-1: `BigEarthNet-S1-v1.0.tar.gz` (~22 GB)
- Extract both to `data/bigearth/`

#### 2. VRSBench (for captioning + grounding)
- HuggingFace: https://huggingface.co/datasets/lx709/VRSBench
```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='lx709/VRSBench', repo_type='dataset', local_dir='data/vrsbench')
"
```

#### 3. CDVQA (for change-VQA)
- Request from authors: https://github.com/YZHJessica/CDVQA
- Email the authors using the form on the GitHub page
- Extract to `data/cdvqa/`

### Verify all datasets:
```bash
python datasets/setup_datasets.py --check
```

---

## Step 1 — BigEarthNet Backbone Fine-tuning

**What:** Fine-tunes BLIP-2 vision encoder on BigEarthNet
**Why:** Makes the model understand satellite imagery, SAR, and RS-specific features
**Input data:** `data/bigearth/BigEarthNet-S2/` + `data/bigearth/BigEarthNet-S1/`

```bash
python training/bigearth/train_backbone.py \
  --data_dir    data/bigearth \
  --output_dir  checkpoints/backbone \
  --epochs      10 \
  --batch_size  16 \
  --lr          2e-4 \
  --use_lora    True \
  --lora_r      16 \
  --max_samples 100000
```

**Output:** `checkpoints/backbone/best/`
**Expected:** Loss drops from ~3.5 → ~1.2

---

## Step 2 — VQA Fine-tuning (RSVQA)

**What:** Fine-tunes BLIP-2 to answer questions about RS images
**Input data:** `data/rsvqa_lr/` (Questions_LR_train.json + Images/)

```bash
python training/vqa/train_vqa.py \
  --data_dir   data/rsvqa_lr \
  --backbone   checkpoints/backbone/best \
  --output_dir checkpoints/vqa \
  --epochs     15 \
  --batch_size 8 \
  --lr         1e-4
```

**Output:** `checkpoints/vqa/best/`
**Expected:** VQA accuracy ~75–85% on RSVQA-LR

---

## Step 3 — ChangeFormer Training (LEVIR-CD)

**What:** Trains Siamese Transformer for bi-temporal change detection
**Input data:** `data/levir_cd/train/A/`, `train/B/`, `train/label/`

```bash
python training/change/train_changeformer.py \
  --data_dir   data/levir_cd \
  --output_dir checkpoints/change \
  --epochs     200 \
  --batch_size 8 \
  --lr         6e-5 \
  --img_size   256 \
  --embed_dim  256 \
  --depth      4 \
  --num_heads  8
```

**Output:** `checkpoints/change/best_changeformer.pth`
**Expected:** F1 ~0.78–0.85 on LEVIR-CD test

---

## Step 4 — Captioning Fine-tuning (VRSBench)

**What:** Fine-tunes BLIP for RS scene captioning
**Input data:** `data/vrsbench/images/` + `captions_train.json`

```bash
python training/caption/train_caption.py \
  --data_dir   data/vrsbench \
  --backbone   checkpoints/backbone/best \
  --output_dir checkpoints/caption \
  --epochs     20 \
  --batch_size 16 \
  --lr         3e-5
```

**Output:** `checkpoints/caption/best/`
**Expected:** BLEU-4 ~0.25–0.38

---

## Step 5 — Grounding Fine-tuning (VRSBench)

**What:** Fine-tunes Grounding DINO for RS text-guided region detection
**Input data:** `data/vrsbench/images/` + `grounding_train.json`

```bash
python training/grounding/train_grounding.py \
  --data_dir   data/vrsbench \
  --output_dir checkpoints/grounding \
  --epochs     30 \
  --batch_size 4 \
  --lr         1e-4
```

**Output:** `checkpoints/grounding/best/`
**Expected:** Acc@0.5 IoU ~0.55–0.70

---

## Step 6 — Evaluate All Models

```bash
python eval/evaluate_all.py \
  --vqa_ckpt       checkpoints/vqa/best \
  --caption_ckpt   checkpoints/caption/best \
  --grounding_ckpt checkpoints/grounding/best \
  --change_ckpt    checkpoints/change/best_changeformer.pth \
  --benchmark      all
```

Results saved to `eval/results/eval_results.json`

---

## Step 7 — Connect Checkpoints to Backend

```bash
python integration/connect_checkpoints.py \
  --vqa_ckpt       checkpoints/vqa/best \
  --caption_ckpt   checkpoints/caption/best \
  --grounding_ckpt checkpoints/grounding/best \
  --change_ckpt    checkpoints/change/best_changeformer.pth \
  --backbone_ckpt  checkpoints/backbone/best \
  --backend_dir    ../satquery/backend
```

This script:
- ✅ Validates all checkpoints
- ✅ Tests each model with a sample image
- ✅ Updates `backend/app/models/loader.py` with your checkpoint paths
- ✅ Updates `change_model.py` with your ChangeFormer weights
- ✅ Generates `checkpoints/model_card.json`

---

## Step 8 — Rebuild and Test Full System

```bash
# Rebuild Docker with updated model paths
cd ../satquery
docker compose up --build

# Open browser
open http://localhost

# Or test API directly
curl http://localhost/api/health
```

---

## 📊 Expected Benchmark Results After Training

| Benchmark | Metric | Expected |
|-----------|--------|----------|
| RSVQA-LR  | Accuracy | 75–85% |
| RSVQA-HR  | Accuracy | 70–80% |
| VRSBench  | BLEU-4   | 0.25–0.38 |
| LEVIR-CD  | F1       | 0.78–0.85 |
| LEVIR-CD  | IoU      | 0.65–0.75 |
| CDVQA     | Accuracy | 68–78% |

---

## 💾 Checkpoint Size Reference

| Model | Size |
|-------|------|
| Backbone (BLIP-2 + LoRA) | ~15–20 GB |
| VQA (BLIP-2 + LoRA)      | ~15–20 GB |
| Caption (BLIP + LoRA)    | ~1–2 GB   |
| Grounding (GDINO tiny)   | ~700 MB   |
| ChangeFormer             | ~200–400 MB |

**Total disk needed:** ~50–60 GB for all checkpoints + datasets ~100 GB

---

## ⚡ GPU Requirements

| Training | Min GPU | Recommended |
|----------|---------|-------------|
| BigEarthNet backbone | 24 GB VRAM | A100 80GB |
| VQA fine-tuning      | 16 GB VRAM | A100 40GB |
| ChangeFormer         | 8 GB VRAM  | RTX 3090   |
| Captioning           | 16 GB VRAM | A100 40GB  |
| Grounding            | 8 GB VRAM  | RTX 3090   |

**Google Colab Pro+ / Kaggle GPU (T4/A100) works for all except backbone.**
**For backbone with limited VRAM: reduce --batch_size to 4 and enable --fp16**

---

## 🔁 Quick Resume After Interruption

```bash
# Resume backbone from step checkpoint
python training/bigearth/train_backbone.py \
  --data_dir checkpoints/backbone/step_5000  # resume from step

# Resume ChangeFormer from epoch
python training/change/train_changeformer.py \
  --data_dir data/levir_cd \
  --resume   checkpoints/change/epoch_100.pth
```

---

*SatQuery AI — ISRO Smart India Hackathon 2025*
