# 🛰️ SatQuery AI
**Agentic Vision-Language Assistant for Multimodal Remote-Sensing Image Analysis**
*ISRO Smart India Hackathon 2025 — Problem Statement #26167*

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    DOCKER COMPOSE STACK                   │
│                                                          │
│  ┌─────────────┐    ┌──────────────┐   ┌─────────────┐  │
│  │  Frontend   │    │  FastAPI     │   │ PostgreSQL  │  │
│  │  (Nginx)    │───▶│  Backend     │──▶│   (DB)      │  │
│  │  Port 80    │    │  Port 8000   │   │  Port 5432  │  │
│  └─────────────┘    └──────┬───────┘   └─────────────┘  │
│                            │                             │
│                     ┌──────▼───────┐   ┌─────────────┐  │
│                     │   Agentic    │   │   Redis     │  │
│                     │  Controller  │   │ (Task Queue)│  │
│                     └──────┬───────┘   └─────────────┘  │
│                            │                             │
│            ┌───────────────┼───────────────┐            │
│            ▼               ▼               ▼            │
│        RS-VQA         ChangeFormer    Opt-SAR           │
│        BLIP-2         (Change Det.)   Fusion            │
│        RS-Caption     RS-Ground                         │
│        (BLIP)         (GDINO+SAM2)                      │
└──────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Docker + Docker Compose
- NVIDIA GPU recommended (falls back to CPU)
- `nvidia-container-toolkit` for GPU support

### 1. Clone and configure
```bash
git clone <your-repo>
cd satquery
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
nano .env
```

### 2. Launch the full stack
```bash
docker compose up --build
```

### 3. Open the app
```
http://localhost
```

API docs available at: `http://localhost:8000/api/docs`

---

## 📁 Project Structure

```
satquery/
├── docker-compose.yml          # Full stack orchestration
├── .env.example                # Environment template
├── scripts/
│   └── init.sql               # PostgreSQL schema
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/
│       ├── main.py            # FastAPI entry point
│       ├── celery_app.py      # Celery task queue
│       ├── core/
│       │   └── config.py      # Settings / env vars
│       ├── db/
│       │   ├── database.py    # SQLAlchemy async engine
│       │   └── models.py      # ORM models
│       ├── api/
│       │   ├── health.py      # GET /api/health
│       │   ├── sessions.py    # /api/sessions
│       │   ├── uploads.py     # /api/uploads
│       │   ├── analyze.py     # /api/analyze  ← CORE
│       │   └── reports.py     # /api/reports
│       ├── agents/
│       │   ├── controller.py  # Agentic orchestrator
│       │   ├── tasks.py       # Task classification + plan
│       │   └── image_utils.py # GeoTIFF loading, preprocessing
│       └── models/
│           ├── loader.py      # ModelRegistry (all HF models)
│           └── change_model.py # ChangeDetection wrapper
└── frontend/
    ├── Dockerfile
    ├── nginx.conf
    └── satquery-ai.html       # Full SPA (connects to API)
```

---

## 🔌 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Model + system status |
| POST | `/api/sessions/` | Create new session |
| GET | `/api/sessions/{id}` | Get session info |
| GET | `/api/sessions/{id}/messages` | Full conversation history |
| POST | `/api/uploads/` | Upload satellite image |
| POST | `/api/analyze/` | **Core: run agentic pipeline** |
| POST | `/api/reports/{id}/generate` | Generate PDF + JSON report |
| GET | `/api/reports/{id}/download/pdf` | Download PDF report |
| GET | `/api/reports/{id}/download/trace` | Download agent trace JSON |

---

## 🤖 Plugging In Your Own Checkpoints

Edit `backend/app/models/loader.py`:

```python
# Replace these with your fine-tuned checkpoint paths or HF repo IDs
VQA_CKPT        = "your-hf-username/rs-vqa-bigearhtnet"
CAPTION_CKPT    = "your-hf-username/rs-caption-vrs"
GROUNDING_CKPT  = "IDEA-Research/grounding-dino-tiny"
CHANGE_CKPT     = "your-hf-username/changeformer-levir"
```

For ChangeFormer, edit `backend/app/models/change_model.py`:
```python
def _build_model(self):
    model = ChangeFormer(...)   # your architecture
    model.load_state_dict(torch.load('your_checkpoint.pth'))
    return model.to(DEVICE).eval()
```

---

## 📊 Datasets

| Dataset | Purpose | Link |
|---------|---------|------|
| BigEarthNet | Primary RS fine-tuning | arxiv.org/abs/2603.29630 |
| VRSBench | Captioning + VQA eval | GitHub |
| RSVQA | Single-image VQA eval | GitHub |
| CDVQA | Change-based VQA eval | GitHub |

---

## 🧪 Evaluation

Run on prescribed test splits:
```bash
# VQA evaluation
python eval/eval_vqa.py --benchmark rsvqa --split test

# Change detection evaluation
python eval/eval_change.py --benchmark cdvqa --split test

# Captioning evaluation
python eval/eval_caption.py --benchmark vrsbench --split test
```

---

## ⚙️ Development Mode (without Docker)

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Frontend: just open frontend/satquery-ai.html in browser
# (Update API = 'http://localhost:8000/api' in the HTML)
```

---

## 🏆 SIH Evaluation Checklist

- [x] Single-image VQA (RSVQA benchmark)
- [x] Scene captioning (VRSBench benchmark)
- [x] Text-guided region grounding (Grounding DINO + SAM2)
- [x] Bi-temporal change detection + C-VQA (CDVQA)
- [x] Optical–SAR cross-modal analysis
- [x] Agentic model/tool orchestration with audit trace
- [x] BigEarthNet fine-tuning (BLIP-2 backbone)
- [x] GeoTIFF / TIFF input support
- [x] Confidence scores per response
- [x] Downloadable PDF report + JSON trace
- [x] PostgreSQL session persistence
- [x] Docker containerised deployment
- [x] Interactive web GUI

---

*Built for ISRO SIH 2025 — Department of Space / Indian Space Research Organisation*
