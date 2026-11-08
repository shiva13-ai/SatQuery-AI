# SatQuery AI

A vision-language assistant for satellite image analysis, started for Smart India Hackathon 2025 (ISRO problem statement 26167). A FastAPI backend classifies a question, runs a matching vision model, and then asks Claude to write up the answer with the model output as context.

The repo has three parts: the backend and Docker setup in `satquery/`, a React frontend in `satquery/frontend/`, and training/evaluation scripts in `satquery-ml/`.

## What it does

- Routes a query to one of five tasks with a keyword-based classifier: VQA, captioning, grounding, change detection, optical-SAR fusion (`satquery/backend/app/agents/tasks.py`).
- VQA with BLIP-2 (`Salesforce/blip2-opt-2.7b`) and captioning with BLIP (`Salesforce/blip-image-captioning-large`).
- Text-guided grounding with Grounding DINO (`IDEA-Research/grounding-dino-tiny`).
- Bi-temporal change detection through a ChangeFormer wrapper. The checkpoint ID in `loader.py` is marked as a placeholder.
- Sends the local model output to Claude (`claude-sonnet-4-6`) to write the final answer, and keeps an agent trace of each step.
- Stores sessions, messages and uploads in PostgreSQL; generates PDF and JSON reports.

Not finished: the optical-SAR fusion step only records trace entries and does not run a model yet. The confidence values in the controller are fixed numbers, not model-calibrated scores. SAM2 is listed as a model used for grounding but is not called in the controller.

## Tech stack

- Backend: Python, FastAPI, SQLAlchemy (asyncpg), Celery, Redis, PostgreSQL
- ML: PyTorch, Hugging Face Transformers, rasterio for GeoTIFF input
- LLM: Anthropic API
- Frontend: React 18, Vite, Tailwind CSS, Radix UI, wouter
- Deployment: Docker Compose

## Project structure

```
satquery/
  docker-compose.yml     postgres, redis, backend, celery worker, frontend
  .env.example
  scripts/init.sql       database schema
  backend/app/
    main.py              FastAPI entry point
    celery_app.py
    core/config.py       settings read from env
    api/                 health, sessions, uploads, analyze, reports
    agents/              controller, task classifier, image utils
    models/              model registry and wrappers
    db/                  engine and ORM models
  frontend/              React + Vite app
satquery-ml/
  datasets/              dataset download script
  training/              backbone, vqa, caption, grounding, change
  eval/evaluate_all.py
  integration/connect_checkpoints.py
  TRAINING_GUIDE.md
```

## Getting started

You need Docker with Compose. The compose file reserves an NVIDIA GPU for the backend, so you need the NVIDIA container toolkit unless you remove that block.

```bash
cd satquery
cp .env.example .env
# set ANTHROPIC_API_KEY in .env
docker compose up --build
```

The backend is on port 8000 and the API docs are at `http://localhost:8000/api/docs`.

Known problem: `docker-compose.yml` builds the frontend from `./frontend/Dockerfile`, but there is no Dockerfile or nginx config in the repo, so the full stack build will fail on that service. You can start the other services by name, for example `docker compose up --build postgres redis backend celery_worker`.

Backend without Docker (needs PostgreSQL and Redis, and `DATABASE_URL` / `REDIS_URL` set; the defaults point at the compose hostnames):

```bash
cd satquery/backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

`main.py` mounts `/app/uploads` and `/app/reports`, so these folders must exist locally or you need to adjust the paths.

Frontend dev server (port 3000, proxies `/api` to `localhost:8000`):

```bash
cd satquery/frontend
npm install
npm run dev
```

Note that the frontend is not fully connected to the backend yet. `src/lib/trpc.ts` imports types from `server/routers`, which is not in the repo, and the upload page posts to `/api/captures/upload`, which the FastAPI app does not define.

## API

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/health` | health check |
| POST | `/api/sessions/` | create a session |
| GET | `/api/sessions/{id}` | session info |
| GET | `/api/sessions/{id}/messages` | conversation history |
| POST | `/api/uploads/` | upload an image |
| POST | `/api/analyze/` | run the agent pipeline |
| POST | `/api/reports/{id}/generate` | generate PDF and JSON report |
| GET | `/api/reports/{id}/download/pdf` | download the PDF |
| GET | `/api/reports/{id}/download/trace` | download the agent trace |

## Environment variables

Set in `satquery/.env` (see `.env.example`):

- `ANTHROPIC_API_KEY` (required)
- `HF_TOKEN` (optional, for gated models)
- `LOAD_VQA_MODEL`, `LOAD_CAPTION_MODEL`, `LOAD_GROUNDING_MODEL`, `LOAD_CHANGE_MODEL`, `LOAD_SAR_MODEL` (set to false to skip loading a model, useful on CPU-only machines)
- `MAX_IMAGE_SIZE_MB`

Also read by the backend: `DATABASE_URL`, `REDIS_URL`, `MODEL_CACHE_DIR`, `UPLOAD_DIR`, `REPORT_DIR`, `CORS_ORIGINS`.

## Training and evaluation

`satquery-ml/` has scripts to fine-tune the backbone (BigEarthNet), VQA, captioning, grounding and change detection models, plus an evaluation script and a script that connects the trained checkpoints to the backend. The order of steps and commands are in `satquery-ml/TRAINING_GUIDE.md`.

```bash
cd satquery-ml
pip install -r requirements_train.txt
python datasets/setup_datasets.py --dataset rsvqa_lr
python eval/evaluate_all.py --benchmark all
```

Checkpoints and data folders are git-ignored.
