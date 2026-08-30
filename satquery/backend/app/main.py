"""
SatQuery AI — FastAPI Backend
ISRO Smart India Hackathon 2025
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import logging

from app.core.config import settings
from app.db.database import init_db
from app.api import sessions, analyze, uploads, reports, health
from app.models.loader import ModelRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🛰️  SatQuery AI starting up...")
    await init_db()
    logger.info("✅ Database initialized")
    await ModelRegistry.load_all()
    logger.info("✅ Model registry loaded")
    yield
    # Shutdown
    logger.info("🛰️  SatQuery AI shutting down...")


app = FastAPI(
    title="SatQuery AI",
    description="Agentic Vision-Language Assistant for Remote-Sensing Image Analysis",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(health.router,   prefix="/api",          tags=["health"])
app.include_router(sessions.router, prefix="/api/sessions", tags=["sessions"])
app.include_router(uploads.router,  prefix="/api/uploads",  tags=["uploads"])
app.include_router(analyze.router,  prefix="/api/analyze",  tags=["analyze"])
app.include_router(reports.router,  prefix="/api/reports",  tags=["reports"])

# Static files (uploaded images, result images)
app.mount("/files", StaticFiles(directory="/app/uploads"), name="uploads")
app.mount("/reports", StaticFiles(directory="/app/reports"), name="reports")
