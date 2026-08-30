from fastapi import APIRouter
from app.models.loader import ModelRegistry

router = APIRouter()


@router.get("/health")
async def health():
    model_status = ModelRegistry.status()
    return {
        "status": "ok",
        "service": "SatQuery AI Backend",
        "version": "1.0.0",
        "models": model_status,
    }
