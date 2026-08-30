from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "postgresql+asyncpg://satquery:satquery_secret_2025@postgres:5432/satquery"

    # Redis
    REDIS_URL: str = "redis://redis:6379"

    # AI
    ANTHROPIC_API_KEY: str = ""
    HF_TOKEN: str = ""

    # Storage
    MODEL_CACHE_DIR: str = "/app/model_cache"
    UPLOAD_DIR: str = "/app/uploads"
    REPORT_DIR: str = "/app/reports"
    MAX_IMAGE_SIZE_MB: int = 100

    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost",
        "http://localhost:80",
        "http://localhost:3000",
        "http://frontend",
    ]

    # Model flags (set False to skip loading on CPU-only machines)
    LOAD_VQA_MODEL: bool = True
    LOAD_CAPTION_MODEL: bool = True
    LOAD_GROUNDING_MODEL: bool = True
    LOAD_CHANGE_MODEL: bool = True
    LOAD_SAR_MODEL: bool = True

    class Config:
        env_file = ".env"


settings = Settings()
