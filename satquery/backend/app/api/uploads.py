"""
/api/uploads — Image upload endpoint.
Saves file to disk, extracts metadata (GeoTIFF or standard), persists to DB.
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession
from pathlib import Path
import uuid
import aiofiles
import logging

from app.db.database import get_db
from app.db.models import Upload, Session
from app.agents.image_utils import get_geotiff_metadata
from app.core.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

ALLOWED_EXTS = {".tif", ".tiff", ".geotiff", ".png", ".jpg", ".jpeg"}
MAX_BYTES    = settings.MAX_IMAGE_SIZE_MB * 1024 * 1024


@router.post("/")
async def upload_image(
    session_id: str = Form(...),
    slot: str       = Form(...),       # img1 | img2
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    # ── Validate session ──────────────────────────────────────────────────────
    session_row = await db.get(Session, uuid.UUID(session_id))
    if not session_row:
        raise HTTPException(status_code=404, detail="Session not found")

    # ── Validate file ─────────────────────────────────────────────────────────
    fname = file.filename or "upload"
    ext   = Path(fname).suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Allowed: {', '.join(ALLOWED_EXTS)}"
        )

    # ── Save file ─────────────────────────────────────────────────────────────
    save_dir = Path(settings.UPLOAD_DIR) / session_id
    save_dir.mkdir(parents=True, exist_ok=True)

    unique_name = f"{slot}_{uuid.uuid4().hex[:8]}{ext}"
    save_path   = save_dir / unique_name

    total_bytes = 0
    async with aiofiles.open(str(save_path), "wb") as out:
        while chunk := await file.read(1024 * 1024):  # 1 MB chunks
            total_bytes += len(chunk)
            if total_bytes > MAX_BYTES:
                save_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"File exceeds {settings.MAX_IMAGE_SIZE_MB} MB limit"
                )
            await out.write(chunk)

    logger.info(f"Saved upload: {save_path} ({total_bytes / 1024:.1f} KB)")

    # ── Extract metadata ──────────────────────────────────────────────────────
    meta = {}
    if ext in (".tif", ".tiff", ".geotiff"):
        meta = get_geotiff_metadata(str(save_path))

    # ── Persist to DB ─────────────────────────────────────────────────────────
    upload_row = Upload(
        session_id      = uuid.UUID(session_id),
        slot            = slot,
        original_name   = fname,
        stored_path     = str(save_path),
        file_size_bytes = total_bytes,
        file_format     = ext.lstrip("."),
        image_width     = meta.get("width"),
        image_height    = meta.get("height"),
        num_bands       = meta.get("bands"),
        crs             = meta.get("crs"),
        modality        = meta.get("modality", "optical"),
        meta            = meta,
    )
    db.add(upload_row)
    await db.commit()
    await db.refresh(upload_row)

    return {
        "upload_id":      str(upload_row.id),
        "slot":           slot,
        "original_name":  fname,
        "file_size_bytes": total_bytes,
        "file_format":    ext.lstrip("."),
        "width":          meta.get("width"),
        "height":         meta.get("height"),
        "bands":          meta.get("bands"),
        "crs":            meta.get("crs"),
        "modality":       meta.get("modality", "optical"),
        "preview_url":    f"/files/{session_id}/{unique_name}",
    }
