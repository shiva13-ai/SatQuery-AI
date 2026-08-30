"""
Image utilities — load, preprocess, save result images for SatQuery AI.
Supports GeoTIFF (via rasterio) and standard formats (via PIL).
"""

import os
import uuid
import logging
import numpy as np
from PIL import Image
from pathlib import Path
from app.core.config import settings

logger = logging.getLogger(__name__)


def load_image_from_path(path: str) -> Image.Image | None:
    """
    Load an image from disk.
    Tries rasterio first (GeoTIFF), falls back to PIL.
    Returns RGB PIL Image or None.
    """
    if not path or not os.path.exists(path):
        return None

    ext = Path(path).suffix.lower()

    # GeoTIFF path
    if ext in (".tif", ".tiff", ".geotiff"):
        try:
            return _load_geotiff(path)
        except Exception as e:
            logger.warning(f"rasterio failed ({e}), falling back to PIL")

    # Standard image path
    try:
        img = Image.open(path).convert("RGB")
        return img
    except Exception as e:
        logger.error(f"PIL load failed for {path}: {e}")
        return None


def _load_geotiff(path: str) -> Image.Image:
    """
    Read a GeoTIFF and return an 8-bit RGB PIL image.
    Handles multi-band Sentinel-1/2 by selecting relevant bands.
    """
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(path) as src:
        n_bands = src.count
        logger.info(f"GeoTIFF: {n_bands} bands, {src.width}×{src.height}, CRS={src.crs}")

        # Cap size to 1024px for memory safety
        scale = min(1.0, 1024 / max(src.width, src.height))
        out_w  = max(1, int(src.width  * scale))
        out_h  = max(1, int(src.height * scale))

        if n_bands >= 3:
            # Use bands 1,2,3 (or Sentinel-2 B4,B3,B2 = RGB)
            data = src.read(
                [1, 2, 3],
                out_shape=(3, out_h, out_w),
                resampling=Resampling.bilinear,
            )
        else:
            # SAR or single-band: replicate to RGB
            band = src.read(
                1,
                out_shape=(1, out_h, out_w),
                resampling=Resampling.bilinear,
            )
            data = np.repeat(band[np.newaxis], 3, axis=0).squeeze(0)
            data = data[np.newaxis]  # (1, H, W)

        # Normalise to 0-255
        data = data.astype(np.float32)
        lo, hi = np.percentile(data, 2), np.percentile(data, 98)
        if hi > lo:
            data = np.clip((data - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
        else:
            data = (data / data.max() * 255).astype(np.uint8) if data.max() > 0 else data.astype(np.uint8)

        # CHW → HWC
        rgb = np.transpose(data, (1, 2, 0))
        if rgb.shape[2] == 1:
            rgb = np.concatenate([rgb] * 3, axis=2)

        return Image.fromarray(rgb, mode="RGB")


def get_geotiff_metadata(path: str) -> dict:
    """Extract metadata from a GeoTIFF for input validation."""
    try:
        import rasterio
        with rasterio.open(path) as src:
            return {
                "width":    src.width,
                "height":   src.height,
                "bands":    src.count,
                "crs":      str(src.crs) if src.crs else None,
                "dtype":    str(src.dtypes[0]),
                "format":   "geotiff",
                "modality": _guess_modality(src),
            }
    except Exception:
        return {}


def _guess_modality(src) -> str:
    """Guess sensor modality from band count and metadata."""
    n = src.count
    desc = [str(d).lower() for d in (src.descriptions or [])]
    if any("vv" in d or "vh" in d for d in desc):
        return "sar"
    if n >= 10:
        return "multispectral"
    if n >= 3:
        return "optical"
    return "sar"


def preprocess_for_model(img: Image.Image, size: int = 224) -> Image.Image:
    """Resize and normalise image for model input."""
    return img.resize((size, size), Image.LANCZOS)


async def save_result_image(
    array: np.ndarray,
    session_id: str,
    label: str = "result",
) -> str:
    """
    Save a numpy array (H×W or H×W×3) as PNG to the reports directory.
    Returns the relative URL path.
    """
    out_dir = Path(settings.REPORT_DIR) / session_id
    out_dir.mkdir(parents=True, exist_ok=True)

    fname = f"{label}_{uuid.uuid4().hex[:8]}.png"
    fpath = out_dir / fname

    if array.ndim == 2:
        img = Image.fromarray(array.astype(np.uint8), mode="L")
    else:
        img = Image.fromarray(array.astype(np.uint8), mode="RGB")

    img.save(str(fpath))
    return f"/reports/{session_id}/{fname}"


def validate_image_pair_compatibility(
    meta1: dict, meta2: dict, mode: str
) -> tuple[bool, str]:
    """
    Check that two images are compatible for the requested mode.
    Returns (is_compatible, message).
    """
    if not meta1 or not meta2:
        return False, "Could not read metadata from one or both images."

    if mode == "crossmodal":
        m1, m2 = meta1.get("modality"), meta2.get("modality")
        if m1 == m2:
            return True, f"Warning: both images appear to be {m1}. Expected one optical + one SAR."
        return True, f"Cross-modal pair validated: {m1} + {m2}"

    if mode == "bitemporal":
        w1, h1 = meta1.get("width", 0), meta1.get("height", 0)
        w2, h2 = meta2.get("width", 0), meta2.get("height", 0)
        if abs(w1 - w2) > w1 * 0.2 or abs(h1 - h2) > h1 * 0.2:
            return False, f"Image dimensions differ significantly: {w1}×{h1} vs {w2}×{h2}. Ensure co-registration."
        return True, f"Bi-temporal pair validated: {w1}×{h1} ≈ {w2}×{h2}"

    return True, "Single image validated."
