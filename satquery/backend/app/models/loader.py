"""
SatQuery AI — Model Registry
Loads and manages all specialist remote-sensing models from HuggingFace.
Swap checkpoint paths here when you have your own fine-tuned weights.
"""

import logging
import torch
from app.core.config import settings

logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"🖥️  Inference device: {DEVICE}")


class ModelRegistry:
    """Singleton registry holding all loaded specialist models."""

    _vqa_processor     = None
    _vqa_model         = None
    _caption_processor = None
    _caption_model     = None
    _grounding_model   = None
    _grounding_proc    = None
    _sam_model         = None
    _change_model      = None
    _sar_model         = None
    _loaded            = False

    # ── Checkpoint configs ────────────────────────────────────────────────────
    # Replace with your own checkpoint paths (local or HF repo id)
    VQA_CKPT        = "Salesforce/blip2-opt-2.7b"          # BLIP-2 for RS-VQA
    CAPTION_CKPT    = "Salesforce/blip-image-captioning-large"  # BLIP for captioning
    GROUNDING_CKPT  = "IDEA-Research/grounding-dino-tiny"   # Grounding DINO
    CHANGE_CKPT     = "chenyang03/ChangeFormer-EF-CD"       # ChangeFormer (placeholder)
    SAR_CKPT        = "microsoft/resnet-50"                 # Fallback fusion backbone

    @classmethod
    async def load_all(cls):
        if cls._loaded:
            return
        logger.info("Loading models… (first run may take a few minutes)")

        if settings.LOAD_VQA_MODEL:
            cls._load_vqa()
        if settings.LOAD_CAPTION_MODEL:
            cls._load_caption()
        if settings.LOAD_GROUNDING_MODEL:
            cls._load_grounding()
        if settings.LOAD_CHANGE_MODEL:
            cls._load_change()

        cls._loaded = True
        logger.info("✅ All models loaded.")

    # ── VQA (BLIP-2) ─────────────────────────────────────────────────────────
    @classmethod
    def _load_vqa(cls):
        try:
            from transformers import Blip2Processor, Blip2ForConditionalGeneration
            logger.info(f"Loading VQA model: {cls.VQA_CKPT}")
            cls._vqa_processor = Blip2Processor.from_pretrained(
                cls.VQA_CKPT, cache_dir=settings.MODEL_CACHE_DIR
            )
            cls._vqa_model = Blip2ForConditionalGeneration.from_pretrained(
                cls.VQA_CKPT,
                cache_dir=settings.MODEL_CACHE_DIR,
                torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                device_map=DEVICE,
            )
            cls._vqa_model.eval()
            logger.info("✅ VQA model ready")
        except Exception as e:
            logger.warning(f"⚠️  VQA model load failed (using Anthropic fallback): {e}")

    # ── Captioning (BLIP) ────────────────────────────────────────────────────
    @classmethod
    def _load_caption(cls):
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            logger.info(f"Loading Caption model: {cls.CAPTION_CKPT}")
            cls._caption_processor = BlipProcessor.from_pretrained(
                cls.CAPTION_CKPT, cache_dir=settings.MODEL_CACHE_DIR
            )
            cls._caption_model = BlipForConditionalGeneration.from_pretrained(
                cls.CAPTION_CKPT,
                cache_dir=settings.MODEL_CACHE_DIR,
                torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            ).to(DEVICE)
            cls._caption_model.eval()
            logger.info("✅ Caption model ready")
        except Exception as e:
            logger.warning(f"⚠️  Caption model load failed: {e}")

    # ── Grounding DINO ────────────────────────────────────────────────────────
    @classmethod
    def _load_grounding(cls):
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            logger.info(f"Loading Grounding model: {cls.GROUNDING_CKPT}")
            cls._grounding_proc = AutoProcessor.from_pretrained(
                cls.GROUNDING_CKPT, cache_dir=settings.MODEL_CACHE_DIR
            )
            cls._grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                cls.GROUNDING_CKPT, cache_dir=settings.MODEL_CACHE_DIR
            ).to(DEVICE)
            cls._grounding_model.eval()
            logger.info("✅ Grounding model ready")
        except Exception as e:
            logger.warning(f"⚠️  Grounding model load failed: {e}")

    # ── ChangeFormer ─────────────────────────────────────────────────────────
    @classmethod
    def _load_change(cls):
        try:
            # ChangeFormer is a custom architecture; we provide a wrapper
            from app.models.change_model import ChangeDetectionModel
            cls._change_model = ChangeDetectionModel()
            logger.info("✅ Change detection model ready")
        except Exception as e:
            logger.warning(f"⚠️  Change model load failed: {e}")

    # ── Accessors ─────────────────────────────────────────────────────────────
    @classmethod
    def get_vqa(cls):
        return cls._vqa_processor, cls._vqa_model

    @classmethod
    def get_caption(cls):
        return cls._caption_processor, cls._caption_model

    @classmethod
    def get_grounding(cls):
        return cls._grounding_proc, cls._grounding_model

    @classmethod
    def get_change(cls):
        return cls._change_model

    @classmethod
    def status(cls):
        return {
            "vqa":       cls._vqa_model       is not None,
            "caption":   cls._caption_model   is not None,
            "grounding": cls._grounding_model is not None,
            "change":    cls._change_model    is not None,
            "device":    DEVICE,
        }
