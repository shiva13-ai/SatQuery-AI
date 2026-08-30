from enum import Enum
import re


class TaskType(Enum):
    VQA        = "vqa"
    CAPTION    = "captioning"
    GROUNDING  = "grounding"
    CHANGE     = "change"
    FUSION     = "fusion"


def classify_task(query: str, mode: str) -> TaskType:
    """Rule-based + keyword task classifier. Extend with a fine-tuned classifier."""
    q = query.lower()

    # Mode-driven overrides
    if mode == "bitemporal":
        return TaskType.CHANGE
    if mode == "crossmodal":
        return TaskType.FUSION

    # Keyword matching
    change_kw    = r'\b(change|differ|before|after|temporal|detect|increase|decrease|grow|shrink|flood|deforest)\b'
    fusion_kw    = r'\b(sar|radar|optical|fusion|built.?up|backscatter|sentinel.?1|risat|cross.?modal)\b'
    grounding_kw = r'\b(highlight|locate|where is|show me|ground|region|area of|bbox|bounding)\b'
    caption_kw   = r'\b(describe|caption|scene|land.?cover|overview|what is visible|major object|classify)\b'

    if re.search(change_kw,    q): return TaskType.CHANGE
    if re.search(fusion_kw,    q): return TaskType.FUSION
    if re.search(grounding_kw, q): return TaskType.GROUNDING
    if re.search(caption_kw,   q): return TaskType.CAPTION
    return TaskType.VQA


def build_agent_plan(task_type: TaskType) -> dict:
    plans = {
        TaskType.VQA: {
            "task": "Remote-Sensing VQA",
            "models": ["RS-VQA"],
            "steps": [
                {"name": "Input validation",                    "model": None},
                {"name": "Route → RS-VQA (BLIP-2)",            "model": "RS-VQA"},
                {"name": "Feature extraction (BigEarthNet backbone)", "model": "RS-VQA"},
                {"name": "Answer generation + confidence",       "model": "RS-VQA"},
                {"name": "LLM synthesis via Claude",             "model": "claude-sonnet-4-6"},
            ]
        },
        TaskType.CAPTION: {
            "task": "Scene Captioning",
            "models": ["RS-Caption"],
            "steps": [
                {"name": "Input validation",              "model": None},
                {"name": "Route → RS-Caption (BLIP)",     "model": "RS-Caption"},
                {"name": "Multi-scale feature extraction", "model": "RS-Caption"},
                {"name": "Caption generation",            "model": "RS-Caption"},
                {"name": "LLM synthesis via Claude",       "model": "claude-sonnet-4-6"},
            ]
        },
        TaskType.GROUNDING: {
            "task": "Text-guided Region Grounding",
            "models": ["Grounding DINO", "SAM2"],
            "steps": [
                {"name": "Input validation",                        "model": None},
                {"name": "Parse grounding query",                   "model": None},
                {"name": "Route → Grounding DINO",                  "model": "Grounding DINO"},
                {"name": "Region proposal generation",              "model": "Grounding DINO"},
                {"name": "SAM2 segmentation mask",                  "model": "SAM2"},
                {"name": "Confidence thresholding + NMS",           "model": None},
                {"name": "LLM synthesis via Claude",                "model": "claude-sonnet-4-6"},
            ]
        },
        TaskType.CHANGE: {
            "task": "Bi-temporal Change Detection",
            "models": ["ChangeFormer", "RS-VQA"],
            "steps": [
                {"name": "Validate image pair (T1, T2)",                 "model": None},
                {"name": "Co-registration check",                        "model": None},
                {"name": "Route → ChangeFormer",                         "model": "ChangeFormer"},
                {"name": "Siamese feature extraction",                   "model": "ChangeFormer"},
                {"name": "Change map generation + pixel stats",          "model": "ChangeFormer"},
                {"name": "Route → RS-VQA for change-VQA",               "model": "RS-VQA"},
                {"name": "LLM synthesis via Claude",                     "model": "claude-sonnet-4-6"},
            ]
        },
        TaskType.FUSION: {
            "task": "Optical–SAR Cross-modal Analysis",
            "models": ["Opt-SAR Fusion", "RS-VQA"],
            "steps": [
                {"name": "Validate optical + SAR pair",                   "model": None},
                {"name": "Modality alignment check",                      "model": None},
                {"name": "Route → Opt-SAR Fusion model",                  "model": "Opt-SAR Fusion"},
                {"name": "Complementary feature extraction",              "model": "Opt-SAR Fusion"},
                {"name": "Built-up index + water body mapping",           "model": "Opt-SAR Fusion"},
                {"name": "Route → RS-VQA for QA",                        "model": "RS-VQA"},
                {"name": "LLM synthesis via Claude",                      "model": "claude-sonnet-4-6"},
            ]
        },
    }
    return plans[task_type]
