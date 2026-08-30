"""
/api/analyze — Core analysis endpoint
Accepts query + session_id, runs agentic pipeline, persists result.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
import uuid
import anthropic
import logging

from app.db.database import get_db
from app.db.models import Session, Message, AnalysisResult, AgentLog
from app.agents.controller import AgentController
from app.core.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

# Shared Anthropic client
_anthropic_client = None

def get_anthropic():
    global _anthropic_client
    if _anthropic_client is None and settings.ANTHROPIC_API_KEY:
        _anthropic_client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    return _anthropic_client


class AnalyzeRequest(BaseModel):
    session_id: str
    query: str
    mode: str = "single"          # single | bitemporal | crossmodal
    img1_upload_id: str | None = None
    img2_upload_id: str | None = None


class AnalyzeResponse(BaseModel):
    message_id: str
    task_type: str
    ai_text: str
    models_used: list
    agent_trace: list
    confidence: float
    latency_ms: int
    change_stats: dict | None = None
    result_image_url: str | None = None
    bounding_boxes: list = []


@router.post("/", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest, db: AsyncSession = Depends(get_db)):
    # ── Validate session ──────────────────────────────────────────────────────
    session_row = await db.get(Session, uuid.UUID(req.session_id))
    if not session_row:
        raise HTTPException(status_code=404, detail="Session not found")

    # ── Resolve image paths from uploads ─────────────────────────────────────
    from app.db.models import Upload
    img1_path = img2_path = None

    if req.img1_upload_id:
        up = await db.get(Upload, uuid.UUID(req.img1_upload_id))
        if up:
            img1_path = up.stored_path

    if req.img2_upload_id:
        up = await db.get(Upload, uuid.UUID(req.img2_upload_id))
        if up:
            img2_path = up.stored_path

    # ── Fetch conversation history for this session ───────────────────────────
    result = await db.execute(
        select(Message)
        .where(Message.session_id == uuid.UUID(req.session_id))
        .order_by(Message.created_at.asc())
        .limit(20)
    )
    history_rows = result.scalars().all()
    conversation_history = [
        {"role": m.role, "content": m.content}
        for m in history_rows
    ]

    # ── Save user message ─────────────────────────────────────────────────────
    user_msg = Message(
        session_id=uuid.UUID(req.session_id),
        role="user",
        content=req.query,
    )
    db.add(user_msg)
    await db.flush()

    # ── Run agentic controller ────────────────────────────────────────────────
    controller = AgentController()
    try:
        result = await controller.run(
            query=req.query,
            mode=req.mode,
            img1_path=img1_path,
            img2_path=img2_path,
            session_id=req.session_id,
            anthropic_client=get_anthropic(),
            conversation_history=conversation_history,
        )
    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent pipeline error: {str(e)}")

    # ── Save assistant message ────────────────────────────────────────────────
    ai_msg = Message(
        session_id=uuid.UUID(req.session_id),
        role="assistant",
        content=result["ai_text"],
        task_type=result["task_type"],
        confidence=result["confidence"],
        latency_ms=result["latency_ms"],
        models_used=result["models_used"],
        agent_trace=result["agent_trace"],
    )
    db.add(ai_msg)
    await db.flush()

    # ── Save analysis result ──────────────────────────────────────────────────
    local = result.get("local_result") or {}
    ar = AnalysisResult(
        message_id=ai_msg.id,
        session_id=uuid.UUID(req.session_id),
        task_type=result["task_type"],
        model_name=", ".join(result["models_used"]),
        result_text=result["ai_text"],
        bounding_boxes=local.get("boxes", []),
        change_stats={
            k: v for k, v in local.items()
            if k in ("change_percent", "change_score", "changed_pixels", "total_pixels")
        },
        confidence=result["confidence"],
        result_image_path=local.get("result_image_path"),
    )
    db.add(ar)

    # ── Save agent logs ───────────────────────────────────────────────────────
    for i, step in enumerate(result["agent_trace"]):
        db.add(AgentLog(
            message_id=ai_msg.id,
            session_id=uuid.UUID(req.session_id),
            step_number=i + 1,
            step_name=step.get("name", ""),
            model_name=step.get("model"),
            status=step.get("status", "done"),
        ))

    # ── Update session stats ──────────────────────────────────────────────────
    await db.execute(
        update(Session)
        .where(Session.id == uuid.UUID(req.session_id))
        .values(
            total_queries=Session.total_queries + 1,
            total_tasks=Session.total_tasks + len(result["agent_trace"]),
            avg_confidence=result["confidence"],
            mode=req.mode,
        )
    )

    await db.commit()

    return AnalyzeResponse(
        message_id=str(ai_msg.id),
        task_type=result["task_type"],
        ai_text=result["ai_text"],
        models_used=result["models_used"],
        agent_trace=result["agent_trace"],
        confidence=round(result["confidence"], 1),
        latency_ms=result["latency_ms"],
        change_stats=ar.change_stats if ar.change_stats else None,
        result_image_url=ar.result_image_path,
        bounding_boxes=ar.bounding_boxes,
    )
