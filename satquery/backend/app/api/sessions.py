from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from app.db.database import get_db
from app.db.models import Session, Message
import uuid

router = APIRouter()


class CreateSessionRequest(BaseModel):
    mode: str = "single"


class SessionResponse(BaseModel):
    id: str
    mode: str
    total_queries: int
    total_tasks: int
    avg_confidence: float
    created_at: str


@router.post("/", response_model=SessionResponse)
async def create_session(req: CreateSessionRequest, db: AsyncSession = Depends(get_db)):
    session = Session(mode=req.mode)
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return SessionResponse(
        id=str(session.id),
        mode=session.mode,
        total_queries=session.total_queries,
        total_tasks=session.total_tasks,
        avg_confidence=session.avg_confidence,
        created_at=session.created_at.isoformat(),
    )


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str, db: AsyncSession = Depends(get_db)):
    row = await db.get(Session, uuid.UUID(session_id))
    if not row:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionResponse(
        id=str(row.id), mode=row.mode,
        total_queries=row.total_queries, total_tasks=row.total_tasks,
        avg_confidence=row.avg_confidence, created_at=row.created_at.isoformat(),
    )


@router.get("/{session_id}/messages")
async def get_messages(session_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Message)
        .where(Message.session_id == uuid.UUID(session_id))
        .order_by(Message.created_at.asc())
    )
    rows = result.scalars().all()
    return [
        {
            "id": str(m.id), "role": m.role, "content": m.content,
            "task_type": m.task_type, "confidence": m.confidence,
            "latency_ms": m.latency_ms, "models_used": m.models_used,
            "agent_trace": m.agent_trace, "created_at": m.created_at.isoformat(),
        }
        for m in rows
    ]


@router.get("/")
async def list_sessions(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Session).order_by(desc(Session.created_at)).limit(50)
    )
    rows = result.scalars().all()
    return [
        {"id": str(r.id), "mode": r.mode, "total_queries": r.total_queries,
         "created_at": r.created_at.isoformat()}
        for r in rows
    ]
