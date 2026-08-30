"""
/api/reports — Generate and download session reports (PDF + JSON).
"""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pathlib import Path
import uuid
import json
import logging

from app.db.database import get_db
from app.db.models import Session, Message, AnalysisResult, Report
from app.core.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/{session_id}/generate")
async def generate_report(session_id: str, db: AsyncSession = Depends(get_db)):
    session_row = await db.get(Session, uuid.UUID(session_id))
    if not session_row:
        raise HTTPException(status_code=404, detail="Session not found")

    # Load messages
    msgs_res = await db.execute(
        select(Message).where(Message.session_id == uuid.UUID(session_id)).order_by(Message.created_at)
    )
    messages = msgs_res.scalars().all()

    # Load analysis results
    res_result = await db.execute(
        select(AnalysisResult).where(AnalysisResult.session_id == uuid.UUID(session_id))
    )
    results = res_result.scalars().all()

    report_dir = Path(settings.REPORT_DIR) / session_id
    report_dir.mkdir(parents=True, exist_ok=True)

    # ── Generate PDF ──────────────────────────────────────────────────────────
    pdf_path = report_dir / "report.pdf"
    try:
        _generate_pdf(session_row, messages, results, str(pdf_path))
    except Exception as e:
        logger.warning(f"PDF generation error (fpdf2): {e}. Generating text report instead.")
        pdf_path = report_dir / "report.txt"
        _generate_txt(session_row, messages, results, str(pdf_path))

    # ── Generate JSON trace ───────────────────────────────────────────────────
    json_path = report_dir / "agent_trace.json"
    trace_data = {
        "session_id":  session_id,
        "mode":        session_row.mode,
        "created_at":  session_row.created_at.isoformat(),
        "total_queries": session_row.total_queries,
        "total_tasks":   session_row.total_tasks,
        "avg_confidence": session_row.avg_confidence,
        "messages": [
            {
                "role": m.role, "content": m.content[:500],
                "task_type": m.task_type, "confidence": m.confidence,
                "latency_ms": m.latency_ms, "models_used": m.models_used,
                "agent_trace": m.agent_trace,
            }
            for m in messages
        ],
        "results": [
            {
                "task_type": r.task_type, "model": r.model_name,
                "confidence": r.confidence, "change_stats": r.change_stats,
                "bounding_boxes": r.bounding_boxes,
            }
            for r in results
        ],
    }
    json_path.write_text(json.dumps(trace_data, indent=2))

    # Persist report record
    fsize = pdf_path.stat().st_size if pdf_path.exists() else 0
    db.add(Report(
        session_id=uuid.UUID(session_id),
        report_type="pdf",
        file_path=str(pdf_path),
        file_size_bytes=fsize,
    ))
    await db.commit()

    return {
        "pdf_url":   f"/reports/{session_id}/report.pdf",
        "json_url":  f"/reports/{session_id}/agent_trace.json",
        "file_size": fsize,
    }


@router.get("/{session_id}/download/pdf")
async def download_pdf(session_id: str):
    path = Path(settings.REPORT_DIR) / session_id / "report.pdf"
    if not path.exists():
        path = Path(settings.REPORT_DIR) / session_id / "report.txt"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Report not generated yet. Call /generate first.")
    return FileResponse(str(path), filename=f"satquery_report_{session_id[:8]}.pdf")


@router.get("/{session_id}/download/trace")
async def download_trace(session_id: str):
    path = Path(settings.REPORT_DIR) / session_id / "agent_trace.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Trace not generated yet.")
    return FileResponse(str(path), filename=f"satquery_trace_{session_id[:8]}.json")


# ── PDF builder ───────────────────────────────────────────────────────────────
def _generate_pdf(session, messages, results, path: str):
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Header
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(30, 60, 120)
    pdf.cell(0, 10, "SatQuery AI — Analysis Report", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, f"ISRO Smart India Hackathon 2025  |  Session: {str(session.id)[:8]}", ln=True)
    pdf.cell(0, 6, f"Generated: {session.created_at.strftime('%Y-%m-%d %H:%M UTC')}  |  Mode: {session.mode}", ln=True)
    pdf.ln(6)

    # Summary box
    pdf.set_fill_color(230, 240, 255)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 60, 120)
    pdf.cell(0, 8, "Session Summary", ln=True, fill=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(50, 50, 50)
    pdf.cell(0, 6, f"Total queries: {session.total_queries}   |   Tasks run: {session.total_tasks}   |   Avg confidence: {session.avg_confidence:.1f}%", ln=True)
    pdf.ln(4)

    # Conversation
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 60, 120)
    pdf.cell(0, 8, "Conversation Log", ln=True)
    pdf.ln(2)

    for m in messages:
        role_label = "USER" if m.role == "user" else "SATQUERY AI"
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(80, 80, 80)
        extras = f" [{m.task_type} | conf: {m.confidence:.0f}%]" if m.task_type else ""
        pdf.cell(0, 5, f"[{role_label}]{extras}", ln=True)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(30, 30, 30)
        safe_text = m.content[:1000].encode("latin-1", errors="replace").decode("latin-1")
        pdf.multi_cell(0, 5, safe_text)
        if m.agent_trace:
            pdf.set_font("Helvetica", "I", 8)
            pdf.set_text_color(120, 120, 120)
            pdf.cell(0, 4, f"  Models: {', '.join(m.models_used or [])}", ln=True)
        pdf.ln(3)

    # Analysis results
    if results:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(30, 60, 120)
        pdf.cell(0, 8, "Analysis Results", ln=True)
        for r in results:
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(50, 50, 50)
            pdf.cell(0, 5, f"Task: {r.task_type}  |  Model: {r.model_name}  |  Confidence: {r.confidence:.1f}%", ln=True)
            if r.change_stats:
                pdf.set_font("Helvetica", "", 9)
                cs = r.change_stats
                pdf.cell(0, 5, f"  Change area: {cs.get('change_percent', 0):.1f}%  |  Changed pixels: {cs.get('changed_pixels', 0):,}", ln=True)
            pdf.ln(2)

    # Footer
    pdf.set_y(-20)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 5, "SatQuery AI — ISRO SIH 2025 | Agentic Vision-Language Assistant for Remote Sensing", align="C")

    pdf.output(path)


def _generate_txt(session, messages, results, path: str):
    lines = [
        "SatQuery AI — Analysis Report",
        "=" * 60,
        f"Session ID: {session.id}",
        f"Mode: {session.mode}",
        f"Created: {session.created_at.isoformat()}",
        f"Total Queries: {session.total_queries}",
        f"Total Tasks: {session.total_tasks}",
        f"Avg Confidence: {session.avg_confidence:.1f}%",
        "", "=" * 60, "CONVERSATION LOG", "=" * 60, "",
    ]
    for m in messages:
        lines.append(f"[{m.role.upper()}]: {m.content}")
        if m.task_type:
            lines.append(f"  Task: {m.task_type} | Confidence: {m.confidence:.1f}% | Models: {m.models_used}")
        lines.append("")
    Path(path).write_text("\n".join(lines))
