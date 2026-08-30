from sqlalchemy import Column, String, Integer, Float, Text, BigInteger, ForeignKey, DateTime, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from app.db.database import Base


class Session(Base):
    __tablename__ = "sessions"

    id            = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at    = Column(DateTime(timezone=True), server_default=func.now())
    updated_at    = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    mode          = Column(String(20), nullable=False, default="single")
    total_queries = Column(Integer, default=0)
    total_tasks   = Column(Integer, default=0)
    avg_confidence= Column(Float, default=0.0)
    meta          = Column(JSON, default=dict)

    messages      = relationship("Message",        back_populates="session", cascade="all, delete-orphan")
    uploads       = relationship("Upload",         back_populates="session", cascade="all, delete-orphan")
    results       = relationship("AnalysisResult", back_populates="session", cascade="all, delete-orphan")
    reports       = relationship("Report",         back_populates="session", cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "messages"

    id          = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id  = Column(UUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    created_at  = Column(DateTime(timezone=True), server_default=func.now())
    role        = Column(String(10), nullable=False)
    content     = Column(Text, nullable=False)
    task_type   = Column(String(30))
    confidence  = Column(Float)
    latency_ms  = Column(Integer)
    models_used = Column(JSON, default=list)
    agent_trace = Column(JSON, default=list)
    meta        = Column(JSON, default=dict)

    session     = relationship("Session",        back_populates="messages")
    results     = relationship("AnalysisResult", back_populates="message", cascade="all, delete-orphan")
    agent_logs  = relationship("AgentLog",       back_populates="message", cascade="all, delete-orphan")


class Upload(Base):
    __tablename__ = "uploads"

    id               = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id       = Column(UUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    created_at       = Column(DateTime(timezone=True), server_default=func.now())
    slot             = Column(String(10), nullable=False)
    original_name    = Column(String(255))
    stored_path      = Column(String(512))
    file_size_bytes  = Column(BigInteger)
    file_format      = Column(String(20))
    image_width      = Column(Integer)
    image_height     = Column(Integer)
    num_bands        = Column(Integer)
    crs              = Column(String(100))
    modality         = Column(String(20))
    meta             = Column(JSON, default=dict)

    session          = relationship("Session", back_populates="uploads")


class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id                = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    message_id        = Column(UUID(as_uuid=True), ForeignKey("messages.id", ondelete="CASCADE"), nullable=False)
    session_id        = Column(UUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    created_at        = Column(DateTime(timezone=True), server_default=func.now())
    task_type         = Column(String(30), nullable=False)
    model_name        = Column(String(100))
    result_text       = Column(Text)
    bounding_boxes    = Column(JSON, default=list)
    change_stats      = Column(JSON, default=dict)
    confidence        = Column(Float)
    result_image_path = Column(String(512))
    meta              = Column(JSON, default=dict)

    message           = relationship("Message", back_populates="results")
    session           = relationship("Session", back_populates="results")


class AgentLog(Base):
    __tablename__ = "agent_logs"

    id             = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    message_id     = Column(UUID(as_uuid=True), ForeignKey("messages.id", ondelete="CASCADE"))
    session_id     = Column(UUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"))
    created_at     = Column(DateTime(timezone=True), server_default=func.now())
    step_number    = Column(Integer)
    step_name      = Column(String(200))
    model_name     = Column(String(100))
    status         = Column(String(20), default="pending")
    duration_ms    = Column(Integer)
    input_summary  = Column(Text)
    output_summary = Column(Text)
    meta           = Column(JSON, default=dict)

    message        = relationship("Message", back_populates="agent_logs")


class Report(Base):
    __tablename__ = "reports"

    id              = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id      = Column(UUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    created_at      = Column(DateTime(timezone=True), server_default=func.now())
    report_type     = Column(String(20), default="pdf")
    file_path       = Column(String(512))
    file_size_bytes = Column(BigInteger)
    meta            = Column(JSON, default=dict)

    session         = relationship("Session", back_populates="reports")
