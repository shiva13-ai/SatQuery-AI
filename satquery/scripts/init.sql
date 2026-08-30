-- SatQuery AI — PostgreSQL Schema
-- ─────────────────────────────────────────────────────────────────────────────

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Sessions table
CREATE TABLE IF NOT EXISTS sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    mode VARCHAR(20) NOT NULL DEFAULT 'single',   -- single | bitemporal | crossmodal
    total_queries INTEGER DEFAULT 0,
    total_tasks INTEGER DEFAULT 0,
    avg_confidence FLOAT DEFAULT 0.0,
    meta JSONB DEFAULT '{}'
);

-- Messages table (conversation history)
CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    role VARCHAR(10) NOT NULL,              -- user | assistant
    content TEXT NOT NULL,
    task_type VARCHAR(30),                  -- vqa | captioning | grounding | change | fusion
    confidence FLOAT,
    latency_ms INTEGER,
    models_used JSONB DEFAULT '[]',
    agent_trace JSONB DEFAULT '[]',
    meta JSONB DEFAULT '{}'
);

-- Uploads table (image file references)
CREATE TABLE IF NOT EXISTS uploads (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    slot VARCHAR(10) NOT NULL,              -- img1 | img2
    original_name VARCHAR(255),
    stored_path VARCHAR(512),
    file_size_bytes BIGINT,
    file_format VARCHAR(20),               -- tiff | geotiff | png | jpeg
    image_width INTEGER,
    image_height INTEGER,
    num_bands INTEGER,
    crs VARCHAR(100),                       -- coordinate reference system
    modality VARCHAR(20),                  -- optical | sar | multispectral
    meta JSONB DEFAULT '{}'
);

-- Analysis results table
CREATE TABLE IF NOT EXISTS analysis_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    message_id UUID NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    task_type VARCHAR(30) NOT NULL,
    model_name VARCHAR(100),
    result_text TEXT,
    bounding_boxes JSONB DEFAULT '[]',
    change_stats JSONB DEFAULT '{}',
    confidence FLOAT,
    result_image_path VARCHAR(512),
    meta JSONB DEFAULT '{}'
);

-- Agent execution log
CREATE TABLE IF NOT EXISTS agent_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    message_id UUID REFERENCES messages(id) ON DELETE CASCADE,
    session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    step_number INTEGER,
    step_name VARCHAR(200),
    model_name VARCHAR(100),
    status VARCHAR(20) DEFAULT 'pending',   -- pending | running | done | error
    duration_ms INTEGER,
    input_summary TEXT,
    output_summary TEXT,
    meta JSONB DEFAULT '{}'
);

-- Reports table
CREATE TABLE IF NOT EXISTS reports (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    report_type VARCHAR(20) DEFAULT 'pdf', -- pdf | json | txt
    file_path VARCHAR(512),
    file_size_bytes BIGINT,
    meta JSONB DEFAULT '{}'
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_created ON messages(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_uploads_session ON uploads(session_id);
CREATE INDEX IF NOT EXISTS idx_results_session ON analysis_results(session_id);
CREATE INDEX IF NOT EXISTS idx_agent_logs_message ON agent_logs(message_id);
CREATE INDEX IF NOT EXISTS idx_sessions_created ON sessions(created_at DESC);

-- Auto-update updated_at on sessions
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER sessions_updated_at
    BEFORE UPDATE ON sessions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();
