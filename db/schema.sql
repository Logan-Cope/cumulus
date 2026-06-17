-- Cumulus schema. One Postgres database does everything:
-- app data, conversation memory/checkpoints, audit, AND vectors (pgvector).
-- This is the "don't add a second system until you need it" choice.

CREATE EXTENSION IF NOT EXISTS vector;

-- Unstructured curriculum chunks for RAG retrieval.
CREATE TABLE IF NOT EXISTS chunks (
    id          BIGSERIAL PRIMARY KEY,
    content     TEXT NOT NULL,
    embedding   VECTOR(1536),          -- 1536 = OpenAI text-embedding-3-small
    module      TEXT,                  -- e.g. "Networking", "IAM"
    lesson      TEXT,                  -- e.g. "VPC Peering"
    source_url  TEXT,                  -- provenance: where this came from (Notion page, etc.)
    fetched_at  TIMESTAMPTZ DEFAULT now()
);

-- HNSW index = fast approximate nearest-neighbour search (same algo the dedicated
-- vector DBs use). Cosine distance to match normalized embeddings.
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- Metadata filter index so "only the Networking module" queries are cheap.
CREATE INDEX IF NOT EXISTS chunks_module_idx ON chunks (module);

-- Full-text (keyword) index for the hybrid-search arm: exact terms like service
-- names, CLI commands, and acronyms that dense embeddings can blur. GIN over the
-- English tsvector of the content.
CREATE INDEX IF NOT EXISTS chunks_content_fts_idx
    ON chunks USING gin (to_tsvector('english', content));

-- LangGraph's PostgresSaver / PostgresStore create their own tables on setup()
-- (checkpoints for conversation memory, store for long-term per-user memory).
-- Nothing to define here for those — see app/graph.py.
