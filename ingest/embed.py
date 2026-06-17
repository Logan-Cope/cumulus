"""Offline ingestion step 2: embed chunks and upsert into pgvector.

Run once (and on every curriculum update). The bot never queries Notion live;
it queries this index. Notion is the source, pgvector is the runtime index.

    uv run python -m ingest.embed        # chunk + embed the whole curriculum
"""
from __future__ import annotations

import psycopg
from langchain_openai import OpenAIEmbeddings
from pgvector.psycopg import register_vector

from app.config import DATABASE_URL, EMBEDDING_MODEL

EMBED_DIM = 1536  # text-embedding-3-small; must match VECTOR(1536) in db/schema.sql
_BATCH = 256


def _embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def embed_and_upsert(chunks: list[dict], replace: bool = True) -> int:
    """Embed each chunk and INSERT into chunks(content, embedding, module, lesson,
    source_url). Returns the count inserted.

    Idempotent by default: `replace=True` first deletes existing rows for every
    source_url in this batch, so re-running ingestion refreshes rather than
    duplicates.
    """
    if not chunks:
        return 0

    embedder = _embeddings()
    vectors = embedder.embed_documents([c["content"] for c in chunks])
    if len(vectors[0]) != EMBED_DIM:
        raise ValueError(
            f"Embedding dim {len(vectors[0])} != {EMBED_DIM}. "
            f"Model {EMBEDDING_MODEL} does not match VECTOR({EMBED_DIM}) in db/schema.sql."
        )

    sources = sorted({c["source_url"] for c in chunks})
    rows = [
        (c["content"], v, c["module"], c["lesson"], c["source_url"])
        for c, v in zip(chunks, vectors)
    ]

    with psycopg.connect(DATABASE_URL) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            if replace:
                cur.execute("DELETE FROM chunks WHERE source_url = ANY(%s);", (sources,))
            cur.executemany(
                """INSERT INTO chunks (content, embedding, module, lesson, source_url)
                   VALUES (%s, %s, %s, %s, %s);""",
                rows,
            )
        conn.commit()
    return len(rows)


def main() -> None:
    from ingest.chunk import chunk_markdown, find_curriculum_files

    files = find_curriculum_files()
    if not files:
        raise SystemExit("No curriculum markdown found under data/curriculum/.")

    all_chunks: list[dict] = []
    for path in files:
        cs = chunk_markdown(path)
        print(f"chunked {len(cs):>4} from {path}")
        all_chunks.extend(cs)

    print(f"embedding {len(all_chunks)} chunks with {EMBEDDING_MODEL} ...")
    n = embed_and_upsert(all_chunks)
    print(f"upserted {n} chunks into pgvector.")


if __name__ == "__main__":
    main()
