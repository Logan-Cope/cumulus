"""Offline ingestion step 2: embed chunks and upsert into pgvector.

Run once (and on every curriculum update). The bot never queries Notion live;
it queries this index. Notion is the source, pgvector is the runtime index.
"""
def embed_and_upsert(chunks: list[dict]) -> int:
    """Embed each chunk, INSERT into chunks(content, embedding, module, lesson, source_url).
    Returns count upserted. TODO: implement.
    """
    raise NotImplementedError
