"""Offline ingestion step 1: turn raw curriculum markdown into chunks.

Source = Notion (exported or pulled) -> data/curriculum/*.md
Use header-aware splitting so a lesson's section stays intact, then size-cap.
"""
# from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

def chunk_markdown(path: str) -> list[dict]:
    """Return [{content, module, lesson, source_url}, ...]. TODO: implement."""
    raise NotImplementedError
