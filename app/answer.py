"""Minimal grounded answer loop (Milestone 1): retrieve -> answer -> cite.

This is the CLI spine before any LangGraph/web layer. The contract Cumulus must
hold: answer ONLY from retrieved curriculum, cite the exact lesson, and refuse
("I don't have enough information from the curriculum") rather than guess.

    uv run python -m app.answer "What is Terraform?"
    uv run python -m app.answer            # interactive prompt
"""
from __future__ import annotations

from dataclasses import dataclass

from langchain_anthropic import ChatAnthropic

from app.config import CHAT_MODEL, RELEVANCE_FLOOR, RETRIEVAL_K
from app.personas.cloud_mentor import CLOUD_MENTOR
from app.retrieval import RetrievedChunk, search_curriculum

REFUSAL = "I don't have enough information from the curriculum to answer that."


@dataclass
class Answer:
    text: str
    sources: list[RetrievedChunk]
    grounded: bool  # False when we refused for lack of curriculum support


def _format_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for i, c in enumerate(chunks, 1):
        blocks.append(
            f"[{i}] Module: {c.module} | Lesson: {c.lesson}\n{c.content}"
        )
    return "\n\n".join(blocks)


def _prompt(query: str, context: str) -> str:
    return (
        "Use ONLY the curriculum sources below to answer the question. Do not use "
        "outside knowledge. If the sources do not contain the answer, reply with "
        f'exactly: "{REFUSAL}"\n\n'
        "When you do answer, cite the lesson(s) you used inline in the form "
        "(Module - Lesson).\n\n"
        f"=== CURRICULUM SOURCES ===\n{context}\n\n"
        f"=== QUESTION ===\n{query}"
    )


def _llm() -> ChatAnthropic:
    return ChatAnthropic(model=CHAT_MODEL, temperature=0, max_tokens=1024)


def select_context(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """The chunks that actually go into the prompt and get cited.

    Currently every retrieved chunk above the refusal gate qualifies; step 4
    tightens this to a higher context bar so weak, off-topic chunks stop
    leaking into citations. The eval measures citation precision through this
    one function, so the before/after numbers move on their own.
    """
    return list(chunks)


def answer_question(query: str, k: int = RETRIEVAL_K) -> Answer:
    chunks = search_curriculum(query, k=k)

    # Relevance gate: nothing close enough means we refuse before spending a
    # model call straining to answer. This is the seed of corrective RAG.
    if not chunks or chunks[0].score < RELEVANCE_FLOOR:
        return Answer(text=REFUSAL, sources=[], grounded=False)

    context = select_context(chunks) or chunks[:1]
    messages = [
        ("system", CLOUD_MENTOR.system_prompt),
        ("human", _prompt(query, _format_context(context))),
    ]
    text = _llm().invoke(messages).content.strip()

    grounded = REFUSAL.rstrip(".") not in text
    return Answer(text=text, sources=context if grounded else [], grounded=grounded)


def _print(ans: Answer) -> None:
    print("\n" + ans.text + "\n")
    if ans.sources:
        print("Sources:")
        seen = set()
        for c in ans.sources:
            key = (c.module, c.lesson)
            if key in seen:
                continue
            seen.add(key)
            print(f"  - {c.module} - {c.lesson}  (similarity {c.score:.2f})")


def main() -> None:
    import sys

    if len(sys.argv) > 1:
        _print(answer_question(" ".join(sys.argv[1:])))
        return
    print("Cumulus (Milestone 1). Ask a cloud question, or Ctrl-C to quit.")
    try:
        while True:
            q = input("\n> ").strip()
            if q:
                _print(answer_question(q))
    except (KeyboardInterrupt, EOFError):
        print()


if __name__ == "__main__":
    main()
