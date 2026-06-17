"""Deterministic retrieval/answer eval for Cumulus.

Loads tests/eval_set.yaml and scores retrieval quality against the lessons each
question should surface:

  * hit-rate@k  - fraction of questions where an expected lesson is in the top-k
  * MRR         - mean reciprocal rank of the first expected lesson
  * citation precision - of the chunks that enter the answer context
                  (app.answer.select_context), the fraction that are relevant
  * refusal     - off-curriculum questions must return the refusal

Run it directly to print the scores:

    uv run python -m tests.eval

The pytest wrapper (tests/test_eval.py) asserts thresholds and is marked
`@pytest.mark.eval`, so the pre-commit hook (`pytest -m "not eval"`) skips it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from app.answer import answer_question, select_context
from app.config import RETRIEVAL_K
from app.retrieval import RetrievedChunk, search_curriculum

EVAL_SET = Path(__file__).with_name("eval_set.yaml")


def _norm(s: str | None) -> str:
    """Loose lesson key: lowercase, drop markdown bold and trailing punctuation."""
    s = (s or "").replace("*", "").strip().lower()
    return re.sub(r"\s+", " ", s).strip(" :.-")


def _relevant(lesson: str | None, expected: list[str]) -> bool:
    nl = _norm(lesson)
    if not nl:
        return False
    return any(ne and (ne == nl or ne in nl or nl in ne) for ne in map(_norm, expected))


def load_cases() -> list[dict]:
    return yaml.safe_load(EVAL_SET.read_text())["cases"]


@dataclass
class CaseResult:
    question: str
    hit: bool
    reciprocal_rank: float
    context_total: int
    context_relevant: int


def eval_retrieval_case(case: dict, k: int) -> CaseResult:
    expected = case["expected_lessons"]
    chunks: list[RetrievedChunk] = search_curriculum(case["question"], k=k)

    rr, hit = 0.0, False
    for rank, c in enumerate(chunks, 1):
        if _relevant(c.lesson, expected):
            hit = True
            rr = 1.0 / rank
            break

    context = select_context(chunks) or chunks[:1]
    rel = sum(1 for c in context if _relevant(c.lesson, expected))
    return CaseResult(case["question"], hit, rr, len(context), rel)


def run_eval(k: int = RETRIEVAL_K) -> dict:
    cases = load_cases()
    retrieval = [c for c in cases if not c.get("should_refuse")]
    refusals = [c for c in cases if c.get("should_refuse")]

    results = [eval_retrieval_case(c, k) for c in retrieval]
    n = len(results) or 1
    hit_rate = sum(r.hit for r in results) / n
    mrr = sum(r.reciprocal_rank for r in results) / n
    ctx_total = sum(r.context_total for r in results)
    citation_precision = (
        sum(r.context_relevant for r in results) / ctx_total if ctx_total else 0.0
    )

    refusal_pass = sum(1 for c in refusals if not answer_question(c["question"]).grounded)

    return {
        "k": k,
        "hit_rate": hit_rate,
        "mrr": mrr,
        "citation_precision": citation_precision,
        "n_retrieval": len(results),
        "refusal_pass": refusal_pass,
        "refusal_total": len(refusals),
        "results": results,
    }


def main() -> None:
    r = run_eval()
    print("\nCumulus retrieval eval")
    print("=" * 60)
    for res in r["results"]:
        mark = "HIT " if res.hit else "MISS"
        print(
            f"  [{mark}] rr={res.reciprocal_rank:.2f} "
            f"ctx {res.context_relevant}/{res.context_total}  {res.question[:48]}"
        )
    print("=" * 60)
    print(f"  hit-rate@{r['k']}:       {r['hit_rate']:.3f}  ({r['n_retrieval']} cases)")
    print(f"  MRR:               {r['mrr']:.3f}")
    print(f"  citation precision: {r['citation_precision']:.3f}")
    print(f"  refusals:          {r['refusal_pass']}/{r['refusal_total']} returned refusal")
    print()


if __name__ == "__main__":
    main()
