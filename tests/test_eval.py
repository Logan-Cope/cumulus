"""Quality eval as pytest assertions. Marked `eval` because it makes real API +
DB calls — excluded from the pre-commit hook (`pytest -m "not eval"`). Run with
`uv run pytest -m eval` or `uv run python -m tests.eval`.
"""
import pytest

from tests.eval import load_cases, run_eval

pytestmark = pytest.mark.eval

# Quality floors. Raise these as retrieval improves so regressions trip the test.
MIN_HIT_RATE = 0.80
MIN_MRR = 0.60


@pytest.fixture(scope="module")
def report():
    return run_eval()


def test_hit_rate_at_k(report):
    assert report["hit_rate"] >= MIN_HIT_RATE, (
        f"hit-rate@{report['k']}={report['hit_rate']:.3f} below {MIN_HIT_RATE}"
    )


def test_mrr(report):
    assert report["mrr"] >= MIN_MRR, f"MRR={report['mrr']:.3f} below {MIN_MRR}"


def test_off_curriculum_refuses(report):
    assert report["refusal_pass"] == report["refusal_total"], (
        f"{report['refusal_total'] - report['refusal_pass']} off-curriculum "
        "question(s) did not refuse"
    )


@pytest.mark.parametrize("case", [c for c in load_cases() if c.get("should_refuse")])
def test_each_off_curriculum_case_refuses(case):
    from app.answer import answer_question

    assert answer_question(case["question"]).grounded is False
