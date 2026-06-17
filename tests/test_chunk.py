"""Unit tests for the curriculum chunker — focused on the two parsing traps:
overloaded `#` levels (day divider vs topic) and `#` lines inside code fences.
"""
from ingest.chunk import chunk_markdown


SAMPLE = """\
# Saturday August 17, 2024 - **(Week 1- Day 1)**

Intro to the day.

# Understanding Computer Hardware

The CPU processes information.

### Software

Software is a set of instructions.

```python
# main.py  <- this is a code comment, not a heading
import boto3
# Create an S3 client
s3 = boto3.client("s3")
```

# docker:x:994:ec2-user

unfenced shell output, not a heading.
"""


def _chunk(tmp_path, text):
    p = tmp_path / "sample.md"
    p.write_text(text)
    return chunk_markdown(str(p), source_url="test://sample")


def test_day_divider_becomes_module_not_lesson(tmp_path):
    chunks = _chunk(tmp_path, SAMPLE)
    modules = {c["module"] for c in chunks}
    assert "Saturday August 17, 2024 - (Week 1- Day 1)" in modules
    # The day header's own preamble is the module Overview...
    overview = [c for c in chunks if c["lesson"] == "Overview"]
    assert overview and "Intro to the day" in overview[0]["content"]


def test_topic_and_subsection_are_lessons_under_the_day(tmp_path):
    chunks = _chunk(tmp_path, SAMPLE)
    day = "Saturday August 17, 2024 - (Week 1- Day 1)"
    lessons = {c["lesson"]: c for c in chunks if c["module"] == day}
    assert "Understanding Computer Hardware" in lessons
    assert "Software" in lessons


def test_code_fence_comments_are_not_headings(tmp_path):
    chunks = _chunk(tmp_path, SAMPLE)
    lessons = {c["lesson"] for c in chunks}
    assert "main.py" not in lessons
    assert "Create an S3 client" not in lessons
    # The fenced code stays inside its section (Software).
    software = next(c for c in chunks if c["lesson"] == "Software")
    assert "boto3.client" in software["content"]


def test_unfenced_shell_output_is_not_a_heading(tmp_path):
    chunks = _chunk(tmp_path, SAMPLE)
    assert "docker:x:994:ec2-user" not in {c["lesson"] for c in chunks}


def test_metadata_present_on_every_chunk(tmp_path):
    chunks = _chunk(tmp_path, SAMPLE)
    assert chunks
    for c in chunks:
        assert c["content"].strip()
        assert c["module"] and c["lesson"]
        assert c["source_url"] == "test://sample"
