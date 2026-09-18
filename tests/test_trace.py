"""Smoke test for the trace data model. No LLM required."""

import pytest
from time import time

from ai_eval.trace import Trace, Score


def test_trace_construction():
    t = Trace(example_id="qa-001", input="What is 2+2?", output="4")
    t.end_time = t.start_time + 0.5
    assert t.trace_id
    assert t.duration_ms == pytest.approx(500.0, abs=1.0)


def test_span_attachment():
    t = Trace(example_id="rag-001", input="Q", output="A")
    start = time()
    t.add_span("retrieval", "retrieval", start, start + 0.1, docs=3)
    t.add_span("generation", "generation", start + 0.1, start + 2.1, model="llama3.2:3b")
    assert len(t.spans) == 2
    assert t.spans[0].duration_ms == pytest.approx(100.0, abs=1.0)
    assert t.spans[0].attributes["docs"] == 3


def test_score_attachment():
    t = Trace(example_id="qa-002", input="Q", output="A")
    t.add_score(Score(
        name="correctness",
        value=0.9,
        passed=True,
        threshold=0.8,
        judge_model="mistral:latest",
        judge_prompt_hash="abc123",
        rubric_version="v1",
        evidence="Matches reference answer.",
    ))
    assert len(t.scores) == 1
    assert t.scores[0].passed is True


def test_serialization_roundtrip():
    t = Trace(example_id="qa-003", input="Q", output="A", reference="A")
    t.end_time = t.start_time + 1.0
    t.add_score(Score(name="exact_match", value=1.0, passed=True))
    d = t.to_dict()
    assert d["example_id"] == "qa-003"
    assert d["scores"][0]["name"] == "exact_match"
    assert d["duration_ms"] == pytest.approx(1000.0, abs=1.0)