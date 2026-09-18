"""Tests for the dataset loader. No LLM required."""

import pytest

from ai_eval.dataset import load_jsonl


QA_PATH = "datasets/qa/general_qa.jsonl"
RAG_PATH = "datasets/rag/clinical_faq.jsonl"
AG_PATH = "datasets/agents/tool_calls.jsonl"


def test_load_qa():
    ds = load_jsonl(QA_PATH)
    assert len(ds) == 10
    first = ds.examples[0]
    assert first.id == "qa-001"
    assert first.reference == "Paris"
    assert first.metadata["category"] == "geography"


def test_load_rag_has_context():
    ds = load_jsonl(RAG_PATH)
    assert len(ds) == 5
    first = ds.examples[0]
    assert first.context, "RAG examples should have context"
    assert isinstance(first.context, list)


def test_load_agents_has_expected_tool():
    ds = load_jsonl(AG_PATH)
    assert len(ds) == 5
    for ex in ds:
        assert ex.metadata.get("expected_tool"), f"{ex.id} missing expected_tool"


def test_filter_by_metadata():
    ds = load_jsonl(QA_PATH)
    easy = ds.filter(difficulty="easy")
    assert len(easy) > 0
    assert all(e.metadata["difficulty"] == "easy" for e in easy)


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        load_jsonl("does/not/exist.jsonl")