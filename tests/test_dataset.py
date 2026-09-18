"""Tests for the dataset loader.

These tests need the eval-datasets corpus. When it's not present (CI),
they skip instead of failing.
"""

import os
import pytest

from ai_eval.dataset import load_jsonl


QA_PATH = "datasets/qa/general_qa.jsonl"
RAG_PATH = "datasets/rag/clinical_faq.jsonl"
AG_PATH = "datasets/agents/tool_calls.jsonl"


def _dataset_available(path: str) -> bool:
    """Return True if the dataset can be resolved."""
    try:
        load_jsonl(path)
        return True
    except FileNotFoundError:
        return False


requires_qa = pytest.mark.skipif(
    not _dataset_available(QA_PATH),
    reason="eval-datasets corpus not available",
)
requires_rag = pytest.mark.skipif(
    not _dataset_available(RAG_PATH),
    reason="eval-datasets corpus not available",
)
requires_agents = pytest.mark.skipif(
    not _dataset_available(AG_PATH),
    reason="eval-datasets corpus not available",
)


@requires_qa
def test_load_qa():
    ds = load_jsonl(QA_PATH)
    assert len(ds) == 10
    first = ds.examples[0]
    assert first.id == "qa-001"
    assert first.reference == "Paris"
    assert first.metadata["category"] == "geography"


@requires_rag
def test_load_rag_has_context():
    ds = load_jsonl(RAG_PATH)
    assert len(ds) == 5
    first = ds.examples[0]
    assert first.context, "RAG examples should have context"
    assert isinstance(first.context, list)


@requires_agents
def test_load_agents_has_expected_tool():
    ds = load_jsonl(AG_PATH)
    assert len(ds) == 5
    for ex in ds:
        assert ex.metadata.get("expected_tool"), f"{ex.id} missing expected_tool"


@requires_qa
def test_filter_by_metadata():
    ds = load_jsonl(QA_PATH)
    easy = ds.filter(difficulty="easy")
    assert len(easy) > 0
    assert all(e.metadata["difficulty"] == "easy" for e in easy)


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        load_jsonl("does/not/exist.jsonl")