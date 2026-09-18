"""Tests for deterministic metrics and the full runner loop."""

import pytest

from ai_eval.client import OllamaClient
from ai_eval.metrics import CorrectnessJudge, ExactMatch, ValidJSON
from ai_eval.runner import generate_and_evaluate, run_metrics
from ai_eval.trace import Trace


def test_exact_match_pass():
    t = Trace(input="Q", output="Paris", reference="paris")
    score = ExactMatch().measure(t)
    assert score.value == 1.0
    assert score.passed is True


def test_exact_match_fail():
    t = Trace(input="Q", output="London", reference="Paris")
    score = ExactMatch().measure(t)
    assert score.value == 0.0
    assert score.passed is False


def test_valid_json_pass():
    t = Trace(input="Q", output='{"a": 1, "b": [2, 3]}')
    score = ValidJSON().measure(t)
    assert score.value == 1.0
    assert score.passed is True


def test_valid_json_fail():
    t = Trace(input="Q", output='{not json}')
    score = ValidJSON().measure(t)
    assert score.value == 0.0


def test_runner_attaches_scores():
    t = Trace(input="Q", output="Paris", reference="paris")
    run_metrics(t, [ExactMatch(), ValidJSON()])
    assert len(t.scores) == 2
    assert {s.name for s in t.scores} == {"exact_match", "valid_json"}


@pytest.mark.skipif(
    not OllamaClient().is_available(),
    reason="Ollama not reachable",
)
def test_full_loop_with_judge():
    trace = generate_and_evaluate(
        input_text="What is the capital of France? Answer in one word.",
        metrics=[CorrectnessJudge()],
        reference="Paris",
        example_id="qa-001",
    )
    assert trace.output
    assert len(trace.spans) == 1
    assert trace.spans[0].name == "generation"
    assert len(trace.scores) == 1

    score = trace.scores[0]
    print(f"\n🤖 output: {trace.output.strip()}")
    print(f"   correctness: {score.value} (threshold {score.threshold})")
    print(f"   judge: {score.judge_model}")
    print(f"   prompt_hash: {score.judge_prompt_hash}")
    print(f"   rubric: {score.rubric_version}")
    print(f"   evidence: {score.evidence}")

    assert score.value >= 0.7, f"judge said: {score.evidence}"