"""Tests for agent metrics. Deterministic only, no LLM."""

from ai_eval.metrics import ToolNameMatch
from ai_eval.trace import Trace


def test_tool_name_match_pass():
    t = Trace(
        input="What's the weather in Tokyo?",
        output="I'll use get_weather for Tokyo.",
        metadata={"expected_tool": "get_weather"},
    )
    score = ToolNameMatch().measure(t)
    assert score.value == 1.0
    assert score.passed is True


def test_tool_name_match_fail():
    t = Trace(
        input="What's the weather in Tokyo?",
        output="I'll search the web.",
        metadata={"expected_tool": "get_weather"},
    )
    score = ToolNameMatch().measure(t)
    assert score.value == 0.0
    assert score.passed is False


def test_tool_name_match_missing_metadata():
    t = Trace(input="Q", output="A", metadata={})
    score = ToolNameMatch().measure(t)
    assert score.value == 0.0
    assert "no expected_tool" in score.evidence