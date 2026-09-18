"""Agent metrics: tool selection correctness.

Tests whether a model picks the right tool from a set of available tools.
Uses metadata.expected_tool as ground truth.
"""

from dataclasses import dataclass

from ai_eval.metrics.base import Metric
from ai_eval.metrics.judge_base import JudgeBase
from ai_eval.text import normalize
from ai_eval.trace import Score, Trace


TOOL_SELECTION_PROMPT = """You are evaluating an AI agent's tool-use decision.

TASK GIVEN TO THE AGENT:
__INPUT__

AVAILABLE TOOLS:
get_weather(city) — returns current weather for a city
search_web(query) — returns web search results
send_email(to, body) — sends an email

EXPECTED TOOL: __EXPECTED__

AGENT'S RESPONSE:
__OUTPUT__

RUBRIC:
Score 1.0 if the agent clearly indicates it would use __EXPECTED__ to accomplish the task.
Score 0.5 if the agent mentions multiple tools including __EXPECTED__, but is ambiguous.
Score 0.0 if the agent picks a different tool, or fails to indicate any tool, or just describes what the tools do without choosing one.

The agent doesn't need to actually call the tool — just indicate which one it would use.
Naming the tool is enough. Describing its purpose without naming it counts as 0.

Think step by step, then output a JSON object with keys score, confidence, reason.
score is a float from 0.0 to 1.0. reason is one short sentence.
"""


@dataclass
class ToolNameMatch(Metric):
    """Deterministic: does the output contain the expected tool name?"""

    name: str = "tool_name_match"
    threshold: float = 1.0

    def measure(self, trace: Trace) -> Score:
        expected = trace.metadata.get("expected_tool")
        if not expected:
            return self._score(0.0, "no expected_tool in metadata")

        text = normalize(trace.output)
        hit = normalize(expected) in text
        return self._score(
            1.0 if hit else 0.0,
            f"tool name '{expected}' {'found' if hit else 'not found'} in output",
        )


@dataclass
class ToolSelection(JudgeBase):
    """Judge: did the agent choose the right tool for the task?"""

    name: str = "tool_selection"
    threshold: float = 0.7
    rubric_version: str = "v1"
    prompt_template: str = TOOL_SELECTION_PROMPT
    score_key: str = "score"

    def __post_init__(self):
        super().__post_init__()

    def build_prompt(self, trace: Trace) -> str:
        expected = trace.metadata.get("expected_tool", "(unspecified)")
        return (
            TOOL_SELECTION_PROMPT
            .replace("__INPUT__", trace.input)
            .replace("__EXPECTED__", expected)
            .replace("__OUTPUT__", trace.output)
        )