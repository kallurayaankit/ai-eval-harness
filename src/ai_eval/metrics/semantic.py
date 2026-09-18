"""Semantic metrics for RAG: faithfulness, relevance, hallucination."""

from dataclasses import dataclass

from ai_eval.metrics.judge_base import JudgeBase
from ai_eval.trace import Trace


FAITHFULNESS_PROMPT = """You are checking whether an AI answer is grounded in the provided context.

CONTEXT (the only source of truth):
__CONTEXT__

QUESTION:
__INPUT__

AI ANSWER:
__OUTPUT__

RUBRIC:
Score 1.0 if every claim in the answer is directly supported by the context.
Score 0.7-0.9 if the answer is mostly grounded but adds minor outside information.
Score 0.4-0.6 if the answer mixes grounded and ungrounded claims.
Score 0.1-0.3 if most of the answer is not supported by the context.
Score 0.0 if the answer contradicts the context or is entirely ungrounded.

Think step by step, then output a JSON object with keys score, confidence, reason.
score is a float from 0.0 to 1.0. reason is one short sentence.
"""


RELEVANCE_PROMPT = """You are checking whether an AI answer addresses the question asked.

QUESTION:
__INPUT__

AI ANSWER:
__OUTPUT__

RUBRIC:
Score 1.0 if the answer directly addresses the question.
Score 0.7-0.9 if the answer is relevant but includes unnecessary detail.
Score 0.4-0.6 if the answer is partially relevant or vague.
Score 0.1-0.3 if the answer is mostly off-topic.
Score 0.0 if the answer is completely unrelated or refuses without cause.

Think step by step, then output a JSON object with keys score, confidence, reason.
score is a float from 0.0 to 1.0. reason is one short sentence.
"""


HALLUCINATION_PROMPT = """You are detecting hallucinations: claims in an AI answer that are NOT supported by the provided context.

CONTEXT:
__CONTEXT__

QUESTION:
__INPUT__

AI ANSWER:
__OUTPUT__

RUBRIC (note: this scale is inverted — high = no hallucination):
Score 1.0 if the answer contains NO claims beyond the context (no hallucination).
Score 0.7-0.9 if the answer contains minor unverifiable embellishments.
Score 0.4-0.6 if the answer contains at least one clear hallucinated claim.
Score 0.1-0.3 if the answer contains several fabricated claims.
Score 0.0 if the answer is entirely fabricated.

Think step by step, then output a JSON object with keys score, confidence, reason.
score is a float from 0.0 to 1.0. reason is one short sentence.
"""


def _context_block(trace: Trace) -> str:
    if not trace.context:
        return "(no context provided)"
    return "\n\n---\n\n".join(trace.context)


@dataclass
class Faithfulness(JudgeBase):
    """Is the answer grounded in the retrieved context?"""

    name: str = "faithfulness"
    threshold: float = 0.8
    rubric_version: str = "v1"
    prompt_template: str = FAITHFULNESS_PROMPT
    score_key: str = "score"

    def __post_init__(self):
        super().__post_init__()

    def build_prompt(self, trace: Trace) -> str:
        return (
            FAITHFULNESS_PROMPT
            .replace("__CONTEXT__", _context_block(trace))
            .replace("__INPUT__", trace.input)
            .replace("__OUTPUT__", trace.output)
        )


@dataclass
class AnswerRelevance(JudgeBase):
    """Does the answer address the question asked?"""

    name: str = "relevance"
    threshold: float = 0.7
    rubric_version: str = "v1"
    prompt_template: str = RELEVANCE_PROMPT
    score_key: str = "score"

    def __post_init__(self):
        super().__post_init__()

    def build_prompt(self, trace: Trace) -> str:
        return (
            RELEVANCE_PROMPT
            .replace("__INPUT__", trace.input)
            .replace("__OUTPUT__", trace.output)
        )


@dataclass
class Hallucination(JudgeBase):
    """Detects unsupported claims (inverted scale — high = no hallucination)."""

    name: str = "hallucination"
    threshold: float = 0.8
    rubric_version: str = "v1"
    prompt_template: str = HALLUCINATION_PROMPT
    score_key: str = "score"

    def __post_init__(self):
        super().__post_init__()

    def build_prompt(self, trace: Trace) -> str:
        return (
            HALLUCINATION_PROMPT
            .replace("__CONTEXT__", _context_block(trace))
            .replace("__INPUT__", trace.input)
            .replace("__OUTPUT__", trace.output)
        )