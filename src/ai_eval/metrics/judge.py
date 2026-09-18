"""LLM-as-a-judge metric with full provenance."""

import hashlib
import json
import os
import re
import time

from dotenv import load_dotenv

from ai_eval.client import OllamaClient
from ai_eval.metrics.base import Metric
from ai_eval.text import truncate
from ai_eval.trace import Score, Trace

load_dotenv()

RUBRIC_VERSION = "v1"

PROMPT_TEMPLATE = """You are an expert evaluator. Grade the model's answer against the reference.

QUESTION:
__INPUT__

MODEL ANSWER:
__OUTPUT__

REFERENCE ANSWER (ground truth):
__REFERENCE__

RUBRIC:
- Score 1.0 if the model answer conveys the same meaning as the reference, even with different wording.
- Score 0.7-0.9 if the answer is mostly correct but misses a minor detail.
- Score 0.4-0.6 if the answer is partially correct or vague.
- Score 0.1-0.3 if the answer is mostly wrong but mentions something relevant.
- Score 0.0 if the answer is wrong, off-topic, or refuses without cause.

Think step by step, then output the JSON verdict on its own line.
Format: a JSON object with keys score, confidence, reason.
score is a float from 0.0 to 1.0.
confidence is a float from 0.0 to 1.0.
reason is one short sentence.
"""


def _prompt_hash() -> str:
    return hashlib.sha256(PROMPT_TEMPLATE.encode()).hexdigest()[:12]


def _extract_json(text: str) -> dict:
    matches = re.findall(r"\{[^{}]*\}", text, re.DOTALL)
    if not matches:
        raise ValueError(f"no JSON in judge response: {text[:200]}")
    return json.loads(matches[-1])


class CorrectnessJudge(Metric):
    """Grade answer correctness against a reference using an LLM judge."""

    name = "correctness"
    threshold = 0.7

    def __init__(self, model: str | None = None):
        self.model = model or os.getenv("JUDGE_MODEL", "mistral:latest")
        self.prompt_hash = _prompt_hash()

    def measure(self, trace: Trace) -> Score:
        if trace.reference is None:
            return self._score(0.0, "no reference provided")

        prompt = (
            PROMPT_TEMPLATE
            .replace("__INPUT__", trace.input)
            .replace("__OUTPUT__", trace.output)
            .replace("__REFERENCE__", trace.reference)
        )

        client = OllamaClient(model=self.model)
        start = time.time()
        raw = client.generate(prompt, timeout=300, temperature=0.0)
        latency_ms = (time.time() - start) * 1000.0

        try:
            data = _extract_json(raw)
            value = float(data.get("score", 0.0))
            evidence = data.get("reason", "")
        except Exception as e:
            value = 0.0
            evidence = f"parse error: {e} | raw: {truncate(raw, 150)}"

        return self._score(
            value,
            evidence=evidence,
            judge_model=self.model,
            judge_prompt_hash=self.prompt_hash,
            rubric_version=RUBRIC_VERSION,
            latency_ms=latency_ms,
        )