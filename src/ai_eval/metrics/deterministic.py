"""Deterministic metrics. No LLM calls, instant results."""

import json
import re

from ai_eval.metrics.base import Metric
from ai_eval.text import normalize
from ai_eval.trace import Score, Trace


class ExactMatch(Metric):
    """Output exactly equals the reference (after normalization)."""

    name = "exact_match"
    threshold = 1.0

    def measure(self, trace: Trace) -> Score:
        if trace.reference is None:
            return self._score(0.0, "no reference provided")
        a = normalize(trace.output)
        b = normalize(trace.reference)
        match = a == b
        return self._score(
            1.0 if match else 0.0,
            f"output {'matches' if match else 'differs from'} reference",
        )


class Contains(Metric):
    """Output contains the reference string (case-insensitive)."""

    name = "contains"
    threshold = 1.0

    def measure(self, trace: Trace) -> Score:
        if trace.reference is None:
            return self._score(0.0, "no reference provided")
        a = normalize(trace.output)
        b = normalize(trace.reference)
        hit = b in a
        return self._score(
            1.0 if hit else 0.0,
            f"reference {'found' if hit else 'not found'} in output",
        )


class RegexMatch(Metric):
    """Output matches a regex pattern from the trace metadata."""

    name = "regex_match"
    threshold = 1.0

    def measure(self, trace: Trace) -> Score:
        pattern = trace.metadata.get("regex")
        if not pattern:
            return self._score(0.0, "no regex provided in metadata")
        hit = re.search(pattern, trace.output) is not None
        return self._score(
            1.0 if hit else 0.0,
            f"pattern '{pattern}' {'matched' if hit else 'not matched'}",
        )


class ValidJSON(Metric):
    """Output parses as valid JSON."""

    name = "valid_json"
    threshold = 1.0

    def measure(self, trace: Trace) -> Score:
        try:
            json.loads(trace.output)
            return self._score(1.0, "parsed successfully")
        except json.JSONDecodeError as e:
            return self._score(0.0, f"parse error: {e.msg}")