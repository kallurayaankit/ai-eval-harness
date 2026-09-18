"""Base metric protocol. Every metric is a class with a measure() method."""

from abc import ABC, abstractmethod

from ai_eval.trace import Score, Trace


class Metric(ABC):
    """Base class for all metrics.

    Subclasses implement measure() and return a Score.
    """

    name: str = "unnamed"
    threshold: float = 0.5

    @abstractmethod
    def measure(self, trace: Trace) -> Score:
        ...

    def _score(self, value: float, evidence: str = "", **kwargs) -> Score:
        value = max(0.0, min(1.0, value))
        return Score(
            name=self.name,
            value=value,
            passed=value >= self.threshold,
            threshold=self.threshold,
            evidence=evidence,
            **kwargs,
        )