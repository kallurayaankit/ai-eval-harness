"""Runs a list of metrics against a trace, attaching scores to the trace."""

from time import time

from ai_eval.client import OllamaClient
from ai_eval.metrics.base import Metric
from ai_eval.trace import Score, Trace


def run_metrics(trace: Trace, metrics: list[Metric]) -> Trace:
    """Attach each metric's score to the trace and return it."""
    for metric in metrics:
        try:
            score = metric.measure(trace)
        except Exception as e:
            score = Score(
                name=metric.name,
                value=0.0,
                passed=False,
                evidence=f"metric raised: {type(e).__name__}: {e}",
            )
        trace.add_score(score)
    return trace


def generate_and_evaluate(
    input_text: str,
    metrics: list[Metric],
    reference: str | None = None,
    context: list[str] | None = None,
    example_id: str = "",
    model: str | None = None,
    metadata: dict | None = None,
) -> Trace:
    """Full loop: send input to the model, capture the trace, run metrics."""
    client = OllamaClient(model=model)

    trace = Trace(
        example_id=example_id,
        input=input_text,
        reference=reference,
        context=context or [],
        metadata=metadata or {},
    )

    start = time()
    output = client.generate(input_text, timeout=300, temperature=0.0)
    end = time()

    trace.output = output
    trace.add_span("generation", "generation", start, end, model=client.model)
    trace.end_time = end

    run_metrics(trace, metrics)
    return trace