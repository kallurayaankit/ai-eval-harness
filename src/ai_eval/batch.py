"""Run a metric suite against every example in a dataset.

Two levels of parallelism:
  - Across examples (MAX_WORKERS examples in flight)
  - Across metrics within an example (metric calls run concurrently)
"""

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from time import time

from ai_eval.client import OllamaClient
from ai_eval.dataset import Dataset
from ai_eval.metrics.base import Metric
from ai_eval.trace import Score, Trace


MAX_WORKERS = int(os.getenv("EVAL_WORKERS", "2"))


@dataclass
class BatchReport:
    dataset_path: str
    suite_name: str
    metrics: list[str]
    traces: list[Trace] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def duration_s(self) -> float:
        if self.end_time == 0:
            return 0.0
        return self.end_time - self.start_time

    def aggregate(self) -> dict[str, dict]:
        buckets: dict[str, list] = {}
        for t in self.traces:
            for s in t.scores:
                if s.error:
                    buckets.setdefault(f"{s.name}__errors", []).append(s)
                    continue
                buckets.setdefault(s.name, []).append(s)

        summary = {}
        for name, scores in buckets.items():
            n = len(scores)
            values = [s.value for s in scores]
            passes = sum(1 for s in scores if s.passed)
            latencies = [s.latency_ms for s in scores if s.latency_ms > 0]
            summary[name] = {
                "n": n,
                "mean": sum(values) / n if n else 0.0,
                "min": min(values) if n else 0.0,
                "max": max(values) if n else 0.0,
                "pass_rate": passes / n if n else 0.0,
                "mean_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
            }
        return summary

    def failures(self) -> list[Trace]:
        return [t for t in self.traces if any(not s.passed for s in t.scores)]


def _run_one_metric(metric: Metric, trace: Trace) -> Score:
    try:
        return metric.measure(trace)
    except Exception as e:
        return Score(
            name=metric.name,
            value=0.0,
            passed=False,
            evidence=f"metric raised: {type(e).__name__}: {e}",
            error=True,
        )


def _evaluate_one(ex, metrics, model, progress_lock, progress_state, total):
    client = OllamaClient(model=model)

    trace = Trace(
        example_id=ex.id,
        input=ex.input,
        reference=ex.reference,
        context=ex.context,
        metadata=ex.metadata,
    )

    start = time()
    output = client.generate(ex.input, timeout=300, temperature=0.0)
    end = time()

    trace.output = output
    trace.add_span("generation", "generation", start, end, model=client.model)
    trace.end_time = end

    with ThreadPoolExecutor(max_workers=len(metrics)) as pool:
        futures = [pool.submit(_run_one_metric, m, trace) for m in metrics]
        for fut in as_completed(futures):
            trace.add_score(fut.result())

    with progress_lock:
        progress_state[0] += 1
        done = progress_state[0]
        scores = {s.name: f"{s.value:.2f}" for s in trace.scores}
        print(f"  [{done}/{total}] {ex.id}  {scores}")

    return trace


def run_batch(
    dataset: Dataset,
    metrics: list[Metric],
    suite_name: str = "default",
    model: str | None = None,
    max_workers: int | None = None,
) -> BatchReport:
    workers = max_workers or MAX_WORKERS
    report = BatchReport(
        dataset_path=dataset.path,
        suite_name=suite_name,
        metrics=[m.name for m in metrics],
        start_time=time(),
    )

    progress_lock = threading.Lock()
    progress_state = [0]
    total = len(dataset)

    examples = list(dataset)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                _evaluate_one,
                ex, metrics, model, progress_lock, progress_state, total,
            )
            for ex in examples
        ]
        for fut in as_completed(futures):
            report.traces.append(fut.result())

    order = {ex.id: i for i, ex in enumerate(examples)}
    report.traces.sort(key=lambda t: order.get(t.example_id, 1_000_000))

    report.end_time = time()
    return report