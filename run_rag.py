"""Full RAG evaluation with semantic metrics and HTML report."""

from ai_eval.batch import run_batch
from ai_eval.dataset import load_jsonl
from ai_eval.metrics import (
    AnswerRelevance,
    CorrectnessJudge,
    Faithfulness,
    Hallucination,
    PIIDetection,
    Toxicity,
)
from ai_eval.report import render_report


ds = load_jsonl("datasets/rag/clinical_faq.jsonl")
print(f"Loaded {len(ds)} RAG examples\n")

report = run_batch(
    ds,
    [
        CorrectnessJudge(),
        Faithfulness(),
        AnswerRelevance(),
        Hallucination(),
        Toxicity(),
        PIIDetection(),
    ],
    "rag-clinical",
)

print("\n--- summary ---")
for name, stats in report.aggregate().items():
    print(f"{name}: mean={stats['mean']:.2f} pass_rate={stats['pass_rate']:.2f}")

print(f"\nTotal time: {report.duration_s:.1f}s")
print(f"Failures: {len(report.failures())}")

path = render_report(report)
print(f"\nReport: {path}")