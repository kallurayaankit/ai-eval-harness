"""Tests for the HTML report. No LLM required."""

from pathlib import Path

from ai_eval.batch import BatchReport
from ai_eval.report import render_report
from ai_eval.trace import Score, Trace


def _make_report():
    r = BatchReport(dataset_path="fake.jsonl", suite_name="test-suite", metrics=["correctness"])

    t1 = Trace(example_id="qa-001", input="What is 2+2?", output="4", reference="4")
    t1.end_time = t1.start_time + 0.5
    t1.add_score(Score(name="correctness", value=1.0, passed=True, threshold=0.7, evidence="exact match"))

    t2 = Trace(example_id="qa-002", input="Capital of France?", output="London", reference="Paris")
    t2.end_time = t2.start_time + 0.5
    t2.add_score(Score(
        name="correctness", value=0.0, passed=False, threshold=0.7,
        evidence="Answer is incorrect.",
        judge_model="mistral:latest",
        judge_prompt_hash="abc123",
        rubric_version="v1",
        latency_ms=1234.5,
    ))

    r.traces = [t1, t2]
    r.start_time = 0
    r.end_time = 1.5
    return r


def test_render_creates_file(tmp_path):
    report = _make_report()
    out = tmp_path / "report.html"
    path = render_report(report, str(out))
    assert Path(path).exists()
    html = Path(path).read_text(encoding="utf-8")
    assert "test-suite" in html
    assert "qa-001" in html
    assert "qa-002" in html


def test_render_includes_provenance(tmp_path):
    report = _make_report()
    out = tmp_path / "report.html"
    render_report(report, str(out))
    html = Path(out).read_text(encoding="utf-8")
    assert "mistral:latest" in html
    assert "abc123" in html
    assert "rubric=v1" in html


def test_render_summary_stats(tmp_path):
    report = _make_report()
    out = tmp_path / "report.html"
    render_report(report, str(out))
    html = Path(out).read_text(encoding="utf-8")
    assert "50%" in html
    assert "0.50" in html