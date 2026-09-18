"""Self-contained HTML traceability report.

One file. Opens in a browser. No server, no CDN, no external assets.
Templates live in src/ai_eval/templates/.
"""

from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import Markup

from ai_eval.batch import BatchReport


TEMPLATES_DIR = Path(__file__).parent / "templates"

_env = Environment(
    loader=FileSystemLoader(str(TEMPLATES_DIR)),
    autoescape=select_autoescape(["html"]),
)


def _render_trace(trace, force_open: bool = False) -> str:
    has_fail = any(not s.passed for s in trace.scores)
    template = _env.get_template("trace.html")
    rendered = template.render(trace=trace, has_fail=has_fail)
    if force_open:
        rendered = rendered.replace("<details class=", "<details open class=", 1)
    return Markup(rendered)


def render_report(report: BatchReport, output_path: str | None = None) -> str:
    """Render a BatchReport to a self-contained HTML file.

    Returns the path to the written file.
    """
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html"]),
    )
    env.globals["render_trace"] = _render_trace

    template = env.get_template("report.html")
    summary = report.aggregate()
    failures = report.failures()
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html_out = template.render(
        suite_name=report.suite_name,
        dataset_path=report.dataset_path,
        n_traces=len(report.traces),
        n_failures=len(failures),
        duration_s=f"{report.duration_s:.1f}",
        generated_at=generated_at,
        summary=summary,
        failures=failures,
        traces=report.traces,
    )

    if output_path is None:
        Path("reports").mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = f"reports/{report.suite_name}-{timestamp}.html"

    Path(output_path).write_text(html_out, encoding="utf-8")
    return output_path