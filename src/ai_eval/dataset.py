"""JSONL dataset loader.

One JSON object per line. Required field: id, input. Optional: reference,
context, metadata.

If the given path doesn't exist and DATASETS_DIR is set, retries the path
relative to DATASETS_DIR. A leading "datasets/" or "datasets\\" is stripped
in the retry so both layouts work:
    datasets/qa/general_qa.jsonl    (local sibling)
    qa/general_qa.jsonl              (external corpus dir)
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Example:
    id: str
    input: str
    reference: str | None = None
    context: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class Dataset:
    """A loaded JSONL dataset."""

    def __init__(self, examples: list[Example], path: str = ""):
        self.examples = examples
        self.path = path

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self):
        return iter(self.examples)

    def filter(self, **criteria) -> "Dataset":
        def matches(ex):
            return all(ex.metadata.get(k) == v for k, v in criteria.items())
        return Dataset([e for e in self.examples if matches(e)], self.path)


def _resolve_path(path: Path) -> Path:
    """If path doesn't exist, retry under DATASETS_DIR (stripping a leading
    'datasets/' prefix if present)."""
    if path.exists():
        return path

    base = os.getenv("DATASETS_DIR")
    if not base:
        return path

    # Strip leading datasets/ if present
    parts = list(path.parts)
    if parts and parts[0] in ("datasets",):
        parts = parts[1:]

    candidate = Path(base).joinpath(*parts)
    if candidate.exists():
        return candidate

    return path


def load_jsonl(path: str | Path) -> Dataset:
    """Load a JSONL file into a Dataset."""
    path = _resolve_path(Path(path))

    if not path.exists():
        raise FileNotFoundError(f"dataset not found: {path}")

    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {e.msg}")

            if "id" not in data or "input" not in data:
                raise ValueError(f"{path}:{lineno}: missing required field (id, input)")

            examples.append(Example(
                id=data["id"],
                input=data["input"],
                reference=data.get("reference"),
                context=data.get("context") or [],
                metadata=data.get("metadata") or {},
            ))

    return Dataset(examples, str(path))