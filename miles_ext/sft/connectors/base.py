"""What every connector shares: the items it yields, and writing them as validated records plus a report."""

from __future__ import annotations

import collections
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from miles_ext.sft.record import RecordError, parse_record

EXAMPLES_PER_REASON = 5


@dataclass(frozen=True)
class Dropped:
    """A source unit, or part of one, that gave no record, and why (``reason`` is a short slug)."""

    source: str
    reason: str
    detail: str = ""


Item = dict[str, Any] | Dropped


@dataclass
class Report:
    records: int = 0
    dropped: collections.Counter = field(default_factory=collections.Counter)
    examples: dict[str, list[dict[str, str]]] = field(default_factory=dict)

    def drop(self, item: Dropped) -> None:
        self.dropped[item.reason] += 1
        examples = self.examples.setdefault(item.reason, [])
        if len(examples) < EXAMPLES_PER_REASON:
            examples.append({"source": item.source, "detail": item.detail})

    def to_json(self) -> dict[str, Any]:
        return {"records": self.records, "dropped": dict(self.dropped), "examples": self.examples}


def report_path(output: Path) -> Path:
    return output.with_name(output.name.removesuffix(".jsonl") + ".report.json")


def write_records(items: Iterable[Item], output: Path) -> Report:
    """Write each record that ``parse_record`` accepts to *output* (JSONL), and the report beside it.

    A record that fails validation is a connector bug: it is reported as ``invalid_record``, not written.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    report = Report()
    with open(output, "w") as f:
        for item in items:
            if isinstance(item, Dropped):
                report.drop(item)
                continue
            try:
                parse_record(item)
            except RecordError as exc:
                source = str((item.get("metadata") or {}).get("trajectory", "?"))
                report.drop(Dropped(source, "invalid_record", str(exc)))
                continue
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            report.records += 1
    report_path(output).write_text(json.dumps(report.to_json(), indent=2) + "\n")
    return report
