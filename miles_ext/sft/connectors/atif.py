"""Convert Harbor trial directories in ATIF (Agent Trajectory Interchange Format) to SFT records."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from miles_ext.sft.connectors.base import Dropped, Item, report_path, write_records

TRAJECTORY_FILE = "trajectory.json"
SCHEMA_PREFIX = "ATIF-v1."
BOUNDARIES = frozenset({"replace", "append"})


class AtifRefused(Exception):
    """The trajectory cannot become a record; ``reason`` is the report's slug."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class AtifOptions:
    """``model``: the model trained on (default: the main trajectory's ``agent.model_name``).
    ``token_ids``: copy ``completion_token_ids``; only when that model has the trained model's tokenizer.
    ``chat_template_kwargs``: the harness's render settings, set on every record."""

    model: str | None = None
    token_ids: bool = False
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _Segment:
    """One trajectory document of a record (the trajectory or a continuation), with where refs resolve."""

    doc: dict[str, Any]
    directory: Path
    chain: tuple[dict[str, Any], ...]  # this document, then the documents whose subagents it may name


def find_trials(root: Path) -> list[Path]:
    """*root* if it is a trial directory, else every trial directory under it."""
    root = Path(root)
    if (root / "agent" / TRAJECTORY_FILE).is_file():
        return [root]
    trials = set()
    for path in root.rglob(f"agent/{TRAJECTORY_FILE}"):
        trial = path.parent.parent
        if trial.parent.name == "steps":  # a stage of a multi-stage trial
            trial = trial.parent.parent
        trials.add(trial)
    return sorted(trials)


def main_trajectories(trial: Path) -> list[Path]:
    """The trajectories a trial starts from; continuations and subagents are reached by reference."""
    paths = [trial / "agent" / TRAJECTORY_FILE] if (trial / "agent" / TRAJECTORY_FILE).is_file() else []
    return paths + sorted(trial.glob(f"steps/*/agent/{TRAJECTORY_FILE}"))


def convert_trial(trial: Path, options: AtifOptions | None = None) -> Iterator[Item]:
    """Records (and drops) for one trial directory: each main trajectory, then its subagents."""
    converter = _TrialConverter(Path(trial), options or AtifOptions())
    for path in main_trajectories(converter.trial):
        yield from converter.convert_file(path, parent=None, chain=(), model=None)


class _TrialConverter:
    def __init__(self, trial: Path, options: AtifOptions):
        self.trial = trial
        self.options = options
        self.seen: set[str] = set()

    def _key(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.trial.resolve()))
        except ValueError:
            return str(path)

    def convert_file(
        self, path: Path, *, parent: str | None, chain: tuple[dict[str, Any], ...], model: str | None
    ) -> Iterator[Item]:
        key = self._key(path)
        if key in self.seen:
            return
        self.seen.add(key)
        try:
            doc = _load(path)
        except AtifRefused as exc:
            yield Dropped(key, exc.reason, exc.detail)
            return
        yield from self.convert_doc(doc, path.parent, key=key, parent=parent, chain=chain, model=model)

    def convert_doc(
        self,
        doc: dict[str, Any],
        directory: Path,
        *,
        key: str,
        parent: str | None,
        chain: tuple[dict[str, Any], ...],
        model: str | None,
    ) -> Iterator[Item]:
        if parent is None:
            model = self.options.model or _agent(doc).get("model_name")
        segments: list[_Segment] = []
        try:
            self._segments(segments, doc, directory, chain, key)
            yield self._record(segments, key=key, parent=parent, model=model)
        except AtifRefused as exc:
            yield Dropped(key, exc.reason, exc.detail)
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            yield Dropped(key, "invalid_trajectory", repr(exc))
        for segment in segments:
            for ref in _subagent_refs(segment.doc):
                yield from self._subagent(ref, segment, parent=key, model=model)

    def _segments(
        self,
        segments: list[_Segment],
        doc: dict[str, Any],
        directory: Path,
        chain: tuple[dict[str, Any], ...],
        key: str,
    ) -> None:
        """Fill *segments* with *doc* and its continuations; what was read stays there if one is refused."""
        segments.append(_Segment(_check_schema(doc), directory, (doc, *chain)))
        visited = {key}
        while ref := segments[-1].doc.get("continued_trajectory_ref"):
            path = segments[-1].directory / ref
            cont_key = self._key(path)
            if cont_key in visited:
                raise AtifRefused("invalid_trajectory", f"continuation loop at {ref}")
            visited.add(cont_key)
            self.seen.add(cont_key)
            if not path.is_file():
                raise AtifRefused("missing_continuation", ref)
            cont = _check_schema(_load(path))
            segments.append(_Segment(cont, path.parent, (cont, doc, *chain)))

    def _subagent(self, ref: Any, holder: _Segment, *, parent: str, model: str | None) -> Iterator[Item]:
        ref = ref if isinstance(ref, dict) else {}
        trajectory_id, trajectory_path = (
            value if isinstance(value, str) else None
            for value in (ref.get("trajectory_id"), ref.get("trajectory_path"))
        )
        if trajectory_id:
            for doc in holder.chain:
                subagents = doc.get("subagent_trajectories")
                for sub in subagents if isinstance(subagents, list) else []:
                    if isinstance(sub, dict) and sub.get("trajectory_id") == trajectory_id:
                        key = f"{parent}#{trajectory_id}"
                        if key in self.seen:
                            return
                        self.seen.add(key)
                        yield from self.convert_doc(
                            sub, holder.directory, key=key, parent=parent, chain=holder.chain, model=model
                        )
                        return
        if trajectory_path:
            path = holder.directory / trajectory_path
            if path.is_file():
                yield from self.convert_file(path, parent=parent, chain=holder.chain, model=model)
                return
        if trajectory_id or trajectory_path:
            yield Dropped(parent, "missing_subagent", str(trajectory_path or trajectory_id))
        else:
            yield Dropped(parent, "unresolvable_subagent", json.dumps(ref)[:200])

    def _record(self, segments: list[_Segment], *, key: str, parent: str | None, model: str | None) -> dict[str, Any]:
        root = segments[0].doc
        agent = _agent(root)
        builder = _RecordBuilder(model=model, token_ids=self.options.token_ids)
        for segment in segments:
            builder.add_steps(segment.doc["steps"], agent_model=_agent(segment.doc).get("model_name"))
        record: dict[str, Any] = {"messages": builder.messages}
        if agent.get("tool_definitions"):
            record["tools"] = agent["tool_definitions"]
        if self.options.chat_template_kwargs:
            record["chat_template_kwargs"] = dict(self.options.chat_template_kwargs)
        steps = builder.finish()
        if steps is not None:
            record["steps"] = steps
        metadata = {
            "source": "atif",
            "trial": str(self.trial),
            "trajectory": key,
            "agent": agent.get("name"),
            "agent_version": agent.get("version"),
            "model": agent.get("model_name"),
            "parent": parent,
        }
        record["metadata"] = {k: v for k, v in metadata.items() if v is not None}
        return record


class _RecordBuilder:
    """Walks ATIF steps in order, keeping the message store and the context of the next model call."""

    def __init__(self, *, model: str | None, token_ids: bool):
        self.model = model
        self.token_ids = token_ids
        self.messages: list[dict[str, Any]] = []
        self.context: list[int] = []
        self.steps: list[dict[str, Any]] = []

    def _add(self, message: dict[str, Any]) -> int:
        self.messages.append(message)
        self.context.append(len(self.messages) - 1)
        return len(self.messages) - 1

    def add_steps(self, steps: list[dict[str, Any]], *, agent_model: str | None) -> None:
        if not isinstance(steps, list):
            raise AtifRefused("invalid_trajectory", "steps is not a list")
        for step in steps:
            calls = step.get("llm_call_count")
            if isinstance(calls, int) and calls > 1:
                raise AtifRefused("aggregated_step", f"step {step.get('step_id')} has llm_call_count {calls}")
            source = step.get("source")
            if source == "user":
                self._add({"role": "user", "content": _text(step.get("message"))})
            elif source == "system":
                self._system(step)
            elif source == "agent":
                self._agent(step, agent_model)
            else:
                raise AtifRefused("invalid_trajectory", f"step {step.get('step_id')} has source {source!r}")

    def _system(self, step: dict[str, Any]) -> None:
        management = (step.get("extra") or {}).get("context_management")
        if management is None:
            text = _text(step.get("message"))
            if text:
                self._add({"role": "system", "content": text})
            return
        boundary = management.get("boundary") if isinstance(management, dict) else None
        if boundary not in BOUNDARIES:
            raise AtifRefused("context_undetermined", f"step {step.get('step_id')} has boundary {boundary!r}")
        if boundary == "replace":
            self.context = []
        for result in (step.get("observation") or {}).get("results") or []:
            text = _text(result.get("content"))
            if text:
                self._add({"role": "user", "content": text})

    def _agent(self, step: dict[str, Any], agent_model: str | None) -> None:
        message: dict[str, Any] = {"role": "assistant", "content": _text(step.get("message"))}
        if step.get("reasoning_content") is not None:
            message["reasoning_content"] = step["reasoning_content"]
        call_ids = set()
        if step.get("tool_calls"):
            message["tool_calls"] = []
            for call in step["tool_calls"]:
                call_ids.add(call["tool_call_id"])
                arguments = json.dumps(call.get("arguments") or {}, ensure_ascii=False)
                message["tool_calls"].append(
                    {
                        "id": call["tool_call_id"],
                        "type": "function",
                        "function": {"name": call["function_name"], "arguments": arguments},
                    }
                )
        ours = _same_model(step.get("model_name") or agent_model, self.model)
        token_ids = (step.get("metrics") or {}).get("completion_token_ids")
        if self.token_ids and ours and token_ids:
            message["completion_token_ids"] = token_ids
        self.messages.append(message)
        index = len(self.messages) - 1
        if step.get("is_copied_context") is not True and step.get("llm_call_count") != 0:
            if not self.context:
                raise AtifRefused("empty_context", f"step {step.get('step_id')} has no context")
            self.steps.append({"context": list(self.context), "completion": index, "train": ours})
        self.context.append(index)
        for result in (step.get("observation") or {}).get("results") or []:
            call_id = result.get("source_call_id")
            if call_id is not None:
                if call_id not in call_ids:
                    raise AtifRefused("unknown_tool_call", f"step {step.get('step_id')} result for {call_id!r}")
                self._add({"role": "tool", "tool_call_id": call_id, "content": _text(result.get("content"))})
            elif result.get("content") is not None:
                self._add({"role": "user", "content": _text(result["content"])})

    def finish(self) -> list[dict[str, Any]] | None:
        """The record's ``steps``, or ``None`` when the append-only shorthand says the same."""
        if not any(step["train"] for step in self.steps):
            raise AtifRefused("no_trained_step")
        shorthand = [
            {"context": list(range(i)), "completion": i, "train": True}
            for i, message in enumerate(self.messages)
            if message["role"] == "assistant" and i > 0
        ]
        return None if self.steps == shorthand else self.steps


def _load(path: Path) -> dict[str, Any]:
    try:
        doc = json.loads(Path(path).read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AtifRefused("unreadable", f"{path}: {exc}") from exc
    if not isinstance(doc, dict):
        raise AtifRefused("unreadable", f"{path}: not a JSON object")
    return doc


def _check_schema(doc: dict[str, Any]) -> dict[str, Any]:
    version = doc.get("schema_version")
    if not isinstance(version, str) or not version.startswith(SCHEMA_PREFIX):
        raise AtifRefused("schema_version", repr(version))
    return doc


def _agent(doc: dict[str, Any]) -> dict[str, Any]:
    agent = doc.get("agent")
    return agent if isinstance(agent, dict) else {}


def _text(content: Any) -> str:
    """ATIF text, or text ``ContentPart``s concatenated; images and audio are refused."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            kind = part.get("type") if isinstance(part, dict) else None
            if kind in ("image", "audio"):
                raise AtifRefused("multimodal", f"{kind} content")
            if kind != "text" or not isinstance(part.get("text"), str):
                raise AtifRefused("invalid_trajectory", f"content part {part!r:.100}")
            parts.append(part["text"])
        return "".join(parts)
    raise AtifRefused("invalid_trajectory", f"content of type {type(content).__name__}")


def _same_model(step_model: str | None, model: str | None) -> bool:
    """Model names compare by their last ``/`` part (``openai/gpt-4o`` is ``gpt-4o``); unknown matches."""
    if not step_model or not model:
        return True
    return step_model.rsplit("/", 1)[-1] == model.rsplit("/", 1)[-1]


def _subagent_refs(doc: dict[str, Any]) -> Iterator[Any]:
    """Every ``subagent_trajectory_ref`` entry of *doc*; malformed steps and results are skipped
    (the record conversion has already reported them)."""

    def objects(value: Any) -> list[dict[str, Any]]:
        return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []

    for step in objects(doc.get("steps")):
        observation = step.get("observation")
        for result in objects(observation.get("results") if isinstance(observation, dict) else None):
            refs = result.get("subagent_trajectory_ref")
            yield from refs if isinstance(refs, list) else []


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, help="a trial or job directory (repeatable)")
    parser.add_argument("--output", required=True, type=Path, help="the records JSONL to write")
    parser.add_argument("--model", default=None, help="the model trained on (default: each trial's agent model)")
    parser.add_argument(
        "--token-ids",
        action="store_true",
        help="copy completion_token_ids; only when the trajectories' model has the trained model's tokenizer",
    )
    parser.add_argument("--chat-template-kwargs", type=json.loads, default={}, help="JSON object set on every record")
    args = parser.parse_args(argv)
    options = AtifOptions(model=args.model, token_ids=args.token_ids, chat_template_kwargs=args.chat_template_kwargs)

    def items() -> Iterator[Item]:
        for root in args.input:
            trials = find_trials(Path(root))
            if not trials:
                yield Dropped(root, "no_trial", f"no agent/{TRAJECTORY_FILE} under it")
            for trial in trials:
                yield from convert_trial(trial, options)

    report = write_records(items(), args.output)
    print(
        f"{report.records} records to {args.output}; dropped {dict(report.dropped)}; report {report_path(args.output)}"
    )


if __name__ == "__main__":
    main()
