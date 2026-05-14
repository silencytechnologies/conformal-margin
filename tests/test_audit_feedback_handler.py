"""Unit tests for conformal_margin.handlers.audit_feedback.AuditFeedbackHandler.

Uses pytest's `tmp_path` fixture for isolated filesystem state per
test and a `_StubLabel` dataclass to satisfy the `FeedbackLabel`
Protocol. Tests verify: construction shape, Protocol satisfaction,
parent-directory validation, signal/label id matching, JSON Lines
output format, append semantics across multiple records, OS error
mapping to `StoreUnavailableError`, and metadata
JSON-serializability rejection.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from conformal_margin.handlers.audit_feedback import AuditFeedbackHandler
from conformal_margin.ontology.domain.exceptions import (StoreUnavailableError,
                                                         ValidationError)
from conformal_margin.ontology.domain.ontology import (DriftEventType, Role,
                                                       Severity, SignalKind)
from conformal_margin.ontology.ports.feedback import FeedbackPort
from conformal_margin.operator_signal import OperatorSignal

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


UTC = timezone.utc
VALID_UUID = "550e8400-e29b-41d4-a716-446655440000"
OTHER_UUID = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
SIGNAL_TIME = datetime(2026, 5, 12, 9, 5, tzinfo=UTC)


def _make_signal(**overrides) -> OperatorSignal:
    base = {
        "signal_id": VALID_UUID,
        "operator_name": "baseline",
        "timestamp": SIGNAL_TIME,
        "kind": SignalKind.ANOMALY,
        "severity": Severity.HIGH,
        "score": 4.7,
        "p_value": 0.003,
        "threshold_exceeded": True,
        "affected_agents": ("agent-7",),
        "drift_event_type": None,
        "metadata": {"region": "eu-west-1"},
    }
    base.update(overrides)
    return OperatorSignal(**base)


@dataclass(frozen=True)
class _StubLabel:
    """Test double satisfying the FeedbackLabel Protocol."""

    signal_id: str
    is_true_positive: bool
    metadata: dict[str, Any] = field(default_factory=dict)


def test_constructs_with_valid_path(tmp_path: Path):
    handler = AuditFeedbackHandler(name="audit", path=tmp_path / "fb.jsonl")
    assert handler.name == "audit"
    assert handler.path == tmp_path / "fb.jsonl"


def test_role_is_sink(tmp_path: Path):
    handler = AuditFeedbackHandler(name="audit", path=tmp_path / "fb.jsonl")
    assert handler.role is Role.SINK


def test_satisfies_feedback_port_protocol(tmp_path: Path):
    handler = AuditFeedbackHandler(name="audit", path=tmp_path / "fb.jsonl")
    assert isinstance(handler, FeedbackPort)


def test_is_frozen_attribute_assignment_raises(tmp_path: Path):
    handler = AuditFeedbackHandler(name="audit", path=tmp_path / "fb.jsonl")
    with pytest.raises(AttributeError):
        handler.name = "renamed"


def test_empty_name_raises_validation_error(tmp_path: Path):
    with pytest.raises(ValidationError) as exc_info:
        AuditFeedbackHandler(name="", path=tmp_path / "fb.jsonl")
    assert exc_info.value.field == "name"


def test_non_path_raises_validation_error(tmp_path: Path):
    with pytest.raises(ValidationError) as exc_info:
        AuditFeedbackHandler(name="audit", path=str(tmp_path / "fb.jsonl"))
    assert exc_info.value.field == "path"


def test_missing_parent_directory_raises_validation_error(tmp_path: Path):
    bogus = tmp_path / "nonexistent-subdir" / "fb.jsonl"
    with pytest.raises(ValidationError) as exc_info:
        AuditFeedbackHandler(name="audit", path=bogus)
    assert exc_info.value.field == "path"
    assert "parent directory does not exist" in exc_info.value.reason


async def test_record_writes_one_json_line_per_call(tmp_path: Path):
    log = tmp_path / "fb.jsonl"
    handler = AuditFeedbackHandler(name="audit", path=log)
    signal = _make_signal()
    label = _StubLabel(signal_id=signal.signal_id, is_true_positive=False)
    await handler.record(signal, label)

    content = log.read_text(encoding="utf-8")
    lines = content.splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert set(record.keys()) == {"recorded_at", "signal", "label"}


async def test_record_serializes_all_eleven_signal_fields(tmp_path: Path):
    log = tmp_path / "fb.jsonl"
    handler = AuditFeedbackHandler(name="audit", path=log)
    signal = _make_signal(
        kind=SignalKind.DRIFT,
        drift_event_type=DriftEventType.MEAN_SHIFT,
    )
    label = _StubLabel(signal_id=signal.signal_id, is_true_positive=True)
    await handler.record(signal, label)

    record = json.loads(log.read_text(encoding="utf-8"))
    sig = record["signal"]
    assert sig["signal_id"] == VALID_UUID
    assert sig["operator_name"] == "baseline"
    assert sig["timestamp"] == SIGNAL_TIME.isoformat()
    assert sig["kind"] == "drift"
    assert sig["severity"] == "high"
    assert sig["score"] == 4.7
    assert sig["p_value"] == 0.003
    assert sig["threshold_exceeded"] is True
    assert sig["affected_agents"] == ["agent-7"]
    assert sig["drift_event_type"] == "mean_shift"
    assert sig["metadata"] == {"region": "eu-west-1"}


async def test_record_serializes_label_fields(tmp_path: Path):
    log = tmp_path / "fb.jsonl"
    handler = AuditFeedbackHandler(name="audit", path=log)
    signal = _make_signal()
    label = _StubLabel(
        signal_id=signal.signal_id,
        is_true_positive=False,
        metadata={"reviewer": "ops-team-1", "ticket": "INC-42"},
    )
    await handler.record(signal, label)

    record = json.loads(log.read_text(encoding="utf-8"))
    lbl = record["label"]
    assert lbl["signal_id"] == VALID_UUID
    assert lbl["is_true_positive"] is False
    assert lbl["metadata"] == {"reviewer": "ops-team-1", "ticket": "INC-42"}


async def test_record_recorded_at_is_recent_utc_iso(tmp_path: Path):
    log = tmp_path / "fb.jsonl"
    handler = AuditFeedbackHandler(name="audit", path=log)
    signal = _make_signal()
    label = _StubLabel(signal_id=signal.signal_id, is_true_positive=True)
    before = datetime.now(UTC)
    await handler.record(signal, label)
    after = datetime.now(UTC)

    record = json.loads(log.read_text(encoding="utf-8"))
    recorded_at = datetime.fromisoformat(record["recorded_at"])
    assert before <= recorded_at <= after


async def test_record_creates_file_on_first_write(tmp_path: Path):
    log = tmp_path / "first-time.jsonl"
    assert not log.exists()
    handler = AuditFeedbackHandler(name="audit", path=log)
    signal = _make_signal()
    label = _StubLabel(signal_id=signal.signal_id, is_true_positive=True)
    await handler.record(signal, label)
    assert log.exists()


async def test_record_appends_multiple_calls_as_separate_lines(tmp_path: Path):
    log = tmp_path / "fb.jsonl"
    handler = AuditFeedbackHandler(name="audit", path=log)
    signal_a = _make_signal()
    signal_b = _make_signal(signal_id=OTHER_UUID, severity=Severity.LOW)
    await handler.record(
        signal_a, _StubLabel(signal_id=signal_a.signal_id, is_true_positive=True)
    )
    await handler.record(
        signal_b, _StubLabel(signal_id=signal_b.signal_id, is_true_positive=False)
    )
    await handler.record(
        signal_a, _StubLabel(signal_id=signal_a.signal_id, is_true_positive=False)
    )

    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    parsed = [json.loads(line) for line in lines]
    assert parsed[0]["signal"]["signal_id"] == VALID_UUID
    assert parsed[1]["signal"]["signal_id"] == OTHER_UUID
    assert parsed[2]["label"]["is_true_positive"] is False


async def test_record_with_mismatched_signal_ids_raises_validation_error(
    tmp_path: Path,
):
    log = tmp_path / "fb.jsonl"
    handler = AuditFeedbackHandler(name="audit", path=log)
    signal = _make_signal()
    mismatched = _StubLabel(signal_id=OTHER_UUID, is_true_positive=True)
    with pytest.raises(ValidationError) as exc_info:
        await handler.record(signal, mismatched)
    assert exc_info.value.field == "label.signal_id"
    assert not log.exists()


async def test_record_with_non_json_serializable_metadata_raises(tmp_path: Path):
    log = tmp_path / "fb.jsonl"
    handler = AuditFeedbackHandler(name="audit", path=log)
    signal = _make_signal()
    label = _StubLabel(
        signal_id=signal.signal_id,
        is_true_positive=True,
        metadata={"callback": lambda x: x},
    )
    with pytest.raises(ValidationError) as exc_info:
        await handler.record(signal, label)
    assert exc_info.value.field == "metadata"


async def test_record_unwritable_path_raises_store_unavailable_error(
    tmp_path: Path,
):
    parent = tmp_path / "vanishing-dir"
    parent.mkdir()
    log = parent / "fb.jsonl"
    handler = AuditFeedbackHandler(name="audit", path=log)
    parent.rmdir()

    signal = _make_signal()
    label = _StubLabel(signal_id=signal.signal_id, is_true_positive=True)
    with pytest.raises(StoreUnavailableError) as exc_info:
        await handler.record(signal, label)
    assert exc_info.value.operation == "record"
    assert exc_info.value.details["path"] == str(log)
