"""Unit tests for conformal_margin.handlers.filtered.FilteredHandler.

Uses a stub `_RecordingHandler` to assert which signals reach the
wrapped handler, and `structlog.testing.capture_logs` to assert on
the rejection log event. Filters in tests are real `SeverityFilter`
and `KindFilter` instances rather than stubs, so we exercise the
genuine `Filter` Protocol composition path.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import ClassVar

import pytest
import structlog

from conformal_margin.filters import KindFilter, SeverityFilter
from conformal_margin.handlers.filtered import FilteredHandler
from conformal_margin.handlers.log import LogHandler
from conformal_margin.ontology.domain.exceptions import ValidationError
from conformal_margin.ontology.domain.ontology import (Role, Severity,
                                                       SignalKind)
from conformal_margin.ontology.ports.alert_handler import AlertHandler
from conformal_margin.ontology.ports.operator_signal import OperatorSignalLike
from conformal_margin.operator_signal import OperatorSignal

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


UTC = timezone.utc
VALID_UUID = "550e8400-e29b-41d4-a716-446655440000"
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
        "metadata": {},
    }
    base.update(overrides)
    return OperatorSignal(**base)


@dataclass
class _RecordingHandler:
    """Stub AlertHandler that records every signal passed to handle()."""

    name: str
    role: ClassVar[Role] = Role.SINK
    received: list[OperatorSignalLike] = field(default_factory=list)

    async def handle(self, signal: OperatorSignalLike) -> None:
        self.received.append(signal)


@dataclass
class _ExplodingHandler:
    """Stub AlertHandler that raises on every call. Verifies exception passthrough."""

    name: str
    role: ClassVar[Role] = Role.SINK

    async def handle(self, signal: OperatorSignalLike) -> None:
        raise RuntimeError("downstream failure")


def test_constructs_with_no_filters():
    wrapped = _RecordingHandler(name="rec")
    handler = FilteredHandler(name="empty-chain", wrapped=wrapped, filters=())
    assert handler.filters == ()
    assert handler.wrapped is wrapped


def test_role_is_sink():
    handler = FilteredHandler(
        name="x", wrapped=_RecordingHandler(name="rec"), filters=()
    )
    assert handler.role is Role.SINK


def test_satisfies_alert_handler_protocol():
    handler = FilteredHandler(
        name="x", wrapped=_RecordingHandler(name="rec"), filters=()
    )
    assert isinstance(handler, AlertHandler)


def test_is_frozen_attribute_assignment_raises():
    handler = FilteredHandler(
        name="x", wrapped=_RecordingHandler(name="rec"), filters=()
    )
    with pytest.raises(AttributeError):
        handler.name = "renamed"


def test_empty_name_raises_validation_error():
    with pytest.raises(ValidationError) as exc_info:
        FilteredHandler(name="", wrapped=_RecordingHandler(name="rec"), filters=())
    assert exc_info.value.field == "name"


def test_filters_iterable_is_frozen_to_tuple():
    wrapped = _RecordingHandler(name="rec")
    mutable_filters: list = [SeverityFilter(name="high+", min_severity=Severity.HIGH)]
    handler = FilteredHandler(name="x", wrapped=wrapped, filters=mutable_filters)
    mutable_filters.append(
        KindFilter(name="anomaly-only", kinds=frozenset({SignalKind.ANOMALY}))
    )
    assert isinstance(handler.filters, tuple)
    assert len(handler.filters) == 1


async def test_empty_filter_chain_passes_every_signal():
    wrapped = _RecordingHandler(name="rec")
    handler = FilteredHandler(name="empty", wrapped=wrapped, filters=())
    signal = _make_signal()
    await handler.handle(signal)
    assert wrapped.received == [signal]


async def test_single_accepting_filter_passes_signal():
    wrapped = _RecordingHandler(name="rec")
    handler = FilteredHandler(
        name="x",
        wrapped=wrapped,
        filters=(SeverityFilter(name="medium+", min_severity=Severity.MEDIUM),),
    )
    signal = _make_signal(severity=Severity.HIGH)
    await handler.handle(signal)
    assert wrapped.received == [signal]


async def test_single_rejecting_filter_blocks_signal():
    wrapped = _RecordingHandler(name="rec")
    handler = FilteredHandler(
        name="x",
        wrapped=wrapped,
        filters=(SeverityFilter(name="critical-only", min_severity=Severity.CRITICAL),),
    )
    signal = _make_signal(severity=Severity.HIGH)
    await handler.handle(signal)
    assert wrapped.received == []


async def test_all_filters_must_accept_for_signal_to_pass():
    wrapped = _RecordingHandler(name="rec")
    handler = FilteredHandler(
        name="x",
        wrapped=wrapped,
        filters=(
            SeverityFilter(name="medium+", min_severity=Severity.MEDIUM),
            KindFilter(name="drift-only", kinds=frozenset({SignalKind.DRIFT})),
        ),
    )
    signal = _make_signal(severity=Severity.HIGH, kind=SignalKind.ANOMALY)
    await handler.handle(signal)
    assert wrapped.received == []


async def test_short_circuits_at_first_rejecting_filter():
    wrapped = _RecordingHandler(name="rec")
    handler = FilteredHandler(
        name="x",
        wrapped=wrapped,
        filters=(
            SeverityFilter(name="filter-A", min_severity=Severity.CRITICAL),
            KindFilter(name="filter-B", kinds=frozenset({SignalKind.DRIFT})),
        ),
    )
    signal = _make_signal(severity=Severity.LOW, kind=SignalKind.ANOMALY)
    with structlog.testing.capture_logs() as captured:
        await handler.handle(signal)
    assert wrapped.received == []
    assert len(captured) == 1
    assert captured[0]["filter_name"] == "filter-A"


async def test_rejection_emits_signal_filtered_out_event():
    wrapped = _RecordingHandler(name="rec-wrapped")
    handler = FilteredHandler(
        name="my-filter-wrapper",
        wrapped=wrapped,
        filters=(SeverityFilter(name="critical-only", min_severity=Severity.CRITICAL),),
    )
    signal = _make_signal(severity=Severity.HIGH)
    with structlog.testing.capture_logs() as captured:
        await handler.handle(signal)
    assert len(captured) == 1
    record = captured[0]
    assert record["event"] == "signal_filtered_out"
    assert record["log_level"] == "info"
    assert record["handler"] == "my-filter-wrapper"
    assert record["wrapped_handler"] == "rec-wrapped"
    assert record["filter_name"] == "critical-only"
    assert record["signal_id"] == VALID_UUID
    assert record["severity"] == "high"
    assert record["kind"] == "anomaly"


async def test_acceptance_does_not_emit_filtered_out_event():
    wrapped = _RecordingHandler(name="rec")
    handler = FilteredHandler(
        name="x",
        wrapped=wrapped,
        filters=(SeverityFilter(name="any", min_severity=Severity.INFO),),
    )
    with structlog.testing.capture_logs() as captured:
        await handler.handle(_make_signal(severity=Severity.HIGH))
    assert all(rec["event"] != "signal_filtered_out" for rec in captured)


async def test_wrapped_handler_exception_propagates_unchanged():
    handler = FilteredHandler(
        name="x", wrapped=_ExplodingHandler(name="boom"), filters=()
    )
    with pytest.raises(RuntimeError, match="downstream failure"):
        await handler.handle(_make_signal())


async def test_composes_with_real_log_handler_when_accepting():
    wrapped = LogHandler(name="default-log")
    handler = FilteredHandler(
        name="filtered-log",
        wrapped=wrapped,
        filters=(SeverityFilter(name="high+", min_severity=Severity.HIGH),),
    )
    with structlog.testing.capture_logs() as captured:
        await handler.handle(_make_signal(severity=Severity.HIGH))
    assert [rec["event"] for rec in captured] == ["signal_handled"]
    assert captured[0]["handler"] == "default-log"


async def test_composes_with_real_log_handler_when_rejecting():
    wrapped = LogHandler(name="default-log")
    handler = FilteredHandler(
        name="filtered-log",
        wrapped=wrapped,
        filters=(SeverityFilter(name="critical-only", min_severity=Severity.CRITICAL),),
    )
    with structlog.testing.capture_logs() as captured:
        await handler.handle(_make_signal(severity=Severity.HIGH))
    assert [rec["event"] for rec in captured] == ["signal_filtered_out"]
