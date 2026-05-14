"""Unit tests for conformal_margin.handlers.log.LogHandler.

Uses `structlog.testing.capture_logs` to assert on the emitted log
record without running through the real renderer. Each test asserts
on one behavior: Protocol satisfaction, the severity-to-level
mapping (all five severities), the bound handler name, the full
eleven-field payload, and the special-case rendering of
`drift_event_type` for anomaly vs drift signals.
"""

from datetime import datetime, timezone

import pytest
import structlog

from conformal_margin.handlers.log import LogHandler
from conformal_margin.ontology.domain.exceptions import ValidationError
from conformal_margin.ontology.domain.ontology import (DriftEventType, Role,
                                                       Severity, SignalKind)
from conformal_margin.ontology.ports.alert_handler import AlertHandler
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
        "metadata": {"calibration_window_size": 1024},
    }
    base.update(overrides)
    return OperatorSignal(**base)


def test_constructs_with_valid_name():
    handler = LogHandler(name="default-log")
    assert handler.name == "default-log"


def test_role_is_sink():
    handler = LogHandler(name="default-log")
    assert handler.role is Role.SINK


def test_satisfies_alert_handler_protocol():
    handler = LogHandler(name="default-log")
    assert isinstance(handler, AlertHandler)


def test_is_frozen_attribute_assignment_raises():
    handler = LogHandler(name="default-log")
    with pytest.raises(AttributeError):
        handler.name = "renamed"


def test_empty_name_raises_validation_error():
    with pytest.raises(ValidationError) as exc_info:
        LogHandler(name="")
    assert exc_info.value.field == "name"


@pytest.mark.parametrize(
    ("severity", "threshold_exceeded", "expected_level"),
    [
        (Severity.INFO, False, "info"),
        (Severity.LOW, True, "info"),
        (Severity.MEDIUM, True, "warning"),
        (Severity.HIGH, True, "error"),
        (Severity.CRITICAL, True, "critical"),
    ],
)
async def test_severity_maps_to_log_level(severity, threshold_exceeded, expected_level):
    handler = LogHandler(name="test")
    signal = _make_signal(severity=severity, threshold_exceeded=threshold_exceeded)
    with structlog.testing.capture_logs() as captured:
        await handler.handle(signal)
    assert len(captured) == 1
    assert captured[0]["log_level"] == expected_level


async def test_event_name_is_signal_handled():
    handler = LogHandler(name="test")
    with structlog.testing.capture_logs() as captured:
        await handler.handle(_make_signal())
    assert captured[0]["event"] == "signal_handled"


async def test_handler_name_is_bound_on_record():
    handler = LogHandler(name="my-named-handler")
    with structlog.testing.capture_logs() as captured:
        await handler.handle(_make_signal())
    assert captured[0]["handler"] == "my-named-handler"


async def test_all_eleven_signal_fields_appear_on_record():
    handler = LogHandler(name="test")
    signal = _make_signal()
    with structlog.testing.capture_logs() as captured:
        await handler.handle(signal)
    record = captured[0]
    assert record["signal_id"] == VALID_UUID
    assert record["operator_name"] == "baseline"
    assert record["timestamp"] == SIGNAL_TIME.isoformat()
    assert record["kind"] == "anomaly"
    assert record["severity"] == "high"
    assert record["score"] == 4.7
    assert record["p_value"] == 0.003
    assert record["threshold_exceeded"] is True
    assert record["affected_agents"] == ("agent-7",)
    assert record["drift_event_type"] is None
    assert record["metadata"] == {"calibration_window_size": 1024}


async def test_anomaly_signal_logs_drift_event_type_as_none():
    handler = LogHandler(name="test")
    with structlog.testing.capture_logs() as captured:
        await handler.handle(_make_signal(kind=SignalKind.ANOMALY))
    assert captured[0]["drift_event_type"] is None


async def test_drift_signal_logs_drift_event_type_value_as_string():
    handler = LogHandler(name="test")
    signal = _make_signal(
        kind=SignalKind.DRIFT,
        drift_event_type=DriftEventType.MEAN_SHIFT,
    )
    with structlog.testing.capture_logs() as captured:
        await handler.handle(signal)
    assert captured[0]["drift_event_type"] == "mean_shift"
