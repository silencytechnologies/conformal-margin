"""Unit tests for conformal_margin.filters.

Three filter classes, each with: happy-path accept, rejection paths,
boundary inclusivity, Protocol satisfaction, frozen behavior, role
attribute check, empty-name rejection. KindFilter and
DriftEventTypeFilter additionally cover empty-set rejection and the
defensive frozenset conversion that immunizes a filter from
mutation of a mutable input set.
"""

from datetime import datetime, timezone

import pytest

from conformal_margin.filters import (DriftEventTypeFilter, KindFilter,
                                      SeverityFilter)
from conformal_margin.ontology.domain.exceptions import ValidationError
from conformal_margin.ontology.domain.ontology import (DriftEventType, Role,
                                                       Severity, SignalKind)
from conformal_margin.ontology.ports.filter import Filter
from conformal_margin.operator_signal import OperatorSignal

UTC = timezone.utc
VALID_UUID = "550e8400-e29b-41d4-a716-446655440000"
SIGNAL_TIME = datetime(2026, 5, 12, 9, 5, tzinfo=UTC)


def _make_signal(**overrides) -> OperatorSignal:
    """Return a valid OperatorSignal; tests override individual fields."""
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


def test_severity_filter_accepts_signal_at_threshold():
    f = SeverityFilter(name="medium-and-above", min_severity=Severity.MEDIUM)
    assert f.accepts(_make_signal(severity=Severity.MEDIUM))


def test_severity_filter_accepts_signal_above_threshold():
    f = SeverityFilter(name="medium-and-above", min_severity=Severity.MEDIUM)
    assert f.accepts(_make_signal(severity=Severity.CRITICAL))


def test_severity_filter_rejects_signal_below_threshold():
    f = SeverityFilter(name="medium-and-above", min_severity=Severity.MEDIUM)
    signal = _make_signal(threshold_exceeded=False, severity=Severity.INFO)
    assert not f.accepts(signal)


def test_severity_filter_satisfies_filter_protocol():
    f = SeverityFilter(name="warn", min_severity=Severity.HIGH)
    assert isinstance(f, Filter)


def test_severity_filter_role_is_sink():
    f = SeverityFilter(name="warn", min_severity=Severity.HIGH)
    assert f.role is Role.SINK


def test_severity_filter_is_frozen_attribute_assignment_raises():
    f = SeverityFilter(name="warn", min_severity=Severity.HIGH)
    with pytest.raises(AttributeError):
        f.min_severity = Severity.LOW


def test_severity_filter_empty_name_raises_validation_error():
    with pytest.raises(ValidationError) as exc_info:
        SeverityFilter(name="", min_severity=Severity.HIGH)
    assert exc_info.value.field == "name"


def test_kind_filter_accepts_signal_with_kind_in_set():
    f = KindFilter(name="anomaly-only", kinds=frozenset({SignalKind.ANOMALY}))
    assert f.accepts(_make_signal(kind=SignalKind.ANOMALY))


def test_kind_filter_rejects_signal_with_kind_not_in_set():
    f = KindFilter(name="anomaly-only", kinds=frozenset({SignalKind.ANOMALY}))
    drift = _make_signal(
        kind=SignalKind.DRIFT, drift_event_type=DriftEventType.MEAN_SHIFT
    )
    assert not f.accepts(drift)


def test_kind_filter_accepts_multi_kind_set():
    f = KindFilter(name="both", kinds=frozenset({SignalKind.ANOMALY, SignalKind.DRIFT}))
    drift = _make_signal(
        kind=SignalKind.DRIFT, drift_event_type=DriftEventType.MEAN_SHIFT
    )
    assert f.accepts(_make_signal(kind=SignalKind.ANOMALY))
    assert f.accepts(drift)


def test_kind_filter_satisfies_filter_protocol():
    f = KindFilter(name="anomaly-only", kinds=frozenset({SignalKind.ANOMALY}))
    assert isinstance(f, Filter)


def test_kind_filter_empty_set_raises_validation_error():
    with pytest.raises(ValidationError) as exc_info:
        KindFilter(name="empty", kinds=frozenset())
    assert exc_info.value.field == "kinds"


def test_kind_filter_empty_name_raises_validation_error():
    with pytest.raises(ValidationError) as exc_info:
        KindFilter(name="", kinds=frozenset({SignalKind.ANOMALY}))
    assert exc_info.value.field == "name"


def test_kind_filter_converts_plain_set_to_frozenset_for_immutability():
    mutable: set[SignalKind] = {SignalKind.ANOMALY}
    f = KindFilter(name="anomaly-only", kinds=mutable)
    mutable.add(SignalKind.DRIFT)
    drift = _make_signal(
        kind=SignalKind.DRIFT, drift_event_type=DriftEventType.MEAN_SHIFT
    )
    assert not f.accepts(drift)
    assert isinstance(f.kinds, frozenset)


def test_drift_event_type_filter_accepts_matching_drift_signal():
    f = DriftEventTypeFilter(
        name="mean-only", types=frozenset({DriftEventType.MEAN_SHIFT})
    )
    signal = _make_signal(
        kind=SignalKind.DRIFT, drift_event_type=DriftEventType.MEAN_SHIFT
    )
    assert f.accepts(signal)


def test_drift_event_type_filter_rejects_non_matching_drift_signal():
    f = DriftEventTypeFilter(
        name="mean-only", types=frozenset({DriftEventType.MEAN_SHIFT})
    )
    signal = _make_signal(
        kind=SignalKind.DRIFT, drift_event_type=DriftEventType.VARIANCE_SHIFT
    )
    assert not f.accepts(signal)


def test_drift_event_type_filter_rejects_anomaly_signal_unconditionally():
    f = DriftEventTypeFilter(
        name="any-drift",
        types=frozenset(DriftEventType),
    )
    assert not f.accepts(_make_signal(kind=SignalKind.ANOMALY))


def test_drift_event_type_filter_satisfies_filter_protocol():
    f = DriftEventTypeFilter(
        name="mean-only", types=frozenset({DriftEventType.MEAN_SHIFT})
    )
    assert isinstance(f, Filter)


def test_drift_event_type_filter_empty_set_raises_validation_error():
    with pytest.raises(ValidationError) as exc_info:
        DriftEventTypeFilter(name="empty", types=frozenset())
    assert exc_info.value.field == "types"


def test_drift_event_type_filter_empty_name_raises_validation_error():
    with pytest.raises(ValidationError) as exc_info:
        DriftEventTypeFilter(name="", types=frozenset({DriftEventType.MEAN_SHIFT}))
    assert exc_info.value.field == "name"
