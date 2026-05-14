"""Unit tests for conformal_margin.operator_signal.OperatorSignal.

Each of the seven `__post_init__` invariants has dedicated tests
covering both the rejection of invalid input and the acceptance of
boundary-legal input (p_value=0, p_value=1, threshold_exceeded=True
with non-INFO severity). Protocol satisfaction, frozen behavior,
JSON roundtrip for both signal kinds, and unknown-field rejection
at JSON decode complete the inventory.
"""

from datetime import datetime, timezone

import msgspec
import pytest

from conformal_margin.ontology.domain.exceptions import ValidationError
from conformal_margin.ontology.domain.ontology import (DriftEventType,
                                                       Severity, SignalKind)
from conformal_margin.ontology.ports.operator_signal import OperatorSignalLike
from conformal_margin.operator_signal import OperatorSignal

UTC = timezone.utc
SIGNAL_TIME = datetime(2026, 5, 12, 9, 5, tzinfo=UTC)
VALID_UUID = "550e8400-e29b-41d4-a716-446655440000"


def _valid_kwargs(**overrides):
    """Return kwargs for a valid anomaly OperatorSignal."""
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
    return base


def test_constructs_valid_anomaly_signal():
    signal = OperatorSignal(**_valid_kwargs())
    assert signal.kind is SignalKind.ANOMALY
    assert signal.drift_event_type is None


def test_constructs_valid_drift_signal_with_drift_event_type():
    signal = OperatorSignal(
        **_valid_kwargs(
            kind=SignalKind.DRIFT,
            drift_event_type=DriftEventType.MEAN_SHIFT,
        )
    )
    assert signal.kind is SignalKind.DRIFT
    assert signal.drift_event_type is DriftEventType.MEAN_SHIFT


def test_constructs_fleet_wide_signal_with_empty_affected_agents():
    signal = OperatorSignal(**_valid_kwargs(affected_agents=()))
    assert signal.affected_agents == ()


def test_satisfies_operator_signal_like_protocol():
    signal = OperatorSignal(**_valid_kwargs())
    assert isinstance(signal, OperatorSignalLike)


def test_is_frozen_attribute_assignment_raises():
    signal = OperatorSignal(**_valid_kwargs())
    with pytest.raises(AttributeError):
        signal.severity = Severity.LOW


def test_signal_id_invalid_uuid_raises_validation_error():
    with pytest.raises(ValidationError) as exc_info:
        OperatorSignal(**_valid_kwargs(signal_id="not-a-uuid"))
    assert exc_info.value.field == "signal_id"


def test_signal_id_uppercase_raises_validation_error():
    upper = VALID_UUID.upper()
    with pytest.raises(ValidationError) as exc_info:
        OperatorSignal(**_valid_kwargs(signal_id=upper))
    assert exc_info.value.field == "signal_id"


def test_signal_id_braces_raise_validation_error():
    braced = "{" + VALID_UUID + "}"
    with pytest.raises(ValidationError) as exc_info:
        OperatorSignal(**_valid_kwargs(signal_id=braced))
    assert exc_info.value.field == "signal_id"


def test_signal_id_urn_prefix_raises_validation_error():
    with pytest.raises(ValidationError) as exc_info:
        OperatorSignal(**_valid_kwargs(signal_id="urn:uuid:" + VALID_UUID))
    assert exc_info.value.field == "signal_id"


def test_signal_id_hex_without_hyphens_raises_validation_error():
    no_hyphens = VALID_UUID.replace("-", "")
    with pytest.raises(ValidationError) as exc_info:
        OperatorSignal(**_valid_kwargs(signal_id=no_hyphens))
    assert exc_info.value.field == "signal_id"


def test_empty_operator_name_raises_validation_error():
    with pytest.raises(ValidationError) as exc_info:
        OperatorSignal(**_valid_kwargs(operator_name=""))
    assert exc_info.value.field == "operator_name"


def test_naive_timestamp_raises_validation_error():
    naive = datetime(2026, 5, 12, 9, 5)
    with pytest.raises(ValidationError) as exc_info:
        OperatorSignal(**_valid_kwargs(timestamp=naive))
    assert exc_info.value.field == "timestamp"


def test_nan_score_raises_validation_error():
    with pytest.raises(ValidationError) as exc_info:
        OperatorSignal(**_valid_kwargs(score=float("nan")))
    assert exc_info.value.field == "score"


def test_positive_infinity_score_raises_validation_error():
    with pytest.raises(ValidationError) as exc_info:
        OperatorSignal(**_valid_kwargs(score=float("inf")))
    assert exc_info.value.field == "score"


def test_negative_infinity_score_raises_validation_error():
    with pytest.raises(ValidationError) as exc_info:
        OperatorSignal(**_valid_kwargs(score=float("-inf")))
    assert exc_info.value.field == "score"


def test_p_value_below_zero_raises_validation_error():
    with pytest.raises(ValidationError) as exc_info:
        OperatorSignal(**_valid_kwargs(p_value=-0.01))
    assert exc_info.value.field == "p_value"


def test_p_value_above_one_raises_validation_error():
    with pytest.raises(ValidationError) as exc_info:
        OperatorSignal(**_valid_kwargs(p_value=1.01))
    assert exc_info.value.field == "p_value"


def test_p_value_zero_is_accepted_as_lower_boundary():
    signal = OperatorSignal(**_valid_kwargs(p_value=0.0))
    assert signal.p_value == 0.0


def test_p_value_one_is_accepted_as_upper_boundary():
    signal = OperatorSignal(
        **_valid_kwargs(
            p_value=1.0,
            threshold_exceeded=False,
            severity=Severity.INFO,
        )
    )
    assert signal.p_value == 1.0


def test_drift_kind_without_drift_event_type_raises_validation_error():
    with pytest.raises(ValidationError) as exc_info:
        OperatorSignal(**_valid_kwargs(kind=SignalKind.DRIFT, drift_event_type=None))
    assert exc_info.value.field == "drift_event_type"


def test_anomaly_kind_with_drift_event_type_raises_validation_error():
    with pytest.raises(ValidationError) as exc_info:
        OperatorSignal(
            **_valid_kwargs(
                kind=SignalKind.ANOMALY,
                drift_event_type=DriftEventType.MEAN_SHIFT,
            )
        )
    assert exc_info.value.field == "drift_event_type"


def test_not_exceeded_with_non_info_severity_raises_validation_error():
    with pytest.raises(ValidationError) as exc_info:
        OperatorSignal(
            **_valid_kwargs(threshold_exceeded=False, severity=Severity.HIGH)
        )
    assert exc_info.value.field == "severity"


def test_not_exceeded_with_info_severity_is_accepted():
    signal = OperatorSignal(
        **_valid_kwargs(threshold_exceeded=False, severity=Severity.INFO)
    )
    assert signal.severity is Severity.INFO


def test_exceeded_with_info_severity_is_accepted():
    signal = OperatorSignal(
        **_valid_kwargs(threshold_exceeded=True, severity=Severity.INFO)
    )
    assert signal.severity is Severity.INFO


def test_json_roundtrip_anomaly_signal():
    signal = OperatorSignal(**_valid_kwargs())
    encoded = msgspec.json.encode(signal)
    decoded = msgspec.json.decode(encoded, type=OperatorSignal)
    assert decoded == signal


def test_json_roundtrip_drift_signal():
    signal = OperatorSignal(
        **_valid_kwargs(
            kind=SignalKind.DRIFT,
            drift_event_type=DriftEventType.VARIANCE_SHIFT,
        )
    )
    encoded = msgspec.json.encode(signal)
    decoded = msgspec.json.decode(encoded, type=OperatorSignal)
    assert decoded == signal


def test_json_decode_rejects_unknown_fields():
    payload = (
        b'{"signal_id":"' + VALID_UUID.encode() + b'",'
        b'"operator_name":"baseline","timestamp":"2026-05-12T09:05:00Z",'
        b'"kind":"anomaly","severity":"high","score":4.7,'
        b'"p_value":0.003,"threshold_exceeded":true,'
        b'"affected_agents":["agent-7"],"drift_event_type":null,'
        b'"metadata":{},"injected_attr":"bogus"}'
    )
    with pytest.raises(msgspec.ValidationError):
        msgspec.json.decode(payload, type=OperatorSignal)
