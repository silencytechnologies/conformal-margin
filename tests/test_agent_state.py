"""Unit tests for conformal_margin.agent_state.AgentState.

Each ValidationError path declared in `AgentState.__post_init__` has
a dedicated test. Frozen behavior, Protocol satisfaction, JSON
roundtrip, and unknown-field rejection at JSON decode are verified
separately. The unknown-field test locks in the security-positive
default for the canonical wire type.

The test inventory mirrors the eleven behaviors that the prior turn
verified ad-hoc via `uv run python -c`; each ad-hoc check is now a
named regression test.
"""

from datetime import datetime, timedelta, timezone

import msgspec
import pytest

from conformal_margin.agent_state import AgentState
from conformal_margin.ontology.domain.exceptions import ValidationError
from conformal_margin.ontology.domain.ontology import ModelTier
from conformal_margin.ontology.ports.agent_state import AgentStateLike

UTC = timezone.utc


def _valid_kwargs(**overrides):
    """Return kwargs for a valid AgentState; tests override individual fields."""
    base = {
        "agent_id": "agent-7",
        "timestamp": datetime(2026, 5, 12, 9, 0, tzinfo=UTC),
        "risk_score": 0.42,
        "model_tier": ModelTier.FOUNDATION,
        "metadata": {"region": "eu-west-1"},
    }
    base.update(overrides)
    return base


def test_constructs_with_valid_inputs():
    state = AgentState(**_valid_kwargs())
    assert state.agent_id == "agent-7"
    assert state.risk_score == 0.42
    assert state.model_tier is ModelTier.FOUNDATION
    assert state.metadata == {"region": "eu-west-1"}


def test_satisfies_agent_state_like_protocol():
    state = AgentState(**_valid_kwargs())
    assert isinstance(state, AgentStateLike)


def test_is_frozen_attribute_assignment_raises():
    state = AgentState(**_valid_kwargs())
    with pytest.raises(AttributeError):
        state.risk_score = 99.9


def test_empty_agent_id_raises_validation_error():
    with pytest.raises(ValidationError) as exc_info:
        AgentState(**_valid_kwargs(agent_id=""))
    assert exc_info.value.field == "agent_id"


def test_naive_timestamp_raises_validation_error():
    naive = datetime(2026, 5, 12, 9, 0)
    with pytest.raises(ValidationError) as exc_info:
        AgentState(**_valid_kwargs(timestamp=naive))
    assert exc_info.value.field == "timestamp"


def test_nan_risk_score_raises_validation_error():
    with pytest.raises(ValidationError) as exc_info:
        AgentState(**_valid_kwargs(risk_score=float("nan")))
    assert exc_info.value.field == "risk_score"


def test_positive_infinity_risk_score_raises_validation_error():
    with pytest.raises(ValidationError) as exc_info:
        AgentState(**_valid_kwargs(risk_score=float("inf")))
    assert exc_info.value.field == "risk_score"


def test_negative_infinity_risk_score_raises_validation_error():
    with pytest.raises(ValidationError) as exc_info:
        AgentState(**_valid_kwargs(risk_score=float("-inf")))
    assert exc_info.value.field == "risk_score"


def test_non_utc_timezone_is_accepted():
    warsaw_summer = timezone(timedelta(hours=2))
    ts = datetime(2026, 5, 12, 9, 0, tzinfo=warsaw_summer)
    state = AgentState(**_valid_kwargs(timestamp=ts))
    assert state.timestamp.utcoffset() == timedelta(hours=2)


def test_json_roundtrip_preserves_equality():
    state = AgentState(**_valid_kwargs())
    encoded = msgspec.json.encode(state)
    decoded = msgspec.json.decode(encoded, type=AgentState)
    assert decoded == state


def test_json_decode_rejects_unknown_fields():
    payload = (
        b'{"agent_id":"a","timestamp":"2026-05-12T00:00:00Z",'
        b'"risk_score":0.1,"model_tier":"foundation",'
        b'"metadata":{},"injected_attr":"bogus"}'
    )
    with pytest.raises(msgspec.ValidationError):
        msgspec.json.decode(payload, type=AgentState)
