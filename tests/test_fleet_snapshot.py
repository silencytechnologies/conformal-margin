"""Unit tests for conformal_margin.fleet_snapshot.FleetSnapshot.

Verifies construction over varied fleet sizes, Protocol satisfaction,
frozen behavior, each `__post_init__` ValidationError path, the
documented independence of agent timestamps from the fleet
timestamp, JSON roundtrip, and the unknown-field rejection at JSON
decode.
"""

from datetime import datetime, timedelta, timezone

import msgspec
import pytest

from conformal_margin.agent_state import AgentState
from conformal_margin.fleet_snapshot import FleetSnapshot
from conformal_margin.ontology.domain.exceptions import ValidationError
from conformal_margin.ontology.domain.ontology import ModelTier
from conformal_margin.ontology.ports.fleet_snapshot import FleetSnapshotLike

UTC = timezone.utc
FLEET_TIME = datetime(2026, 5, 12, 9, 5, tzinfo=UTC)
AGENT_TIME = datetime(2026, 5, 12, 9, 0, tzinfo=UTC)


def _make_agent(agent_id: str = "agent-1", **overrides) -> AgentState:
    """Return a valid AgentState; tests override individual fields."""
    base = {
        "agent_id": agent_id,
        "timestamp": AGENT_TIME,
        "risk_score": 0.42,
        "model_tier": ModelTier.FOUNDATION,
        "metadata": {},
    }
    base.update(overrides)
    return AgentState(**base)


def test_constructs_with_single_agent():
    snapshot = FleetSnapshot(timestamp=FLEET_TIME, agents=(_make_agent(),))
    assert len(snapshot.agents) == 1
    assert snapshot.agents[0].agent_id == "agent-1"


def test_constructs_with_empty_fleet():
    snapshot = FleetSnapshot(timestamp=FLEET_TIME, agents=())
    assert snapshot.agents == ()


def test_constructs_with_multiple_agents():
    snapshot = FleetSnapshot(
        timestamp=FLEET_TIME,
        agents=(_make_agent("a"), _make_agent("b"), _make_agent("c")),
    )
    assert tuple(a.agent_id for a in snapshot.agents) == ("a", "b", "c")


def test_satisfies_fleet_snapshot_like_protocol():
    snapshot = FleetSnapshot(timestamp=FLEET_TIME, agents=(_make_agent(),))
    assert isinstance(snapshot, FleetSnapshotLike)


def test_is_frozen_attribute_assignment_raises():
    snapshot = FleetSnapshot(timestamp=FLEET_TIME, agents=())
    with pytest.raises(AttributeError):
        snapshot.agents = (_make_agent(),)


def test_naive_timestamp_raises_validation_error():
    naive = datetime(2026, 5, 12, 9, 5)
    with pytest.raises(ValidationError) as exc_info:
        FleetSnapshot(timestamp=naive, agents=())
    assert exc_info.value.field == "timestamp"


def test_duplicate_agent_id_raises_validation_error():
    a1 = _make_agent("dup")
    a2 = _make_agent("dup", risk_score=0.7)
    with pytest.raises(ValidationError) as exc_info:
        FleetSnapshot(timestamp=FLEET_TIME, agents=(a1, a2))
    assert exc_info.value.field == "agents"
    assert exc_info.value.value == "dup"


def test_duplicate_detection_handles_three_or_more_agents():
    a, b, c, d = (
        _make_agent("a"),
        _make_agent("b"),
        _make_agent("c"),
        _make_agent("b"),
    )
    with pytest.raises(ValidationError) as exc_info:
        FleetSnapshot(timestamp=FLEET_TIME, agents=(a, b, c, d))
    assert exc_info.value.value == "b"


def test_non_utc_timezone_is_accepted():
    warsaw_summer = timezone(timedelta(hours=2))
    ts = datetime(2026, 5, 12, 11, 5, tzinfo=warsaw_summer)
    snapshot = FleetSnapshot(timestamp=ts, agents=())
    assert snapshot.timestamp.utcoffset() == timedelta(hours=2)


def test_agent_timestamp_may_differ_from_fleet_timestamp():
    agent = _make_agent(timestamp=AGENT_TIME)
    snapshot = FleetSnapshot(timestamp=FLEET_TIME, agents=(agent,))
    assert snapshot.timestamp != agent.timestamp


def test_json_roundtrip_preserves_equality():
    snapshot = FleetSnapshot(
        timestamp=FLEET_TIME,
        agents=(_make_agent("a"), _make_agent("b")),
    )
    encoded = msgspec.json.encode(snapshot)
    decoded = msgspec.json.decode(encoded, type=FleetSnapshot)
    assert decoded == snapshot


def test_json_roundtrip_with_empty_fleet():
    snapshot = FleetSnapshot(timestamp=FLEET_TIME, agents=())
    encoded = msgspec.json.encode(snapshot)
    decoded = msgspec.json.decode(encoded, type=FleetSnapshot)
    assert decoded == snapshot


def test_json_decode_rejects_unknown_fields():
    payload = (
        b'{"timestamp":"2026-05-12T09:05:00Z","agents":[],' b'"extra_field":"bogus"}'
    )
    with pytest.raises(msgspec.ValidationError):
        msgspec.json.decode(payload, type=FleetSnapshot)
