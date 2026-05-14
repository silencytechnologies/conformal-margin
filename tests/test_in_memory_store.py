"""Unit tests for conformal_margin.stores.in_memory.InMemoryStore.

Async tests use the anyio pytest plugin (registered automatically
when `anyio>=4.0` is installed). The module-level `pytestmark`
applies `@pytest.mark.anyio` to every test; the `anyio_backend`
fixture pins the backend to asyncio (we do not exercise trio in
v0.1).

Test inventory mirrors the `StateStore` Protocol surface plus the
v0.1 design decisions:

- Construction shape (Protocol satisfaction, role, SUPPORTED_METRICS)
- save: persistence, idempotency on equality, conflict on collision
- load_recent: window selection, ordering, empty cases
- load_range: bounds (inclusive start, exclusive end), validation
  of timezone awareness and start/end ordering
- evict: explicit eviction by relative age
- query_trend: each of three metrics, unknown-metric rejection,
  empty-window behavior
"""

from datetime import datetime, timedelta, timezone

import pytest

from conformal_margin.agent_state import AgentState
from conformal_margin.fleet_snapshot import FleetSnapshot
from conformal_margin.ontology.domain.exceptions import (OperationFailedError,
                                                         UnknownMetricError,
                                                         ValidationError)
from conformal_margin.ontology.domain.ontology import ModelTier, Role
from conformal_margin.ontology.ports.state_store import StateStore
from conformal_margin.stores.in_memory import InMemoryStore

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    """Pin the anyio backend to asyncio for the v0.1 test suite."""
    return "asyncio"


UTC = timezone.utc


def _make_agent(agent_id: str = "agent-1", risk_score: float = 0.5) -> AgentState:
    return AgentState(
        agent_id=agent_id,
        timestamp=datetime(2026, 5, 12, 9, 0, tzinfo=UTC),
        risk_score=risk_score,
        model_tier=ModelTier.FOUNDATION,
        metadata={},
    )


def _make_snapshot(ts: datetime, agents: tuple[AgentState, ...] = ()) -> FleetSnapshot:
    """Construct a FleetSnapshot. Defaults to an empty fleet if no agents."""
    return FleetSnapshot(timestamp=ts, agents=agents)


def test_constructs_with_no_arguments():
    store = InMemoryStore()
    assert store is not None


def test_role_is_store():
    assert InMemoryStore.role is Role.STORE


def test_supported_metrics_is_the_documented_triple():
    assert InMemoryStore.SUPPORTED_METRICS == (
        "mean_risk_score",
        "p95_risk_score",
        "unique_agent_count",
    )


def test_satisfies_state_store_protocol():
    store = InMemoryStore()
    assert isinstance(store, StateStore)


async def test_save_persists_snapshot():
    store = InMemoryStore()
    ts = datetime.now(UTC) - timedelta(seconds=1)
    snapshot = _make_snapshot(ts)
    await store.save(snapshot)
    loaded = await store.load_recent(timedelta(minutes=1))
    assert loaded == (snapshot,)


async def test_save_same_snapshot_twice_is_idempotent():
    store = InMemoryStore()
    ts = datetime.now(UTC) - timedelta(seconds=1)
    snapshot = _make_snapshot(ts, agents=(_make_agent(),))
    await store.save(snapshot)
    await store.save(snapshot)
    loaded = await store.load_recent(timedelta(minutes=1))
    assert loaded == (snapshot,)


async def test_save_different_snapshot_at_same_timestamp_raises():
    store = InMemoryStore()
    ts = datetime.now(UTC) - timedelta(seconds=1)
    first = _make_snapshot(ts, agents=(_make_agent("a"),))
    second = _make_snapshot(ts, agents=(_make_agent("b"),))
    await store.save(first)
    with pytest.raises(OperationFailedError) as exc_info:
        await store.save(second)
    assert exc_info.value.operation == "save"
    assert exc_info.value.details["reason"] == (
        "different snapshot exists at this timestamp"
    )
    assert exc_info.value.details["timestamp"] == ts.isoformat()


async def test_load_recent_returns_only_snapshots_within_window():
    store = InMemoryStore()
    now = datetime.now(UTC)
    recent = _make_snapshot(now - timedelta(seconds=5))
    old = _make_snapshot(now - timedelta(hours=2))
    await store.save(recent)
    await store.save(old)
    loaded = await store.load_recent(timedelta(minutes=1))
    assert loaded == (recent,)


async def test_load_recent_orders_by_timestamp_ascending():
    store = InMemoryStore()
    now = datetime.now(UTC)
    s1 = _make_snapshot(now - timedelta(seconds=30))
    s2 = _make_snapshot(now - timedelta(seconds=20))
    s3 = _make_snapshot(now - timedelta(seconds=10))
    await store.save(s2)
    await store.save(s3)
    await store.save(s1)
    loaded = await store.load_recent(timedelta(minutes=1))
    assert loaded == (s1, s2, s3)


async def test_load_recent_returns_empty_tuple_on_empty_store():
    store = InMemoryStore()
    assert await store.load_recent(timedelta(hours=1)) == ()


async def test_load_recent_returns_empty_tuple_when_nothing_in_window():
    store = InMemoryStore()
    now = datetime.now(UTC)
    await store.save(_make_snapshot(now - timedelta(hours=2)))
    assert await store.load_recent(timedelta(seconds=1)) == ()


async def test_load_range_returns_snapshots_in_half_open_interval():
    store = InMemoryStore()
    s1 = _make_snapshot(datetime(2026, 5, 12, 9, 0, tzinfo=UTC))
    s2 = _make_snapshot(datetime(2026, 5, 12, 9, 5, tzinfo=UTC))
    s3 = _make_snapshot(datetime(2026, 5, 12, 9, 10, tzinfo=UTC))
    for s in (s1, s2, s3):
        await store.save(s)
    loaded = await store.load_range(
        start=datetime(2026, 5, 12, 9, 0, tzinfo=UTC),
        end=datetime(2026, 5, 12, 9, 10, tzinfo=UTC),
    )
    assert loaded == (s1, s2)


async def test_load_range_orders_by_timestamp_ascending():
    store = InMemoryStore()
    s_early = _make_snapshot(datetime(2026, 5, 12, 9, 0, tzinfo=UTC))
    s_mid = _make_snapshot(datetime(2026, 5, 12, 9, 5, tzinfo=UTC))
    s_late = _make_snapshot(datetime(2026, 5, 12, 9, 9, tzinfo=UTC))
    await store.save(s_late)
    await store.save(s_early)
    await store.save(s_mid)
    loaded = await store.load_range(
        start=datetime(2026, 5, 12, 9, 0, tzinfo=UTC),
        end=datetime(2026, 5, 12, 9, 10, tzinfo=UTC),
    )
    assert loaded == (s_early, s_mid, s_late)


async def test_load_range_naive_start_raises_validation_error():
    store = InMemoryStore()
    with pytest.raises(ValidationError) as exc_info:
        await store.load_range(
            start=datetime(2026, 5, 12, 9, 0),
            end=datetime(2026, 5, 12, 9, 10, tzinfo=UTC),
        )
    assert exc_info.value.field == "start"


async def test_load_range_naive_end_raises_validation_error():
    store = InMemoryStore()
    with pytest.raises(ValidationError) as exc_info:
        await store.load_range(
            start=datetime(2026, 5, 12, 9, 0, tzinfo=UTC),
            end=datetime(2026, 5, 12, 9, 10),
        )
    assert exc_info.value.field == "end"


async def test_load_range_end_equal_to_start_raises_validation_error():
    store = InMemoryStore()
    when = datetime(2026, 5, 12, 9, 0, tzinfo=UTC)
    with pytest.raises(ValidationError) as exc_info:
        await store.load_range(start=when, end=when)
    assert exc_info.value.field == "end"


async def test_load_range_end_before_start_raises_validation_error():
    store = InMemoryStore()
    with pytest.raises(ValidationError) as exc_info:
        await store.load_range(
            start=datetime(2026, 5, 12, 9, 10, tzinfo=UTC),
            end=datetime(2026, 5, 12, 9, 0, tzinfo=UTC),
        )
    assert exc_info.value.field == "end"


async def test_evict_removes_snapshots_older_than_cutoff():
    store = InMemoryStore()
    now = datetime.now(UTC)
    fresh = _make_snapshot(now - timedelta(seconds=5))
    stale = _make_snapshot(now - timedelta(hours=2))
    await store.save(fresh)
    await store.save(stale)
    await store.evict(older_than=timedelta(minutes=10))
    loaded = await store.load_recent(timedelta(hours=24))
    assert loaded == (fresh,)


async def test_evict_with_zero_duration_drops_everything():
    store = InMemoryStore()
    now = datetime.now(UTC)
    await store.save(_make_snapshot(now - timedelta(seconds=5)))
    await store.save(_make_snapshot(now - timedelta(seconds=1)))
    await store.evict(older_than=timedelta(seconds=0))
    assert await store.load_recent(timedelta(hours=24)) == ()


async def test_query_trend_unknown_metric_raises_unknown_metric_error():
    store = InMemoryStore()
    with pytest.raises(UnknownMetricError) as exc_info:
        await store.query_trend(metric="median_risk", window=timedelta(minutes=1))
    assert exc_info.value.metric == "median_risk"
    assert exc_info.value.available == InMemoryStore.SUPPORTED_METRICS


async def test_query_trend_empty_window_returns_empty_mapping():
    store = InMemoryStore()
    result = await store.query_trend(
        metric="mean_risk_score", window=timedelta(minutes=1)
    )
    assert result == {}


async def test_query_trend_mean_risk_score():
    store = InMemoryStore()
    now = datetime.now(UTC)
    snap = _make_snapshot(
        now - timedelta(seconds=5),
        agents=(
            _make_agent("a", risk_score=0.2),
            _make_agent("b", risk_score=0.4),
            _make_agent("c", risk_score=0.6),
        ),
    )
    await store.save(snap)
    result = await store.query_trend(
        metric="mean_risk_score", window=timedelta(minutes=1)
    )
    assert result["value"] == pytest.approx(0.4)
    assert result["n_samples"] == 3


async def test_query_trend_p95_risk_score_uses_nearest_rank():
    store = InMemoryStore()
    now = datetime.now(UTC)
    agents = tuple(_make_agent(f"agent-{i}", risk_score=i * 0.05) for i in range(20))
    await store.save(_make_snapshot(now - timedelta(seconds=5), agents=agents))
    result = await store.query_trend(
        metric="p95_risk_score", window=timedelta(minutes=1)
    )
    assert result["value"] == pytest.approx(0.90)
    assert result["n_samples"] == 20


async def test_query_trend_unique_agent_count():
    store = InMemoryStore()
    now = datetime.now(UTC)
    snap1 = _make_snapshot(
        now - timedelta(seconds=30),
        agents=(_make_agent("a"), _make_agent("b")),
    )
    snap2 = _make_snapshot(
        now - timedelta(seconds=10),
        agents=(_make_agent("b"), _make_agent("c")),
    )
    await store.save(snap1)
    await store.save(snap2)
    result = await store.query_trend(
        metric="unique_agent_count", window=timedelta(minutes=1)
    )
    assert result["value"] == 3
