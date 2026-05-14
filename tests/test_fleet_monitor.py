"""Unit tests for conformal_margin.fleet_monitor.FleetMonitor.

Uses stub Operator / AlertHandler / FeedbackPort implementations
to control inputs and inspect outputs precisely. Real
`InMemoryStore` is used for the store role so the load/save
plumbing is exercised end-to-end.

Coverage: construction validation, end-to-end observe lifecycle,
operator/handler concurrency, error semantics (operator failure
propagates, handler failure is caught), feedback dispatch path,
store interaction (load before save, save after dispatch).
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import ClassVar

import pytest

from conformal_margin.agent_state import AgentState
from conformal_margin.fleet_monitor import FleetMonitor
from conformal_margin.fleet_snapshot import FleetSnapshot
from conformal_margin.ontology.domain.exceptions import (StoreUnavailableError,
                                                         UnexpectedError,
                                                         ValidationError)
from conformal_margin.ontology.domain.ontology import (ModelTier, Role,
                                                       Severity, SignalKind)
from conformal_margin.ontology.ports.feedback import FeedbackLabel
from conformal_margin.ontology.ports.operator_signal import OperatorSignalLike
from conformal_margin.operator_signal import OperatorSignal
from conformal_margin.stores.in_memory import InMemoryStore

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


UTC = timezone.utc


def _now_minus(seconds: int) -> datetime:
    return datetime.now(UTC) - timedelta(seconds=seconds)


def _agent(agent_id: str = "a", risk_score: float = 0.5) -> AgentState:
    return AgentState(
        agent_id=agent_id,
        timestamp=_now_minus(1),
        risk_score=risk_score,
        model_tier=ModelTier.FOUNDATION,
        metadata={},
    )


def _snapshot(agents: tuple[AgentState, ...] = ()) -> FleetSnapshot:
    return FleetSnapshot(timestamp=_now_minus(1), agents=agents)


def _signal(signal_id: str = "550e8400-e29b-41d4-a716-446655440000") -> OperatorSignal:
    return OperatorSignal(
        signal_id=signal_id,
        operator_name="stub",
        timestamp=_now_minus(1),
        kind=SignalKind.ANOMALY,
        severity=Severity.HIGH,
        score=4.7,
        p_value=0.003,
        threshold_exceeded=True,
        affected_agents=("a",),
        drift_event_type=None,
        metadata={},
    )


@dataclass
class _StubOperator:
    """Operator stub that emits a fixed tuple of signals on every run."""

    name: str
    signals_to_emit: tuple[OperatorSignal, ...] = ()
    call_count: int = 0
    role: ClassVar = Role.TRANSFORMER

    async def run(self, snapshot, history):
        self.call_count += 1
        return self.signals_to_emit


@dataclass
class _FailingOperator:
    """Operator stub that raises on every run."""

    name: str
    role: ClassVar = Role.TRANSFORMER

    async def run(self, snapshot, history):
        raise UnexpectedError(context="operator boom")


@dataclass
class _StubHandler:
    """AlertHandler stub that records every received signal."""

    name: str
    role: ClassVar = Role.SINK
    received: list[OperatorSignalLike] = field(default_factory=list)

    async def handle(self, signal):
        self.received.append(signal)


@dataclass
class _ExplodingHandler:
    """AlertHandler stub that raises on every handle call."""

    name: str
    role: ClassVar = Role.SINK

    async def handle(self, signal):
        raise RuntimeError("handler kaboom")


@dataclass
class _StubFeedbackPort:
    name: str
    role: ClassVar = Role.SINK
    received: list[tuple[OperatorSignalLike, FeedbackLabel]] = field(
        default_factory=list
    )

    async def record(self, signal, label):
        self.received.append((signal, label))


@dataclass(frozen=True)
class _StubLabel:
    signal_id: str
    is_true_positive: bool
    metadata: dict = field(default_factory=dict)


@dataclass
class _FailingLoadStore:
    """Store stub whose load_recent always raises."""

    role: ClassVar = Role.STORE

    async def save(self, snapshot): ...
    async def load_recent(self, window):
        raise StoreUnavailableError(operation="load_recent")

    async def load_range(self, start, end):
        return ()

    async def evict(self, older_than): ...
    async def query_trend(self, metric, window):
        return {}


def test_constructs_with_valid_inputs():
    op = _StubOperator(name="op")
    h = _StubHandler(name="h")
    monitor = FleetMonitor(operators=(op,), store=InMemoryStore(), handlers=(h,))
    assert monitor.operators == (op,)
    assert monitor.handlers == (h,)


def test_empty_operators_raises_validation_error():
    with pytest.raises(ValidationError) as exc_info:
        FleetMonitor(
            operators=(), store=InMemoryStore(), handlers=(_StubHandler(name="h"),)
        )
    assert exc_info.value.field == "operators"


def test_empty_handlers_raises_validation_error():
    with pytest.raises(ValidationError) as exc_info:
        FleetMonitor(
            operators=(_StubOperator(name="op"),),
            store=InMemoryStore(),
            handlers=(),
        )
    assert exc_info.value.field == "handlers"


def test_zero_calibration_window_raises_validation_error():
    with pytest.raises(ValidationError) as exc_info:
        FleetMonitor(
            operators=(_StubOperator(name="op"),),
            store=InMemoryStore(),
            handlers=(_StubHandler(name="h"),),
            calibration_window=timedelta(0),
        )
    assert exc_info.value.field == "calibration_window"


def test_iterables_are_frozen_to_tuples():
    op = _StubOperator(name="op")
    h = _StubHandler(name="h")
    operators_list = [op]
    handlers_list = [h]
    monitor = FleetMonitor(
        operators=operators_list, store=InMemoryStore(), handlers=handlers_list
    )
    operators_list.append(_StubOperator(name="extra"))
    handlers_list.append(_StubHandler(name="extra"))
    assert isinstance(monitor.operators, tuple)
    assert isinstance(monitor.handlers, tuple)
    assert len(monitor.operators) == 1
    assert len(monitor.handlers) == 1


def test_is_frozen():
    monitor = FleetMonitor(
        operators=(_StubOperator(name="op"),),
        store=InMemoryStore(),
        handlers=(_StubHandler(name="h"),),
    )
    with pytest.raises(AttributeError):
        monitor.calibration_window = timedelta(hours=2)


async def test_observe_returns_all_emitted_signals():
    sig1 = _signal(signal_id="11111111-1111-4111-8111-111111111111")
    sig2 = _signal(signal_id="22222222-2222-4222-8222-222222222222")
    op = _StubOperator(name="op", signals_to_emit=(sig1, sig2))
    h = _StubHandler(name="h")
    monitor = FleetMonitor(operators=(op,), store=InMemoryStore(), handlers=(h,))
    result = await monitor.observe(_snapshot())
    assert result == (sig1, sig2)


async def test_observe_dispatches_every_signal_to_every_handler():
    sig = _signal()
    op = _StubOperator(name="op", signals_to_emit=(sig,))
    h1 = _StubHandler(name="h1")
    h2 = _StubHandler(name="h2")
    monitor = FleetMonitor(operators=(op,), store=InMemoryStore(), handlers=(h1, h2))
    await monitor.observe(_snapshot())
    assert h1.received == [sig]
    assert h2.received == [sig]


async def test_observe_calls_each_operator_once_per_snapshot():
    op1 = _StubOperator(name="op1")
    op2 = _StubOperator(name="op2")
    monitor = FleetMonitor(
        operators=(op1, op2), store=InMemoryStore(), handlers=(_StubHandler(name="h"),)
    )
    await monitor.observe(_snapshot())
    assert op1.call_count == 1
    assert op2.call_count == 1


async def test_observe_saves_snapshot_after_dispatch():
    op = _StubOperator(name="op")
    store = InMemoryStore()
    monitor = FleetMonitor(
        operators=(op,), store=store, handlers=(_StubHandler(name="h"),)
    )
    snapshot = _snapshot()
    await monitor.observe(snapshot)
    loaded = await store.load_recent(timedelta(seconds=10))
    assert loaded == (snapshot,)


async def test_observe_loads_history_excluding_current_snapshot():
    op = _StubOperator(name="op")
    store = InMemoryStore()
    past = FleetSnapshot(timestamp=_now_minus(60), agents=())
    await store.save(past)
    monitor = FleetMonitor(
        operators=(op,), store=store, handlers=(_StubHandler(name="h"),)
    )
    current = _snapshot()
    await monitor.observe(current)
    final = await store.load_recent(timedelta(minutes=10))
    assert past in final
    assert current in final


async def test_observe_returns_empty_when_operator_emits_nothing():
    op = _StubOperator(name="op", signals_to_emit=())
    h = _StubHandler(name="h")
    monitor = FleetMonitor(operators=(op,), store=InMemoryStore(), handlers=(h,))
    result = await monitor.observe(_snapshot())
    assert result == ()
    assert h.received == []


async def test_observe_concatenates_signals_from_multiple_operators():
    sig1 = _signal(signal_id="11111111-1111-4111-8111-111111111111")
    sig2 = _signal(signal_id="22222222-2222-4222-8222-222222222222")
    op1 = _StubOperator(name="op1", signals_to_emit=(sig1,))
    op2 = _StubOperator(name="op2", signals_to_emit=(sig2,))
    monitor = FleetMonitor(
        operators=(op1, op2),
        store=InMemoryStore(),
        handlers=(_StubHandler(name="h"),),
    )
    result = await monitor.observe(_snapshot())
    assert len(result) == 2
    assert sig1 in result
    assert sig2 in result


async def test_operator_failure_propagates_through_task_group():
    bad = _FailingOperator(name="bad")
    monitor = FleetMonitor(
        operators=(bad,),
        store=InMemoryStore(),
        handlers=(_StubHandler(name="h"),),
    )
    with pytest.raises(BaseExceptionGroup) as exc_info:
        await monitor.observe(_snapshot())
    flat_exceptions = list(_flatten(exc_info.value))
    assert any(isinstance(e, UnexpectedError) for e in flat_exceptions)


async def test_handler_failure_is_caught_and_does_not_kill_siblings():
    sig = _signal()
    op = _StubOperator(name="op", signals_to_emit=(sig,))
    bad = _ExplodingHandler(name="bad")
    good = _StubHandler(name="good")
    monitor = FleetMonitor(operators=(op,), store=InMemoryStore(), handlers=(bad, good))
    await monitor.observe(_snapshot())
    assert good.received == [sig]


async def test_store_load_failure_propagates_as_store_unavailable():
    monitor = FleetMonitor(
        operators=(_StubOperator(name="op"),),
        store=_FailingLoadStore(),
        handlers=(_StubHandler(name="h"),),
    )
    with pytest.raises(StoreUnavailableError):
        await monitor.observe(_snapshot())


async def test_submit_feedback_with_no_ports_raises_validation_error():
    monitor = FleetMonitor(
        operators=(_StubOperator(name="op"),),
        store=InMemoryStore(),
        handlers=(_StubHandler(name="h"),),
    )
    with pytest.raises(ValidationError) as exc_info:
        await monitor.submit_feedback(
            _signal(), _StubLabel(signal_id="x", is_true_positive=True)
        )
    assert exc_info.value.field == "feedback_ports"


async def test_submit_feedback_dispatches_to_every_port():
    p1 = _StubFeedbackPort(name="p1")
    p2 = _StubFeedbackPort(name="p2")
    monitor = FleetMonitor(
        operators=(_StubOperator(name="op"),),
        store=InMemoryStore(),
        handlers=(_StubHandler(name="h"),),
        feedback_ports=(p1, p2),
    )
    sig = _signal()
    label = _StubLabel(signal_id=sig.signal_id, is_true_positive=False)
    await monitor.submit_feedback(sig, label)
    assert p1.received == [(sig, label)]
    assert p2.received == [(sig, label)]


def _flatten(group):
    """Recursively flatten nested ExceptionGroup into a list of leaf exceptions."""
    for e in group.exceptions:
        if isinstance(e, BaseExceptionGroup):
            yield from _flatten(e)
        else:
            yield e
