"""Unit tests for conformal_margin.operators.baseline.BaselineOperator.

Coverage: construction and validation, Operator Protocol
satisfaction, identity generation (UUID4 + snapshot.timestamp),
the four-tier severity mapping from p-value, calibration
threshold cutoffs (min_calibration_size), empty inputs, drift
signal emission when Page-Hinkley triggers, end-to-end formal
coverage verification on exchangeable data.
"""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from conformal_margin.agent_state import AgentState
from conformal_margin.fleet_snapshot import FleetSnapshot
from conformal_margin.ontology.domain.exceptions import ValidationError
from conformal_margin.ontology.domain.ontology import (DriftEventType,
                                                       ModelTier, Role,
                                                       Severity, SignalKind)
from conformal_margin.ontology.ports.operator import Operator
from conformal_margin.operators.baseline import BaselineOperator

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


UTC = timezone.utc
BASELINE_TIME = datetime(2026, 5, 13, 12, 0, tzinfo=UTC)


def _agent(
    agent_id: str, risk_score: float, ts: datetime = BASELINE_TIME
) -> AgentState:
    return AgentState(
        agent_id=agent_id,
        timestamp=ts,
        risk_score=risk_score,
        model_tier=ModelTier.FOUNDATION,
        metadata={},
    )


def _snapshot_with_agents(
    ts: datetime, agents: tuple[AgentState, ...]
) -> FleetSnapshot:
    return FleetSnapshot(timestamp=ts, agents=agents)


def _stable_history(
    n_snapshots: int = 30,
    agents_per_snapshot: int = 5,
    base_score: float = 0.5,
    noise_scale: float = 0.05,
    rng_seed: int = 2026,
) -> tuple[FleetSnapshot, ...]:
    """Build a stable calibration history (no drift, scores around base_score)."""
    rng = np.random.default_rng(seed=rng_seed)
    history: list[FleetSnapshot] = []
    for i in range(n_snapshots):
        ts = BASELINE_TIME - timedelta(minutes=n_snapshots - i)
        agents = tuple(
            _agent(
                agent_id=f"cal-{i}-{j}",
                risk_score=float(rng.normal(base_score, noise_scale)),
                ts=ts,
            )
            for j in range(agents_per_snapshot)
        )
        history.append(_snapshot_with_agents(ts, agents))
    return tuple(history)


def test_constructs_with_valid_inputs():
    op = BaselineOperator(name="baseline")
    assert op.name == "baseline"
    assert op.alpha == 0.01
    assert op.min_calibration_size == 30


def test_role_is_transformer():
    assert BaselineOperator.role is Role.TRANSFORMER


def test_satisfies_operator_protocol():
    op = BaselineOperator(name="baseline")
    assert isinstance(op, Operator)


def test_is_frozen():
    op = BaselineOperator(name="baseline")
    with pytest.raises(AttributeError):
        op.alpha = 0.5


def test_empty_name_raises_validation_error():
    with pytest.raises(ValidationError) as exc_info:
        BaselineOperator(name="")
    assert exc_info.value.field == "name"


@pytest.mark.parametrize("bad_alpha", [-0.1, 0.0, 1.0, 1.5])
def test_alpha_out_of_open_unit_interval_raises_validation_error(bad_alpha):
    with pytest.raises(ValidationError) as exc_info:
        BaselineOperator(name="baseline", alpha=bad_alpha)
    assert exc_info.value.field == "alpha"


def test_min_calibration_size_below_one_raises_validation_error():
    with pytest.raises(ValidationError) as exc_info:
        BaselineOperator(name="baseline", min_calibration_size=0)
    assert exc_info.value.field == "min_calibration_size"


async def test_run_with_empty_snapshot_returns_no_signals():
    op = BaselineOperator(name="baseline")
    snapshot = _snapshot_with_agents(BASELINE_TIME, ())
    history = _stable_history(n_snapshots=30, agents_per_snapshot=5)
    signals = await op.run(snapshot, history)
    assert signals == ()


async def test_run_with_too_small_calibration_returns_no_signals():
    op = BaselineOperator(name="baseline", min_calibration_size=30)
    snapshot = _snapshot_with_agents(BASELINE_TIME, (_agent("a", 0.5),))
    history = _stable_history(n_snapshots=1, agents_per_snapshot=5)
    signals = await op.run(snapshot, history)
    assert signals == ()


async def test_run_with_empty_history_returns_no_signals():
    op = BaselineOperator(name="baseline")
    snapshot = _snapshot_with_agents(BASELINE_TIME, (_agent("a", 0.5),))
    signals = await op.run(snapshot, ())
    assert signals == ()


async def test_signals_carry_uuid4_signal_id_and_snapshot_timestamp():
    import uuid as _uuid

    op = BaselineOperator(name="baseline", min_calibration_size=10)
    snapshot = _snapshot_with_agents(BASELINE_TIME, (_agent("a", 0.5),))
    history = _stable_history(n_snapshots=5, agents_per_snapshot=4)
    signals = await op.run(snapshot, history)
    assert len(signals) >= 1
    for sig in signals:
        parsed = _uuid.UUID(sig.signal_id)
        assert parsed.version == 4
        assert sig.timestamp == snapshot.timestamp


async def test_signal_ids_are_distinct_across_emitted_signals():
    op = BaselineOperator(name="baseline", min_calibration_size=10)
    snapshot = _snapshot_with_agents(
        BASELINE_TIME,
        (_agent("a", 0.5), _agent("b", 0.55), _agent("c", 0.6)),
    )
    history = _stable_history(n_snapshots=5, agents_per_snapshot=4)
    signals = await op.run(snapshot, history)
    ids = {sig.signal_id for sig in signals}
    assert len(ids) == len(signals)


async def test_anomaly_signal_for_far_outlier_yields_critical_severity():
    op = BaselineOperator(name="baseline", alpha=0.01, min_calibration_size=10)
    history = _stable_history(n_snapshots=400, agents_per_snapshot=5, rng_seed=11)
    snapshot = _snapshot_with_agents(BASELINE_TIME, (_agent("outlier", 100.0),))
    signals = await op.run(snapshot, history)
    anomaly = signals[0]
    assert anomaly.kind is SignalKind.ANOMALY
    assert anomaly.threshold_exceeded is True
    assert anomaly.severity in {Severity.HIGH, Severity.CRITICAL}


async def test_anomaly_signal_for_typical_value_yields_info_severity():
    op = BaselineOperator(name="baseline", alpha=0.01, min_calibration_size=10)
    history = _stable_history(
        n_snapshots=40, agents_per_snapshot=5, base_score=0.5, noise_scale=0.05
    )
    snapshot = _snapshot_with_agents(BASELINE_TIME, (_agent("typical", 0.5),))
    signals = await op.run(snapshot, history)
    anomaly = signals[0]
    assert anomaly.kind is SignalKind.ANOMALY
    assert anomaly.threshold_exceeded is False
    assert anomaly.severity is Severity.INFO


async def test_one_signal_per_agent():
    op = BaselineOperator(name="baseline", min_calibration_size=10)
    snapshot = _snapshot_with_agents(
        BASELINE_TIME,
        (
            _agent("a", 0.5),
            _agent("b", 0.55),
            _agent("c", 0.6),
            _agent("d", 0.45),
        ),
    )
    history = _stable_history(n_snapshots=20, agents_per_snapshot=5)
    signals = await op.run(snapshot, history)
    anomaly_signals = [s for s in signals if s.kind is SignalKind.ANOMALY]
    assert len(anomaly_signals) == 4
    affected_ids = {s.affected_agents[0] for s in anomaly_signals}
    assert affected_ids == {"a", "b", "c", "d"}


async def test_each_anomaly_signal_targets_one_agent():
    op = BaselineOperator(name="baseline", min_calibration_size=10)
    snapshot = _snapshot_with_agents(
        BASELINE_TIME, (_agent("a", 0.5), _agent("b", 0.55))
    )
    history = _stable_history(n_snapshots=20, agents_per_snapshot=5)
    signals = await op.run(snapshot, history)
    for sig in signals:
        if sig.kind is SignalKind.ANOMALY:
            assert len(sig.affected_agents) == 1


async def test_drift_signal_emitted_when_upward_shift_present():
    op = BaselineOperator(
        name="baseline",
        min_calibration_size=20,
        drift_delta=0.005,
        drift_threshold=5.0,
    )
    early = _stable_history(
        n_snapshots=20, agents_per_snapshot=10, base_score=0.5, rng_seed=1
    )
    late = _stable_history(
        n_snapshots=20, agents_per_snapshot=10, base_score=2.0, rng_seed=2
    )
    history = early + late
    snapshot = _snapshot_with_agents(BASELINE_TIME, (_agent("a", 1.0),))
    signals = await op.run(snapshot, history)
    drift_signals = [s for s in signals if s.kind is SignalKind.DRIFT]
    assert len(drift_signals) == 1
    drift = drift_signals[0]
    assert drift.severity is Severity.HIGH
    assert drift.drift_event_type is DriftEventType.MEAN_SHIFT
    assert drift.affected_agents == ()
    assert drift.p_value == 0.0


async def test_no_drift_signal_on_stable_history():
    op = BaselineOperator(
        name="baseline",
        min_calibration_size=20,
        drift_delta=0.005,
        drift_threshold=5.0,
    )
    history = _stable_history(n_snapshots=40, agents_per_snapshot=10)
    snapshot = _snapshot_with_agents(BASELINE_TIME, (_agent("a", 0.5),))
    signals = await op.run(snapshot, history)
    drift_signals = [s for s in signals if s.kind is SignalKind.DRIFT]
    assert drift_signals == []


async def test_end_to_end_empirical_coverage_on_stable_history():
    op = BaselineOperator(name="baseline", alpha=0.05, min_calibration_size=50)
    history = _stable_history(
        n_snapshots=200, agents_per_snapshot=5, base_score=0.5, noise_scale=0.1
    )
    rng = np.random.default_rng(seed=999)
    snapshot = _snapshot_with_agents(
        BASELINE_TIME,
        tuple(_agent(f"t-{i}", float(rng.normal(0.5, 0.1))) for i in range(500)),
    )
    signals = await op.run(snapshot, history)
    anomaly_signals = [s for s in signals if s.kind is SignalKind.ANOMALY]
    flagged = sum(1 for s in anomaly_signals if s.threshold_exceeded)
    rate = flagged / len(anomaly_signals)
    assert (
        rate <= 0.05 + 0.03
    ), f"empirical FPR {rate:.4f} exceeds alpha 0.05 plus slack"
