"""Unit tests for conformal_margin.factory module.

Coverage: `from_settings` returns a working `FleetMonitor` with
the expected default topology (one BaselineOperator, one
InMemoryStore, one LogHandler); `from_env` round-trips through
environment variables; the resulting monitor is exercise-able
end-to-end with a real snapshot.
"""

from datetime import datetime, timedelta, timezone

import pytest

from conformal_margin import AgentState, FleetSnapshot
from conformal_margin.config import AppSettings
from conformal_margin.factory import from_env, from_settings
from conformal_margin.fleet_monitor import FleetMonitor
from conformal_margin.handlers.log import LogHandler
from conformal_margin.ontology.domain.ontology import ModelTier
from conformal_margin.operators.baseline import BaselineOperator
from conformal_margin.stores.in_memory import InMemoryStore

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


UTC = timezone.utc


def test_from_settings_returns_fleet_monitor():
    monitor = from_settings(AppSettings())
    assert isinstance(monitor, FleetMonitor)


def test_from_settings_wires_one_baseline_operator():
    monitor = from_settings(AppSettings())
    assert len(monitor.operators) == 1
    assert isinstance(monitor.operators[0], BaselineOperator)


def test_from_settings_wires_in_memory_store():
    monitor = from_settings(AppSettings())
    assert isinstance(monitor.store, InMemoryStore)


def test_from_settings_wires_one_log_handler():
    monitor = from_settings(AppSettings())
    assert len(monitor.handlers) == 1
    assert isinstance(monitor.handlers[0], LogHandler)


def test_from_settings_no_feedback_ports_by_default():
    monitor = from_settings(AppSettings())
    assert monitor.feedback_ports == ()


def test_operator_name_from_settings_propagates():
    settings = AppSettings(operator_name="custom-op")
    monitor = from_settings(settings)
    assert monitor.operators[0].name == "custom-op"


def test_alpha_from_settings_propagates():
    settings = AppSettings(alpha=0.05)
    monitor = from_settings(settings)
    assert monitor.operators[0].alpha == 0.05


def test_min_calibration_size_from_settings_propagates():
    settings = AppSettings(min_calibration_size=50)
    monitor = from_settings(settings)
    assert monitor.operators[0].min_calibration_size == 50


def test_drift_parameters_from_settings_propagate():
    settings = AppSettings(drift_delta=0.02, drift_threshold=10.0)
    monitor = from_settings(settings)
    assert monitor.operators[0].drift_delta == 0.02
    assert monitor.operators[0].drift_threshold == 10.0


def test_calibration_window_seconds_propagates_as_timedelta():
    settings = AppSettings(calibration_window_seconds=900)
    monitor = from_settings(settings)
    assert monitor.calibration_window == timedelta(seconds=900)


def test_log_handler_name_from_settings_propagates():
    settings = AppSettings(log_handler_name="my-log")
    monitor = from_settings(settings)
    assert monitor.handlers[0].name == "my-log"


def test_from_env_reads_environment(monkeypatch):
    monkeypatch.setenv("CONFORMAL_MARGIN_ALPHA", "0.03")
    monkeypatch.setenv("CONFORMAL_MARGIN_OPERATOR_NAME", "from-env-op")
    monitor = from_env()
    assert monitor.operators[0].alpha == 0.03
    assert monitor.operators[0].name == "from-env-op"


def test_from_env_falls_back_to_defaults_when_env_absent():
    monitor = from_env()
    assert monitor.operators[0].alpha == 0.01
    assert monitor.operators[0].name == "baseline"


async def test_monitor_from_factory_observes_snapshot_without_error():
    monitor = from_settings(AppSettings(min_calibration_size=1))
    snapshot = FleetSnapshot(
        timestamp=datetime.now(UTC) - timedelta(seconds=1),
        agents=(
            AgentState(
                agent_id="agent-1",
                timestamp=datetime.now(UTC) - timedelta(seconds=2),
                risk_score=0.5,
                model_tier=ModelTier.FOUNDATION,
                metadata={},
            ),
        ),
    )
    first = await monitor.observe(snapshot)
    assert first == ()


async def test_monitor_from_factory_emits_signals_with_enough_history():
    monitor = from_settings(AppSettings(min_calibration_size=2, alpha=0.5))
    now = datetime.now(UTC)
    first_snapshot = FleetSnapshot(
        timestamp=now - timedelta(seconds=30),
        agents=(
            AgentState(
                agent_id="a",
                timestamp=now - timedelta(seconds=30),
                risk_score=0.3,
                model_tier=ModelTier.FOUNDATION,
                metadata={},
            ),
            AgentState(
                agent_id="b",
                timestamp=now - timedelta(seconds=30),
                risk_score=0.7,
                model_tier=ModelTier.FOUNDATION,
                metadata={},
            ),
        ),
    )
    await monitor.observe(first_snapshot)

    second_snapshot = FleetSnapshot(
        timestamp=now - timedelta(seconds=10),
        agents=(
            AgentState(
                agent_id="c",
                timestamp=now - timedelta(seconds=10),
                risk_score=0.5,
                model_tier=ModelTier.FOUNDATION,
                metadata={},
            ),
        ),
    )
    signals = await monitor.observe(second_snapshot)
    assert len(signals) == 1
