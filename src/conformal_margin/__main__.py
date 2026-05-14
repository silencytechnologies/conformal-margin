"""conformal-margin runnable quickstart.

Run with:

    uv run python -m conformal_margin

This module is the entry point for `python -m conformal_margin`. It
demonstrates the minimal end-to-end pipeline: build a FleetMonitor
via `from_settings`, observe a calibration history snapshot, then
observe a test snapshot containing one clear outlier. The library
emits one `OperatorSignal` per agent in the test snapshot; the
`LogHandler` renders each one as a colorized structlog record on
stdout.

For richer demos see the examples/ directory in the repository:

- `examples/log_handler_demo.py`         - five signals across every severity tier
- `examples/blocking_handler_demo.py`    - reference enforcement pattern
- `examples/smolagents_demo/main.py`     - end-to-end with HuggingFace SmolLM2
- `examples/dashboard/main.py`           - live FastAPI dashboard
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone

import structlog

from conformal_margin import (AgentState, AppSettings, FleetSnapshot,
                              from_settings)
from conformal_margin.fleet_monitor import FleetMonitor
from conformal_margin.handlers.log import LogHandler
from conformal_margin.ontology.domain.ontology import ModelTier

UTC = timezone.utc


def _configure_logging() -> None:
    """Human-readable console output for the quickstart."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    )


def _agent(agent_id: str, risk_score: float, ts: datetime) -> AgentState:
    return AgentState(
        agent_id=agent_id,
        timestamp=ts,
        risk_score=risk_score,
        model_tier=ModelTier.FOUNDATION,
        metadata={},
    )


async def _run_quickstart() -> None:
    _configure_logging()
    now = datetime.now(UTC)
    base = from_settings(
        AppSettings(
            operator_name="quickstart",
            alpha=0.10,
            min_calibration_size=3,
        )
    )
    monitor = FleetMonitor(
        operators=base.operators,
        store=base.store,
        handlers=(LogHandler(name="quickstart-log"),),
        calibration_window=base.calibration_window,
    )

    print()
    print("=" * 60)
    print("  conformal-margin quickstart")
    print("=" * 60)
    print()
    print("Step 1: observe a calibration history snapshot")
    print("        (five typical agents around risk_score 0.5)")
    history = FleetSnapshot(
        timestamp=now - timedelta(minutes=1),
        agents=tuple(
            _agent(
                agent_id=f"cal-{i}",
                risk_score=0.45 + 0.025 * i,
                ts=now - timedelta(minutes=1),
            )
            for i in range(5)
        ),
    )
    signals_history = await monitor.observe(history)
    print(f"        -> {len(signals_history)} signals emitted on calibration pass")
    print()

    print("Step 2: observe a test snapshot containing one outlier")
    print("        (normal agent at 0.5, outlier agent at 100.0)")
    test = FleetSnapshot(
        timestamp=now,
        agents=(
            _agent(agent_id="normal-1", risk_score=0.5, ts=now),
            _agent(agent_id="outlier-1", risk_score=100.0, ts=now),
        ),
    )
    signals = await monitor.observe(test)
    print(f"        -> {len(signals)} signals emitted (rendered above)")
    print()
    print("For richer demos see the examples/ directory:")
    print("  - examples/log_handler_demo.py")
    print("  - examples/blocking_handler_demo.py")
    print("  - examples/smolagents_demo/main.py")
    print("  - examples/dashboard/main.py")
    print()


if __name__ == "__main__":
    asyncio.run(_run_quickstart())
