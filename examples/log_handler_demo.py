"""LogHandler demo - dispatch five signals at every Severity tier to stdout.

Run from the workspace root:
    uv run python examples/log_handler_demo.py

Configures structlog with a console renderer so each signal is
rendered as one human-readable, colour-coded line in the terminal.
Demonstrates the Severity -> log-level mapping documented in
`conformal_margin.handlers.log`:

    Severity.INFO     -> info
    Severity.LOW      -> info
    Severity.MEDIUM   -> warning
    Severity.HIGH     -> error
    Severity.CRITICAL -> critical

Also dispatches one drift signal so the drift_event_type field is
visible in the output.
"""

import asyncio
import logging
from datetime import datetime, timezone

import structlog

from conformal_margin.handlers.log import LogHandler
from conformal_margin.ontology.domain.ontology import (DriftEventType,
                                                       Severity, SignalKind)
from conformal_margin.operator_signal import OperatorSignal

UTC = timezone.utc
DEMO_UUID = "550e8400-e29b-41d4-a716-446655440000"
DEMO_TIME = datetime(2026, 5, 13, 12, 0, tzinfo=UTC)


def _configure_structlog() -> None:
    """Configure structlog for human-readable colorized console output."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        cache_logger_on_first_use=True,
    )


def _build_signal(
    severity: Severity,
    *,
    kind: SignalKind = SignalKind.ANOMALY,
    drift_event_type: DriftEventType | None = None,
    operator_name: str = "baseline",
) -> OperatorSignal:
    """Construct a valid OperatorSignal at the requested severity."""
    threshold_exceeded = severity is not Severity.INFO
    p_value = 1.0 if severity is Severity.INFO else 0.003
    return OperatorSignal(
        signal_id=DEMO_UUID,
        operator_name=operator_name,
        timestamp=DEMO_TIME,
        kind=kind,
        severity=severity,
        score=4.7,
        p_value=p_value,
        threshold_exceeded=threshold_exceeded,
        affected_agents=("agent-7",),
        drift_event_type=drift_event_type,
        metadata={"calibration_window_size": 1024},
    )


async def _main() -> None:
    _configure_structlog()
    handler = LogHandler(name="demo-log")

    print()
    print("=== Five severity tiers (anomaly signals) ===")
    print()
    for severity in (
        Severity.INFO,
        Severity.LOW,
        Severity.MEDIUM,
        Severity.HIGH,
        Severity.CRITICAL,
    ):
        await handler.handle(_build_signal(severity))

    print()
    print("=== One drift signal (mean shift, HIGH severity) ===")
    print()
    await handler.handle(
        _build_signal(
            Severity.HIGH,
            kind=SignalKind.DRIFT,
            drift_event_type=DriftEventType.MEAN_SHIFT,
            operator_name="drift-detector",
        )
    )


if __name__ == "__main__":
    asyncio.run(_main())
