"""Reference enforcement pattern: a custom `BlockingHandler`.

The library MONITORS agents - it does not BLOCK them.
This demo shows the is exact pattern a customer copies into
their own codebase: a class that implements the `AlertHandler` Protocol,
dispatches on `signal.severity`, and calls the customer's agent runtime API.

The runtime in this demo is a `FakeAgentRuntime` stub that just
prints what it would do; in a real deployment, the same calls
hit LangChain / LangGraph / your own orchestrator API. The
BlockingHandler logic is what you keep.

Run from the workspace root:
    uv run python examples/blocking_handler_demo.py
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import ClassVar

from conformal_margin import OperatorSignal
from conformal_margin.fleet_monitor import FleetMonitor
from conformal_margin.handlers.log import LogHandler
from conformal_margin.ontology.domain.ontology import (Role, Severity,
                                                       SignalKind)
from conformal_margin.ontology.ports.operator_signal import OperatorSignalLike
from conformal_margin.operators.baseline import BaselineOperator
from conformal_margin.stores.in_memory import InMemoryStore

UTC = timezone.utc


@dataclass
class FakeAgentRuntime:
    """Stub runtime. Prints what it would do; tracks state in-memory."""

    suspended: set[str] = field(default_factory=set)
    throttled: dict[str, float] = field(default_factory=dict)

    async def suspend(self, agent_id: str, reason: str) -> None:
        if agent_id in self.suspended:
            return
        self.suspended.add(agent_id)
        print(f"  [runtime] WOULD SUSPEND {agent_id} (reason: {reason})")

    async def throttle(self, agent_id: str, requests_per_minute: float) -> None:
        self.throttled[agent_id] = requests_per_minute
        print(
            f"  [runtime] WOULD THROTTLE {agent_id} to "
            f"{requests_per_minute} req/min"
        )

    async def resume(self, agent_id: str) -> None:
        self.suspended.discard(agent_id)
        self.throttled.pop(agent_id, None)
        print(f"  [runtime] WOULD RESUME {agent_id}")


@dataclass
class BlockingHandler:
    """AlertHandler that enforces severity-tier response on the runtime."""

    name: str
    runtime: FakeAgentRuntime
    role: ClassVar = Role.SINK

    async def handle(self, signal: OperatorSignalLike) -> None:
        if not signal.threshold_exceeded:
            return

        if signal.severity is Severity.CRITICAL:
            for agent_id in signal.affected_agents:
                await self.runtime.suspend(
                    agent_id,
                    reason=f"conformal-margin anomaly p={signal.p_value:.4f}",
                )
        elif signal.severity is Severity.HIGH:
            for agent_id in signal.affected_agents:
                await self.runtime.throttle(agent_id, requests_per_minute=1.0)
        elif signal.severity is Severity.MEDIUM:
            for agent_id in signal.affected_agents:
                print(
                    f"  [policy] MEDIUM signal on {agent_id} - "
                    f"queued for human review (not blocked)"
                )


def _build_signal(
    agent_id: str,
    severity: Severity,
    *,
    threshold_exceeded: bool = True,
    p_value: float | None = None,
) -> OperatorSignal:
    """Construct a finalized OperatorSignal for demo purposes."""
    if p_value is None:
        match severity:
            case Severity.CRITICAL:
                p_value = 0.00005
            case Severity.HIGH:
                p_value = 0.0008
            case Severity.MEDIUM:
                p_value = 0.008
            case _:
                p_value = 0.5
    if severity is Severity.INFO:
        threshold_exceeded = False
    return OperatorSignal(
        signal_id=str(uuid.uuid4()),
        operator_name="demo",
        timestamp=datetime.now(UTC),
        kind=SignalKind.ANOMALY,
        severity=severity,
        score=3.0,
        p_value=p_value,
        threshold_exceeded=threshold_exceeded,
        affected_agents=(agent_id,),
        drift_event_type=None,
        metadata={},
    )


async def _drive_demo() -> None:
    runtime = FakeAgentRuntime()
    blocker = BlockingHandler(name="blocking-handler", runtime=runtime)

    demo_signals = [
        _build_signal("agent-001", Severity.INFO),
        _build_signal("agent-002", Severity.LOW),
        _build_signal("agent-003", Severity.MEDIUM),
        _build_signal("agent-004", Severity.HIGH),
        _build_signal("agent-005", Severity.CRITICAL),
        _build_signal("agent-006", Severity.CRITICAL),
        _build_signal("agent-004", Severity.HIGH),
    ]

    print()
    print("=" * 64)
    print("  BlockingHandler demo - severity-tier enforcement policy")
    print("=" * 64)
    print()
    print("  This demo feeds synthetic OperatorSignal values directly to a")
    print("  BlockingHandler. The handler maps severity to action via the")
    print("  customer-owned policy. The 'runtime' is a stub that prints what")
    print("  it would do; in a real deployment those calls hit your")
    print("  orchestrator API.")
    print()
    print("=" * 64)

    for signal in demo_signals:
        print(
            f"\n  signal: severity={signal.severity.value:8s} "
            f"agent={signal.affected_agents[0]} "
            f"p_value={signal.p_value:.5f} "
            f"threshold_exceeded={signal.threshold_exceeded}"
        )
        await blocker.handle(signal)

    print()
    print("=" * 64)
    print("  Runtime state after demo")
    print("=" * 64)
    print(f"  Suspended agents:  {sorted(runtime.suspended) or '(none)'}")
    print(f"  Throttled agents:  {dict(sorted(runtime.throttled.items())) or '(none)'}")
    print()
    print("=" * 64)
    print("  Note: this BlockingHandler is the EXACT skeleton a customer")
    print("  copies into their codebase. The Protocol contract is preserved;")
    print("  the body of `handle` is your policy.")
    print("=" * 64)


def _build_monitor_with_blocking(runtime: FakeAgentRuntime) -> FleetMonitor:
    """Reference wiring: BlockingHandler alongside LogHandler."""
    return FleetMonitor(
        operators=(BaselineOperator(name="baseline", min_calibration_size=3),),
        store=InMemoryStore(),
        handlers=(
            LogHandler(name="primary-log"),
            BlockingHandler(name="blocker", runtime=runtime),
        ),
    )


if __name__ == "__main__":
    asyncio.run(_drive_demo())
