"""LogHandler - structlog-backed `AlertHandler` Protocol implementation.

The simplest concrete sink the library ships. Each finalized
`OperatorSignal` is rendered as one structured log event whose
log level is driven by the signal's `severity`.

Design decisions backing this file:

- Sole logging library: structlog. Event name follows the
  `entity_action_outcome` convention; the event emitted by
  this handler is `signal_handled`.
- **Severity to log level mapping** (hardcoded in v0.1;
  customizable in a later version if a customer needs it):

      Severity.INFO     -> info
      Severity.LOW      -> info       (5 severities collapse onto
                                       4 standard log levels)
      Severity.MEDIUM   -> warning
      Severity.HIGH     -> error
      Severity.CRITICAL -> critical

- **Filter lane vs handler**: LogHandler does not take a
  `min_severity` parameter. Customers who want to suppress some
  signals compose a `SeverityFilter` upstream via the
  `FilteredHandler` wrapper (which lands later in this folder).
- **Bind per call**: the handler's registered `name` is bound onto
  the logger inside `handle()` rather than stored on the instance.
  This keeps the dataclass truly frozen (no mutable
  `_logger` slot) and the perf cost is microseconds per call.
- **Async semantics**: `handle()` is declared `async def` because
  the `AlertHandler` Protocol requires it. The body is
  synchronous because structlog's BoundLogger is fast-sync. A
  customer who needs non-blocking log emission configures
  structlog with non-blocking processors at application startup;
  the handler itself does not wrap calls in `anyio.to_thread`.
- **Errors**: structlog rarely fails in practice. v0.1 lets any
  underlying exception propagate as-is; a customer wrapping in
  their own error policy can catch and translate to
  `SinkRejectionError`. Tracked for v0.2+ if a real failure mode
  surfaces.

Structlog must be configured at application startup (renderer,
processors, log level threshold). LogHandler does not configure
structlog; it only emits events.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar

import structlog

from conformal_margin.ontology.domain.exceptions import ValidationError
from conformal_margin.ontology.domain.ontology import Role, Severity
from conformal_margin.ontology.ports.operator_signal import OperatorSignalLike

__all__ = ["LogHandler"]

_SEVERITY_TO_LEVEL: Mapping[Severity, str] = {
    Severity.INFO: "info",
    Severity.LOW: "info",
    Severity.MEDIUM: "warning",
    Severity.HIGH: "error",
    Severity.CRITICAL: "critical",
}


@dataclass(frozen=True, slots=True)
class LogHandler:
    """Emit each signal as one structlog event with severity-driven level.

    Attributes:
        name: registered handler name. Bound onto the log record as
            `handler=<name>` so downstream log aggregation can
            filter by which sink processed the signal. Non-empty.

    Examples:
        >>> handler = LogHandler(name="default-log")
        >>> handler.role.value
        'sink'
    """

    name: str

    role: ClassVar[Role] = Role.SINK

    def __post_init__(self) -> None:
        """Validate that the registered name is non-empty.

        Raises:
            ValidationError: name is empty.
        """
        if not self.name:
            raise ValidationError(
                field="name",
                reason="must be a non-empty string",
                value=self.name,
            )

    async def handle(self, signal: OperatorSignalLike) -> None:
        """Emit the signal as a single structured log event.

        The log level is selected from the signal's `severity` via
        the hardcoded mapping documented in the module docstring.
        Every field of the signal is bound onto the log record as
        a structured key; the record's event name is
        `signal_handled`.

        Args:
            signal: a finalized `OperatorSignalLike`. The library
                guarantees all eleven invariants hold by the time
                the signal reaches a sink.
        """
        logger = structlog.get_logger().bind(handler=self.name)
        level_name = _SEVERITY_TO_LEVEL[signal.severity]
        log_method = getattr(logger, level_name)
        log_method(
            "signal_handled",
            signal_id=signal.signal_id,
            operator_name=signal.operator_name,
            timestamp=signal.timestamp.isoformat(),
            kind=signal.kind.value,
            severity=signal.severity.value,
            score=signal.score,
            p_value=signal.p_value,
            threshold_exceeded=signal.threshold_exceeded,
            affected_agents=signal.affected_agents,
            drift_event_type=(
                signal.drift_event_type.value
                if signal.drift_event_type is not None
                else None
            ),
            metadata=signal.metadata,
        )
