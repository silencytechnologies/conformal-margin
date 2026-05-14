"""AlertHandler Protocol - the Sink role contract.

Terminal consumer of finalized OperatorSignals. The library dispatches
signals to one or more AlertHandlers; each handler decides what to do
with the signal in its own lane (paging, triage queue, dashboard,
audit log - the four customer alert consumption patterns).

Concrete implementations: `LogHandler` (library default in v0.1),
`AuditFeedbackHandler` (writes finalized signals to the feedback
audit log), and
customer-supplied handlers wrapping PagerDuty, Slack, OpenSearch,
Splunk, or any other external sink.

The `role: Final = Role.SINK` class attribute is informational
(documents the role, Semgrep hook for boundary checks).

See `ontology/01_roles.md` Sink section, `customer_alert_consumption.md`
for the four severity-driven consumption patterns, and
`control_layers_design.md` for the Sink lane in the control plane.
"""

from typing import Final, Protocol, runtime_checkable

from conformal_margin.ontology.domain.ontology import Role
from conformal_margin.ontology.ports.operator_signal import OperatorSignalLike

__all__ = ["AlertHandler"]


@runtime_checkable
class AlertHandler(Protocol):
    """Terminal sink for finalized OperatorSignal values.

        AlertHandlers receive signals after the FleetMonitor has injected
        identity (signal_id, timestamp) and after the Filter lane has
        decided the signal is worth delivering. They are the last library
        boundary before the signal reaches an external system or the
        customer's own code.
    async via anyio. The FleetMonitor dispatches to all
        registered handlers concurrently inside an `anyio.TaskGroup`; a
        slow or failing handler does not block the others.
    handlers do not return a value. The library does not
        chain handler output; each handler is a leaf of the dispatch tree.
        Errors from one handler are isolated (logged, surfaced via the
        operational error category) and do not poison sibling handlers.

        Attributes:
            role: always `Role.SINK` (documentation; read by Semgrep
                rules and future chain introspection).
            name: registered name of this handler. Used in structured
                logs to identify which sink processed which signal, and
                in the handler registry. Distinct from `operator_name`
                on the signal itself.
    """

    role: Final = Role.SINK
    name: str

    async def handle(self, signal: OperatorSignalLike) -> None:
        """Consume a finalized signal.

        The handler decides what to do based on the signal's content
        (typically routing on `severity`). The library does not
        prescribe behavior; built-in handlers in the
        `conformal_margin` library each document their own routing.

        Args:
            signal: a fully-finalized `OperatorSignalLike`. Identity
                fields (`signal_id`, `timestamp`) are populated;
                severity and threshold_exceeded reflect the final
                routing decision; the metadata mapping is whatever
                the operator and FleetMonitor placed there.

        Raises:
            SinkRejectionError: the handler refused this signal for
                a documented reason (for example, downstream queue
                is full, external service returned 4xx). Caught by
                the FleetMonitor and logged; sibling handlers are
                unaffected.
            StoreUnavailableError: handler depends on a Store
                (for example, `AuditFeedbackHandler` writing to
                feedback log) and the Store is unreachable.
            UnexpectedError: library invariant violated inside the
                handler.
        """
        ...
