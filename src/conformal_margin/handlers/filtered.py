"""FilteredHandler - wraps an `AlertHandler` behind a chain of `Filter`s.

The bridge between the `filters.py` predicates and the Sink lane.
A FilteredHandler is itself an `AlertHandler` (so it composes
recursively if a customer wants chained filtering), but delegates
to a wrapped handler only after every registered filter accepts.

Design decisions backing this file:

- Filter composition is conjunctive (logical AND). The
  wrapped handler is invoked iff every filter returns `True`.
- Filter ordering is preserved for diagnostic logging
  but does not affect the boolean outcome. Evaluation is
  short-circuited at the first rejecting filter so later filters
  do not run; the rejecting filter is what gets logged.
- Filters are pure synchronous predicates; FilteredHandler
  calls `.accepts(signal)` (not `await`). Async I/O work belongs
  in the wrapped handler.
- **Empty filter tuple is legal**: vacuous AND is `True`, every
  signal passes. A customer may construct
  `FilteredHandler(name=..., wrapped=h, filters=())` as a
  placeholder while building up a filter chain, or to keep the
  wrapper as a named diagnostic boundary even without filtering.

Rejection logging: every filter rejection emits one structured
log event `signal_filtered_out` at `info` level (rejection is a
normal flow, not an error). The event carries the FilteredHandler
name, the wrapped handler name, the rejecting filter name, and
identity fields of the signal. Customers see in their structured
logs exactly which filter dropped which signal.
"""

from dataclasses import dataclass
from typing import ClassVar

import structlog

from conformal_margin.ontology.domain.exceptions import ValidationError
from conformal_margin.ontology.domain.ontology import Role
from conformal_margin.ontology.ports.alert_handler import AlertHandler
from conformal_margin.ontology.ports.filter import Filter
from conformal_margin.ontology.ports.operator_signal import OperatorSignalLike

__all__ = ["FilteredHandler"]


@dataclass(frozen=True, slots=True)
class FilteredHandler:
    """`AlertHandler` wrapper gating delivery through an ordered Filter chain.

    Itself satisfies the `AlertHandler` Protocol, so chaining two
    FilteredHandlers (one inside another's `wrapped` slot) is legal
    and produces a longer effective filter chain.

    Attributes:
        name: registered name of this wrapper, used in structured
            logs. Distinct from `wrapped.name`; both appear in the
            rejection log so the customer can trace either layer.
            Non-empty.
        wrapped: the underlying `AlertHandler` that receives any
            signal which passes every filter. The Protocol is the
            contract, so any concrete `LogHandler`,
            `AuditFeedbackHandler`, or customer-supplied handler
            slots in here.
        filters: tuple of `Filter` predicates evaluated in order.
            Any iterable of filters is accepted and converted to a
            tuple in `__post_init__` so external mutation of the
            input cannot change behavior after construction. May
            be empty.

    Examples:
        >>> from conformal_margin.handlers.log import LogHandler
        >>> from conformal_margin.filters import SeverityFilter
        >>> from conformal_margin.ontology.domain.ontology import Severity
        >>> wrapped = LogHandler(name="default-log")
        >>> filtered = FilteredHandler(
        ...     name="high-and-above-log",
        ...     wrapped=wrapped,
        ...     filters=(SeverityFilter(name="high+", min_severity=Severity.HIGH),),
        ... )
        >>> filtered.role.value
        'sink'
    """

    name: str
    wrapped: AlertHandler
    filters: tuple[Filter, ...]

    role: ClassVar[Role] = Role.SINK

    def __post_init__(self) -> None:
        """Validate name and freeze the filters iterable into a tuple.

        Raises:
            ValidationError: name is empty.
        """
        if not self.name:
            raise ValidationError(
                field="name",
                reason="must be a non-empty string",
                value=self.name,
            )
        object.__setattr__(self, "filters", tuple(self.filters))

    async def handle(self, signal: OperatorSignalLike) -> None:
        """Run the filter chain and delegate to the wrapped handler.

        Short-circuits at the first rejecting filter. The
        rejection emits one structured log event at `info` level
        with `filter_name` of the rejecting filter; later filters
        are not evaluated and the wrapped handler is not called.

        Args:
            signal: a finalized `OperatorSignalLike` from the
                FleetMonitor.
        """
        for filter_ in self.filters:
            if not filter_.accepts(signal):
                structlog.get_logger().info(
                    "signal_filtered_out",
                    handler=self.name,
                    wrapped_handler=self.wrapped.name,
                    filter_name=filter_.name,
                    signal_id=signal.signal_id,
                    severity=signal.severity.value,
                    kind=signal.kind.value,
                )
                return
        await self.wrapped.handle(signal)
