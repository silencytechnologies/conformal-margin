"""Filter Protocol - synchronous predicate in the Sink lane.

A Filter decides whether a given OperatorSignal should be delivered
to its wrapped AlertHandler. Filters compose: `FilteredHandler` in
the `conformal_margin` library wraps an `AlertHandler` with an
ordered tuple of `Filter` predicates; the handler is only invoked
when every filter returns `True`.

Filters are deliberately synchronous. They are pure
predicates over a single signal value, not I/O steps. Synchronous
predicates compose without coloring the calling context and make
the dispatch path easier to reason about. Any side-effectful
behavior (rate-limit counters, deduplication caches) is the
responsibility of the wrapped handler or a dedicated upstream stage,
not the Filter.

The Filter is part of the Sink role's surface even though it is not
itself a terminal sink. It earns its `role: Final = Role.SINK`
attribute because it participates in the same lane and lives in the
same dispatch path as the AlertHandler it gates.

Concrete implementations: `SeverityFilter`, `DriftEventTypeFilter`,
`AgentAllowlistFilter` in the `conformal_margin` library; customers
may bring their own.
"""

from typing import Final, Protocol, runtime_checkable

from conformal_margin.ontology.domain.ontology import Role
from conformal_margin.ontology.ports.operator_signal import OperatorSignalLike

__all__ = ["Filter"]


@runtime_checkable
class Filter(Protocol):
    """Pure synchronous predicate gating delivery to an AlertHandler.
    filters are `def`, not `async def`. They must not
        perform I/O, must not raise on ordinary input, and must be
        referentially transparent over the signal value. A filter that
        needs to consult external state belongs in the handler layer.
    filter composition is conjunctive (logical AND).
        `FilteredHandler` invokes the wrapped `AlertHandler` only when
        every registered filter returns `True`. Order is preserved for
        diagnostic logging but does not affect the boolean result.

        Attributes:
            role: always `Role.SINK` (documentation; the Filter lives in
                the Sink lane).
            name: registered name of this filter, used in structured
                logs to identify which filter rejected which signal.
    """

    role: Final = Role.SINK
    name: str

    def accepts(self, signal: OperatorSignalLike) -> bool:
        """Return whether this filter allows the signal to proceed.

        Args:
            signal: the finalized `OperatorSignalLike` being
                considered for delivery.

        Returns:
            `True` to allow delivery to the wrapped handler;
            `False` to drop the signal at this stage. Filters do
            not raise on ordinary input - a malformed signal is a
            library invariant violation handled upstream, not a
            filter concern.
        """
        ...
