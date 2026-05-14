"""FleetSnapshotLike Protocol - data shape for the whole fleet at one moment.

A FleetSnapshot is the unit of input for Operators: a tuple of
AgentStateLike values plus a single timestamp identifying when the
snapshot was captured.

Constructed by customer Source code; consumed by library Operators.
The library ships a concrete `FleetSnapshot` msgspec.Struct in the
`conformal_margin` library that satisfies this Protocol.
"""

from datetime import datetime
from typing import Protocol, runtime_checkable

from conformal_margin.ontology.ports.agent_state import AgentStateLike

__all__ = ["FleetSnapshotLike"]


@runtime_checkable
class FleetSnapshotLike(Protocol):
    """Immutable collection of AgentStateLike values at one timestamp.

    The unit of input passed to `Operator.run(snapshot, history)`.
    Concrete implementations are frozen (msgspec.Struct frozen) and
    do not own the lifecycle of the contained agents - the agents
    are passed by reference.

    Data shape Protocol - no `role` attribute.

    Attributes:
        timestamp: when this snapshot was captured. Timezone-aware.
            Distinct from each agent's own timestamp; the fleet
            timestamp is when the customer chose to observe.
        agents: tuple of AgentStateLike values. Tuple, not list, for
            immutability. Can be empty (an empty fleet is a legal
            snapshot; the operator produces no signals).
    """

    timestamp: datetime
    agents: tuple[AgentStateLike, ...]
