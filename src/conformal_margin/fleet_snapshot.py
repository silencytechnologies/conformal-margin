"""FleetSnapshot - concrete implementation of FleetSnapshotLike.

A frozen msgspec.Struct representing one observation moment of the
whole fleet: a tuple of `AgentState` values plus a single fleet
timestamp. Satisfies the `FleetSnapshotLike` Protocol declared in
`conformal_margin.ontology.ports.fleet_snapshot`.

Construction-time validation in `__post_init__` enforces:

- `timestamp`: timezone-aware datetime (the fleet observation moment).
- `agents`: an iterable of unique `agent_id` values. Duplicate
  identifiers within a single snapshot raise `ValidationError`. This
  invariant goes beyond the bare Protocol (which is silent on
  uniqueness) because library operators aggregate over the
  `affected_agents` field of emitted signals; counting the same
  agent twice within one observation moment would silently corrupt
  every downstream aggregation.

Type-level note: `agents` is `tuple[AgentState, ...]` (the concrete
library struct) rather than `tuple[AgentStateLike, ...]` (the
ontology Protocol). msgspec requires concrete types at JSON decode
time; it cannot decode into a Protocol. `FleetSnapshot` still
satisfies the abstract `FleetSnapshotLike` Protocol at the
`@runtime_checkable` boundary because Protocol checks are
structural (attribute presence) rather than type-deep.

Snapshot timestamp vs agent timestamps: independent. Per the
Protocol docstring, the fleet timestamp is when the customer chose
to observe, while each agent carries its own measurement moment.
They may differ, and the library does not relate them.

See `conformal_margin.ontology.ports.fleet_snapshot` for the
abstract Protocol contract.
"""

from datetime import datetime

import msgspec

from conformal_margin.agent_state import AgentState
from conformal_margin.ontology.domain.exceptions import ValidationError

__all__ = ["FleetSnapshot"]


class FleetSnapshot(
    msgspec.Struct,
    frozen=True,
    gc=False,
    forbid_unknown_fields=True,
):
    """Immutable observation of the fleet at one moment.

    Concrete frozen `msgspec.Struct` satisfying the
    `FleetSnapshotLike` Protocol. The library passes
    `FleetSnapshot` values into operators via
    `Operator.run(snapshot, history)`.

    Attributes:
        timestamp: when the customer chose to observe the fleet.
            Timezone-aware.
        agents: tuple of pre-validated `AgentState` values. May be
            empty (an empty fleet is legal; the operator emits no
            signals). Every `agent_id` within the tuple must be
            unique; duplicates raise `ValidationError` at
            construction.

    Examples:
        >>> from datetime import UTC, datetime
        >>> from conformal_margin.agent_state import AgentState
        >>> from conformal_margin.ontology.domain.ontology import ModelTier
        >>> agent = AgentState(
        ...     agent_id="agent-1",
        ...     timestamp=datetime(2026, 5, 12, 9, 0, tzinfo=UTC),
        ...     risk_score=0.2,
        ...     model_tier=ModelTier.FOUNDATION,
        ...     metadata={},
        ... )
        >>> snapshot = FleetSnapshot(
        ...     timestamp=datetime(2026, 5, 12, 9, 5, tzinfo=UTC),
        ...     agents=(agent,),
        ... )
        >>> len(snapshot.agents)
        1
    """

    timestamp: datetime
    agents: tuple[AgentState, ...]

    def __post_init__(self) -> None:
        """Enforce timezone-aware timestamp and unique agent_ids.

        Raises:
            ValidationError: `timestamp` is naive, or two agents in
                `agents` share the same `agent_id`. The `value`
                field of the exception carries the duplicated id so
                the boundary handler can log which agent collided.
        """
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValidationError(
                field="timestamp",
                reason="must be timezone-aware (a naive datetime is forbidden)",
                value=self.timestamp,
            )
        seen: set[str] = set()
        for agent in self.agents:
            if agent.agent_id in seen:
                raise ValidationError(
                    field="agents",
                    reason="duplicate agent_id; each agent must appear at most once per snapshot",
                    value=agent.agent_id,
                )
            seen.add(agent.agent_id)
