"""AgentState - concrete implementation of AgentStateLike.

A frozen msgspec.Struct representing one agent's observed state at
one moment. Satisfies the `AgentStateLike` Protocol declared in
`conformal_margin.ontology.ports.agent_state`.

Construction-time validation runs in `__post_init__` and enforces
the documented invariants from the Protocol:

- `agent_id`: non-empty string
- `timestamp`: timezone-aware datetime
- `risk_score`: finite float (no NaN, no positive or negative
  infinity); library operators do arithmetic on this value and
  NaN or infinity would silently poison every downstream
  computation
- `model_tier`: msgspec type-checks at JSON decode; Python
  construction trusts the static type
- `metadata`: msgspec type-checks at JSON decode; Python
  construction trusts the static type

Customers construct `AgentState` values at the Source boundary
(from their telemetry source: LangChain, LangGraph, custom
orchestrator) and pass them to the library via `FleetSnapshot`.
The library never mutates an `AgentState` after construction;
`frozen=True` guards against accidental in-place edits.

The msgspec.Struct base buys us schema-driven JSON and MessagePack
encoding, slot-based memory layout, structural equality, and the
`__post_init__` invariant hook. `gc=False` opts out of Python's
cycle collector because `AgentState` cannot form reference cycles.
`forbid_unknown_fields=True` rejects extra fields at JSON decode
to prevent silent schema drift across library versions.
"""

import math
from collections.abc import Mapping
from datetime import datetime
from typing import Any

import msgspec

from conformal_margin.ontology.domain.exceptions import ValidationError
from conformal_margin.ontology.domain.ontology import ModelTier

__all__ = ["AgentState"]


class AgentState(
    msgspec.Struct,
    frozen=True,
    gc=False,
    forbid_unknown_fields=True,
):
    """One agent's observed state at a single timestamp.

    Concrete frozen `msgspec.Struct` that satisfies the
    `AgentStateLike` Protocol. Constructing an invalid `AgentState`
    raises `ValidationError` synchronously from `__post_init__`.

    Field semantics mirror the `AgentStateLike` Protocol exactly;
    consult that Protocol's docstring for per-field meaning. This
    class docstring documents only the additional contracts this
    concrete type adds on top of the Protocol:

    - Frozen: attributes cannot be reassigned after construction.
    - Hashable: yes, except when `metadata` happens to be a
      non-hashable Mapping (a plain `dict`). Customers who need a
      hashable `AgentState` pass a hashable mapping such as a
      `types.MappingProxyType` over a frozen mapping; the library
      itself never hashes `AgentState` values.
    - JSON encoding: handled by `msgspec.json.Encoder()`; the
      schema is the field declaration below.
    - Unknown fields at JSON decode are rejected
      (`forbid_unknown_fields=True`).

    Attributes:
        agent_id: stable identifier for the agent. Non-empty.
        timestamp: when this state was observed. Timezone-aware.
        risk_score: aggregated risk score. Finite float. Customer
            computes from their threat-modeling source; library
            does not prescribe a range.
        model_tier: capability bucket of the underlying LLM.
        metadata: customer-extensible JSON-serializable Mapping.
            Library passes it through unchanged.

    Examples:
        >>> from datetime import UTC, datetime
        >>> from conformal_margin.ontology.domain.ontology import ModelTier
        >>> state = AgentState(
        ...     agent_id="agent-7",
        ...     timestamp=datetime(2026, 5, 12, 9, 0, tzinfo=UTC),
        ...     risk_score=0.42,
        ...     model_tier=ModelTier.FOUNDATION,
        ...     metadata={"region": "eu-west-1"},
        ... )
        >>> state.agent_id
        'agent-7'
        >>> state.risk_score
        0.42
    """

    agent_id: str
    timestamp: datetime
    risk_score: float
    model_tier: ModelTier
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Enforce the boundary invariants documented above.

        Runs once at construction (msgspec.Struct invokes it after
        slot assignment, before the struct is observable to the
        caller). Raises `ValidationError` on the first violation;
        the field carrying the violation is the `field` attribute
        of the exception so boundary handlers can route by field
        without parsing the message.

        Raises:
            ValidationError: any documented invariant is violated.
        """
        if not self.agent_id:
            raise ValidationError(
                field="agent_id",
                reason="must be a non-empty string",
                value=self.agent_id,
            )
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValidationError(
                field="timestamp",
                reason="must be timezone-aware (a naive datetime is forbidden)",
                value=self.timestamp,
            )
        if not math.isfinite(self.risk_score):
            raise ValidationError(
                field="risk_score",
                reason="must be a finite float (no NaN, no positive or negative infinity)",
                value=self.risk_score,
            )
