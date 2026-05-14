"""AgentStateLike Protocol - data shape for one agent at one moment.

The minimal surface the library reads from any agent state source.
Customers construct values satisfying this Protocol from their agent
runtime (LangChain, LangGraph, custom orchestrator, or any other
software that emits agent telemetry).

The library does not require a specific concrete class. The
`conformal_margin` library ships an `AgentState` msgspec.Struct that
satisfies this Protocol as a default implementation, but customers
may bring their own.

See `ontology/01_roles.md` for Source role context and
`silency_stack_inventory.md` for how this Protocol relates to
`silencytechnologies/probabilistic-threat-modeling`.
"""

from collections.abc import Mapping
from datetime import datetime
from typing import Protocol, runtime_checkable

from conformal_margin.ontology.domain.ontology import ModelTier

__all__ = ["AgentStateLike"]


@runtime_checkable
class AgentStateLike(Protocol):
    """Minimal surface of an agent state value the library reads.

    (tentative resolution): minimal fields plus a free-form
    `metadata` bag for opaque customer context. The library does not
    branch on metadata keys; it passes the mapping through unchanged.

    Data shape Protocol - no `role` attribute. AgentStateLike values
    are the data that flows from the Source role into the library;
    the Source role itself has no Protocol in v0.1.

    Attributes:
        agent_id: stable identifier for the agent. Non-empty string.
        timestamp: when this state was observed. Timezone-aware.
        risk_score: aggregated risk score for this agent. Customer
            computes from their threat-modeling source (for example,
            probabilistic-threat-modeling's StochasticSTRIDE
            aggregated to one number); library does not prescribe.
        model_tier: capability bucket of the underlying language model.
        metadata: customer-extensible context bag. Must be JavaScript
            Object Notation serializable for TaskIQ-future
            compatibility.
    """

    agent_id: str
    timestamp: datetime
    risk_score: float
    model_tier: ModelTier
    metadata: Mapping[str, object]
