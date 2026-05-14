"""OperatorSignalLike Protocol - data shape of every signal the library emits.

A signal flows from Operator (Transformer role) through the
FleetMonitor to AlertHandler (Sink role). Concrete implementations
in the `conformal_margin` library are msgspec.Struct frozen for
serialization and audit fidelity.

Eleven fields total, grouped into five families (identity, category,
measurement, scope, extensibility).

Data shape Protocol - no `role` attribute. Signals are the payload
that flows between roles, not a role themselves.
"""

from collections.abc import Mapping
from datetime import datetime
from typing import Protocol, runtime_checkable

from conformal_margin.ontology.domain.ontology import (DriftEventType,
                                                       Severity, SignalKind)

__all__ = ["OperatorSignalLike"]


@runtime_checkable
class OperatorSignalLike(Protocol):
    """The eleven-field shape every emitted signal must satisfy.

    Concrete implementations enforce the validation invariants from
    `ontology/06b_operator_signal_invariants.md` at construction
    (drift_event_type set iff kind == DRIFT, timezone-aware
    timestamp, p_value in [0, 1], signal_id is canonical Universally
    Unique Identifier, etc.).

    Attributes:
        signal_id: Universally Unique Identifier in canonical form
            (for example `"550e8400-e29b-41d4-a716-446655440000"`).
            Library-generated at FleetMonitor finalization; operators
            do not produce identity.
        operator_name: registered name of the emitting operator.
            Matches `Operator.name` attribute.
        timestamp: when the signal was finalized by the FleetMonitor.
            Timezone-aware.
        kind: SignalKind member (`anomaly` or `drift` in v0.1).
        severity: routing key for downstream sinks.
        score: raw robust score from the scorer layer. No fixed range;
            depends on the operator's scoring method.
        p_value: calibrated p-value carrying the formal False Positive
            Rate guarantee. Closed range `[0.0, 1.0]`.
        threshold_exceeded: did the customer-configured alpha
            threshold get crossed? When `False`, severity defaults to
            `Severity.INFO`.
        affected_agents: tuple of agent identifiers this signal
            pertains to. Tuple, not list, for immutability. Can be
            empty for fleet-wide signals.
        drift_event_type: set if and only if `kind == SignalKind.DRIFT`.
            Names the statistical subtype of the drift.
        metadata: customer-extensible context. Library pass-through;
            must be JavaScript Object Notation serializable.
    """

    signal_id: str
    operator_name: str
    timestamp: datetime
    kind: SignalKind
    severity: Severity
    score: float
    p_value: float
    threshold_exceeded: bool
    affected_agents: tuple[str, ...]
    drift_event_type: DriftEventType | None
    metadata: Mapping[str, object]
