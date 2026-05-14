"""OperatorSignal - concrete implementation of OperatorSignalLike.

The canonical wire-format type for every signal the library emits.
Frozen `msgspec.Struct` satisfying the `OperatorSignalLike` Protocol
from `conformal_margin.ontology.ports.operator_signal`.

Eleven fields total, grouped into five families per the Protocol:

1. Identity: `signal_id`, `operator_name`, `timestamp`.
2. Category: `kind`.
3. Routing: `severity`, `threshold_exceeded`.
4. Measurement: `score`, `p_value`.
5. Scope: `affected_agents`.
6. Drift detail: `drift_event_type`.
7. Extensibility: `metadata`.

`__post_init__` enforces the seven validation invariants`:

1. `signal_id` is a canonical lowercase UUID with hyphens (no
   braces, no `urn:uuid:` prefix, no uppercase). Verified by
   parsing with `uuid.UUID` and checking the canonical roundtrip
   `str(parsed) == signal_id`.
2. `operator_name` is a non-empty string.
3. `timestamp` is timezone-aware.
4. `score` is a finite float (no NaN, no positive or negative
   infinity).
5. `p_value` is in the closed interval `[0.0, 1.0]`. Boundaries
   are inclusive because formal conformal calibration can produce
   p-values of exactly zero or exactly one.
6. `drift_event_type` is set if and only if `kind` is
   `SignalKind.DRIFT`. Both directions raise `ValidationError`:
   a drift signal without a drift_event_type, and a non-drift
   signal with a drift_event_type set.
7. `threshold_exceeded=False` implies `severity=Severity.INFO`.
   The reverse direction is intentionally not enforced because an
   operator may legitimately cross the alpha threshold and still
   report `Severity.INFO` for audit-only signals.

Library invariants beyond the bare Protocol that we deliberately
do NOT enforce here:

- JSON-serializability of `metadata`: validating at construction
  would require deep traversal of the mapping; `msgspec.json`
  raises at encode time if any value is unserializable, so the
  failure surfaces at the right boundary.
- Range of `score`: the Protocol explicitly says "No fixed range;
  depends on the operator's scoring method".
- Non-empty `affected_agents`: the Protocol explicitly allows the
  empty tuple for fleet-wide signals.

See `conformal_margin.ontology.ports.operator_signal` for the
abstract Protocol contract.
"""

import math
import uuid
from collections.abc import Mapping
from datetime import datetime
from typing import Any

import msgspec

from conformal_margin.ontology.domain.exceptions import ValidationError
from conformal_margin.ontology.domain.ontology import (DriftEventType,
                                                       Severity, SignalKind)

__all__ = ["OperatorSignal"]


class OperatorSignal(
    msgspec.Struct,
    frozen=True,
    gc=False,
    forbid_unknown_fields=True,
):
    """A finalized signal ready for dispatch to AlertHandlers.

    Concrete frozen `msgspec.Struct` satisfying the
    `OperatorSignalLike` Protocol. Operators return draft signals;
    the FleetMonitor finalizes a draft by injecting `signal_id`
    (canonical UUID) and `timestamp` before construction. Once
    constructed, the signal flows through the Filter lane to one or
    more AlertHandlers; sinks never see partial signals because the
    `__post_init__` invariants reject any inconsistent draft.

    Field semantics mirror the `OperatorSignalLike` Protocol; this
    docstring documents only contracts the concrete type adds:

    - Frozen: attributes cannot be reassigned after construction.
    - Hashable: yes, except when `metadata` is a non-hashable
      Mapping (which is the typical case for plain dicts). The
      library itself never hashes signals; customers building
      deduplication caches pass an immutable hashable Mapping.
    - JSON encoding: `msgspec.json.Encoder()`; unknown fields at
      decode time are rejected (`forbid_unknown_fields=True`).

    Attributes:
        signal_id: canonical UUID string (lowercase, hyphenated,
            no braces, no urn prefix).
        operator_name: non-empty registered name of the emitting
            operator.
        timestamp: tz-aware finalization time set by the
            FleetMonitor.
        kind: SignalKind member (anomaly or drift in v0.1).
        severity: routing key for downstream sinks.
        score: raw robust score; finite. No fixed range.
        p_value: calibrated p-value in `[0.0, 1.0]` inclusive.
        threshold_exceeded: whether the alpha threshold was
            crossed. When False, severity must be Severity.INFO.
        affected_agents: tuple of agent identifiers. Empty for
            fleet-wide signals.
        drift_event_type: DriftEventType member when kind is DRIFT,
            else None.
        metadata: customer-extensible JSON-serializable Mapping.

    Examples:
        >>> from datetime import UTC, datetime
        >>> signal = OperatorSignal(
        ...     signal_id="550e8400-e29b-41d4-a716-446655440000",
        ...     operator_name="baseline",
        ...     timestamp=datetime(2026, 5, 12, 9, 5, tzinfo=UTC),
        ...     kind=SignalKind.ANOMALY,
        ...     severity=Severity.HIGH,
        ...     score=4.7,
        ...     p_value=0.003,
        ...     threshold_exceeded=True,
        ...     affected_agents=("agent-7",),
        ...     drift_event_type=None,
        ...     metadata={"calibration_window_size": 1024},
        ... )
        >>> signal.severity > Severity.LOW
        True
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
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Enforce the seven validation invariants documented above.

        Raises `ValidationError` on the first violation. Field order
        of checks is identity, then independent-field, then
        cross-field invariants; this gives the most informative
        error when a draft signal has multiple problems (the
        identity error surfaces first rather than a cross-field
        error that depends on a field that itself is invalid).

        Raises:
            ValidationError: any of the seven invariants is violated.
        """
        try:
            parsed = uuid.UUID(self.signal_id)
        except (ValueError, AttributeError, TypeError) as exc:
            raise ValidationError(
                field="signal_id",
                reason="must be a valid UUID string",
                value=self.signal_id,
            ) from exc
        if str(parsed) != self.signal_id:
            raise ValidationError(
                field="signal_id",
                reason="must be in canonical lowercase form with hyphens (no braces, no urn prefix, no uppercase)",
                value=self.signal_id,
            )

        if not self.operator_name:
            raise ValidationError(
                field="operator_name",
                reason="must be a non-empty string",
                value=self.operator_name,
            )

        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValidationError(
                field="timestamp",
                reason="must be timezone-aware (a naive datetime is forbidden)",
                value=self.timestamp,
            )

        if not math.isfinite(self.score):
            raise ValidationError(
                field="score",
                reason="must be a finite float (no NaN, no positive or negative infinity)",
                value=self.score,
            )

        if not (0.0 <= self.p_value <= 1.0):
            raise ValidationError(
                field="p_value",
                reason="must be in the closed interval [0.0, 1.0]",
                value=self.p_value,
            )

        if self.kind is SignalKind.DRIFT and self.drift_event_type is None:
            raise ValidationError(
                field="drift_event_type",
                reason="must be set when kind is SignalKind.DRIFT",
                value=None,
            )
        if self.kind is not SignalKind.DRIFT and self.drift_event_type is not None:
            raise ValidationError(
                field="drift_event_type",
                reason="must be None when kind is not SignalKind.DRIFT",
                value=self.drift_event_type,
            )

        if not self.threshold_exceeded and self.severity is not Severity.INFO:
            raise ValidationError(
                field="severity",
                reason="must be Severity.INFO when threshold_exceeded is False",
                value=self.severity,
            )
