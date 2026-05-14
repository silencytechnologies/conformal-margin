"""Concrete Filter implementations gating delivery to AlertHandlers.

Filters are configuration values, not wire-format data, so they use
`@dataclass(frozen=True, slots=True)` rather than `msgspec.Struct`.
This keeps the runtime library free of any serialization expectation
on filter instances; they live in process, not on the wire.

Per the `Filter` Protocol: each filter is a pure synchronous
predicate. No I/O. No raising on ordinary input. Composition by
`FilteredHandler` is conjunctive (AND); a signal reaches the wrapped
`AlertHandler` only when every registered filter returns `True`.

Three filters ship in v0.1:

- `SeverityFilter`: accept iff `signal.severity` is at least the
  configured minimum. The comparison uses the `Severity` ordering
  by `numeric_weight` (not lexicographic - see the `Severity`
  class docstring for the regression context).
- `KindFilter`: accept iff `signal.kind` is in the configured set.
- `DriftEventTypeFilter`: accept iff `signal.drift_event_type` is
  non-None and in the configured set. Anomaly signals (where
  `drift_event_type` is None by invariant) are rejected; this filter
  is "drill into drift subtypes". A customer who wants anomaly plus
  drift composes with `KindFilter` upstream.

`AgentAllowlistFilter` is mentioned in the ontology Filter Protocol
docstring but deferred from v0.1. Reason: the semantics for
fleet-wide signals (empty `affected_agents`) is a design decision
worth its own pass and is tracked in the Decision Queue.

Each filter validates its construction inputs in `__post_init__`
and raises `ValidationError` on violation. The
`role: ClassVar[Role] = Role.SINK` attribute documents the
boundary lane the filter belongs to and satisfies the Protocol's
`role` attribute check at `@runtime_checkable` isinstance time.
"""

from dataclasses import dataclass
from typing import ClassVar

from conformal_margin.ontology.domain.exceptions import ValidationError
from conformal_margin.ontology.domain.ontology import (DriftEventType, Role,
                                                       Severity, SignalKind)
from conformal_margin.ontology.ports.operator_signal import OperatorSignalLike

__all__ = [
    "DriftEventTypeFilter",
    "KindFilter",
    "SeverityFilter",
]


@dataclass(frozen=True, slots=True)
class SeverityFilter:
    """Accept signals at or above a configured minimum severity.

    The comparison uses `Severity.__ge__`, which orders by
    `numeric_weight` rather than the lexicographic ordering that
    `str` would provide. This is the operator that the
    `@total_ordering` decorator failed to install correctly; the
    explicit `__ge__` on `Severity` is what makes this filter work.

    Attributes:
        name: registered name of this filter, used in structured
            logs to identify which filter rejected which signal.
            Non-empty.
        min_severity: signals with `severity` strictly below this
            are rejected; signals at this severity or higher pass.

    Examples:
        >>> from conformal_margin.ontology.domain.ontology import Severity
        >>> f = SeverityFilter(name="high-and-above", min_severity=Severity.HIGH)
        >>> f.role.value
        'sink'
    """

    name: str
    min_severity: Severity

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

    def accepts(self, signal: OperatorSignalLike) -> bool:
        """Return whether the signal's severity meets the minimum.

        Args:
            signal: a finalized OperatorSignalLike under consideration.

        Returns:
            `True` if `signal.severity >= self.min_severity`,
            `False` otherwise.
        """
        return signal.severity >= self.min_severity


@dataclass(frozen=True, slots=True)
class KindFilter:
    """Accept signals whose `kind` is in the configured set.

    The set is frozen on construction: any iterable of `SignalKind`
    is accepted and converted to a `frozenset` in `__post_init__`,
    so the filter cannot be mutated externally by a customer who
    happened to pass in a plain `set`.

    Attributes:
        name: registered name, non-empty.
        kinds: frozenset of `SignalKind` members. Non-empty
            (an empty set would reject every signal, which is
            indistinguishable from a misconfiguration; the
            constructor rejects it).

    Examples:
        >>> from conformal_margin.ontology.domain.ontology import SignalKind
        >>> f = KindFilter(name="drift-only", kinds=frozenset({SignalKind.DRIFT}))
        >>> f.role.value
        'sink'
    """

    name: str
    kinds: frozenset[SignalKind]

    role: ClassVar[Role] = Role.SINK

    def __post_init__(self) -> None:
        """Validate name, freeze the kinds set, and reject empty configuration.

        Raises:
            ValidationError: name is empty, or `kinds` is empty.
        """
        if not self.name:
            raise ValidationError(
                field="name",
                reason="must be a non-empty string",
                value=self.name,
            )
        frozen_kinds = frozenset(self.kinds)
        if not frozen_kinds:
            raise ValidationError(
                field="kinds",
                reason="must be a non-empty set of SignalKind members",
                value=self.kinds,
            )
        object.__setattr__(self, "kinds", frozen_kinds)

    def accepts(self, signal: OperatorSignalLike) -> bool:
        """Return whether the signal's kind is in the allowed set.

        Args:
            signal: a finalized OperatorSignalLike under consideration.

        Returns:
            `True` if `signal.kind in self.kinds`, `False` otherwise.
        """
        return signal.kind in self.kinds


@dataclass(frozen=True, slots=True)
class DriftEventTypeFilter:
    """Accept drift signals whose `drift_event_type` is in the configured set.

    Designed to drill into drift subtypes. Anomaly signals - where
    `drift_event_type` is None by the `OperatorSignal` invariant -
    are rejected unconditionally; a customer who wants "all
    anomaly signals plus some drift subtypes" composes this filter
    with a `KindFilter` upstream or downstream.

    Attributes:
        name: registered name, non-empty.
        types: frozenset of `DriftEventType` members. Non-empty.

    Examples:
        >>> from conformal_margin.ontology.domain.ontology import DriftEventType
        >>> f = DriftEventTypeFilter(
        ...     name="mean-only",
        ...     types=frozenset({DriftEventType.MEAN_SHIFT}),
        ... )
        >>> f.role.value
        'sink'
    """

    name: str
    types: frozenset[DriftEventType]

    role: ClassVar[Role] = Role.SINK

    def __post_init__(self) -> None:
        """Validate name and types, freeze the set.

        Raises:
            ValidationError: name is empty, or `types` is empty.
        """
        if not self.name:
            raise ValidationError(
                field="name",
                reason="must be a non-empty string",
                value=self.name,
            )
        frozen_types = frozenset(self.types)
        if not frozen_types:
            raise ValidationError(
                field="types",
                reason="must be a non-empty set of DriftEventType members",
                value=self.types,
            )
        object.__setattr__(self, "types", frozen_types)

    def accepts(self, signal: OperatorSignalLike) -> bool:
        """Return whether the signal carries an allowed drift event type.

        Anomaly signals (where `drift_event_type` is None) are
        rejected; this filter only passes drift signals whose
        subtype is in the configured set.

        Args:
            signal: a finalized OperatorSignalLike under consideration.

        Returns:
            `True` if `signal.drift_event_type is not None and
            signal.drift_event_type in self.types`, `False`
            otherwise.
        """
        return (
            signal.drift_event_type is not None
            and signal.drift_event_type in self.types
        )
