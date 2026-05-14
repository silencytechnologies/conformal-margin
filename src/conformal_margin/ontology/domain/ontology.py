"""Canonical vocabulary of conformal-margin.

Six closed StrEnum sets every public component references. The set
of members is fixed at this ontology version; growing a member set
requires a minor version bump of the ontology package, renaming
or removing a member requires a major bump.

Pattern across all six enums: StrEnum plus optional `description`
property backed by a Final lookup table. Severity additionally
defines the four rich-comparison operators explicitly (not via
`functools.total_ordering`) because StrEnum's str base class already
provides lexicographic operators that occupy the decorator's slots
and silently win over a single `__lt__` override. The other enums
have no natural ordering and do not implement one.

"""

from enum import StrEnum
from typing import Final

__all__ = [
    "Role",
    "Severity",
    "SignalKind",
    "ErrorCategory",
    "DriftEventType",
    "ModelTier",
]


class Role(StrEnum):
    """The four data flow roles.

    Every public component fits exactly one role. The set is closed
    at this ontology version; growing it requires a major bump.

    See `ontology/01_roles.md` for per-member rationale.

    Examples:
        >>> Role.SOURCE.value
        'source'
        >>> len(list(Role))
        4
    """

    SOURCE = "source"
    TRANSFORMER = "transformer"
    STORE = "store"
    SINK = "sink"


class Severity(StrEnum):
    """Five-tier urgency label carried by every OperatorSignal.

    Members are ordered by `numeric_weight`. Customers compare with
    `<`, `<=`, `>`, `>=`, `==`; the library uses ordering in
    tier-comparison sinks.

    All four rich-comparison operators are defined explicitly. We do
    not use `functools.total_ordering` because StrEnum inherits from
    `str`, which already provides lexicographic versions of all four
    operators. The decorator sees those inherited operators as "already
    defined" and refuses to overwrite them with weight-based versions.
    The result of that subtle interaction would be `Severity.HIGH >
    Severity.LOW == False` (because `'h' < 'l'` lexicographically),
    which is exactly the failure mode we caught at the integration
    boundary the first time we exercised the package via uv sync.

    Operations that are not meaningful and will raise `TypeError` or
    return `NotImplemented`: arithmetic addition (the weights are
    ordinal, not interval) and cross-type comparison
    (`Severity.HIGH > 3` returns NotImplemented).

    See `ontology/02_severity.md` and `02_severity_bounds.md`.

    Examples:
        >>> Severity.HIGH > Severity.LOW
        True
        >>> Severity.CRITICAL.numeric_weight
        4
        >>> sorted([Severity.HIGH, Severity.INFO])[0]
        <Severity.INFO: 'info'>
    """

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def numeric_weight(self) -> int:
        """Integer rank for ordering. INFO is 0, CRITICAL is 4."""
        return _SEVERITY_WEIGHTS[self]

    @property
    def description(self) -> str:
        """Human-readable description of the tier."""
        return _SEVERITY_DESCRIPTIONS[self]

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        return self.numeric_weight < other.numeric_weight

    def __le__(self, other: object) -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        return self.numeric_weight <= other.numeric_weight

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        return self.numeric_weight > other.numeric_weight

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        return self.numeric_weight >= other.numeric_weight


_SEVERITY_WEIGHTS: Final[dict[Severity, int]] = {
    Severity.INFO: 0,
    Severity.LOW: 1,
    Severity.MEDIUM: 2,
    Severity.HIGH: 3,
    Severity.CRITICAL: 4,
}

_SEVERITY_DESCRIPTIONS: Final[dict[Severity, str]] = {
    Severity.INFO: (
        "Informational signal. No action required. Recorded for "
        "audit and trend analysis only."
    ),
    Severity.LOW: (
        "Minor deviation from baseline. Worth noting in periodic "
        "review. No immediate response required."
    ),
    Severity.MEDIUM: (
        "Notable deviation. Reviewable within the working day. "
        "Investigate when convenient."
    ),
    Severity.HIGH: (
        "Significant deviation. Surface to an analyst within the "
        "hour. May warrant scoped investigation."
    ),
    Severity.CRITICAL: (
        "Severe deviation. Immediate response required. Should "
        "trigger paging or equivalent oncall escalation."
    ),
}


class SignalKind(StrEnum):
    """Categorical label for what kind of finding a signal represents.

    Two members in v0.1: anomaly and drift. The set is closed at this
    ontology version. SignalKind is orthogonal to Severity: any kind
    can carry any severity (no kind-severity coupling, mirroring the
    Common Vulnerability Scoring System convention).

    See `ontology/03_signal_kinds.md` and `03_signal_kinds_bounds.md`.

    Examples:
        >>> SignalKind.ANOMALY.value
        'anomaly'
        >>> len(list(SignalKind))
        2
    """

    ANOMALY = "anomaly"
    DRIFT = "drift"

    @property
    def description(self) -> str:
        """One-sentence description of what this kind represents."""
        return _SIGNAL_KIND_DESCRIPTIONS[self]


_SIGNAL_KIND_DESCRIPTIONS: Final[dict[SignalKind, str]] = {
    SignalKind.ANOMALY: (
        "A point or population deviation flagged by a calibrated "
        "anomaly operator. Carries a calibrated p-value and a raw "
        "robust score."
    ),
    SignalKind.DRIFT: (
        "A distribution shift detected over the calibration window. "
        "Carries a drift event type field that names the kind of "
        "shift (mean, variance, distribution)."
    ),
}


class ErrorCategory(StrEnum):
    """Response-policy classification for every library exception.

    Each exception class declares its category as a class attribute.
    Boundary handlers dispatch on category, so callers do not have to
    know every specific subclass to handle a failure correctly.

    See `ontology/04_error_categories.md` and
    `04_error_categories_bounds.md`.

    Examples:
        >>> ErrorCategory.VALIDATION.value
        'validation'
        >>> len(list(ErrorCategory))
        3
    """

    VALIDATION = "validation"
    OPERATIONAL = "operational"
    UNEXPECTED = "unexpected"


class DriftEventType(StrEnum):
    """Statistical subtype of a drift signal.

    Carried on `OperatorSignal` if and only if `kind == SignalKind.DRIFT`.
    Three members in v0.1; `distribution_shift` doubles as the
    catch-all when neither mean nor variance alone explains the
    shift.

    See `ontology/05_drift_event_types.md` and
    `05_drift_event_types_bounds.md`.

    Examples:
        >>> DriftEventType.MEAN_SHIFT.value
        'mean_shift'
        >>> len(list(DriftEventType))
        3
    """

    MEAN_SHIFT = "mean_shift"
    VARIANCE_SHIFT = "variance_shift"
    DISTRIBUTION_SHIFT = "distribution_shift"

    @property
    def description(self) -> str:
        """One-sentence description of the statistical subtype."""
        return _DRIFT_EVENT_DESCRIPTIONS[self]


_DRIFT_EVENT_DESCRIPTIONS: Final[dict[DriftEventType, str]] = {
    DriftEventType.MEAN_SHIFT: (
        "The central tendency of risk scores in the calibration "
        "window has shifted relative to the previously stable "
        "baseline. Average behaviour changed."
    ),
    DriftEventType.VARIANCE_SHIFT: (
        "The spread of risk scores in the calibration window has "
        "shifted: agents are now either more or less consistent "
        "than the previously stable baseline. Average behaviour "
        "may be unchanged."
    ),
    DriftEventType.DISTRIBUTION_SHIFT: (
        "The shape of the score distribution has shifted in a "
        "way not isolated to mean or variance alone. Also serves "
        "as the catch-all when the detector identifies a change "
        "but cannot pin it to a specific moment of the "
        "distribution."
    ),
}


class ModelTier(StrEnum):
    """Capability bucket of the underlying language model.

    Values mirror github.com/silencytechnologies/probabilistic-threat-modeling
    one-to-one. Carried as a field on `AgentStateLike`;
    library operators read it for context but do not branch on it
    in v0.1. No `numeric_weight` because the library does not order
    by ModelTier; customers can compare with equality.

    String comparison (`if tier < ModelTier.FRONTIER:`) raises
    `TypeError` to prevent lexicographic comparison from lying.

    See `ontology/08_model_tier.md`.

    Examples:
        >>> ModelTier.FRONTIER.value
        'frontier'
        >>> len(list(ModelTier))
        3
    """

    SLM_EDGE = "slm_edge"
    FOUNDATION = "foundation"
    FRONTIER = "frontier"

    @property
    def description(self) -> str:
        """One-sentence description of the capability bucket."""
        return _MODEL_TIER_DESCRIPTIONS[self]


_MODEL_TIER_DESCRIPTIONS: Final[dict[ModelTier, str]] = {
    ModelTier.SLM_EDGE: (
        "Small language model deployed at the edge. Under seven "
        "billion parameters. Lower capability, lower attack "
        "surface, often runs in constrained environments."
    ),
    ModelTier.FOUNDATION: (
        "Mid-size foundation model. Seven to seventy billion "
        "parameters. Typical production deployment for enterprise "
        "agentic workflows."
    ),
    ModelTier.FRONTIER: (
        "Frontier-scale model. Very large or Mixture of Experts. "
        "Highest capability, broadest attack surface, often "
        "subject to vendor-specific safety controls."
    ),
}
