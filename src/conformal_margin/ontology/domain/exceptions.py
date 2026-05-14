"""Exception hierarchy for conformal-margin.

Three response categories:
- `validation`: caller passed bad input. Fix the caller.
- `operational`: transient or environmental failure. Caller decides
  retry policy.
- `unexpected`: library invariant violated. Surface and crash; let
  the process manager restart.

Each exception class declares its category as a class attribute.
Boundary handlers dispatch on category without knowing every
specific subclass:

    try:
        await monitor.observe(snapshot)
    except ConformalMarginError as exc:
        match exc.category:
            case ErrorCategory.VALIDATION: ...
            case ErrorCategory.OPERATIONAL: ...
            case ErrorCategory.UNEXPECTED: ...

Always chain on re-raise: `raise NewError(...) from original`.
"""

from typing import Final

from conformal_margin.ontology.domain.ontology import ErrorCategory

__all__ = [
    "ConformalMarginError",
    "ValidationError",
    "UnknownMetricError",
    "OperationFailedError",
    "StoreUnavailableError",
    "SinkRejectionError",
    "OperatorTimeoutError",
    "UnexpectedError",
]


class ConformalMarginError(Exception):
    """Base for all library exceptions.

    Catch this at the outermost boundary to handle any error from the
    library. Inherits from `Exception`, never `BaseException`.

    The default category is `UNEXPECTED` as a safe default: a new
    exception subclass that forgets to declare a category is treated
    as a panic until proven otherwise.
    """

    category: Final[ErrorCategory] = ErrorCategory.UNEXPECTED


class ValidationError(ConformalMarginError):
    """The caller passed input that violates a documented precondition.

    Library does not retry, does not silently coerce, does not
    log-and-continue. Public docstrings list the ValidationError
    conditions they can raise.

    Attributes:
        field: name of the offending input (for example `agent_id`).
        reason: short human-readable description of the violation.
        value: the offending value if safe to include in error
            output; `None` if the value is sensitive or unavailable.
    """

    category: Final[ErrorCategory] = ErrorCategory.VALIDATION

    def __init__(
        self,
        field: str,
        reason: str,
        *,
        value: object | None = None,
    ) -> None:
        self.field = field
        self.reason = reason
        self.value = value
        super().__init__(f"invalid {field}: {reason}")


class UnknownMetricError(ValidationError):
    """Requested metric name is not registered with this Store implementation.

    Raised by any `StateStore.query_trend(metric, window)` call when
    the `metric` string is not in the Store implementation's
    documented registry. The library does not standardize metric
    names across Stores (per the `query_trend` Protocol docstring);
    each implementation publishes its own.

    Inherits from `ValidationError` so that boundary handlers
    catching the parent class keep working unchanged. The
    specialization adds the `available` attribute so the caller can
    surface the recognized set without parsing the reason string.

    Attributes:
        metric: the unrecognized metric name passed by the caller.
        available: tuple of metric names this Store recognizes.
            Ordering is implementation-defined but stable within a
            single Store instance.
    """

    def __init__(self, metric: str, available: tuple[str, ...]) -> None:
        self.metric = metric
        self.available = available
        super().__init__(
            field="metric",
            reason=(
                f"unknown metric {metric!r}; this Store recognizes: " f"{available}"
            ),
            value=metric,
        )


class OperationFailedError(ConformalMarginError):
    """Transient or environmental failure, not caused by caller or library.

    Examples: network partition, downstream service unavailable, race
    condition. Library wraps the underlying cause and surfaces; caller
    decides retry policy via `tenacity` or their own logic.

    Always chain the original cause: `raise OperationFailedError(...) from original`.

    Attributes:
        operation: short name of what was attempted (for example
            `save`, `load_recent`).
        attempt: attempt counter if known to the caller; defaults to 1.
        details: structured details for debugging.
    """

    category: Final[ErrorCategory] = ErrorCategory.OPERATIONAL

    def __init__(
        self,
        operation: str,
        *,
        attempt: int = 1,
        details: dict[str, object] | None = None,
    ) -> None:
        self.operation = operation
        self.attempt = attempt
        self.details = details if details is not None else {}
        super().__init__(f"{operation} failed (attempt {attempt})")


class StoreUnavailableError(OperationFailedError):
    """StateStore implementation unreachable.

    Raised when network, disk, or database backing the Store cannot
    be reached. The customer's HybridStateStore composition (per
    `data_flow_resilience.md`) is the recommended path to handle
    this gracefully.
    """


class SinkRejectionError(OperationFailedError):
    """AlertHandler rejected the signal.

    Raised by a Sink when its downstream system (PagerDuty, SIEM,
    custom webhook) is unavailable or refused the request. The
    customer's BufferedAlertHandler composition (per
    `data_flow_resilience.md`) is the recommended path to handle
    this gracefully.
    """


class OperatorTimeoutError(OperationFailedError):
    """Operator exceeded its configured timeout waiting on a Port.

    Typical cause: history load from a slow Store. The library does
    not retry; the caller decides whether to retry or fail.
    """


class UnexpectedError(ConformalMarginError):
    """Library invariant violated or unexpected condition.

    Library logs a structured event and exits its own frame. Caller's
    outer handler decides process-level response (alert oncall, fail
    fast, restart with backoff).

    Examples: operator produces NaN, internal cache corrupted,
    invariant on OperatorSignal fields violated.

    Attributes:
        context: short description of what was unexpected (for
            example `operator produced NaN score`).
        details: structured details for debugging.
    """

    category: Final[ErrorCategory] = ErrorCategory.UNEXPECTED

    def __init__(
        self,
        context: str,
        *,
        details: dict[str, object] | None = None,
    ) -> None:
        self.context = context
        self.details = details if details is not None else {}
        super().__init__(f"unexpected: {context}")
