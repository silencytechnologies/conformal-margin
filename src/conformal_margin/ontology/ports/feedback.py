"""FeedbackPort Protocol - audit-only feedback ingestion in v0.1.

A FeedbackPort accepts labeled judgments about previously emitted
signals - typically "this was a false positive" or "this was a true
positive" - and persists them for later analysis. In v0.1 the
feedback path is intentionally audit-only: labels are written to an
append-only log and never feed back into the calibrator at runtime.

Why audit-only in v0.1 (per in the Decision Queue): allowing
customer-supplied labels to mutate the calibration set at runtime
opens an adversarial poisoning surface.
An attacker who can influence the feedback channel can
shift thresholds and degrade the formal FPR guarantee that motivates
the whole library. Until we have a defended ingestion path
(provenance, rate limiting, calibrated-trust weighting), the safe
default is to record judgments without acting on them.

The FeedbackPort sits in the Sink role lane in the sense that it is
a terminal consumer of paired (signal, label) values. It is
distinguished from `AlertHandler` because its input is a labeled
pair, not a raw signal, and because its purpose is durable record
keeping rather than alert delivery.

Concrete implementation in v0.1: `AuditFeedbackHandler` writes
records to an append-only log; nothing reads the log back into the
calibrator. v0.2+ work is tracked in .
"""

from typing import Final, Protocol, runtime_checkable

from conformal_margin.ontology.domain.ontology import Role
from conformal_margin.ontology.ports.operator_signal import OperatorSignalLike

__all__ = ["FeedbackPort", "FeedbackLabel"]


class FeedbackLabel(Protocol):
    """Customer judgment paired with a previously emitted signal.

    Minimal shape: which signal, what verdict, and a free-form
    metadata bag for customer context (reviewer identity, ticket
    URL, notes). The library does not branch on label content in
    v0.1; the FeedbackPort treats the value as an opaque record
    to persist.

    Attributes:
        signal_id: identifier of the signal this label refers to.
            Matches `OperatorSignalLike.signal_id`.
        is_true_positive: customer verdict. `True` if the signal
            corresponded to a real condition; `False` if it was a
            false positive.
        metadata: customer-extensible context bag. Must be
            JavaScript Object Notation serializable.
    """

    signal_id: str
    is_true_positive: bool
    metadata: "object"


@runtime_checkable
class FeedbackPort(Protocol):
    """Append-only sink for labeled judgments about prior signals.

        Per : v0.1 implementations
        persist labels for audit and offline analysis only. They must not
        expose a path that lets the persisted labels mutate runtime
        calibration. A future v0.2+ implementation may add a defended
        ingestion path, gated by separate Decision Queue work.
    async via anyio.

        Attributes:
            role: always `Role.SINK` (documentation; FeedbackPort lives
                in the Sink lane alongside `AlertHandler`).
            name: registered name of this feedback port, used in
                structured logs.
    """

    role: Final = Role.SINK
    name: str

    async def record(
        self,
        signal: OperatorSignalLike,
        label: FeedbackLabel,
    ) -> None:
        """Persist a (signal, label) pair to the audit log.

        Args:
            signal: the previously emitted signal the label refers
                to. Passed by reference; the FeedbackPort may
                serialize whichever fields it needs.
            label: customer verdict and metadata.

        Raises:
            ValidationError: `label.signal_id` does not match
                `signal.signal_id`, or the label metadata is not
                JavaScript Object Notation serializable.
            StoreUnavailableError: the backing audit log is
                unreachable.
        """
        ...
