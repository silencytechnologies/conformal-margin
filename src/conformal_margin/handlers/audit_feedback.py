"""AuditFeedbackHandler - append-only JSON Lines audit log for feedback labels.

`FeedbackPort` Protocol implementation. Persists each `(signal,
label)` pair to a file in JSON Lines format. The library does not
expose any read API on persisted labels; this handler is the
write-side of the FeedbackPort only.

Why audit-only in v0.1: allowing customer-supplied labels to
mutate the conformal calibration set at runtime is adversarially
unsafe (calibration-set poisoning, see arXiv 2604.17984). v0.1
records labels for offline analysis; runtime feedback ingestion is
a separate v0.2+ design pass with provenance, rate limiting, and
trust scoring.

Design decisions:

- **Format**: JSON Lines (one JSON object per line, UTF-8 encoded,
  `\\n` line separator on every platform). Opened in append-binary
  mode so the binary write of bytes plus a literal `\\n` byte is
  not converted to `\\r\\n` on Windows.
- **Record schema**: each line is a single JSON object with three
  top-level keys: `recorded_at` (handler-side UTC timestamp at
  write moment), `signal` (the eleven fields of `OperatorSignalLike`
  serialized), `label` (the three Protocol fields of
  `FeedbackLabel`). Nesting under `signal` and `label` lets
  downstream readers parse with a single deserialization step.
- **Path**: strict `pathlib.Path`; customers convert strings
  themselves. Cleaner typing surface than `Path | str`.
- **Parent directory must exist at construction**. The library
  does not create directories; missing parent raises
  `ValidationError` at construction time so the failure is
  immediate and clear, not deferred until the first write.
- **Concurrency**: a single `anyio.Lock` serializes writes within
  one process. Multi-process is not supported; one writer per file.
- **Blocking I/O off-loaded** to a thread via
  `anyio.to_thread.run_sync` so the event loop is not blocked by
  filesystem latency.
- **Errors**:
  - signal_id mismatch -> `ValidationError(field="label.signal_id")`
  - non-JSON-serializable metadata -> `ValidationError(field="metadata")`
    chained from the underlying `msgspec.EncodeError`
  - file I/O failure (permission denied, disk full, parent dir
    disappeared) -> `StoreUnavailableError`
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

import anyio
import msgspec

from conformal_margin.ontology.domain.exceptions import (StoreUnavailableError,
                                                         ValidationError)
from conformal_margin.ontology.domain.ontology import Role
from conformal_margin.ontology.ports.feedback import FeedbackLabel
from conformal_margin.ontology.ports.operator_signal import OperatorSignalLike

__all__ = ["AuditFeedbackHandler"]


@dataclass(frozen=True, slots=True)
class AuditFeedbackHandler:
    """Append `(signal, label)` pairs to a JSON Lines audit log.

    Attributes:
        name: registered name of this handler, used in structured
            logs to identify which sink processed which feedback.
            Non-empty.
        path: filesystem path to the audit log file. The parent
            directory must exist at construction. The file itself
            is created on the first write if it does not exist.

    Examples:
        >>> from pathlib import Path
        >>> handler = AuditFeedbackHandler(
        ...     name="primary-audit",
        ...     path=Path("/var/log/conformal-margin/feedback.jsonl"),
        ... )
        >>> handler.role.value
        'sink'
    """

    name: str
    path: Path

    role: ClassVar[Role] = Role.SINK

    _lock: anyio.Lock = field(default_factory=anyio.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate name and parent-directory existence.

        Raises:
            ValidationError: name is empty, or the parent directory
                of the configured path does not exist.
        """
        if not self.name:
            raise ValidationError(
                field="name",
                reason="must be a non-empty string",
                value=self.name,
            )
        if not isinstance(self.path, Path):
            raise ValidationError(
                field="path",
                reason="must be a pathlib.Path instance",
                value=type(self.path).__name__,
            )
        if not self.path.parent.exists():
            raise ValidationError(
                field="path",
                reason=(
                    f"parent directory does not exist: {self.path.parent}; "
                    "create it before constructing the handler"
                ),
                value=str(self.path),
            )

    async def record(self, signal: OperatorSignalLike, label: FeedbackLabel) -> None:
        """Append one `(signal, label)` record to the audit log.

        Args:
            signal: the previously-emitted signal the label refers to.
            label: the customer verdict, with `signal_id` matching
                `signal.signal_id`.

        Raises:
            ValidationError: `label.signal_id` does not match
                `signal.signal_id`, or either metadata bag is not
                JSON-serializable.
            StoreUnavailableError: file I/O failed (permission
                denied, disk full, parent directory removed).
        """
        if label.signal_id != signal.signal_id:
            raise ValidationError(
                field="label.signal_id",
                reason=(f"does not match signal.signal_id ({signal.signal_id!r})"),
                value=label.signal_id,
            )

        record_dict: dict[str, Any] = {
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "signal": _signal_to_dict(signal),
            "label": _label_to_dict(label),
        }

        try:
            encoded = msgspec.json.encode(record_dict)
        except (msgspec.EncodeError, TypeError) as exc:
            raise ValidationError(
                field="metadata",
                reason=(
                    "signal or label metadata is not "
                    "JavaScript Object Notation serializable"
                ),
                value=None,
            ) from exc

        async with self._lock:
            try:
                await anyio.to_thread.run_sync(self._write_record, encoded)
            except OSError as exc:
                raise StoreUnavailableError(
                    operation="record",
                    details={
                        "reason": "failed to write to audit log",
                        "path": str(self.path),
                    },
                ) from exc

    def _write_record(self, encoded: bytes) -> None:
        """Synchronous append of one JSON Lines record to the file.

        Called via `anyio.to_thread.run_sync` so the event loop is
        not blocked by filesystem latency. Open mode is append-binary
        (`"ab"`) to preserve `\\n` as a single byte on Windows where
        text mode would emit `\\r\\n`.
        """
        with open(self.path, "ab") as f:
            f.write(encoded)
            f.write(b"\n")


def _signal_to_dict(signal: OperatorSignalLike) -> dict[str, Any]:
    """Project an `OperatorSignalLike` onto a plain JSON-serializable dict."""
    return {
        "signal_id": signal.signal_id,
        "operator_name": signal.operator_name,
        "timestamp": signal.timestamp.isoformat(),
        "kind": signal.kind.value,
        "severity": signal.severity.value,
        "score": signal.score,
        "p_value": signal.p_value,
        "threshold_exceeded": signal.threshold_exceeded,
        "affected_agents": list(signal.affected_agents),
        "drift_event_type": (
            signal.drift_event_type.value
            if signal.drift_event_type is not None
            else None
        ),
        "metadata": (
            dict(signal.metadata)
            if isinstance(signal.metadata, Mapping)
            else signal.metadata
        ),
    }


def _label_to_dict(label: FeedbackLabel) -> dict[str, Any]:
    """Project a `FeedbackLabel` onto a plain JSON-serializable dict."""
    return {
        "signal_id": label.signal_id,
        "is_true_positive": label.is_true_positive,
        "metadata": (
            dict(label.metadata)
            if isinstance(label.metadata, Mapping)
            else label.metadata
        ),
    }
