"""InMemoryStore - in-process `StateStore` Protocol implementation.

A `dict[datetime, FleetSnapshotLike]` plus a single `anyio.Lock`.
The simplest Store the library ships, intended for development,
testing, and small-scale deployments. Production-scale customers
swap it for `RedisStore` or `DuckDBStore` (deferred) without
changing operator or handler code, because both implementations
satisfy the same `StateStore` Protocol.

Design decisions backing this file:

- Eviction is explicit only. `save` never auto-evicts;
  no background task. Customer schedules `evict(older_than)`
  periodically or accepts unbounded memory growth.
- Single `anyio.Lock` covers every method. Fine-grained
  locking is YAGNI for v0.1 scale (under ten thousand agents, one
  snapshot per minute).
- Plain `dict[datetime, FleetSnapshotLike]` keyed by
  snapshot timestamp. Range queries sort keys on demand.
- Idempotency in `save` is by structural `==`. Same
  snapshot at same timestamp: silent no-op. Different snapshot at
  same timestamp: `OperationFailedError`. No content hash, no UUID.
- `query_trend` ships three metrics: `mean_risk_score`,
  `p95_risk_score`, `unique_agent_count`. Each returns a Mapping
  with at minimum a `"value"` key. Empty window returns empty
  mapping.
- Unknown metric raises `UnknownMetricError` (subclass of
  `ValidationError`) carrying the available registry.

"Now" reference for time-relative queries (`load_recent`, `evict`,
`query_trend`) is `datetime.now(timezone.utc)`. The InMemoryStore is
a wall-clock-dependent component; tests that exercise time-relative
queries either use real `time.sleep`-style waits or construct
snapshots at deterministic offsets from "now".
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import ClassVar

import anyio

from conformal_margin.ontology.domain.exceptions import (OperationFailedError,
                                                         UnknownMetricError,
                                                         ValidationError)
from conformal_margin.ontology.domain.ontology import Role
from conformal_margin.ontology.ports.fleet_snapshot import FleetSnapshotLike

__all__ = ["InMemoryStore"]


@dataclass(slots=True)
class InMemoryStore:
    """In-process `StateStore` for dev, testing, and small-scale runs.

    Holds mutable state (the snapshot dict and a lock) so the class
    is not frozen. `slots=True` keeps memory layout tight; the two
    fields are `init=False` because they have intrinsic defaults
    that the constructor cannot meaningfully override.

    Attributes:
        role: always `Role.STORE` (Protocol contract).
        SUPPORTED_METRICS: tuple of metric names accepted by
            `query_trend`. Public for introspection; customers may
            assert against it in their integration tests.

    Examples:
        >>> store = InMemoryStore()
        >>> store.role.value
        'store'
        >>> InMemoryStore.SUPPORTED_METRICS
        ('mean_risk_score', 'p95_risk_score', 'unique_agent_count')
    """

    _snapshots: dict[datetime, FleetSnapshotLike] = field(
        default_factory=dict, init=False
    )
    _lock: anyio.Lock = field(default_factory=anyio.Lock, init=False)

    role: ClassVar[Role] = Role.STORE

    SUPPORTED_METRICS: ClassVar[tuple[str, ...]] = (
        "mean_risk_score",
        "p95_risk_score",
        "unique_agent_count",
    )

    async def save(self, snapshot: FleetSnapshotLike) -> None:
        """Persist a snapshot keyed by its `timestamp`.

        Idempotent on structural equality: saving the same
        snapshot twice is a silent no-op. Saving a different
        snapshot at the same `timestamp` raises `OperationFailedError`
        - this catches coordination bugs that would otherwise
        silently overwrite history.

        Args:
            snapshot: a value satisfying `FleetSnapshotLike`.

        Raises:
            OperationFailedError: a different snapshot already
                exists at `snapshot.timestamp`. `details` carries
                `reason` and the ISO-formatted timestamp string.
        """
        async with self._lock:
            existing = self._snapshots.get(snapshot.timestamp)
            if existing is not None:
                if existing == snapshot:
                    return
                raise OperationFailedError(
                    operation="save",
                    details={
                        "reason": "different snapshot exists at this timestamp",
                        "timestamp": snapshot.timestamp.isoformat(),
                    },
                )
            self._snapshots[snapshot.timestamp] = snapshot

    async def load_recent(self, window: timedelta) -> tuple[FleetSnapshotLike, ...]:
        """Return snapshots from the most recent `window`, oldest first.

        "Now" is `datetime.now(timezone.utc)`; the returned set is
        every snapshot with `timestamp >= now - window`. Empty tuple
        if no snapshots fall within the window.

        Args:
            window: how far back to look from now.

        Returns:
            Tuple of snapshots in ascending timestamp order.
        """
        cutoff = datetime.now(timezone.utc) - window
        async with self._lock:
            in_window = [
                (ts, snap) for ts, snap in self._snapshots.items() if ts >= cutoff
            ]
        in_window.sort(key=lambda pair: pair[0])
        return tuple(snap for _, snap in in_window)

    async def load_range(
        self, start: datetime, end: datetime
    ) -> tuple[FleetSnapshotLike, ...]:
        """Return snapshots with `start <= timestamp < end`, oldest first.

        Args:
            start: inclusive lower bound, timezone-aware.
            end: exclusive upper bound, timezone-aware. Must be
                strictly after `start`.

        Returns:
            Tuple of snapshots in ascending timestamp order.

        Raises:
            ValidationError: `start` or `end` is naive, or `end` is
                not strictly after `start`.
        """
        if start.tzinfo is None or start.utcoffset() is None:
            raise ValidationError(
                field="start",
                reason="must be timezone-aware",
                value=start,
            )
        if end.tzinfo is None or end.utcoffset() is None:
            raise ValidationError(
                field="end",
                reason="must be timezone-aware",
                value=end,
            )
        if end <= start:
            raise ValidationError(
                field="end",
                reason="must be strictly after start",
                value=end,
            )
        async with self._lock:
            in_range = [
                (ts, snap) for ts, snap in self._snapshots.items() if start <= ts < end
            ]
        in_range.sort(key=lambda pair: pair[0])
        return tuple(snap for _, snap in in_range)

    async def evict(self, older_than: timedelta) -> None:
        """Remove snapshots with `timestamp < now - older_than`.

        Args:
            older_than: snapshots older than this relative age are
                removed. Use `timedelta(seconds=0)` to drop every
                snapshot.
        """
        cutoff = datetime.now(timezone.utc) - older_than
        async with self._lock:
            to_remove = [ts for ts in self._snapshots if ts < cutoff]
            for ts in to_remove:
                del self._snapshots[ts]

    async def query_trend(self, metric: str, window: timedelta) -> Mapping[str, object]:
        """Compute an aggregate metric over snapshots in the recent window.

        Three metrics are supported in v0.1; each returns a
        Mapping with a `"value"` key plus optional implementation
        details. Empty window returns an empty mapping per the
        Protocol contract.

        Args:
            metric: one of `mean_risk_score`, `p95_risk_score`,
                `unique_agent_count`.
            window: how far back to aggregate from now.

        Returns:
            Mapping[str, object] with a `"value"` key. For
            `mean_risk_score` and `p95_risk_score` also includes
            `"n_samples"` (total agent-observations aggregated).
            Empty mapping if no snapshots fall in the window or if
            the window contains zero agents.

        Raises:
            UnknownMetricError: `metric` is not in `SUPPORTED_METRICS`.
        """
        if metric not in self.SUPPORTED_METRICS:
            raise UnknownMetricError(metric=metric, available=self.SUPPORTED_METRICS)
        cutoff = datetime.now(timezone.utc) - window
        async with self._lock:
            in_window = [snap for ts, snap in self._snapshots.items() if ts >= cutoff]
        if not in_window:
            return {}

        if metric == "mean_risk_score":
            scores = [a.risk_score for snap in in_window for a in snap.agents]
            if not scores:
                return {}
            return {
                "value": sum(scores) / len(scores),
                "n_samples": len(scores),
            }
        if metric == "p95_risk_score":
            scores = sorted(a.risk_score for snap in in_window for a in snap.agents)
            if not scores:
                return {}
            idx = max(0, math.ceil(0.95 * len(scores)) - 1)
            return {"value": scores[idx], "n_samples": len(scores)}
        agent_ids = {a.agent_id for snap in in_window for a in snap.agents}
        return {"value": len(agent_ids)}
