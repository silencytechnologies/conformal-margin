"""StateStore Protocol - the Store role contract.

Persists FleetSnapshot history and answers queries against it. The
only role allowed to hold mutable state across snapshot boundaries.

Concrete implementations: `InMemoryStore` (library default in v0.1),
future `DuckDBStore`, `RedisStore`, `HybridStateStore`. Customers
may bring their own.

The `role: Final = Role.STORE` class attribute is informational
(documents the role, Semgrep hook for boundary checks).
"""

from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import Final, Protocol, runtime_checkable

from conformal_margin.ontology.domain.ontology import Role
from conformal_margin.ontology.ports.fleet_snapshot import FleetSnapshotLike

__all__ = ["StateStore"]


@runtime_checkable
class StateStore(Protocol):
    """Persist FleetSnapshot history and answer history queries.

    All methods are `async def`. Implementations may be
    in-process (`InMemoryStore`) or network-backed (`RedisStore`);
    the Protocol does not distinguish.

    Five methods total - the documented Repository pattern exception
    to the five-public-method cap (per `public-method-contract.md`).
    All five methods are the same shape of work (history access), so
    cohesion holds.

    Attributes:
        role: always `Role.STORE` (documentation, Semgrep hook).
    """

    role: Final = Role.STORE

    async def save(self, snapshot: FleetSnapshotLike) -> None:
        """Persist a FleetSnapshot.

        Args:
            snapshot: the snapshot to persist. Implementations key
                by snapshot timestamp; saving the same snapshot
                twice is idempotent.

        Raises:
            StoreUnavailableError: backing storage is unreachable.
        """
        ...

    async def load_recent(
        self,
        window: timedelta,
    ) -> tuple[FleetSnapshotLike, ...]:
        """Load snapshots from the most recent window.

        Args:
            window: how far back to look from "now". For example,
                `timedelta(hours=1)` returns snapshots from the last
                hour.

        Returns:
            Tuple of snapshots in ascending timestamp order. Empty
            tuple if no snapshots fall within the window.

        Raises:
            StoreUnavailableError: backing storage is unreachable.
        """
        ...

    async def load_range(
        self,
        start: datetime,
        end: datetime,
    ) -> tuple[FleetSnapshotLike, ...]:
        """Load snapshots within a specific time range.

        Args:
            start: inclusive lower bound, timezone-aware.
            end: exclusive upper bound, timezone-aware.

        Returns:
            Tuple of snapshots in ascending timestamp order.

        Raises:
            ValidationError: start or end is naive (no timezone), or
                end is not strictly after start.
            StoreUnavailableError: backing storage is unreachable.
        """
        ...

    async def evict(self, older_than: timedelta) -> None:
        """Remove snapshots older than the given relative age.

        Args:
            older_than: snapshots with timestamp before
                `now - older_than` are removed.

        Raises:
            StoreUnavailableError: backing storage is unreachable.
        """
        ...

    async def query_trend(
        self,
        metric: str,
        window: timedelta,
    ) -> Mapping[str, object]:
        """Aggregated query over the recent window.

        Implementation-specific. The metric name identifies the
        aggregation kind; the returned mapping is metric-specific.
        Customers parse based on the metric they requested.

        Args:
            metric: identifier of the aggregation to compute.
                Library does not standardize metric names in v0.1;
                each Store implementation documents its own set.
            window: how far back to aggregate.

        Returns:
            Mapping of result keys to values. Shape depends on the
            metric. Empty mapping if no data falls within the window.

        Raises:
            ValidationError: metric name is not recognized by the
                Store implementation.
            StoreUnavailableError: backing storage is unreachable.
        """
        ...
