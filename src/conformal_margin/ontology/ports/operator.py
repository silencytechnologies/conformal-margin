"""Operator Protocol - the Transformer role contract.

A pure async function from a FleetSnapshot (plus optional history
loaded from the StateStore) to draft OperatorSignal values. Operators
are stateless; identity (signal_id, timestamp) is injected
by the FleetMonitor at finalization.

The `role: Final = Role.TRANSFORMER` class attribute is informational.
It documents what role any concrete `Operator` plays and gives
Semgrep a hook for boundary checks.
"""

from typing import Final, Protocol, runtime_checkable

from conformal_margin.ontology.domain.ontology import Role
from conformal_margin.ontology.ports.fleet_snapshot import FleetSnapshotLike
from conformal_margin.ontology.ports.operator_signal import OperatorSignalLike

__all__ = ["Operator"]


@runtime_checkable
class Operator(Protocol):
    """A pure async transformation from FleetSnapshot to draft signals.

        Concrete implementations (`BaselineOperator` in v0.1, future
        `CorrelationOperator`, `TemporalOperator`) live in the
        `conformal_margin` library.
    operators are stateless. Any history they need is
        loaded from the Store.

        (2026-05-13 revision of an earlier draft): operators
        are responsible for generating the full identity of each emitted
        signal. They produce `signal_id` (a fresh canonical UUID per
        signal, typically `uuid.uuid4()`) and set `timestamp` to
        `snapshot.timestamp` (the observation moment, not the processing
        moment). The FleetMonitor does not inject identity at
        finalization; the operator's output is a fully-formed
        `OperatorSignal` ready for dispatch. An earlier draft of this
        docstring claimed FleetMonitor finalization; that design did
        not survive integration with the OperatorSignal invariants
        (canonical UUID required at construction).
    async via anyio. The library invokes operators inside
        an `anyio.TaskGroup` for structured concurrency.

        Attributes:
            role: always `Role.TRANSFORMER` (documentation; read by
                Semgrep rules and future chain introspection).
            name: registered name of this operator. Becomes the
                `operator_name` field on emitted signals. Used for the
                operator registry: operators are identifiable by
                string name.
    """

    role: Final = Role.TRANSFORMER
    name: str

    async def run(
        self,
        snapshot: FleetSnapshotLike,
        history: tuple[FleetSnapshotLike, ...],
    ) -> tuple[OperatorSignalLike, ...]:
        """Run the operator and return zero, one, or more fully-formed signals.

        Args:
            snapshot: current FleetSnapshot to analyze.
            history: prior snapshots loaded from the StateStore. The
                window of interest is operator-specific; the
                FleetMonitor passes the configured calibration window
                or whatever the operator requested.

        Returns:
            Tuple of fully-formed `OperatorSignalLike` values, each
            with `signal_id` and `timestamp` already populated by
            the operator. Empty tuple is legal
            (operator observed nothing worth reporting). The
            FleetMonitor dispatches these to sinks without modifying
            identity.

        Raises:
            ValidationError: input violates a documented precondition
                (for example, snapshot contains an agent with missing
                `agent_id`).
            OperatorTimeoutError: history load exceeded the configured
                timeout.
            StoreUnavailableError: history requested from a Store that
                is currently unreachable. Propagated unchanged from
                the Store.
            UnexpectedError: library invariant violated during the
                operator's own computation (for example, calibrator
                produced NaN).
        """
        ...
