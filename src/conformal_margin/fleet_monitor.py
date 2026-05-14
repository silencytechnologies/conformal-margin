"""FleetMonitor - the orchestrator for the conformal-margin pipeline.

Customer-facing entry point. Receives a `FleetSnapshot`, loads
history from the configured `StateStore`, dispatches the snapshot
plus history to each configured `Operator` concurrently, then
dispatches every emitted signal to each configured `AlertHandler`
concurrently, and finally persists the snapshot to the store.

Lifecycle of one `observe(snapshot)` call:

1. Load history: `store.load_recent(calibration_window)`. The
   returned tuple is the calibration data passed to operators.
   Crucially, history is loaded **before** the current snapshot
   is saved, so operators never see the test point in their
   calibration set (a precondition of the conformal coverage
   guarantee).
2. Dispatch to operators: inside an `anyio.TaskGroup`, each
   operator runs `op.run(snapshot, history)` concurrently. Each
   operator generates the full identity of its emitted signals.
3. Collect signals: results from all operators are concatenated
   in operator order.
4. Dispatch to handlers: inside a second `anyio.TaskGroup`, each
   `(handler, signal)` pair runs in its own task. Handler
   failures are caught and logged via structlog at error level;
   sibling handlers continue. This is the "best-effort delivery"
   semantic for the Sink lane.
5. Save snapshot: `store.save(snapshot)`. Saving after dispatch
   means a slow handler does not delay snapshot persistence's
   visibility to future runs, while still preserving the
   load-before-save ordering that guarantees calibration-set
   exchangeability.

Error semantics:

- Operator failure: the operator's task raises; structured
  concurrency propagates as `ExceptionGroup` that `observe()`
  re-raises. Pipeline failure is loud because operator
  correctness is the value-add; silent operator drops would
  conceal real problems.
- Handler failure: per-handler `try / except` catches every
  exception, logs `signal_dispatch_failed` at error level, and
  the task returns normally. The pipeline as a whole reports
  success even when individual handlers refuse a signal.
- Store failure on load: `StoreUnavailableError` propagates;
  caller retries `observe()`.
- Store failure on save: `StoreUnavailableError` propagates
  **after** signals have already reached handlers. Caller is
  expected to retry the save (the snapshot is idempotent under
  `InMemoryStore.save` semantics) or accept the gap.

Feedback path (`submit_feedback(signal, label)`):

Separate method from `observe()`. Customer calls when a label
becomes available. Dispatches to each configured `FeedbackPort`
concurrently; same per-task error catching as handlers (loud per
port, quiet for sibling ports).
"""

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, ClassVar

import anyio
import structlog

from conformal_margin.ontology.domain.exceptions import ValidationError
from conformal_margin.ontology.ports.alert_handler import AlertHandler
from conformal_margin.ontology.ports.feedback import (FeedbackLabel,
                                                      FeedbackPort)
from conformal_margin.ontology.ports.fleet_snapshot import FleetSnapshotLike
from conformal_margin.ontology.ports.operator import Operator
from conformal_margin.ontology.ports.operator_signal import OperatorSignalLike
from conformal_margin.ontology.ports.state_store import StateStore
from conformal_margin.operator_signal import OperatorSignal

__all__ = ["FleetMonitor"]


@dataclass(frozen=True, slots=True)
class FleetMonitor:
    """Orchestrate one observe-snapshot-to-emit-signals cycle.

    Attributes:
        operators: tuple of `Operator` instances. Non-empty.
            Each operator runs concurrently per snapshot inside an
            `anyio.TaskGroup`.
        store: the single `StateStore` instance. Customers who
            want layered behaviour (cache plus durable, buffered
            plus persistent) compose those at the Store boundary
            via their own implementation of the `StateStore`
            Protocol.
        handlers: tuple of `AlertHandler` instances. Non-empty.
            Every emitted signal is dispatched to every handler.
        calibration_window: how far back `observe()` asks the
            store to load history. Same window for every
            operator; per-operator override is not supported in
            v0.1. Typical: one hour.
        feedback_ports: tuple of `FeedbackPort` instances for the
            optional `submit_feedback` path. May be empty when
            feedback is not part of the deployment; in that case
            calling `submit_feedback` raises `ValidationError`.

    Class attribute `role` is informational; FleetMonitor itself
    is not a `Role`-tagged component (it is the orchestrator that
    moves data between roles).

    Examples:
        >>> from datetime import timedelta
        >>> from conformal_margin.stores.in_memory import InMemoryStore
        >>> from conformal_margin.handlers.log import LogHandler
        >>> from conformal_margin.operators.baseline import BaselineOperator
        >>> monitor = FleetMonitor(
        ...     operators=(BaselineOperator(name="baseline"),),
        ...     store=InMemoryStore(),
        ...     handlers=(LogHandler(name="primary-log"),),
        ...     calibration_window=timedelta(hours=1),
        ... )
        >>> len(monitor.operators)
        1
    """

    operators: tuple[Operator, ...]
    store: StateStore
    handlers: tuple[AlertHandler, ...]
    calibration_window: timedelta = timedelta(hours=1)
    feedback_ports: tuple[FeedbackPort, ...] = ()

    _logger: Any = field(init=False, repr=False, compare=False)

    role: ClassVar[str] = "orchestrator"

    def __post_init__(self) -> None:
        """Validate that operators and handlers are non-empty.

        Raises:
            ValidationError: `operators` is empty, `handlers` is
                empty, or `calibration_window` is not strictly
                positive.
        """
        if not self.operators:
            raise ValidationError(
                field="operators",
                reason="must contain at least one Operator",
                value=len(self.operators),
            )
        if not self.handlers:
            raise ValidationError(
                field="handlers",
                reason="must contain at least one AlertHandler",
                value=len(self.handlers),
            )
        if self.calibration_window.total_seconds() <= 0:
            raise ValidationError(
                field="calibration_window",
                reason="must be strictly positive",
                value=str(self.calibration_window),
            )
        object.__setattr__(self, "operators", tuple(self.operators))
        object.__setattr__(self, "handlers", tuple(self.handlers))
        object.__setattr__(self, "feedback_ports", tuple(self.feedback_ports))
        object.__setattr__(
            self,
            "_logger",
            structlog.get_logger().bind(component="fleet_monitor"),
        )

    async def observe(self, snapshot: FleetSnapshotLike) -> tuple[OperatorSignal, ...]:
        """Run one full observe-dispatch-save cycle.

        Args:
            snapshot: the current `FleetSnapshotLike` to process.

        Returns:
            Tuple of every `OperatorSignal` emitted by every
            operator, in operator-then-emission order.

        Raises:
            StoreUnavailableError: store load or save failed.
            ExceptionGroup: at least one operator raised. Sibling
                operators are cancelled per `anyio.TaskGroup`
                semantics.
            ValidationError: snapshot violates a documented
                precondition (propagated from operator or store).
        """
        history = await self.store.load_recent(self.calibration_window)

        operator_results: list[tuple[OperatorSignalLike, ...]] = [
            () for _ in self.operators
        ]
        async with anyio.create_task_group() as tg:
            for index, op in enumerate(self.operators):
                tg.start_soon(
                    self._run_operator,
                    op,
                    snapshot,
                    history,
                    operator_results,
                    index,
                )

        all_signals: tuple[OperatorSignal, ...] = tuple(
            sig for result in operator_results for sig in result
        )

        if all_signals:
            async with anyio.create_task_group() as tg:
                for sig in all_signals:
                    for handler in self.handlers:
                        tg.start_soon(self._safe_handle, handler, sig)

        await self.store.save(snapshot)
        return all_signals

    async def submit_feedback(
        self, signal: OperatorSignalLike, label: FeedbackLabel
    ) -> None:
        """Dispatch one (signal, label) pair to every configured FeedbackPort.

        Args:
            signal: the previously-emitted signal the label
                refers to.
            label: the customer verdict.

        Raises:
            ValidationError: no feedback ports were configured at
                FleetMonitor construction.
        """
        if not self.feedback_ports:
            raise ValidationError(
                field="feedback_ports",
                reason=(
                    "no FeedbackPort configured on this FleetMonitor; "
                    "construct with feedback_ports=(...) to enable feedback"
                ),
                value=0,
            )
        async with anyio.create_task_group() as tg:
            for port in self.feedback_ports:
                tg.start_soon(self._safe_record, port, signal, label)

    async def _run_operator(
        self,
        op: Operator,
        snapshot: FleetSnapshotLike,
        history: tuple[FleetSnapshotLike, ...],
        results: list[tuple[OperatorSignalLike, ...]],
        index: int,
    ) -> None:
        """Run one operator and store its emitted signals at `results[index]`.

        Operator exceptions are not caught here: they propagate up
        the TaskGroup so the pipeline fails loud. Operator failures
        are pipeline-critical.
        """
        results[index] = await op.run(snapshot, history)

    async def _safe_handle(
        self, handler: AlertHandler, signal: OperatorSignalLike
    ) -> None:
        """Invoke one handler with one signal, catching and logging any failure.

        Per the best-effort delivery semantic: handler exceptions
        are logged at error level; the task returns normally so
        sibling handler tasks continue.
        """
        try:
            await handler.handle(signal)
        except Exception as exc:
            self._logger.error(
                "signal_dispatch_failed",
                handler=handler.name,
                signal_id=signal.signal_id,
                error_type=type(exc).__name__,
                error=str(exc),
            )

    async def _safe_record(
        self,
        port: FeedbackPort,
        signal: OperatorSignalLike,
        label: FeedbackLabel,
    ) -> None:
        """Invoke one feedback port with one (signal, label) pair, catching failures.

        Same best-effort semantic as `_safe_handle`.
        """
        try:
            await port.record(signal, label)
        except Exception as exc:
            self._logger.error(
                "feedback_record_failed",
                port=port.name,
                signal_id=signal.signal_id,
                error_type=type(exc).__name__,
                error=str(exc),
            )
