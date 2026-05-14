"""BaselineOperator - the v0.1 conformal-margin operator.

Composes the three sub-components of the Robust-Conformal Baseline
architecture:

1. `RobustScorer` produces nonconformity scores using median plus
   scaled MAD (robust to calibration-set outliers).
2. `ConformalCalibrator` converts raw scores to formal conformal
   p-values with the `P(false positive rate > alpha) <= alpha`
   coverage guarantee.
3. `DriftDetector` (Page-Hinkley) raises a separate alarm when
   the calibration set is no longer exchangeable with current
   data - which is the precondition for the coverage guarantee.

Per-snapshot workflow inside `run()`:

1. Flatten the history window into a 1D array of historical
   `risk_score` values. This is the calibration pool.
2. Compute median plus scaled MAD on that array (`scorer.fit`).
3. Score each calibration point against those stats: this is the
   calibration score distribution.
4. Score each agent in the current snapshot against the same
   stats; obtain test scores.
5. For each test score, compute its conformal p-value against the
   calibration score distribution.
6. Map p-value into severity tier (per C.1 design):
   - p < alpha * 0.01: Severity.CRITICAL
   - p < alpha * 0.10: Severity.HIGH
   - p <= alpha:       Severity.MEDIUM
   - p > alpha:        Severity.INFO   (threshold_exceeded=False)
7. Emit one `OperatorSignal` per agent with full identity
   (UUID4 + snapshot.timestamp).
8. Run `DriftDetector` on the calibration score distribution
   against the older half's mean as baseline. If drift is
   detected, emit one additional drift `OperatorSignal`.

Configuration parameters at construction:

- `name`: registered operator name (`Operator.name`)
- `alpha`: customer-configured target FPR
- `min_calibration_size`: below this many historical observations
  the operator emits nothing. Conformal coverage is unreliable
  below ~30 samples (classical statistical heuristic).
- `drift_delta`, `drift_threshold`: Page-Hinkley parameters

Operator is stateless. All history is loaded via the
`StateStore` Protocol before `run()` is invoked; the operator
sees the calibration window as an immutable input.
"""

import uuid
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from conformal_margin.ontology.domain.exceptions import ValidationError
from conformal_margin.ontology.domain.ontology import (DriftEventType, Role,
                                                       Severity, SignalKind)
from conformal_margin.ontology.ports.fleet_snapshot import FleetSnapshotLike
from conformal_margin.operator_signal import OperatorSignal
from conformal_margin.operators.calibrator import ConformalCalibrator
from conformal_margin.operators.drift import DriftDetector
from conformal_margin.operators.scorer import RobustScorer

__all__ = ["BaselineOperator"]


@dataclass(frozen=True, slots=True)
class BaselineOperator:
    """The v0.1 Robust-Conformal Baseline operator.

    Implements the `Operator` Protocol from
    `conformal_margin.ontology.ports.operator`.

    Attributes:
        name: registered operator name. Becomes `operator_name` on
            every emitted signal. Non-empty.
        alpha: customer-configured target False Positive Rate
            threshold. Signals with p-value <= alpha have
            `threshold_exceeded=True`. Must be in `(0, 1)`. Typical
            values: `0.01`, `0.001`.
        min_calibration_size: minimum number of historical
            observations needed before the operator emits any
            signal. Below this size the conformal coverage
            guarantee is unreliable and we return zero signals
            rather than mislead the customer. Must be at least 1.
        drift_delta: Page-Hinkley allowable mean shift parameter
            for `DriftDetector`.
        drift_threshold: Page-Hinkley detection threshold for
            `DriftDetector`.

    Class attribute `role` is always `Role.TRANSFORMER` per the
    Operator Protocol.
    """

    name: str
    alpha: float = 0.01
    min_calibration_size: int = 30
    drift_delta: float = 0.005
    drift_threshold: float = 50.0

    _scorer: RobustScorer = field(default_factory=RobustScorer, init=False, repr=False)
    _calibrator: ConformalCalibrator = field(
        default_factory=ConformalCalibrator, init=False, repr=False
    )
    _drift_detector: DriftDetector = field(init=False, repr=False)

    role: ClassVar[Role] = Role.TRANSFORMER

    def __post_init__(self) -> None:
        """Validate configuration and bind sub-operators.

        Raises:
            ValidationError: any parameter violates its precondition.
        """
        if not self.name:
            raise ValidationError(
                field="name",
                reason="must be a non-empty string",
                value=self.name,
            )
        if not (0.0 < self.alpha < 1.0):
            raise ValidationError(
                field="alpha",
                reason="must be in the open interval (0, 1)",
                value=self.alpha,
            )
        if self.min_calibration_size < 1:
            raise ValidationError(
                field="min_calibration_size",
                reason="must be at least 1",
                value=self.min_calibration_size,
            )
        object.__setattr__(
            self,
            "_drift_detector",
            DriftDetector(delta=self.drift_delta, threshold=self.drift_threshold),
        )

    async def run(
        self,
        snapshot: FleetSnapshotLike,
        history: tuple[FleetSnapshotLike, ...],
    ) -> tuple[OperatorSignal, ...]:
        """Score every agent in `snapshot` against `history`.

                Returns zero or more `OperatorSignal` values. The signal
                identity (`signal_id`, `timestamp`) is generated here per
        `signal_id` is a fresh UUID4 per signal,
                `timestamp` is `snapshot.timestamp`. The FleetMonitor does
                not modify these.

                Args:
                    snapshot: the current observation moment to analyse.
                    history: prior snapshots loaded from the `StateStore`.
                        Used as the calibration window.

                Returns:
                    Tuple of `OperatorSignal` values. Empty if the
                    calibration window is too small
                    (`< min_calibration_size`) or if the current snapshot
                    has no agents.

                Raises:
                    ValidationError: snapshot or history is structurally
                        invalid (propagated from upstream).
        """
        if len(snapshot.agents) == 0:
            return ()

        cal_values = _flatten_history_risk_scores(history)
        if len(cal_values) < self.min_calibration_size:
            return ()

        median, scaled_mad = self._scorer.fit(cal_values)
        cal_scores = self._scorer.score_array(cal_values, median, scaled_mad)

        agent_values = np.array(
            [a.risk_score for a in snapshot.agents], dtype=np.float64
        )
        test_scores = self._scorer.score_array(agent_values, median, scaled_mad)
        p_values = self._calibrator.p_value_array(test_scores, cal_scores)

        agent_signals = tuple(
            OperatorSignal(
                signal_id=str(uuid.uuid4()),
                operator_name=self.name,
                timestamp=snapshot.timestamp,
                kind=SignalKind.ANOMALY,
                severity=self._severity_from_p_value(float(p)),
                score=float(s),
                p_value=float(p),
                threshold_exceeded=bool(p <= self.alpha),
                affected_agents=(agent.agent_id,),
                drift_event_type=None,
                metadata={
                    "calibration_window_size": int(len(cal_values)),
                },
            )
            for agent, s, p in zip(snapshot.agents, test_scores, p_values)
        )

        drift_signals = self._maybe_emit_drift(
            cal_values=cal_values,
            snapshot=snapshot,
            calibration_window_size=int(len(cal_values)),
        )

        return agent_signals + drift_signals

    def _severity_from_p_value(self, p: float) -> Severity:
        """Map a conformal p-value to a severity tier.

        Per section C.1: four tiers, mapping conservative-anomalous
        on a logarithmic scale relative to `alpha`. The lowest tier
        (`INFO`) is also the only legal severity when
        `threshold_exceeded=False` (OperatorSignal invariant 7).
        """
        if p > self.alpha:
            return Severity.INFO
        if p < self.alpha * 0.01:
            return Severity.CRITICAL
        if p < self.alpha * 0.10:
            return Severity.HIGH
        return Severity.MEDIUM

    def _maybe_emit_drift(
        self,
        cal_values: npt.NDArray[np.float64],
        snapshot: FleetSnapshotLike,
        calibration_window_size: int,
    ) -> tuple[OperatorSignal, ...]:
        """Run Page-Hinkley on raw `cal_values`; emit zero or one drift signal.

        Operates on raw historical `risk_score` values, not on the
        derived nonconformity scores. Reason: if the calibration
        location (median) is estimated from the full window and
        drift is present, the median naturally lands between the
        pre- and post-drift regimes, and the nonconformity scores
        of both halves become similarly distributed. Page-Hinkley
        would then see no shift in the score stream. Running on
        raw values keeps the drift signal sensitive regardless of
        how the location estimate is computed.

        Baseline mean for Page-Hinkley is the mean of the older
        half of `cal_values`. The newer half is tested for an
        upward shift relative to that baseline. With a single
        window we use a 50/50 split.
        """
        n = len(cal_values)
        if n < 4:
            return ()
        split = n // 2
        baseline_mean = float(np.mean(cal_values[:split]))
        drift_detected, ph_statistic = self._drift_detector.detect(
            cal_values[split:], baseline_mean=baseline_mean
        )
        if not drift_detected:
            return ()
        drift_signal = OperatorSignal(
            signal_id=str(uuid.uuid4()),
            operator_name=self.name,
            timestamp=snapshot.timestamp,
            kind=SignalKind.DRIFT,
            severity=Severity.HIGH,
            score=ph_statistic,
            p_value=0.0,
            threshold_exceeded=True,
            affected_agents=(),
            drift_event_type=DriftEventType.MEAN_SHIFT,
            metadata={
                "calibration_window_size": calibration_window_size,
                "page_hinkley_baseline_mean": baseline_mean,
                "page_hinkley_threshold": self.drift_threshold,
            },
        )
        return (drift_signal,)


def _flatten_history_risk_scores(
    history: tuple[FleetSnapshotLike, ...],
) -> npt.NDArray[np.float64]:
    """Flatten history into a 1D numpy array of all historical risk_scores."""
    return np.array(
        [a.risk_score for s in history for a in s.agents],
        dtype=np.float64,
    )
