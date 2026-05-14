"""DriftDetector - Page-Hinkley streaming test for mean shifts.

Page-Hinkley (Page 1954, "Continuous Inspection Schemes") is a
sequential test for changes in the mean of a stream. Given a
sequence of observations `x_1, ..., x_t` with known baseline mean
`mu_0` and an "allowable shift" parameter `delta`, the test
statistic is:

    m_t = sum_{i=1..t} (x_i - mu_0 - delta)
    M_t = min_{i=1..t} m_t
    PH_t = m_t - M_t                                   (always >= 0)
    drift_detected_at_t  <==>  PH_t > lambda

where `lambda` is the detection threshold. The test rejects the
null "the mean is stable at `mu_0`" in favour of "the mean has
shifted upward by at least `delta` somewhere in the window".

This implementation is **stateless**: every call recomputes
the test from scratch over the provided window. State-carrying
"online" streaming implementations are intentionally avoided
because they would force `BaselineOperator` to be stateful too,
violating the stateless-operator contract. The recompute cost is O(n) per call where n is
the window length; for typical calibration windows (under one
thousand observations) this is microseconds.

**One-directional only.** Standard Page-Hinkley detects mean
INCREASE only. The mirror version for mean decrease is a separate
implementation. For `conformal-margin`'s use case (drift detection
on risk-score nonconformity), upward drift ("fleet is getting
riskier") is the operational concern; downward drift ("fleet got
safer") is not actionable. Two-directional support is tracked for
v0.2+ if a customer surfaces a need.

**No formal p-value.** Page-Hinkley does not produce a calibrated
p-value the way conformal calibration does; it produces a test
statistic and a binary decision. `BaselineOperator` emits drift
signals with `p_value=0.0` as a placeholder. The drift
statistic is exposed on the signal via the `score` field for
downstream interpretation.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from conformal_margin.ontology.domain.exceptions import ValidationError

__all__ = ["DriftDetector"]


@dataclass(frozen=True, slots=True)
class DriftDetector:
    """Stateless Page-Hinkley drift detector for upward mean shifts.

    Attributes:
        delta: the "allowable" mean shift before the test starts
            accumulating evidence of drift. Larger `delta` makes
            the detector less sensitive (fewer false alarms,
            slower to detect real drift). Typical: `0.005` for
            nonconformity scores in `[0, 5]` range.
        threshold: detection threshold `lambda` on the PH statistic.
            When `PH_t > threshold`, the call reports drift.
            Larger `threshold` makes the detector less sensitive.
            Typical: `50.0`.

    Examples:
        >>> import numpy as np
        >>> detector = DriftDetector()
        >>>
        >>> values = np.array([0.5, 0.49, 0.51, 0.50, 0.48])
        >>> drift, statistic = detector.detect(values, baseline_mean=0.5)
        >>> drift
        False
    """

    delta: float = 0.005
    threshold: float = 50.0

    def __post_init__(self) -> None:
        """Validate parameters are non-negative.

        Raises:
            ValidationError: `delta` or `threshold` is negative.
        """
        if self.delta < 0:
            raise ValidationError(
                field="delta",
                reason="must be non-negative (Page-Hinkley allowable shift)",
                value=self.delta,
            )
        if self.threshold <= 0:
            raise ValidationError(
                field="threshold",
                reason="must be strictly positive (Page-Hinkley detection threshold)",
                value=self.threshold,
            )

    def detect(
        self,
        values: npt.NDArray[np.float64],
        baseline_mean: float,
    ) -> tuple[bool, float]:
        """Run Page-Hinkley over `values` against `baseline_mean`.

        Args:
            values: 1D numpy array of streaming observations in
                arrival order. Empty array returns no-drift, zero
                statistic.
            baseline_mean: the `mu_0` reference value against which
                drift is measured. Typically the empirical mean of
                an older calibration window believed to be stable.

        Returns:
            Tuple `(drift_detected, ph_statistic)`. `drift_detected`
            is `True` iff the most recent Page-Hinkley statistic
            exceeded `self.threshold`. `ph_statistic` is the final
            PH value (always non-negative).
        """
        if len(values) == 0:
            return False, 0.0
        deviations = values - baseline_mean - self.delta
        cumulative = np.cumsum(deviations)
        min_so_far = np.minimum.accumulate(cumulative)
        ph_statistic = float(cumulative[-1] - min_so_far[-1])
        return ph_statistic > self.threshold, ph_statistic
