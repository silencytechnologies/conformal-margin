"""RobustScorer - median plus scaled-MAD nonconformity scorer.

The nonconformity score function for `BaselineOperator`. Given a
calibration set and a value to score, returns:

    s(x) = |x - median(C)| / max(1.4826 * MAD(C), epsilon)

where `C` is the calibration set, `MAD(C) = median(|c - median(C)|)`
is the Median Absolute Deviation, and the `1.4826` scaling factor
makes the denominator approximately equal to the standard
deviation when `C` is Gaussian (Gaussian-consistency scaling).

Why MAD and not standard deviation: MAD is **robust** in the
statistical sense - a few outliers in the calibration set do not
inflate it. Standard deviation is not robust; a small fraction of
adversarial high-magnitude scores would balloon `std(C)` and mask
genuine anomalies. the library positions itself as
"hardened against calibration poisoning"; MAD is the first line of
that hardening.

The `epsilon` floor protects against the degenerate case where
every calibration value is identical (`MAD = 0`). Without the
floor, the score function would divide by zero for any non-zero
deviation.

`RobustScorer` has no state and exposes a `fit()` method that
returns the location-and-scale statistics, plus `score()` and
`score_array()` that consume those statistics. Splitting fit from
score lets `BaselineOperator` compute the statistics once per
snapshot and apply them to every agent (single inference) plus the
entire calibration history (population of calibration scores)
without recomputation.
"""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from conformal_margin.ontology.domain.exceptions import ValidationError

__all__ = ["RobustScorer"]


@dataclass(frozen=True, slots=True)
class RobustScorer:
    """Robust nonconformity scorer using median and scaled MAD.

    Attributes:
        epsilon: floor on the denominator scaled-MAD value. When the
            calibration set is degenerate (all identical values),
            scaled MAD is zero and dividing by it would error;
            `epsilon` is substituted instead. Must be strictly
            positive. Default `1e-9` is small enough not to bias
            scores in well-conditioned cases.

    Class attributes:
        MAD_SCALE: the `1.4826` Gaussian-consistency scaling
            constant. Public for reference; not customer-tunable
            because changing it breaks the "scaled MAD approximates
            standard deviation for Gaussian data" interpretation
            that the rest of the documentation assumes.

    Examples:
        >>> import numpy as np
        >>> scorer = RobustScorer()
        >>> calibration = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> median, scaled_mad = scorer.fit(calibration)
        >>> median
        3.0
        >>> round(scaled_mad, 4)
        1.4826
        >>> scorer.score(3.0, median, scaled_mad)
        0.0
    """

    epsilon: float = 1e-9

    MAD_SCALE: ClassVar[float] = 1.4826

    def __post_init__(self) -> None:
        """Validate that epsilon is strictly positive.

        Raises:
            ValidationError: epsilon is zero or negative.
        """
        if not (self.epsilon > 0):
            raise ValidationError(
                field="epsilon",
                reason="must be strictly positive",
                value=self.epsilon,
            )

    def fit(self, calibration: npt.NDArray[np.float64]) -> tuple[float, float]:
        """Compute location (median) and scale (scaled MAD) of calibration.

        Args:
            calibration: 1D numpy array of finite floats. The
                library trusts that elements are finite because
                `AgentState.risk_score` is finite by construction
                (`AgentState.__post_init__` enforces it).

        Returns:
            Tuple `(median, scaled_mad)`. The returned `scaled_mad`
            is the raw `1.4826 * median(|x - median|)` value; the
            `epsilon` floor is applied at `score` time so callers
            see the genuine scale of the calibration distribution
            in this return value.

        Raises:
            ValidationError: `calibration` is empty.
        """
        if len(calibration) == 0:
            raise ValidationError(
                field="calibration",
                reason="must be non-empty for fit",
                value=0,
            )
        median = float(np.median(calibration))
        mad = float(np.median(np.abs(calibration - median)))
        return median, self.MAD_SCALE * mad

    def score(self, value: float, median: float, scaled_mad: float) -> float:
        """Compute the nonconformity score of a single value.

        Args:
            value: the observation to score.
            median: location parameter from `fit()`.
            scaled_mad: scale parameter from `fit()`.

        Returns:
            Non-negative float `|value - median| / max(scaled_mad,
            epsilon)`. Equal to `0.0` when `value == median`.
        """
        denominator = max(scaled_mad, self.epsilon)
        return abs(value - median) / denominator

    def score_array(
        self,
        values: npt.NDArray[np.float64],
        median: float,
        scaled_mad: float,
    ) -> npt.NDArray[np.float64]:
        """Vectorized nonconformity scores for an array of values.

        Equivalent to applying `score()` to each element but
        computed in a single numpy expression (the typical hot
        path in `BaselineOperator` which scores the entire
        calibration set plus the current snapshot in one pass).

        Args:
            values: 1D numpy array of finite floats.
            median: location parameter from `fit()`.
            scaled_mad: scale parameter from `fit()`.

        Returns:
            1D numpy array of non-negative floats, same length as
            `values`.
        """
        denominator = max(scaled_mad, self.epsilon)
        return np.abs(values - median) / denominator
