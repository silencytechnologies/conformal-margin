"""ConformalCalibrator - inductive split conformal p-value computer.

Implements the core mathematical step that gives `conformal-margin`
its formal `P(false positive rate > alpha) <= alpha` coverage
guarantee. Given a set of calibration nonconformity scores
`S_1, ..., S_n` and a new nonconformity score `S_new`, returns:

    p_value(S_new) = (1 + |{i : S_i >= S_new}|) / (n + 1)

This is the canonical inductive conformal p-value (Vovk,
Gammerman, Shafer; "Algorithmic Learning in a Random World",
chapters 2-4). Under the exchangeability assumption between
calibration and test data, the resulting p-value has the property:

    P(p_value <= alpha) <= alpha    for any alpha in (0, 1)

This is **exact**, not asymptotic; it holds for finite n. The
guarantee is what differentiates the library from heuristic
anomaly detectors. the formula is **non-negotiable on
the critical path**: streaming approximate quantile structures
(t-digest, Greenwald-Khanna, etc.) are forbidden here because
they replace the exact rank statistic with an approximation that
weakens the coverage guarantee.

Range of returned p-values: `[1 / (n + 1), 1]`. Never zero. The
minimum value `1 / (n + 1)` is achieved when the new score
strictly exceeds every calibration score (maximum surprise). For
example, with `n = 99` calibration scores and `alpha = 0.01` the
minimum p-value is exactly `0.01`, so the calibration set must be
at least 99 large for any anomaly to ever cross the alpha
threshold. `BaselineOperator` exposes a `min_calibration_size`
parameter that enforces this practically.

`ConformalCalibrator` is stateless and frozen. It exists as a
dataclass (not a free function) so that future calibrator
variants (jackknife+, cross-validation+) can be added
as sibling classes with the same API surface.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from conformal_margin.ontology.domain.exceptions import ValidationError

__all__ = ["ConformalCalibrator"]


@dataclass(frozen=True)
class ConformalCalibrator:
    """Stateless inductive split conformal p-value computer.

    The class has no fields in v0.1. It is a dataclass so that
    future variants (jackknife+, cross-validation+) can be sibling
    classes with the same API; if those variants land they may
    introduce configuration fields without breaking the call site.

    `slots=True` is intentionally **not** used here even though
    other dataclasses in the library do use it. Reason: the
    combination of `slots=True` + `frozen=True` + zero fields
    produces a known dataclass-generated `__setattr__` that raises
    `TypeError` from a broken `super()` call rather than
    `FrozenInstanceError`. `slots` saves no memory for fieldless
    classes anyway. The moment we add a configuration field, `slots=True` becomes meaningful and is added back.

    Examples:
        >>> import numpy as np
        >>> calibrator = ConformalCalibrator()
        >>> cal_scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        >>>
        >>> calibrator.p_value(2.0, cal_scores)
        0.1
        >>>
        >>> calibrator.p_value(0.0, cal_scores)
        1.0
    """

    def p_value(
        self,
        score: float,
        cal_scores: npt.NDArray[np.float64],
    ) -> float:
        """Compute the inductive split conformal p-value of one score.

        Implements `p = (1 + |{i : S_i >= s}|) / (n + 1)` exactly,
        with `>=` (not `>`) on the comparison so that ties contribute
        to the count. Using `>` would give a slightly looser
        guarantee.

        Args:
            score: the new (test) nonconformity score.
            cal_scores: 1D numpy array of calibration nonconformity
                scores. Must be non-empty; otherwise the formula is
                undefined (division by zero is the technical issue;
                the deeper issue is that without any calibration
                data the coverage guarantee has no basis).

        Returns:
            p-value in the half-open interval `(0, 1]`. Specifically
            in the closed interval `[1 / (n + 1), 1]` where
            `n = len(cal_scores)`.

        Raises:
            ValidationError: `cal_scores` is empty.
        """
        n = len(cal_scores)
        if n == 0:
            raise ValidationError(
                field="cal_scores",
                reason="must be non-empty to compute a conformal p-value",
                value=0,
            )
        count_at_or_above = int(np.sum(cal_scores >= score))
        return (1 + count_at_or_above) / (n + 1)

    def p_value_array(
        self,
        scores: npt.NDArray[np.float64],
        cal_scores: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Vectorized p-value for an array of new scores against one calibration.

        Sorts `cal_scores` once and uses `numpy.searchsorted` for
        O(k log n) total cost where `k = len(scores)` and
        `n = len(cal_scores)`. This is the hot path for
        `BaselineOperator` which evaluates every agent in the
        current snapshot against the same calibration set.

        Args:
            scores: 1D numpy array of new (test) scores.
            cal_scores: 1D numpy array of calibration scores. Must
                be non-empty.

        Returns:
            1D numpy array of p-values, same length as `scores`.
            Each element is in `[1 / (n + 1), 1]`.

        Raises:
            ValidationError: `cal_scores` is empty.
        """
        n = len(cal_scores)
        if n == 0:
            raise ValidationError(
                field="cal_scores",
                reason="must be non-empty to compute a conformal p-value",
                value=0,
            )
        sorted_cal = np.sort(cal_scores)
        insertion_indices = np.searchsorted(sorted_cal, scores, side="left")
        counts_at_or_above = n - insertion_indices
        return (1 + counts_at_or_above) / (n + 1)
