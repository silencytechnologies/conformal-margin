"""Unit tests for conformal_margin.operators.scorer.RobustScorer.

Coverage: construction, epsilon validation, fit on known datasets,
score at and away from median, MAD-zero degenerate case using
epsilon floor, vectorized score_array consistency with scalar
score, empty calibration rejection, frozen behavior.
"""

import numpy as np
import pytest

from conformal_margin.ontology.domain.exceptions import ValidationError
from conformal_margin.operators.scorer import RobustScorer


def test_default_epsilon():
    scorer = RobustScorer()
    assert scorer.epsilon == 1e-9


def test_custom_epsilon():
    scorer = RobustScorer(epsilon=1e-6)
    assert scorer.epsilon == 1e-6


def test_zero_epsilon_raises_validation_error():
    with pytest.raises(ValidationError) as exc_info:
        RobustScorer(epsilon=0.0)
    assert exc_info.value.field == "epsilon"


def test_negative_epsilon_raises_validation_error():
    with pytest.raises(ValidationError) as exc_info:
        RobustScorer(epsilon=-0.1)
    assert exc_info.value.field == "epsilon"


def test_is_frozen():
    scorer = RobustScorer()
    with pytest.raises(AttributeError):
        scorer.epsilon = 0.5


def test_mad_scale_is_classvar_1_4826():
    assert RobustScorer.MAD_SCALE == 1.4826


def test_fit_returns_median_and_scaled_mad_on_symmetric_data():
    scorer = RobustScorer()
    cal = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    median, scaled_mad = scorer.fit(cal)
    assert median == 5.0
    assert scaled_mad == pytest.approx(1.4826 * 2.0)


def test_fit_on_single_element_returns_that_value_and_zero_mad():
    scorer = RobustScorer()
    cal = np.array([42.0])
    median, scaled_mad = scorer.fit(cal)
    assert median == 42.0
    assert scaled_mad == 0.0


def test_fit_on_identical_values_returns_zero_mad():
    scorer = RobustScorer()
    cal = np.array([7.0, 7.0, 7.0, 7.0, 7.0])
    median, scaled_mad = scorer.fit(cal)
    assert median == 7.0
    assert scaled_mad == 0.0


def test_fit_empty_calibration_raises_validation_error():
    scorer = RobustScorer()
    with pytest.raises(ValidationError) as exc_info:
        scorer.fit(np.array([], dtype=np.float64))
    assert exc_info.value.field == "calibration"


def test_fit_returns_floats_not_numpy_scalars():
    scorer = RobustScorer()
    median, scaled_mad = scorer.fit(np.array([1.0, 2.0, 3.0]))
    assert type(median) is float
    assert type(scaled_mad) is float


def test_fit_resistant_to_extreme_outlier():
    scorer = RobustScorer()
    cal = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 1_000_000.0])
    median, scaled_mad = scorer.fit(cal)
    assert median == 3.5
    assert scaled_mad == pytest.approx(1.4826 * 1.5)


def test_score_at_median_is_zero():
    scorer = RobustScorer()
    assert scorer.score(5.0, median=5.0, scaled_mad=2.0) == 0.0


def test_score_one_scaled_mad_above_median_is_one():
    scorer = RobustScorer()
    assert scorer.score(7.0, median=5.0, scaled_mad=2.0) == 1.0


def test_score_one_scaled_mad_below_median_is_one():
    scorer = RobustScorer()
    assert scorer.score(3.0, median=5.0, scaled_mad=2.0) == 1.0


def test_score_three_scaled_mads_above_median_is_three():
    scorer = RobustScorer()
    assert scorer.score(11.0, median=5.0, scaled_mad=2.0) == 3.0


def test_score_with_zero_scaled_mad_uses_epsilon_floor():
    scorer = RobustScorer(epsilon=1e-6)
    score = scorer.score(value=1.0, median=0.0, scaled_mad=0.0)
    assert score == 1.0 / 1e-6


def test_score_with_zero_deviation_and_zero_mad_is_zero():
    scorer = RobustScorer()
    assert scorer.score(value=5.0, median=5.0, scaled_mad=0.0) == 0.0


def test_score_array_matches_per_element_score():
    scorer = RobustScorer()
    values = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
    median, scaled_mad = 5.0, 2.0
    array_scores = scorer.score_array(values, median, scaled_mad)
    loop_scores = np.array([scorer.score(float(v), median, scaled_mad) for v in values])
    np.testing.assert_array_equal(array_scores, loop_scores)


def test_score_array_uses_epsilon_floor_when_scaled_mad_is_zero():
    scorer = RobustScorer(epsilon=1e-3)
    values = np.array([0.5, 1.0, 1.5])
    result = scorer.score_array(values, median=0.0, scaled_mad=0.0)
    expected = np.array([0.5 / 1e-3, 1.0 / 1e-3, 1.5 / 1e-3])
    np.testing.assert_array_almost_equal(result, expected)


def test_score_array_preserves_length_and_returns_numpy_array():
    scorer = RobustScorer()
    values = np.array([1.0, 2.0, 3.0, 4.0])
    result = scorer.score_array(values, median=2.5, scaled_mad=1.0)
    assert isinstance(result, np.ndarray)
    assert len(result) == 4


def test_fit_then_score_on_same_data_centers_calibration_at_zero():
    scorer = RobustScorer()
    cal = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    median, scaled_mad = scorer.fit(cal)
    cal_scores = scorer.score_array(cal, median, scaled_mad)
    assert cal_scores[2] == 0.0
