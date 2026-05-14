"""Unit tests for conformal_margin.operators.calibrator.ConformalCalibrator.

Coverage: construction, p-value boundary values (min and max),
monotonicity (higher nonconformity score yields lower p-value),
tie handling on the `>=` comparison, batch consistency with scalar
path, empty-calibration rejection, the formal coverage guarantee
verified empirically (Monte Carlo on exchangeable data).
"""

import numpy as np
import pytest

from conformal_margin.ontology.domain.exceptions import ValidationError
from conformal_margin.operators.calibrator import ConformalCalibrator


def test_constructs_with_no_arguments():
    calibrator = ConformalCalibrator()
    assert calibrator is not None


def test_is_frozen():
    calibrator = ConformalCalibrator()
    with pytest.raises(AttributeError):
        calibrator.some_field = 42


def test_p_value_equals_one_when_new_score_below_all_calibration():
    calibrator = ConformalCalibrator()
    cal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert calibrator.p_value(0.0, cal) == 1.0


def test_p_value_equals_min_when_new_score_above_all_calibration():
    calibrator = ConformalCalibrator()
    cal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert calibrator.p_value(10.0, cal) == pytest.approx(1 / 6)


def test_p_value_at_exact_max_calibration_includes_tie():
    calibrator = ConformalCalibrator()
    cal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert calibrator.p_value(5.0, cal) == pytest.approx(2 / 6)


def test_p_value_with_many_ties_counts_each_one():
    calibrator = ConformalCalibrator()
    cal = np.array([1.0, 5.0, 5.0, 5.0, 5.0])
    assert calibrator.p_value(5.0, cal) == pytest.approx(5 / 6)


def test_p_value_strictly_decreases_with_higher_score():
    calibrator = ConformalCalibrator()
    cal = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    p_low = calibrator.p_value(2.0, cal)
    p_mid = calibrator.p_value(5.0, cal)
    p_high = calibrator.p_value(11.0, cal)
    assert p_high <= p_mid <= p_low


def test_p_value_is_bounded_within_interval():
    calibrator = ConformalCalibrator()
    rng = np.random.default_rng(seed=42)
    cal = rng.normal(size=100)
    n = len(cal)
    for test_score in rng.normal(size=50):
        p = calibrator.p_value(float(test_score), cal)
        assert 1.0 / (n + 1) <= p <= 1.0


def test_p_value_empty_calibration_raises_validation_error():
    calibrator = ConformalCalibrator()
    with pytest.raises(ValidationError) as exc_info:
        calibrator.p_value(0.5, np.array([], dtype=np.float64))
    assert exc_info.value.field == "cal_scores"


def test_p_value_array_empty_calibration_raises_validation_error():
    calibrator = ConformalCalibrator()
    with pytest.raises(ValidationError) as exc_info:
        calibrator.p_value_array(np.array([0.5, 1.5]), np.array([], dtype=np.float64))
    assert exc_info.value.field == "cal_scores"


def test_p_value_array_matches_scalar_p_value_per_element():
    calibrator = ConformalCalibrator()
    rng = np.random.default_rng(seed=123)
    cal = rng.normal(size=50)
    scores = rng.normal(size=20)
    batch_result = calibrator.p_value_array(scores, cal)
    scalar_result = np.array([calibrator.p_value(float(s), cal) for s in scores])
    np.testing.assert_array_almost_equal(batch_result, scalar_result)


def test_p_value_array_handles_ties_consistently_with_scalar():
    calibrator = ConformalCalibrator()
    cal = np.array([1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0])
    scores = np.array([2.0, 3.0, 3.5])
    batch = calibrator.p_value_array(scores, cal)
    scalar = np.array([calibrator.p_value(float(s), cal) for s in scores])
    np.testing.assert_array_equal(batch, scalar)


def test_p_value_array_preserves_input_length():
    calibrator = ConformalCalibrator()
    cal = np.array([1.0, 2.0, 3.0])
    scores = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    result = calibrator.p_value_array(scores, cal)
    assert len(result) == 5


def test_empirical_coverage_holds_for_exchangeable_data():
    calibrator = ConformalCalibrator()
    rng = np.random.default_rng(seed=2026)
    alpha = 0.05
    n_cal = 200
    n_test = 5000

    cal_raw = rng.normal(size=n_cal)
    test_raw = rng.normal(size=n_test)
    cal_scores = np.abs(cal_raw)
    test_scores = np.abs(test_raw)

    p_values = calibrator.p_value_array(test_scores, cal_scores)
    false_positive_rate = float(np.mean(p_values <= alpha))

    assert (
        false_positive_rate <= alpha + 0.015
    ), f"FPR {false_positive_rate:.4f} exceeded alpha {alpha} + slack"
