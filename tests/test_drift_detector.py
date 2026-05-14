"""Unit tests for conformal_margin.operators.drift.DriftDetector.

Coverage: construction, parameter validation, statelessness (each
call returns based only on inputs), Page-Hinkley statistic
correctness on stable and drifting streams, one-directional
behavior (decrease does not trigger), empty input handling,
threshold sensitivity, frozen behavior.
"""

import numpy as np
import pytest

from conformal_margin.ontology.domain.exceptions import ValidationError
from conformal_margin.operators.drift import DriftDetector


def test_default_parameters():
    detector = DriftDetector()
    assert detector.delta == 0.005
    assert detector.threshold == 50.0


def test_custom_parameters():
    detector = DriftDetector(delta=0.1, threshold=10.0)
    assert detector.delta == 0.1
    assert detector.threshold == 10.0


def test_negative_delta_raises_validation_error():
    with pytest.raises(ValidationError) as exc_info:
        DriftDetector(delta=-0.01)
    assert exc_info.value.field == "delta"


def test_zero_threshold_raises_validation_error():
    with pytest.raises(ValidationError) as exc_info:
        DriftDetector(threshold=0.0)
    assert exc_info.value.field == "threshold"


def test_negative_threshold_raises_validation_error():
    with pytest.raises(ValidationError) as exc_info:
        DriftDetector(threshold=-5.0)
    assert exc_info.value.field == "threshold"


def test_zero_delta_is_accepted():
    detector = DriftDetector(delta=0.0, threshold=10.0)
    assert detector.delta == 0.0


def test_is_frozen():
    detector = DriftDetector()
    with pytest.raises(AttributeError):
        detector.delta = 0.1


def test_detect_empty_values_returns_no_drift_zero_statistic():
    detector = DriftDetector()
    drift, statistic = detector.detect(
        np.array([], dtype=np.float64), baseline_mean=0.5
    )
    assert drift is False
    assert statistic == 0.0


def test_detect_single_value_at_baseline_no_drift():
    detector = DriftDetector(delta=0.01, threshold=10.0)
    drift, statistic = detector.detect(np.array([0.5]), baseline_mean=0.5)
    assert drift is False
    assert statistic == 0.0


def test_stable_stream_around_baseline_yields_no_drift():
    detector = DriftDetector(delta=0.005, threshold=10.0)
    rng = np.random.default_rng(seed=2026)
    values = rng.normal(loc=0.5, scale=0.01, size=200)
    drift, statistic = detector.detect(values, baseline_mean=0.5)
    assert drift is False
    assert statistic < 10.0


def test_upward_step_change_triggers_drift():
    detector = DriftDetector(delta=0.01, threshold=2.0)
    baseline_segment = np.full(100, 0.5)
    drifted_segment = np.full(100, 1.0)
    values = np.concatenate([baseline_segment, drifted_segment])
    drift, statistic = detector.detect(values, baseline_mean=0.5)
    assert drift is True
    assert statistic > 2.0


def test_decreasing_stream_does_not_trigger_one_directional_detector():
    detector = DriftDetector(delta=0.01, threshold=2.0)
    baseline_segment = np.full(100, 0.5)
    drifted_down = np.full(100, 0.1)
    values = np.concatenate([baseline_segment, drifted_down])
    drift, statistic = detector.detect(values, baseline_mean=0.5)
    assert drift is False


def test_statistic_grows_with_drift_magnitude():
    detector = DriftDetector(delta=0.01, threshold=1000.0)
    baseline_segment = np.full(50, 0.5)

    _, small_stat = detector.detect(
        np.concatenate([baseline_segment, np.full(50, 0.6)]),
        baseline_mean=0.5,
    )
    _, big_stat = detector.detect(
        np.concatenate([baseline_segment, np.full(50, 2.0)]),
        baseline_mean=0.5,
    )
    assert big_stat > small_stat


def test_statistic_is_always_non_negative():
    detector = DriftDetector(delta=0.5, threshold=1000.0)
    rng = np.random.default_rng(seed=99)
    values = rng.normal(loc=0.5, scale=0.5, size=100)
    _, statistic = detector.detect(values, baseline_mean=0.5)
    assert statistic >= 0.0


def test_threshold_sensitivity_higher_threshold_means_less_sensitive():
    baseline_segment = np.full(100, 0.5)
    moderate_drift = np.full(100, 0.55)
    values = np.concatenate([baseline_segment, moderate_drift])

    sensitive = DriftDetector(delta=0.005, threshold=1.0)
    loose = DriftDetector(delta=0.005, threshold=1000.0)

    drift_sensitive, _ = sensitive.detect(values, baseline_mean=0.5)
    drift_loose, _ = loose.detect(values, baseline_mean=0.5)

    assert drift_sensitive is True
    assert drift_loose is False


def test_two_calls_with_same_input_return_same_result():
    detector = DriftDetector(delta=0.01, threshold=5.0)
    values = np.concatenate([np.full(50, 0.5), np.full(50, 1.0)])
    first = detector.detect(values, baseline_mean=0.5)
    second = detector.detect(values, baseline_mean=0.5)
    assert first == second
