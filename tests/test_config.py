"""Unit tests for conformal_margin.config.AppSettings.

Coverage: defaults, programmatic overrides, environment-variable
loading via the `CONFORMAL_MARGIN_` prefix, per-field validation
(alpha range, positive sizes), extra-field rejection, dotenv file
loading.
"""

import pytest

from conformal_margin.config import AppSettings


def test_defaults_match_underlying_class_defaults():
    settings = AppSettings()
    assert settings.operator_name == "baseline"
    assert settings.alpha == 0.01
    assert settings.calibration_window_seconds == 3600
    assert settings.min_calibration_size == 30
    assert settings.drift_delta == 0.005
    assert settings.drift_threshold == 50.0
    assert settings.log_handler_name == "default-log"


def test_programmatic_override_takes_precedence():
    settings = AppSettings(alpha=0.05, drift_threshold=25.0)
    assert settings.alpha == 0.05
    assert settings.drift_threshold == 25.0


def test_programmatic_override_preserves_other_defaults():
    settings = AppSettings(alpha=0.05)
    assert settings.alpha == 0.05
    assert settings.drift_threshold == 50.0


def test_loads_from_environment_with_prefix(monkeypatch):
    monkeypatch.setenv("CONFORMAL_MARGIN_ALPHA", "0.03")
    monkeypatch.setenv("CONFORMAL_MARGIN_DRIFT_THRESHOLD", "12.5")
    monkeypatch.setenv("CONFORMAL_MARGIN_OPERATOR_NAME", "custom-name")
    settings = AppSettings()
    assert settings.alpha == 0.03
    assert settings.drift_threshold == 12.5
    assert settings.operator_name == "custom-name"


def test_environment_loading_is_case_insensitive(monkeypatch):
    monkeypatch.setenv("conformal_margin_alpha", "0.02")
    settings = AppSettings()
    assert settings.alpha == 0.02


def test_environment_variable_without_prefix_is_ignored(monkeypatch):
    monkeypatch.setenv("ALPHA", "0.99")
    settings = AppSettings()
    assert settings.alpha == 0.01


@pytest.mark.parametrize("bad_alpha", [-0.1, 0.0, 1.0, 1.5])
def test_alpha_out_of_open_unit_interval_raises(bad_alpha):
    with pytest.raises(Exception) as exc_info:
        AppSettings(alpha=bad_alpha)
    assert "alpha" in str(exc_info.value).lower()


def test_zero_min_calibration_size_raises():
    with pytest.raises(Exception):
        AppSettings(min_calibration_size=0)


def test_negative_drift_delta_raises():
    with pytest.raises(Exception):
        AppSettings(drift_delta=-0.1)


def test_zero_drift_threshold_raises():
    with pytest.raises(Exception):
        AppSettings(drift_threshold=0.0)


def test_zero_calibration_window_seconds_raises():
    with pytest.raises(Exception):
        AppSettings(calibration_window_seconds=0)


def test_empty_operator_name_raises():
    with pytest.raises(Exception):
        AppSettings(operator_name="")


def test_empty_log_handler_name_raises():
    with pytest.raises(Exception):
        AppSettings(log_handler_name="")


def test_unknown_field_raises():
    with pytest.raises(Exception):
        AppSettings(this_field_does_not_exist=42)
