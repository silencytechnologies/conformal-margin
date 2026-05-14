"""Unit tests for conformal_margin.ontology.domain.ontology.

Lock-in tests for the six closed enums. Severity ordering tests are
regression tests for the bug caught at the first integration touch
in `uv sync`: `functools.total_ordering` does not work with StrEnum
because the str base class already provides lexicographic operators
that occupy the decorator's slots. See the Severity class docstring
for the full explanation.
"""

import pytest

from conformal_margin.ontology.domain.ontology import (DriftEventType,
                                                       ErrorCategory,
                                                       ModelTier, Role,
                                                       Severity, SignalKind)


def test_role_has_exactly_four_members():
    assert {r.value for r in Role} == {
        "source",
        "transformer",
        "store",
        "sink",
    }


def test_severity_has_exactly_five_members():
    assert {s.value for s in Severity} == {
        "info",
        "low",
        "medium",
        "high",
        "critical",
    }


def test_severity_numeric_weight_is_zero_indexed():
    assert Severity.INFO.numeric_weight == 0
    assert Severity.LOW.numeric_weight == 1
    assert Severity.MEDIUM.numeric_weight == 2
    assert Severity.HIGH.numeric_weight == 3
    assert Severity.CRITICAL.numeric_weight == 4


def test_severity_orders_by_numeric_weight_not_lexicographic():
    assert Severity.HIGH > Severity.LOW
    assert Severity.LOW < Severity.HIGH


def test_severity_supports_all_four_comparison_operators():
    assert Severity.HIGH > Severity.LOW
    assert Severity.LOW < Severity.HIGH
    assert Severity.HIGH >= Severity.HIGH
    assert Severity.HIGH >= Severity.MEDIUM
    assert Severity.LOW <= Severity.LOW
    assert Severity.LOW <= Severity.MEDIUM


def test_severity_sort_is_info_low_medium_high_critical():
    shuffled = [
        Severity.CRITICAL,
        Severity.INFO,
        Severity.HIGH,
        Severity.LOW,
        Severity.MEDIUM,
    ]
    expected = [
        Severity.INFO,
        Severity.LOW,
        Severity.MEDIUM,
        Severity.HIGH,
        Severity.CRITICAL,
    ]
    assert sorted(shuffled) == expected


def test_severity_equality_with_string_value():
    assert Severity.HIGH == "high"


def test_severity_cross_type_comparison_raises_type_error():
    with pytest.raises(TypeError):
        _ = Severity.HIGH > 3


def test_severity_descriptions_are_distinct_and_nonempty():
    descriptions = {s.description for s in Severity}
    assert len(descriptions) == len(list(Severity))
    assert all(d.strip() for d in descriptions)


def test_signal_kind_has_exactly_two_members():
    assert {s.value for s in SignalKind} == {"anomaly", "drift"}


def test_signal_kind_descriptions_are_nonempty():
    for kind in SignalKind:
        assert kind.description.strip()


def test_error_category_has_exactly_three_members():
    assert {e.value for e in ErrorCategory} == {
        "validation",
        "operational",
        "unexpected",
    }


def test_drift_event_type_has_exactly_three_members():
    assert {d.value for d in DriftEventType} == {
        "mean_shift",
        "variance_shift",
        "distribution_shift",
    }


def test_drift_event_type_descriptions_are_nonempty():
    for dt in DriftEventType:
        assert dt.description.strip()


def test_model_tier_has_exactly_three_members():
    assert {m.value for m in ModelTier} == {
        "slm_edge",
        "foundation",
        "frontier",
    }


def test_model_tier_descriptions_are_nonempty():
    for mt in ModelTier:
        assert mt.description.strip()
