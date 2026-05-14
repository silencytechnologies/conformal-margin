"""Unit tests for conformal_margin.ontology.domain.exceptions.

Verifies the exception hierarchy, category attributes, and the
field attributes that boundary handlers read to dispatch on error
kind without parsing exception messages.
"""

import pytest

from conformal_margin.ontology.domain.exceptions import (ConformalMarginError,
                                                         OperationFailedError,
                                                         OperatorTimeoutError,
                                                         SinkRejectionError,
                                                         StoreUnavailableError,
                                                         UnexpectedError,
                                                         UnknownMetricError,
                                                         ValidationError)
from conformal_margin.ontology.domain.ontology import ErrorCategory


def test_base_inherits_from_exception_not_baseexception():
    assert issubclass(ConformalMarginError, Exception)
    assert not issubclass(ConformalMarginError, BaseException) or issubclass(
        ConformalMarginError, Exception
    )


def test_base_default_category_is_unexpected():
    assert ConformalMarginError.category is ErrorCategory.UNEXPECTED


def test_every_library_exception_inherits_from_base():
    for cls in (
        ValidationError,
        UnknownMetricError,
        OperationFailedError,
        StoreUnavailableError,
        SinkRejectionError,
        OperatorTimeoutError,
        UnexpectedError,
    ):
        assert issubclass(cls, ConformalMarginError)


def test_validation_error_caught_by_base_class():
    with pytest.raises(ConformalMarginError):
        raise ValidationError(field="x", reason="empty")


def test_validation_error_category_is_validation():
    assert ValidationError.category is ErrorCategory.VALIDATION


def test_validation_error_carries_field_reason_value():
    exc = ValidationError(field="agent_id", reason="empty string", value="")
    assert exc.field == "agent_id"
    assert exc.reason == "empty string"
    assert exc.value == ""


def test_validation_error_value_defaults_to_none():
    exc = ValidationError(field="agent_id", reason="missing")
    assert exc.value is None


def test_validation_error_message_includes_field_and_reason():
    exc = ValidationError(field="agent_id", reason="empty string")
    text = str(exc)
    assert "agent_id" in text
    assert "empty string" in text


def test_unknown_metric_error_inherits_from_validation_error():
    assert issubclass(UnknownMetricError, ValidationError)


def test_unknown_metric_error_category_is_validation():
    assert UnknownMetricError.category is ErrorCategory.VALIDATION


def test_unknown_metric_error_carries_metric_and_available_attributes():
    exc = UnknownMetricError(
        metric="median_risk", available=("mean_risk_score", "p95_risk_score")
    )
    assert exc.metric == "median_risk"
    assert exc.available == ("mean_risk_score", "p95_risk_score")


def test_unknown_metric_error_inherits_validation_error_fields():
    exc = UnknownMetricError(metric="median_risk", available=())
    assert exc.field == "metric"
    assert "median_risk" in exc.reason
    assert exc.value == "median_risk"


def test_unknown_metric_error_caught_by_validation_error():
    with pytest.raises(ValidationError):
        raise UnknownMetricError(metric="nope", available=("mean_risk_score",))


def test_unknown_metric_error_caught_by_base_class():
    with pytest.raises(ConformalMarginError):
        raise UnknownMetricError(metric="nope", available=())


def test_operation_failed_error_category_is_operational():
    assert OperationFailedError.category is ErrorCategory.OPERATIONAL


def test_operation_failed_error_carries_operation_and_attempt():
    exc = OperationFailedError(operation="save", attempt=3)
    assert exc.operation == "save"
    assert exc.attempt == 3


def test_operation_failed_error_attempt_defaults_to_one():
    exc = OperationFailedError(operation="save")
    assert exc.attempt == 1


def test_operation_failed_error_details_defaults_to_empty_dict():
    exc = OperationFailedError(operation="save")
    assert exc.details == {}


def test_operation_failed_error_preserves_provided_details():
    exc = OperationFailedError(operation="save", details={"host": "store-1"})
    assert exc.details == {"host": "store-1"}


@pytest.mark.parametrize(
    "subclass",
    [StoreUnavailableError, SinkRejectionError, OperatorTimeoutError],
)
def test_operational_subclass_chains_to_operation_failed_error(subclass):
    assert issubclass(subclass, OperationFailedError)
    assert subclass.category is ErrorCategory.OPERATIONAL


def test_chain_via_from_preserves_original_cause():
    original = RuntimeError("connection refused")
    try:
        try:
            raise original
        except RuntimeError as err:
            raise StoreUnavailableError(operation="load_recent") from err
    except StoreUnavailableError as exc:
        assert exc.__cause__ is original


def test_unexpected_error_category_is_unexpected():
    assert UnexpectedError.category is ErrorCategory.UNEXPECTED


def test_unexpected_error_carries_context_and_details():
    exc = UnexpectedError(context="NaN score", details={"operator": "baseline"})
    assert exc.context == "NaN score"
    assert exc.details == {"operator": "baseline"}


def test_unexpected_error_details_defaults_to_empty_dict():
    exc = UnexpectedError(context="cache corrupted")
    assert exc.details == {}
