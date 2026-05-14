"""AppSettings - customer-facing boundary configuration.

Pydantic v2 + pydantic-settings model that customers fill in
either programmatically, from environment variables prefixed
`CONFORMAL_MARGIN_`, or from a `.env` file in the working
directory. The model is the input boundary; values are validated
at construction (alpha in `(0, 1)`, drift_threshold > 0, etc.)
and the resulting `AppSettings` instance is what the factory
functions consume.

Pydantic for boundary validation, msgspec for canonical wire
types: the two libraries play complementary roles.
Configuration is heterogeneous, sometimes-fuzzy customer input
that benefits from pydantic's coercion and rich error messages.
Wire-format values that cross process boundaries are msgspec
struct, which is faster and stricter.

The schema is intentionally **flat** in v0.1: no nested
sub-configs, no per-operator overrides. This keeps environment
variable layout obvious (`CONFORMAL_MARGIN_ALPHA=0.05`) and
matches the v0.1 surface (one operator, one store, one log
handler). Nested structure can be added in v0.2+ without
breaking the flat form.

The defaults shipped here are the same defaults declared by the
underlying classes (`BaselineOperator`, `LogHandler`, etc.).
Centralizing them in `AppSettings` gives customers one place to
read or override, and keeps the factory functions simple
(`from_settings` does not need to know fallback values).
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["AppSettings"]


class AppSettings(BaseSettings):
    """Boundary configuration for the default `FleetMonitor` pipeline.

    Customers construct this either programmatically:

        AppSettings(alpha=0.05, drift_threshold=30.0)

    or from environment variables:

        CONFORMAL_MARGIN_ALPHA=0.05 CONFORMAL_MARGIN_DRIFT_THRESHOLD=30.0 python ...

    or from a `.env` file in the working directory. The factory
    function `conformal_margin.factory.from_env` does the bare
    `AppSettings()` call so customers only need one import:

        from conformal_margin import from_env
        monitor = from_env()

    Attributes:
        operator_name: registered name of the single `BaselineOperator`
            wired by the factory. Becomes `operator_name` on every
            emitted signal.
        alpha: target False Positive Rate per the conformal coverage
            guarantee. Customer-configurable. Must be in
            the open interval `(0, 1)`.
        calibration_window_seconds: how far back the FleetMonitor
            asks the Store to load history, in seconds. Default 3600
            (one hour).
        min_calibration_size: minimum number of historical
            observations the operator requires before emitting any
            signal. Below this, the conformal coverage guarantee is
            unreliable so the operator returns zero signals.
        drift_delta: Page-Hinkley allowable mean shift parameter.
        drift_threshold: Page-Hinkley detection threshold.
        log_handler_name: registered name of the default `LogHandler`
            wired by the factory.

    Examples:
        >>> settings = AppSettings(alpha=0.05)
        >>> settings.alpha
        0.05
        >>> settings.operator_name
        'baseline'
    """

    model_config = SettingsConfigDict(
        env_prefix="CONFORMAL_MARGIN_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="forbid",
    )

    operator_name: str = Field(
        default="baseline",
        min_length=1,
        description="Registered operator name; appears on every emitted signal.",
    )
    alpha: float = Field(
        default=0.01,
        gt=0.0,
        lt=1.0,
        description="Target False Positive Rate threshold per the conformal coverage guarantee.",
    )
    calibration_window_seconds: int = Field(
        default=3600,
        gt=0,
        description="How far back to load history from the Store, in seconds.",
    )
    min_calibration_size: int = Field(
        default=30,
        ge=1,
        description="Minimum historical observations before the operator emits signals.",
    )
    drift_delta: float = Field(
        default=0.005,
        ge=0.0,
        description="Page-Hinkley allowable mean shift parameter.",
    )
    drift_threshold: float = Field(
        default=50.0,
        gt=0.0,
        description="Page-Hinkley detection threshold above which drift is flagged.",
    )
    log_handler_name: str = Field(
        default="default-log",
        min_length=1,
        description="Registered name of the default LogHandler.",
    )
