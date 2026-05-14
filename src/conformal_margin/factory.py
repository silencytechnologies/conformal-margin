"""Factory functions for building a default `FleetMonitor` from configuration.

Two customer-facing entry points:

- `from_settings(settings)`: explicit construction from a
  pre-built `AppSettings` instance.
- `from_env()`: convenience wrapper that builds `AppSettings`
  from environment variables and `.env`, then calls
  `from_settings`.

The factory wires the **default v0.1 pipeline**:

- `BaselineOperator` configured per `AppSettings`
- `InMemoryStore` (no configurable Store choice in v0.1; production
  deployments construct their `RedisStore` or `DuckDBStore` directly
  and pass to `FleetMonitor` without going through the factory)
- `LogHandler` configured per `AppSettings`

Customers who need different topology (multiple operators, custom
filters via `FilteredHandler`, `AuditFeedbackHandler` for feedback
audit, custom Store) skip the factory and construct `FleetMonitor`
directly. The factory exists for one-liner onboarding, not as the
only way to build the monitor.
"""

from datetime import timedelta

from conformal_margin.config import AppSettings
from conformal_margin.fleet_monitor import FleetMonitor
from conformal_margin.handlers.log import LogHandler
from conformal_margin.operators.baseline import BaselineOperator
from conformal_margin.stores.in_memory import InMemoryStore

__all__ = ["from_env", "from_settings"]


def from_settings(settings: AppSettings) -> FleetMonitor:
    """Wire a default `FleetMonitor` from an `AppSettings` instance.

    Args:
        settings: validated configuration. Construct via
            `AppSettings(...)` programmatically or via
            `AppSettings()` to read from environment / `.env`.

    Returns:
        `FleetMonitor` wired with one `BaselineOperator`, an
        `InMemoryStore`, and one `LogHandler`. Ready to call
        `await monitor.observe(snapshot)`.
    """
    operator = BaselineOperator(
        name=settings.operator_name,
        alpha=settings.alpha,
        min_calibration_size=settings.min_calibration_size,
        drift_delta=settings.drift_delta,
        drift_threshold=settings.drift_threshold,
    )
    store = InMemoryStore()
    handler = LogHandler(name=settings.log_handler_name)
    return FleetMonitor(
        operators=(operator,),
        store=store,
        handlers=(handler,),
        calibration_window=timedelta(seconds=settings.calibration_window_seconds),
    )


def from_env() -> FleetMonitor:
    """Wire a default `FleetMonitor` from environment variables and `.env`.

    Equivalent to `from_settings(AppSettings())`. The bare
    `AppSettings()` call reads any environment variable prefixed
    `CONFORMAL_MARGIN_` and any keys in `.env` in the working
    directory; values not present fall back to documented defaults.

    Returns:
        `FleetMonitor` wired per the resolved settings. The
        one-line onboarding path:

            from conformal_margin import from_env
            monitor = from_env()
            await monitor.observe(snapshot)
    """
    return from_settings(AppSettings())
