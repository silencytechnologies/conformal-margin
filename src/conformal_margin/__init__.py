"""conformal-margin: fleet-level monitoring of AI agent deployments.

This is the runtime library. It provides concrete implementations of
the contracts defined in the `conformal_margin.ontology` submodule:

- Concrete data types (`AgentState`, `FleetSnapshot`, `OperatorSignal`)
- Operators (`BaselineOperator` composing `RobustScorer`,
  `ConformalCalibrator`, `DriftDetector`)
- Stores (`InMemoryStore` in v0.1)
- Handlers (`LogHandler`, `FilteredHandler`, `AuditFeedbackHandler`)
- Runtime (`FleetMonitor`) and configuration entry points.

The library deliberately exposes a small public surface in v0.1.
For the canonical vocabulary, import from `conformal_margin.ontology`.
"""

from conformal_margin.agent_state import AgentState
from conformal_margin.config import AppSettings
from conformal_margin.factory import from_env, from_settings
from conformal_margin.fleet_monitor import FleetMonitor
from conformal_margin.fleet_snapshot import FleetSnapshot
from conformal_margin.operator_signal import OperatorSignal

VERSION = "0.1.0.dev0"

__all__ = [
    "VERSION",
    "AgentState",
    "AppSettings",
    "FleetMonitor",
    "FleetSnapshot",
    "OperatorSignal",
    "from_env",
    "from_settings",
]
