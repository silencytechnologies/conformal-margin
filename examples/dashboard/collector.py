"""Dashboard state: in-memory signal/agent collector and template engine.

The `_Collector` dataclass plays two roles. As an `AlertHandler` it
receives every emitted `OperatorSignal` from the `FleetMonitor` and
retains the most recent 1000 records. As a state container it tracks
per-agent metadata (last prompt, last response, last risk score) and
demo-loop status so the templates can render meaningful empty-state
copy ("starting", "loading", "running", "failed").

Two module-level singletons live here:

- `_COLLECTOR`: the one-and-only handler instance shared by the
  FleetMonitor (writer) and every HTTP endpoint (reader). A demo is
  single-process, single-tenant, so module-level state is fine.
- `_TEMPLATES`: the Jinja2 environment. Templates render against the
  collector state directly through context dicts.
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import ClassVar

from fastapi.templating import Jinja2Templates

from conformal_margin.ontology.domain.ontology import Role
from conformal_margin.ontology.ports.operator_signal import OperatorSignalLike

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
TEMPLATES_DIR = Path(__file__).parent / "templates"


@dataclass
class _Collector:
    """AlertHandler that retains recent signals for the API endpoints.

    Also tracks per-agent stats and demo-loop status so the dashboard
    can render meaningful empty states ("no agents yet", "model
    warming up", "load failed").
    """

    name: str = "dashboard-collector"
    role: ClassVar = Role.SINK
    max_signals: int = 1000
    signals: deque[OperatorSignalLike] = field(default_factory=deque)
    agent_signal_counts: dict[str, int] = field(default_factory=dict)
    agent_last_risk: dict[str, float] = field(default_factory=dict)
    agent_last_seen: dict[str, datetime] = field(default_factory=dict)
    agent_last_prompt: dict[str, str] = field(default_factory=dict)
    agent_last_response: dict[str, str] = field(default_factory=dict)
    status: str = "starting"
    status_message: str = "Dashboard starting up."
    model_id: str = MODEL_ID

    async def handle(self, signal: OperatorSignalLike) -> None:
        self.signals.appendleft(signal)
        while len(self.signals) > self.max_signals:
            self.signals.pop()
        for agent_id in signal.affected_agents:
            self.agent_signal_counts[agent_id] = (
                self.agent_signal_counts.get(agent_id, 0) + 1
            )


_COLLECTOR = _Collector()
_TEMPLATES = Jinja2Templates(directory=str(TEMPLATES_DIR))
