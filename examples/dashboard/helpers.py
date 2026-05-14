"""Aggregations over `_COLLECTOR` state.

Every function here is read-only: it inspects current collector state
and returns a derived value. No mutations. No side effects. Each one
is fast (O(N) over recent signals, N <= 1000).

These exist so the HTTP endpoint code in `api.py` stays focused on
request/response wiring and template selection. Anything that boils
down to "summarise the deque" lives here.
"""

from datetime import datetime, timedelta, timezone
from typing import Any

from conformal_margin.ontology.domain.ontology import Severity, SignalKind

from .collector import _COLLECTOR

UTC = timezone.utc


def _agent_rows() -> list[dict[str, Any]]:
    """Per-agent summary rows, sorted by signal count descending."""
    ordered = sorted(
        _COLLECTOR.agent_last_seen.keys(),
        key=lambda a: (-_COLLECTOR.agent_signal_counts.get(a, 0), a),
    )
    return [
        {
            "agent_id": aid,
            "signal_count": _COLLECTOR.agent_signal_counts.get(aid, 0),
            "last_risk": _COLLECTOR.agent_last_risk.get(aid, 0.0),
            "last_seen": _COLLECTOR.agent_last_seen.get(aid),
            "last_prompt": _COLLECTOR.agent_last_prompt.get(aid, ""),
            "last_response": _COLLECTOR.agent_last_response.get(aid, ""),
        }
        for aid in ordered
    ]


def _count_signals_since(window: timedelta) -> int:
    cutoff = datetime.now(UTC) - window
    return sum(1 for s in _COLLECTOR.signals if s.timestamp >= cutoff)


def _signals_by_severity_last(window: timedelta) -> dict[str, int]:
    cutoff = datetime.now(UTC) - window
    counts = {sev.value: 0 for sev in Severity}
    for s in _COLLECTOR.signals:
        if s.timestamp >= cutoff:
            counts[s.severity.value] += 1
    return counts


def _mean_p_value_last(window: timedelta) -> float | None:
    cutoff = datetime.now(UTC) - window
    ps = [
        s.p_value
        for s in _COLLECTOR.signals
        if s.timestamp >= cutoff and s.kind is SignalKind.ANOMALY
    ]
    return sum(ps) / len(ps) if ps else None


def _drift_active_recently() -> bool:
    cutoff = datetime.now(UTC) - timedelta(minutes=5)
    return any(
        s.kind is SignalKind.DRIFT and s.timestamp >= cutoff for s in _COLLECTOR.signals
    )
