"""HTMX fragment endpoints, mounted under `/api`.

Each endpoint returns an HTML fragment (or JSON for the Chart.js
data feed). The pages in `routes.py` use `hx-get` to poll these
fragments and swap them into the DOM without a full page reload.

All endpoints are read-only and synchronous over the in-memory
`_COLLECTOR` deque. Filtering and pagination happen here in Python;
the templates render the already-prepared subset.
"""

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse

from conformal_margin.ontology.domain.ontology import SignalKind
from conformal_margin.ontology.ports.operator_signal import OperatorSignalLike

from .collector import _COLLECTOR, _TEMPLATES
from .helpers import (_agent_rows, _count_signals_since,
                      _drift_active_recently, _mean_p_value_last,
                      _signals_by_severity_last)

UTC = timezone.utc
router = APIRouter()


@router.get("/cards", response_class=HTMLResponse)
async def cards_fragment(request: Request) -> HTMLResponse:
    return _TEMPLATES.TemplateResponse(
        request=request,
        name="_cards.html",
        context={
            "active_agents": len(_COLLECTOR.agent_last_seen),
            "signals_last_hour": _count_signals_since(timedelta(hours=1)),
            "signals_by_severity": _signals_by_severity_last(timedelta(hours=1)),
            "mean_p_value": _mean_p_value_last(timedelta(hours=1)),
            "drift_active": _drift_active_recently(),
            "collector": _COLLECTOR,
        },
    )


@router.get("/top-agents", response_class=HTMLResponse)
async def top_agents_fragment(request: Request) -> HTMLResponse:
    return _TEMPLATES.TemplateResponse(
        request=request,
        name="_top_agents.html",
        context={"rows": _agent_rows()[:10], "collector": _COLLECTOR},
    )


@router.get("/agents-table", response_class=HTMLResponse)
async def agents_table_fragment(request: Request) -> HTMLResponse:
    return _TEMPLATES.TemplateResponse(
        request=request,
        name="_agents_table.html",
        context={"rows": _agent_rows(), "collector": _COLLECTOR},
    )


@router.get("/signals-list", response_class=HTMLResponse)
async def signals_list_fragment(
    request: Request,
    page: int = 1,
    per_page: int = 50,
    severity: list[str] = Query(default=[]),
    kind: str = "",
    agent_query: str = "",
    p_value_max: str = "",
    score_min: str = "",
) -> HTMLResponse:
    """Filterable, paginated signal list. Max 50 records per page.

    Numeric filter fields are typed `str = ""` rather than `float | None`
    because an HTML form posts empty number inputs as empty strings,
    which FastAPI's float coercion treats as a 422 validation error.
    Parsing them manually keeps "no filter" and "filter set to 0"
    distinguishable.
    """
    per_page = max(1, min(per_page, 50))
    severity_set = {s for s in severity if s}
    agent_q = agent_query.strip().lower()
    p_value_max_f: float | None = float(p_value_max) if p_value_max.strip() else None
    score_min_f: float | None = float(score_min) if score_min.strip() else None

    filtered: list[OperatorSignalLike] = []
    for sig in _COLLECTOR.signals:
        if severity_set and sig.severity.value not in severity_set:
            continue
        if kind and sig.kind.value != kind:
            continue
        if agent_q and not any(agent_q in a.lower() for a in sig.affected_agents):
            continue
        if p_value_max_f is not None and sig.p_value > p_value_max_f:
            continue
        if score_min_f is not None and sig.score < score_min_f:
            continue
        filtered.append(sig)

    total = len(filtered)
    total_pages = max(1, (total + per_page - 1) // per_page)
    page = max(1, min(page, total_pages))
    start = (page - 1) * per_page
    end = start + per_page

    return _TEMPLATES.TemplateResponse(
        request=request,
        name="_signals_list.html",
        context={
            "signals": filtered[start:end],
            "page": page,
            "per_page": per_page,
            "total": total,
            "total_pages": total_pages,
            "start_index": (start + 1) if total > 0 else 0,
            "end_index": min(end, total),
            "collector": _COLLECTOR,
        },
    )


@router.get("/drift-events", response_class=HTMLResponse)
async def drift_events_fragment(request: Request) -> HTMLResponse:
    drift_signals = [s for s in _COLLECTOR.signals if s.kind is SignalKind.DRIFT]
    return _TEMPLATES.TemplateResponse(
        request=request,
        name="_drift_events.html",
        context={"signals": drift_signals[:50], "collector": _COLLECTOR},
    )


@router.get("/charts/signal-rate")
async def signal_rate_chart_data() -> JSONResponse:
    now = datetime.now(UTC)
    buckets: list[int] = [0] * 30
    for sig in _COLLECTOR.signals:
        delta_min = int((now - sig.timestamp).total_seconds() // 60)
        if 0 <= delta_min < 30:
            buckets[29 - delta_min] += 1
    labels = [f"-{29 - i}m" for i in range(30)]
    return JSONResponse(
        {
            "labels": labels,
            "datasets": [
                {
                    "label": "Signals per minute",
                    "data": buckets,
                    "borderColor": "rgb(59, 130, 246)",
                    "backgroundColor": "rgba(59, 130, 246, 0.1)",
                    "tension": 0.3,
                    "fill": True,
                }
            ],
        }
    )
