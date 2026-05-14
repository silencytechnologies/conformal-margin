"""Page routes: full-HTML responses for each top-level navigation tab.

Each route renders one template and returns the full document. The
template extends `base.html` (the sidebar + outer chrome) and fills
the `{% block content %}` placeholder.

HTMX fragment endpoints live in `api.py` and are mounted under the
`/api` prefix by `main.py`.
"""

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from conformal_margin import AppSettings

from .collector import _COLLECTOR, _TEMPLATES

router = APIRouter()


def _ctx(active: str, **extra: Any) -> dict[str, Any]:
    """Build a template context with the active-page marker and collector."""
    return {"active_page": active, "collector": _COLLECTOR, **extra}


@router.get("/", response_class=HTMLResponse)
async def overview(request: Request) -> HTMLResponse:
    return _TEMPLATES.TemplateResponse(
        request=request, name="overview.html", context=_ctx("overview")
    )


@router.get("/agents", response_class=HTMLResponse)
async def agents_page(request: Request) -> HTMLResponse:
    return _TEMPLATES.TemplateResponse(
        request=request, name="agents.html", context=_ctx("agents")
    )


@router.get("/signals", response_class=HTMLResponse)
async def signals_page(request: Request) -> HTMLResponse:
    return _TEMPLATES.TemplateResponse(
        request=request, name="signals.html", context=_ctx("signals")
    )


@router.get("/drift", response_class=HTMLResponse)
async def drift_page(request: Request) -> HTMLResponse:
    return _TEMPLATES.TemplateResponse(
        request=request, name="drift.html", context=_ctx("drift")
    )


@router.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request) -> HTMLResponse:
    settings = AppSettings()
    return _TEMPLATES.TemplateResponse(
        request=request,
        name="settings.html",
        context=_ctx("settings", settings=settings),
    )
