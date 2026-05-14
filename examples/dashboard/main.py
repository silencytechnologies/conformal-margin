"""conformal-margin example dashboard.

Live web dashboard wired to a real HuggingFace SmolLM2-135M-Instruct
language model. On startup the dashboard pins HF_HOME to the
project-local `models/` directory, loads the model in a background
thread, and starts cycling through a fixed batch of prompts. Each
model response becomes one `AgentState`; snapshots flow through
`BaselineOperator` and emit real `OperatorSignal` values that
populate the dashboard panels.

No synthetic data, no dummy values. When the dashboard has not yet
seen any signals, panels show explicit "no agents yet" or "model
warming up" placeholders.

Stack:

- FastAPI for HTTP routes (anyio compatible)
- Jinja2 for HTML templates
- HTMX (CDN) for HTML-fragment polling
- Tailwind CSS (Play CDN) for white-theme styling
- Chart.js (CDN) for live charts

Run with:
    uv run python examples/dashboard/main.py

The dashboard is available at http://127.0.0.1:8000/ and loads the
SmolLM2-135M weights (~270 megabytes) into `<project>/models/` on
first start; subsequent starts use the local cache.

The implementation is split across this entry point and five
sibling modules so each piece stays small enough to read in one
sitting:

- `collector` - the in-memory state plus the Jinja2 environment
- `helpers`   - read-only aggregations over collector state
- `llm_loop`  - model load and the background prompt cycle
- `routes`    - full-page HTML responses for the nav tabs
- `api`       - HTMX fragment endpoints, mounted under `/api`

This file is the entry point: HF_HOME pinning, FastAPI app
construction, lifespan, port handling, and the `uvicorn.run` call.
"""

import os
import signal as _signal
import socket
import subprocess
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MODELS_DIR = _PROJECT_ROOT / "models"
_MODELS_DIR.mkdir(exist_ok=True)
os.environ.setdefault("HF_HOME", str(_MODELS_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(_MODELS_DIR / "hub"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

sys.path.insert(0, str(_PROJECT_ROOT))

import asyncio
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from conformal_margin import AppSettings, from_settings
from conformal_margin.fleet_monitor import FleetMonitor
from examples.dashboard.api import router as api_router
from examples.dashboard.collector import _COLLECTOR, MODEL_ID
from examples.dashboard.llm_loop import demo_loop
from examples.dashboard.routes import router as pages_router


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Build the monitor and spawn the LLM demo loop on startup."""
    base = from_settings(
        AppSettings(min_calibration_size=2, alpha=0.10, operator_name="dashboard")
    )
    monitor = FleetMonitor(
        operators=base.operators,
        store=base.store,
        handlers=(_COLLECTOR,),
        calibration_window=base.calibration_window,
    )
    task = asyncio.create_task(demo_loop(monitor))
    try:
        yield
    finally:
        task.cancel()


app = FastAPI(title="conformal-margin dashboard", lifespan=_lifespan)
app.include_router(pages_router)
app.include_router(api_router, prefix="/api")


def _port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Return True if anything is currently bound to (host, port)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((host, port))
        sock.close()
        return False
    except OSError:
        return True


def _kill_port_owner(port: int) -> int | None:
    """Find and kill whoever owns `port`. Returns the killed PID or None.

    On Windows: `netstat -ano` + `taskkill /F /PID`.
    On POSIX: `lsof -t -i:PORT` + `os.kill(pid, SIGTERM)`.
    """
    if sys.platform == "win32":
        result = subprocess.run(
            ["netstat", "-ano"], capture_output=True, text=True, check=False
        )
        for line in result.stdout.splitlines():
            parts = line.split()
            if (
                len(parts) >= 5
                and parts[1].endswith(f":{port}")
                and parts[3] == "LISTENING"
            ):
                pid = int(parts[4])
                subprocess.run(
                    ["taskkill", "/F", "/PID", str(pid)],
                    capture_output=True,
                    check=False,
                )
                return pid
        return None
    else:
        result = subprocess.run(
            ["lsof", "-t", f"-i:{port}"], capture_output=True, text=True, check=False
        )
        for raw_pid in result.stdout.splitlines():
            try:
                pid = int(raw_pid.strip())
                os.kill(pid, _signal.SIGTERM)
                return pid
            except (ValueError, ProcessLookupError):
                continue
        return None


def _free_port(port: int) -> None:
    """Ensure `port` is free; if something is bound, try to kill it."""
    if not _port_in_use(port):
        return
    print(f"[startup] Port {port} already in use - attempting to free it...")
    killed = _kill_port_owner(port)
    if killed is None:
        print(
            f"[startup] WARNING: could not identify the process holding port "
            f"{port}. uvicorn will fail to bind. Free the port manually."
        )
        return
    for _ in range(20):
        time.sleep(0.1)
        if not _port_in_use(port):
            print(f"[startup] Freed port {port} (killed PID {killed}).")
            return
    print(
        f"[startup] WARNING: port {port} still in use after killing PID "
        f"{killed}; uvicorn may fail to bind."
    )


def _print_startup_banner(host: str, port: int) -> None:
    separator = "=" * 64
    url = f"http://{host}:{port}"
    print()
    print(separator)
    print("  conformal-margin v0.1 demo dashboard")
    print(separator)
    print()
    print(f"  Open in browser:   {url}")
    print(f"  Settings page:     {url}/settings")
    print()
    print(f"  Model:             {MODEL_ID}")
    print(f"  Model cache dir:   {_MODELS_DIR}")
    print()
    print("  First start downloads ~270 MB on first launch.")
    print("  Press CTRL+C to stop.")
    print(separator)
    print()


def main() -> None:
    """Entry point: free the port, print the banner, run uvicorn."""
    host, port = "127.0.0.1", 8000
    _free_port(port)
    _print_startup_banner(host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
