<div align="center">

<br>

# Conformal Margin

<br>

Defend the false-positive rate<br>before the incident review starts.

<br>

<a href="#about"><img src="img/discover-button.svg" alt="Discover"></a>

<br>

<sup>calibrated anomaly detection for autonomous ai agents</sup>


<a href="#about">About</a>&nbsp;&nbsp;&nbsp;&bull;&nbsp;&nbsp;&nbsp;<a href="#the-question">The Question</a>&nbsp;&nbsp;&nbsp;&bull;&nbsp;&nbsp;&nbsp;<a href="#architecture">Architecture</a>&nbsp;&nbsp;&nbsp;&bull;&nbsp;&nbsp;&nbsp;<a href="#baselineoperator">BaselineOperator</a>&nbsp;&nbsp;&nbsp;&bull;&nbsp;&nbsp;&nbsp;<a href="#detection-vs-enforcement">Detection vs Enforcement</a>&nbsp;&nbsp;&nbsp;&bull;&nbsp;&nbsp;&nbsp;<a href="#live-observability">Live Observability</a>&nbsp;&nbsp;&nbsp;&bull;&nbsp;&nbsp;&nbsp;<a href="#implementation">Implementation</a>&nbsp;&nbsp;&nbsp;&bull;&nbsp;&nbsp;&nbsp;<a href="#connect">Connect</a>

<br>

![Python](https://img.shields.io/badge/Python_3.12+-1A191A?style=flat-square&logo=python&logoColor=EDE5DA) ![numpy](https://img.shields.io/badge/numpy-1A191A?style=flat-square&logo=numpy&logoColor=EDE5DA) ![scipy](https://img.shields.io/badge/scipy-1A191A?style=flat-square&logo=scipy&logoColor=EDE5DA) ![Conformal Prediction](https://img.shields.io/badge/Conformal_Prediction-1A191A?style=flat-square&logoColor=EDE5DA) ![Robust Statistics](https://img.shields.io/badge/Robust_Statistics-1A191A?style=flat-square&logoColor=EDE5DA) ![anyio](https://img.shields.io/badge/anyio-1A191A?style=flat-square&logoColor=EDE5DA) ![Protocols](https://img.shields.io/badge/Protocols-1A191A?style=flat-square&logoColor=EDE5DA)

</div>

<br>

<div align="center">

<img src="img/conformal-margin-cover.png" alt="conformal-margin" width="900">

</div>

<br>

<br>

## About

Most AI anomaly detection tools promise "alerts when agents misbehave" and quietly degrade to heuristics: hard-coded thresholds, fitted Gaussians, keyword filters. None can answer the question that matters at incident review: **how many of yesterday's alerts were false?**

This repository answers that question with a single formal bound, and gates every design decision on preserving it. Under exchangeability between calibration data and current observations, the probability that the realised false-positive rate exceeds a customer-chosen alpha is itself bounded by alpha. This is **exact**, not asymptotic. It is what split conformal prediction (Vovk, Gammerman, Shafer) delivers when the implementation refuses shortcuts.

Not an alerting library. A library that defends the false-positive rate of every alert it emits. Before the incident review starts.

<br>

<br>

## The Question

A useful anomaly detector answers three questions for the team that has to act on its output:

<table>
<tr>
<td width="33%" valign="top">

### What is anomalous?

Each agent in the fleet carries a customer-computed `risk_score`. The library scores how unusual that value is relative to the calibration history, using a robust statistic (median plus scaled Median Absolute Deviation) that survives a few poisoned calibration points.

</td>
<td width="33%" valign="top">

### How sure are we?

Each nonconformity score becomes a **calibrated p-value** via inductive split conformal prediction. The p-value carries the formal coverage bound `P(FPR > alpha) <= alpha`. No approximate quantiles, no asymptotic claims. Exact rank statistics over a sorted calibration set.

</td>
<td width="33%" valign="top">

### When does the model stop working?

The coverage bound assumes exchangeability between calibration and current observations. A drift detector (Page-Hinkley, one-directional) flags upward mean shifts in the score stream. When drift fires, the bound is no longer reliable, and you know to recalibrate.

</td>
</tr>
</table>

<br>

The three components compose into one operator. Each is replaceable through a Protocol contract. The library never adapts to a specific agent runtime; you adapt your runtime to the library.

<br>

<br>

## Architecture

Four data-flow roles, each defined by a Protocol contract in the `conformal_margin.ontology` submodule. The library ships components for **Transformer**, **Store**, and **Sink**. **Source** is your integration. Any agent runtime (LangChain, LangGraph, custom orchestrator) adapts its telemetry into the library's vocabulary.

<br>

<div align="center">

<img src="img/conformal-margin-architecture-Silency-Technologies.png" alt="conformal-margin architecture" width="900">

<sub>Source &rarr; Transformer &rarr; Store &rarr; Sink. Three roles ship in the library; Source is your integration.</sub>

</div>

<br>

| Role | Protocol | Library ships | Customer owns |
|:---|:---|:---|:---|
| **Source** | `AgentStateLike`, `FleetSnapshotLike` | `AgentState`, `FleetSnapshot` msgspec.Struct | Telemetry adapter |
| **Transformer** | `Operator` | `BaselineOperator` (default) | Custom operators if needed |
| **Store** | `StateStore` | `InMemoryStore` (dev) | `RedisStore`, `DuckDBStore`, distributed |
| **Sink** | `AlertHandler`, `FeedbackPort` | `LogHandler`, `FilteredHandler`, `AuditFeedbackHandler` | Enforcement handlers, paging adapters |

<br>

Every public surface is `async def`. Every canonical type is a frozen `msgspec.Struct`. Every numerical path runs on `numpy` plus `scipy`. Streaming-approximate quantile libraries (t-digest, Greenwald-Khanna) are barred from the critical path. They break the FPR bound.

<br>

<br>

## BaselineOperator

The Robust-Conformal Baseline architecture composes three sub-components in a strict order: scoring, then calibration, then drift detection. Each step has its own responsibility and its own failure mode.

<br>

<div align="center">

<img src="img/conformal-margin-baseline-operator-Silency-Technologies.png" alt="BaselineOperator internals" width="700">

<sub>RobustScorer + ConformalCalibrator + DriftDetector compose the Robust-Conformal Baseline.</sub>

</div>

<br>

### Why Robust, Why Conformal, Why Drift

A standard z-score detector fails on agentic data for three structural reasons. The library addresses each one:

- **Mean and standard deviation are not robust.** A handful of poisoned calibration points inflates the standard deviation and lowers every subsequent z-score, masking real anomalies. `RobustScorer` uses median and scaled Median Absolute Deviation; both are statistically robust to a fraction of outliers.
- **Gaussian assumptions do not hold.** Risk-score distributions are skewed, multimodal, and shift under regime change. `ConformalCalibrator` is distribution-free: the coverage bound holds for any underlying distribution, as long as calibration and test data are exchangeable.
- **Exchangeability breaks under drift.** When the data-generating distribution shifts, the calibration set no longer represents current behaviour, and the formal bound stops applying. `DriftDetector` flags the break before it silently degrades the bound.

<br>

### Sub-component Responsibilities

| Component | Math | Output |
|:---|:---|:---|
| **RobustScorer** | median(C) + 1.4826 × MAD(C); `s(x) = |x - median| / scaled_MAD` | per-agent nonconformity score |
| **ConformalCalibrator** | `p = (1 + count(c >= s)) / (n + 1)` over sorted C | calibrated p-value in `[1/(n+1), 1]` |
| **DriftDetector** | `m_t = sum(x_i - mu_0 - delta)`; `PH_t = m_t - min(m_<=t)`; alarm at `PH_t > lambda` | boolean drift flag + statistic |

<br>

<details>
<summary><strong>Example - Configuring the operator</strong></summary>

<br>

```python
from conformal_margin import AppSettings, FleetMonitor, from_settings

settings = AppSettings(
    operator_name="baseline",
    alpha=0.01,                          # target false-positive rate
    calibration_window_seconds=3600,     # one hour of history
    min_calibration_size=30,             # below 30 samples, emit nothing
    drift_delta=0.005,                   # Page-Hinkley allowable mean shift
    drift_threshold=50.0,                # detection threshold
)
monitor: FleetMonitor = from_settings(settings)
```

Every parameter has a sensible default, every default is overridable via environment variables prefixed `CONFORMAL_MARGIN_`. The `AppSettings` class is the boundary; the rest of the library reads from it.

</details>

<br>

<br>

## Detection vs Enforcement

The library detects. The library does not enforce. That split is a deliberate architectural commitment. Suspending an agent, throttling its request rate, quarantining its outputs, paging the on-call: these are your responsibility, implemented as a custom `AlertHandler` that talks to your runtime.

<br>

<div align="center">

<img src="img/conformal-margin-detection-vs-enforcement-Silency-Technologies.png" alt="Detection vs Enforcement boundary" width="800">

<sub>Library detects with a formal FPR bound. You enforce via a custom AlertHandler that talks to your runtime.</sub>

</div>

<br>

### Five Reasons This Split Exists

| Reason | Why it matters |
|:---|:---|
| **Ownership** | Your agent runtime is yours. The library has no admin API into LangChain or LangGraph and should not pretend to. |
| **FPR cost** | At alpha = 0.01, a fleet of one thousand agents observed each minute produces around ten false positives per day. Auto-blocking on every one would suspend legitimate agents for statistical noise. |
| **Adversarial robustness** | An attacker who can shape one agent's risk-score distribution can deliberately push false positives onto a competitor or victim. Enforcement decisions need access to identity, request patterns, and business context the library does not have. |
| **Performance** | Per-snapshot detection is cheap; per-request inline blocking would put the library in the agent inference hot path. The latency budget is yours, not ours. |
| **Defense in depth** | Conventional security architecture keeps detection, enforcement, and audit as separate layers, often owned by separate teams. The library is the detection layer. |

<br>

### Reference Enforcement Pattern

The [`examples/blocking_handler_demo.py`](examples/blocking_handler_demo.py) script ships a 25-line `BlockingHandler` skeleton. Severity-tier policy: `CRITICAL` suspends, `HIGH` throttles, `MEDIUM` queues for review, `LOW` and `INFO` are observation-only. Copy it, swap the fake runtime for your real one, plug the handler in next to `LogHandler` on the `FleetMonitor`.

<br>

Full design rationale in [`docs/enforcement.md`](docs/enforcement.md).

<br>

<br>

## Live Observability

The reference dashboard at [`examples/dashboard/`](examples/dashboard/) is a FastAPI + HTMX + Tailwind + Chart.js single-page app that loads SmolLM2-135M-Instruct in a background thread, drives it through a fixed prompt cycle, and surfaces every emitted signal in real time. White theme. Sidebar navigation. Mobile responsive. Five pages.

<br>

<div align="center">

<img src="img/conformal-margin-observability-1-overview.png" alt="Overview page" width="900">

<sub>Overview &mdash; Pipeline health, loaded model, four status cards, top-10 agents, signal-rate chart.</sub>

</div>

<br>

<div align="center">

<img src="img/conformal-margin-observability-2-agents.png" alt="Agents page" width="900">

<sub>Agents &mdash; Full list with last prompt, last response, last risk score per agent.</sub>

</div>

<br>

<div align="center">

<img src="img/conformal-margin-observability-3-signals.png" alt="Signals page" width="900">

<sub>Signals &mdash; Filterable, paginated stream. Filter by severity, kind, agent, p-value, score.</sub>

</div>

<br>

<div align="center">

<img src="img/conformal-margin-observability-4-drift-detection.png" alt="Drift Detection page" width="900">

<sub>Drift Detection &mdash; Page-Hinkley events with statistic value, baseline mean, drift event type.</sub>

</div>

<br>

<div align="center">

<img src="img/conformal-margin-observability-5-settings.png" alt="Settings page" width="900">

<sub>Settings &mdash; Read-only view of AppSettings as loaded from environment and `.env` defaults.</sub>

</div>

<br>

The dashboard is one example, not a product. The Sink layer Protocol means any external observability stack (Loki, Datadog, Splunk, Elastic, Prometheus via a v0.2 `MetricsHandler`) plugs in the same way. See [`docs/observability.md`](docs/observability.md) for the vendor-neutral guide.

<br>

<br>

## Implementation

<br>

### Quick Start

Prerequisites: Python 3.12 or newer, plus [`uv`](https://docs.astral.sh/uv/getting-started/installation/) installed.

**Step 1.** Clone the repository and enter it:

```bash
git clone https://github.com/silencytechnologies/conformal-margin
cd conformal-margin
```

**Step 2.** Install everything - runtime library plus the `examples` and `llm-demo` extras. Run this **once**; every demo below assumes it has been done:

```bash
uv sync --all-extras
```

This pulls FastAPI, Uvicorn, Jinja2 (dashboard), and `smolagents`, `transformers`, `torch`, `accelerate` (LLM demos). One-time download; subsequent runs use the cache.

<br>

**Step 3a - Five-second quickstart.** Minimal end-to-end pipeline, no LLM, runs in about a second:

```bash
uv run python -m conformal_margin
```

Builds a `FleetMonitor`, observes a five-agent calibration snapshot, then observes one outlier and prints the emitted `OperatorSignal` records to stdout as colourised structlog events.

<br>

**Step 3b - Full live dashboard.** FastAPI + HTMX + Chart.js with a real `SmolLM2-135M-Instruct` cycling eight prompts in the background:

```bash
uv sync --all-extras   # one-time, if you have not already
uv run python examples/dashboard/main.py
```

Open `http://127.0.0.1:8000/` in your browser. The startup banner prints the URL. Five pages (Overview, Agents, Signals, Drift, Settings) refresh themselves via HTMX every few seconds as the LLM produces responses and the pipeline emits signals. First start downloads roughly 270 MB of model weights into `<project>/models/`; subsequent starts use the local cache. Press `Ctrl+C` to stop.

<br>

**Step 3c - Lighter demos.** Instant, no LLM dependency - these run as soon as `uv sync --all-extras` is done:

```bash
uv run python examples/log_handler_demo.py        # five signals across every severity tier
uv run python examples/blocking_handler_demo.py   # reference enforcement pattern
```

<br>

### End-to-end

A complete runnable script that builds a `FleetMonitor`, fills its calibration history, and detects one outlier against twenty normal agents:

```python
import asyncio
from datetime import datetime, timedelta, timezone

from conformal_margin import AgentState, AppSettings, FleetSnapshot, from_settings
from conformal_margin.ontology.domain.ontology import ModelTier


def _agent(aid: str, risk: float, ts: datetime) -> AgentState:
    return AgentState(
        agent_id=aid, timestamp=ts, risk_score=risk,
        model_tier=ModelTier.FOUNDATION, metadata={},
    )


async def main() -> None:
    monitor = from_settings(AppSettings(alpha=0.05, min_calibration_size=5))
    now = datetime.now(timezone.utc)
    past = now - timedelta(seconds=10)

    # 1. Calibration history: twenty agents at normal risk levels.
    await monitor.observe(FleetSnapshot(
        timestamp=past,
        agents=tuple(
            _agent(f"agent-{i:02d}", 0.5 + (i - 10) * 0.01, past)
            for i in range(20)
        ),
    ))

    # 2. Current snapshot: same fleet plus one clear outlier.
    signals = await monitor.observe(FleetSnapshot(
        timestamp=now,
        agents=tuple(
            _agent(f"agent-{i:02d}", 0.5 + (i - 10) * 0.01, now)
            for i in range(20)
        ) + (_agent("agent-bad", 5.0, now),),
    ))

    # 3. Each signal carries a calibrated p-value with the formal FPR bound.
    for sig in signals:
        if sig.threshold_exceeded:
            print(
                f"{sig.severity.value}: {sig.affected_agents[0]} "
                f"(p_value={sig.p_value:.4f}, score={sig.score:.2f})"
            )


asyncio.run(main())
```

Expected output:

```
medium: agent-bad (p_value=0.0476, score=60.77)
```

The first `observe` populates the calibration history; nothing is emitted because there is no history to score against. The second observation produces one `OperatorSignal` per agent, with `threshold_exceeded` set when the calibrated p-value falls at or below the configured alpha. The outlier crosses the boundary at the minimum p-value the calibration size allows (`1 / (n_cal + 1) = 1 / 21 = 0.0476`); the twenty calibration agents do not. Severity tier is `medium` because the p-value sits between `alpha` and `alpha * 0.10`; larger calibration sets push it deeper into the tail and escalate the tier to `high` or `critical`.

<br>

### Four Runnable Demonstrations

| Demo | Purpose | Run |
|:---|:---|:---|
| **log_handler_demo** | Five signals at every severity tier, console structlog rendering | `uv run python examples/log_handler_demo.py` |
| **blocking_handler_demo** | Reference enforcement pattern. Severity-tier policy against a fake runtime stub | `uv run python examples/blocking_handler_demo.py` |
| **smolagents_demo** | End-to-end with HuggingFace SmolLM2-135M-Instruct. Load model, generate, score, emit signals | `uv run python examples/smolagents_demo/main.py` |
| **dashboard** | FastAPI + HTMX + Tailwind + Chart.js live dashboard at `http://127.0.0.1:8000/` | `uv run python examples/dashboard/main.py` |

<br>

The dashboard and smolagents demos cache SmolLM2-135M-Instruct weights in `<project>/models/`, not in the user's HuggingFace home cache. First run downloads roughly 270 megabytes; subsequent runs use the local cache.

<br>

<details>
<summary><strong>Example - Wiring a FleetMonitor in three lines</strong></summary>

<br>

```python
from conformal_margin import from_env

monitor = from_env()                              # AppSettings from environment
signals = await monitor.observe(snapshot)         # detect anomalies
# enforcement: register your own BlockingHandler on the monitor
```

`from_env()` reads `CONFORMAL_MARGIN_*` environment variables and `.env` file, builds an `AppSettings` instance, wires a `BaselineOperator` plus `InMemoryStore` plus `LogHandler`, and returns a ready `FleetMonitor`. Customers who need custom topology (multiple operators, custom filters, audit feedback) construct `FleetMonitor` directly.

</details>

<br>

### Project Structure

```
conformal-margin/
│
├── pyproject.toml                              # single package; hatchling + uv
├── README.md                                   # this file
├── LICENSE                                     # Silency Non-Commercial v1.0
│
├── docs/                                       # user-facing guides
│   ├── observability.md                        # how to pipe events into your log stack
│   └── enforcement.md                          # how to wire a BlockingHandler
│
├── src/
│   └── conformal_margin/
│       ├── __init__.py                         # public API surface
│       ├── __main__.py                         # `python -m conformal_margin` quickstart
│       ├── agent_state.py                      # AgentState msgspec.Struct
│       ├── fleet_snapshot.py                   # FleetSnapshot msgspec.Struct
│       ├── operator_signal.py                  # OperatorSignal msgspec.Struct
│       ├── config.py                           # AppSettings (pydantic-settings)
│       ├── factory.py                          # from_settings, from_env
│       ├── fleet_monitor.py                    # the orchestrator
│       ├── filters.py                          # SeverityFilter, KindFilter, DriftEventTypeFilter
│       ├── operators/                          # BaselineOperator + RobustScorer + ConformalCalibrator + DriftDetector
│       ├── stores/                             # InMemoryStore
│       ├── handlers/                           # LogHandler, FilteredHandler, AuditFeedbackHandler
│       └── ontology/                           # canonical vocabulary submodule
│           ├── domain/                         # closed enums + exception hierarchy
│           └── ports/                          # Protocol contracts (zero non-stdlib deps in source)
│
├── tests/                                      # 308 tests, every public surface
├── examples/
│   ├── log_handler_demo.py                     # 5 signals across every severity tier
│   ├── blocking_handler_demo.py                # reference enforcement pattern
│   ├── smolagents_demo/main.py                 # end-to-end with SmolLM2-135M-Instruct
│   └── dashboard/                              # FastAPI + HTMX + Chart.js live dashboard
│       ├── main.py                             # entry point, FastAPI app, lifespan, port handling
│       ├── collector.py                        # in-memory signal/agent collector
│       ├── helpers.py                          # read-only aggregations over collector state
│       ├── llm_loop.py                         # model load + background prompt cycle
│       ├── routes.py                           # full-HTML page routes
│       ├── api.py                              # HTMX fragment endpoints (mounted at /api)
│       └── templates/                          # Jinja2 HTML templates
└── img/                                        # architecture and dashboard images
```

<br>

<br>

## Scope &amp; Limitations

`conformal-margin` v0.1 is a single-tenant, batch-snapshot anomaly detection library. It intentionally does not cover:

- **In-line enforcement** &mdash; the library never blocks, throttles, or terminates agents on its own. Enforcement is yours; see `docs/enforcement.md`.
- **Real-time streaming inference** &mdash; operators run on `FleetSnapshot` batches at observation cadence (typical: once per minute). Per-request inline scoring is out of scope for v0.1.
- **Calibration set adjustment from feedback** &mdash; `AuditFeedbackHandler` logs labelled judgements. The library does not feed them back into the calibrator. Calibration-poisoning resistance is the reason.
- **Production-scale Store** &mdash; `InMemoryStore` is dev-grade. `RedisStore`, `DuckDBStore`, distributed backends are deferred to v0.2 and beyond.
- **Multi-tenant trust boundaries** &mdash; one fleet, one operator chain, one observability surface.

Two sibling repositories define the theory this library builds on:

- **[ai-trust-boundary-patterns](https://github.com/silencytechnologies/ai-trust-boundary-patterns)** &mdash; architectural trust-boundary patterns for agentic systems. Places `conformal-margin` at the SOC and observability boundary of the reference architecture.
- **[probabilistic-threat-modeling](https://github.com/silencytechnologies/probabilistic-threat-modeling)** &mdash; StochasticSTRIDE and probabilistic threat-modeling primitives. Defines the upstream that produces the per-agent `risk_score` this library calibrates. The two repositories share the `AgentState` and `ModelTier` vocabulary.

Adapting these patterns to streaming workloads, multi-tenant deployments, and active feedback ingestion is future work for v0.2 and beyond.

<br>

<br>

## Connect

<div align="center">

<a href="https://silency.io?utm_source=github&utm_medium=readme&utm_campaign=conformal-margin">
  <img src="https://img.shields.io/badge/Website-1A191A?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IiNhNjliOGUiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIj48Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSIxMCIvPjxwYXRoIGQ9Ik0yIDEyaDIwIi8+PHBhdGggZD0iTTEyIDJhMTUuMyAxNS4zIDAgMCAxIDQgMTAgMTUuMyAxNS4zIDAgMCAxLTQgMTAgMTUuMyAxNS4zIDAgMCAxLTQtMTAgMTUuMyAxNS4zIDAgMCAxIDQtMTAiLz48L3N2Zz4=&logoColor=a69b8e" alt="Website">
</a>&nbsp;
<a href="https://www.linkedin.com/in/silency-technologies-a2378239a/">
  <img src="https://img.shields.io/badge/LinkedIn-1A191A?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0iI2E2OWI4ZSI+PHBhdGggZD0iTTIwLjQ1IDIwLjQ1aC0zLjU1di01LjU3YzAtMS4zMy0uMDItMy4wNC0xLjg1LTMuMDQtMS44NSAwLTIuMTQgMS40NS0yLjE0IDIuOTR2NS42N0g5LjM0VjloMy40MXYxLjU2aC4wNWMuNDgtLjkgMS42NC0xLjg1IDMuMzctMS44NSAzLjYgMCA0LjI3IDIuMzcgNC4yNyA1LjQ1djYuMjlaTTUuMzQgNy40M2MtMS4xNCAwLTIuMDYtLjkyLTIuMDYtMi4wNiAwLTEuMTQuOTItMi4wNiAyLjA2LTIuMDYgMS4xNCAwIDIuMDYuOTIgMi4wNiAyLjA2IDAgMS4xNC0uOTIgMi4wNi0yLjA2IDIuMDZ6TTcuMTIgMjAuNDVIMy41NlY5aDMuNTZ2MTEuNDV6Ii8+PC9zdmc+" alt="LinkedIn">
</a>&nbsp;

<br>

<sub>Built by <a href="https://silency.io?utm_source=github&utm_medium=readme&utm_campaign=conformal-margin">Silency Technologies</a></sub>
<br>
<sub>&copy; 2026 Silency Technologies. All rights reserved.</sub>

<br><br>

</div>
