# Observability with conformal-margin

> User-facing guide. Where to plug the library's signals, errors,
> and internal events into your monitoring stack. Vendor-neutral by
> design: we do not ship a "conformal-margin dashboard" and we do
> not depend on Prometheus, OpenTelemetry, Datadog, or any other
> specific backend in the runtime library.

---

## TL;DR

```text
[v0.1, default] 
Pattern 1: pipe structured logs to your existing log aggregator.

[v0.2+] 
Pattern 2: implement MetricsHandler Protocol for counters/histograms.

[optional] 
Pattern 3: build a custom dashboard from the structured event stream.
```

If you already have a log aggregator (Loki, Datadog Logs, Splunk,
Elastic, OpenSearch, Cloudwatch Logs, ...), Pattern 1 gets you 80
percent of useful observability with zero new code. The rest of
this guide explains the three patterns in detail and shows what
goes where.

---

## Pattern 1: structured logs (the default, available today)

The library emits every signal and every internal event as one
`structlog` record. You configure `structlog` once at application startup and choose where the
events go.

### What gets emitted

`LogHandler` (the simplest sink) emits one event per signal:

| Event name           | Severity | What it carries (bound fields)                                          |
|----------------------|----------|-------------------------------------------------------------------------|
| `signal_handled`     | per `Severity` mapping (info / warning / error / critical) | all eleven OperatorSignal fields plus `handler` |
| `signal_filtered_out`| info     | `handler`, `wrapped_handler`, `filter_name`, `signal_id`, `severity`, `kind` |
| `signal_dispatch_failed` | error | `handler`, `signal_id`, `error_type`, `error` (from FleetMonitor)        |
| `feedback_record_failed` | error | `port`, `signal_id`, `error_type`, `error`                              |

All fields are structured (not embedded in a message string), so
log aggregators can query and aggregate without parsing.

### Customer wiring example: pipe to JSON for Loki / Datadog / Splunk

```python
import logging
import structlog

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
)
```

Now every emitted record is one JSON line written to stdout.
Pipe that to:

- **Loki + Grafana**: ship via Promtail / Vector. Grafana LogQL
  panels show signal rate by severity, top noisy operators, drift
  events over time.
- **Datadog**: stdout to Datadog agent. Datadog "Log to Metric"
  generates counters / histograms automatically.
- **Splunk**: forward via HEC. SPL searches and dashboards.
- **Elastic / Kibana**: ship via Filebeat. Kibana visualizations.

### Customer wiring example: human-readable console

For development, replace the JSON renderer with the console one:

```python
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.dev.ConsoleRenderer(colors=True),
    ],
)
```

You get the colourised output shown in
`examples/log_handler_demo.py`.

### What Pattern 1 covers

- Signal rate over time, broken down by severity / kind / operator
- Top affected agents
- Drift events
- Filter rejections (which filter dropped which signal)
- Handler failure rate
- Per-handler latency, if you wrap handlers with timing logs

This is real observability. No new dependencies, no extra Python
classes to write. Just configure `structlog` to emit JSON and
point your existing log shipper at stdout.

---

## Pattern 2: native metrics via the MetricsHandler Protocol (v0.2+)

For deployments that want **first-class counters, histograms, and
gauges** rather than log-derived metrics, v0.2 ships a
`MetricsHandler` Protocol in `conformal_margin.ontology`. The
shape:

```python
@runtime_checkable
class MetricsHandler(Protocol):
    role: Final = Role.SINK
    name: str

    async def record_counter(self, name, value, labels): ...
    async def record_histogram(self, name, value, labels): ...
    async def record_gauge(self, name, value, labels): ...
```

The `FleetMonitor` will call these on emitted signals (and on
its own internal events like operator latency).

### How you plug your stack in

The library will ship **optional adapters** as Python extras:

| You install...                       | You get                                                |
|-------------------------------------|--------------------------------------------------------|
| `conformal-margin[prometheus]`       | `PrometheusMetricsHandler` exposing `/metrics` endpoint |
| `conformal-margin[opentelemetry]`    | `OpenTelemetryMetricsHandler` exporting via OTLP        |
| `conformal-margin[statsd]`           | `StatsDMetricsHandler` for legacy stacks               |
| (nothing)                            | no metrics; you can still write your own adapter        |

The library core **never** pulls any of these as a runtime
dependency. You opt in.

### Status

- Protocol shape: planned for v0.2+.
- Distributed tracing (`TracingHandler`-style for OpenTelemetry
  spans): planned for v0.2+.
- Both arrive together as a coordinated observability pass.

---

## Pattern 3: build your own dashboard (optional)

Some customers want a single-pane-of-glass dashboard showing
recent signals, severity distribution, and drift indicator in a
browser tab. The library does not ship one in the runtime
package, but `/examples/dashboard/` shows how to build one with
FastAPI + HTMX + Tailwind. Open at `localhost:8000`
after `uv run python -m examples.dashboard`.

The example is intended as a starting point, not a product. Fork
it and brand it as part of your own deployment.

---

## Error and incident observability

The library's exception model classifies every raised exception
into one of three categories:

- **`ErrorCategory.VALIDATION`**: caller passed bad input. Fix the
  caller. Aggregate by `field` for "top fields that customers get
  wrong".
- **`ErrorCategory.OPERATIONAL`**: transient or environmental
  failure (Store unreachable, sink rejecting, network timeout).
  Alert on rate, not on individual occurrences.
- **`ErrorCategory.UNEXPECTED`**: library invariant violated.
  Page oncall immediately; this should be zero in steady state.

Every exception has `exc.category` (an `ErrorCategory` enum) and
`exc.__cause__` chains preserve the underlying cause. Boundary
handlers in your code can dispatch on `category` without parsing
exception messages:

```python
try:
    signals = await monitor.observe(snapshot)
except ConformalMarginError as exc:
    match exc.category:
        case ErrorCategory.VALIDATION:
            logger.warning("validation_failed", field=exc.field, reason=exc.reason)
        case ErrorCategory.OPERATIONAL:
            logger.error("operational_failure", operation=exc.operation)
            # retry policy
        case ErrorCategory.UNEXPECTED:
            logger.critical("library_invariant_violated", context=exc.context)
            raise  # propagate to process manager
```

---

## What we explicitly do NOT do

- We do **not** include a UI / dashboard in the runtime library.
  Bundling one would force a tech stack on every deployment.
- We do **not** depend on Prometheus, OpenTelemetry, Datadog,
  Grafana, or any other specific vendor SDK in the runtime
  library. Customers pick their stack.
- We do **not** ship pre-canned Grafana / Kibana / Splunk
  dashboards (yet). Once a few customers ship and we see common
  patterns, we may add a `dashboards/` folder under `/examples`
  with reference JSON / panels. Until then, the structured event
  catalogue above is the source of truth for what to query.
- We do **not** enforce anything on the agent runtime. Observability
  is detection plus alerting; enforcement (suspending, throttling,
  quarantining agents) is a separate concern handled by the
  customer's own `AlertHandler` implementation. See
  `docs/enforcement.md` for the architectural commitment and a
  copy-paste reference handler.

---

## Quick reference: which pattern when

| You already have...                             | Use Pattern... | Estimated wiring time |
|------------------------------------------------|----------------|-----------------------|
| Any log aggregator (Loki, Datadog, Splunk, ...) | 1              | ~10 minutes           |
| Prometheus / OTel infrastructure                | 1 today, 2 in v0.2 | ~30 minutes once v0.2 |
| Nothing yet, just want to see signals          | 3 (`/examples/dashboard/`) | ~5 minutes copy-paste |
| Custom internal tooling                        | 1 with JSONRenderer, then write own consumer | hours              |
