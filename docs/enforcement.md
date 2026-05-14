# Enforcement with conformal-margin

> User-facing guide. How to translate emitted `OperatorSignal`
> values into real-world actions: suspending agents, throttling
> their request rate, quarantining outputs, paging the on-call.
> The library does not enforce anything on its own; this guide
> explains why, and how you wire enforcement into your own
> infrastructure.

---

## TL;DR

```text
conformal-margin DETECTS anomalies with a formal FPR guarantee.
conformal-margin does NOT block, throttle, or terminate agents.
Enforcement is your job, implemented as a custom AlertHandler.
```

If a customer-supplied AlertHandler talks to your agent runtime
(LangChain, LangGraph, your custom orchestrator), and your
runtime exposes "suspend agent" / "throttle agent" / "reject
output" APIs, you get full enforcement. The library gives you
the signal; you call the runtime.

---

## Why the separation is deliberate

conformal-margin is positioned as a **monitoring layer**, not an
**enforcement layer**. The library commits to this split for v0.1
and beyond. Five reasons:

1. **Authority and ownership.** Your agent runtime is yours.
   The library has no access to its process, its message queue,
   or its admin API. Asking the library to "block an agent"
   would require coupling it to a specific runtime, which is
   exactly what the Protocol-based architecture avoids.
2. **False Positive Rate has a cost.** With the formal guarantee
   `P(FPR > alpha) <= alpha` at `alpha = 0.01`, a fleet of one
   thousand agents observed every minute produces roughly ten
   false-positive signals per day. Automatic blocking on every
   one of them would suspend legitimate agents for statistical
   noise. The human (or business-policy) layer above the library
   decides which severity tiers warrant which response.
3. **Adversarial robustness.** If the library auto-blocked,
   adversaries could deliberately inject inputs designed to
   trigger false-positive signals on legitimate competitor or
   victim agents. The result: denial of service via FPR
   injection. The customer's enforcement layer has access to
   identity, request patterns, business context, and audit
   history. The library does not, and should not.
4. **Performance.** Per-snapshot detection costs are
   acceptable for monitoring (one snapshot per minute, one
   conformal calibration pass). Per-request in-line blocking
   would put the library in the agent inference hot path; the
   p-value computation is not free, and the calibration set
   lookup adds latency. Monitoring is asynchronous; enforcement
   is in-line, and the customer's stack is the right place for
   in-line decisions.
5. **Defense in depth.** Conventional security architecture
   keeps detection, enforcement, and audit as separate layers.
   The library is the detection layer. Customers add the
   enforcement and audit layers on top, often with separate
   teams and separate change-management.

---

## What conformal-margin guarantees vs what is your job

| Concern | conformal-margin | Customer |
| --- | --- | --- |
| Compute calibrated p-value per agent | yes | no |
| Emit `OperatorSignal` with severity | yes | no |
| Formal `P(FPR > alpha) <= alpha` | yes | no |
| Dispatch signals to one or more handlers | yes | no |
| Decide which severity warrants which action | no | yes |
| Talk to the agent runtime API | no | yes |
| Maintain agent-suspension state | no | yes |
| Audit: persist who-blocked-whom-and-when | no (only signal-level) | yes |
| Reverse a suspension (resume agent) | no | yes |

The boundary is sharp: the library emits a signal and the
signal is the contract. Everything after that lives in your
infrastructure.

---

## How to wire enforcement in: the AlertHandler Protocol

Every emitted signal is dispatched concurrently to every
`AlertHandler` registered on the `FleetMonitor`. Your
enforcement handler is one more handler alongside the
LogHandler, the AuditFeedbackHandler, and any other sink:

```python
from conformal_margin import FleetMonitor
from conformal_margin.handlers.log import LogHandler
from conformal_margin.handlers.audit_feedback import AuditFeedbackHandler

monitor = FleetMonitor(
    operators=(BaselineOperator(name="baseline"),),
    store=InMemoryStore(),
    handlers=(
        LogHandler(name="primary-log"),          # observability
        AuditFeedbackHandler(name="audit", ...), # post-hoc review
        BlockingHandler(runtime=my_runtime),     # YOUR enforcement
    ),
)
```

Per the FleetMonitor contract, handler failures are caught and
logged at error level; a misbehaving enforcement handler will
not block sibling handlers. Best-effort delivery applies to
enforcement too.

---

## A concrete BlockingHandler

A minimum-viable enforcement handler:

```python
from dataclasses import dataclass
from typing import ClassVar

from conformal_margin.ontology.domain.ontology import Role, Severity
from conformal_margin.ontology.ports.operator_signal import OperatorSignalLike


@dataclass
class BlockingHandler:
    """Customer enforcement: suspend agents on critical anomaly signals."""

    name: str
    runtime: object  # your agent runtime client; must expose `await runtime.suspend(agent_id)`
    role: ClassVar = Role.SINK

    async def handle(self, signal: OperatorSignalLike) -> None:
        if not signal.threshold_exceeded:
            return  # below alpha, library says "normal", we do nothing
        if signal.severity is Severity.CRITICAL:
            for agent_id in signal.affected_agents:
                await self.runtime.suspend(
                    agent_id,
                    reason=f"conformal-margin anomaly p={signal.p_value:.4f}",
                )
        elif signal.severity is Severity.HIGH:
            for agent_id in signal.affected_agents:
                await self.runtime.throttle(agent_id, requests_per_minute=1)
        # Medium and Low: log only (handled by LogHandler).
```

Copy this skeleton, swap `self.runtime` for your concrete agent
orchestrator client, and the enforcement is wired.

---

## Three reference response policies

The mapping from `OperatorSignal.severity` to a real-world
action is policy, not engineering. Pick whichever fits your
operations.

### Pattern 1: Severity-tier response

The simplest. One action per severity:

```
CRITICAL  -> suspend agent immediately, page on-call
HIGH      -> throttle agent to 1 request per minute, notify SRE
MEDIUM    -> log to audit trail, daily review by SOC
LOW       -> log only
INFO      -> log only
```

Implementation: a single `match` on `signal.severity` inside
`handle()`. Easy to reason about, easy to tune by editing one
table.

### Pattern 2: Scope-based policy

Different responses for different agent populations. Useful
when not every agent has the same business importance:

```
agent.model_tier == ModelTier.FRONTIER     -> human review at MEDIUM and above
agent.model_tier == ModelTier.FOUNDATION   -> auto-throttle at HIGH
agent.model_tier == ModelTier.SLM_EDGE     -> auto-suspend at CRITICAL only
```

Customer joins on `signal.affected_agents` and pulls the
underlying agent's `model_tier` from their inventory before
deciding. The library does not surface `model_tier` on the
signal itself by design (signals stay slim and serialisable);
the enforcement handler reads from the customer's source of
truth.

### Pattern 3: Time-bounded suspension with cooldown

Avoid permanent blocks from one signal. Enforcement state
machine:

```
on CRITICAL:    suspend(agent, duration=60 minutes)
on next signal: if HIGH or higher,  extend suspension; else, observe
on observation: after duration without signal, automatically resume
```

Implementation: enforcement handler keeps its own
`agent_id -> suspended_until_timestamp` map. The map is the
customer's state, not the library's. Library signals are the
input; suspension is the output; the map records the bridge.

---

## What the library deliberately does NOT do

This list is part of the architectural commitment:

- Maintain agent-suspension state. The map of "who is currently
  suspended" lives in the customer's system. If the dashboard
  shows a "suspended agents" panel, that panel reads from the
  customer's enforcement layer, not from the library.
- Auto-resume an agent after a quiet period. Same reason.
- Throttle agent request rates at the network layer. The
  library has no network presence.
- Reject agent outputs. The library never sees agent outputs;
  only the customer-computed `risk_score` field on AgentState.
- Page humans (PagerDuty, Opsgenie, etc.). LogHandler emits
  structured events your existing log-to-alerting pipeline
  picks up. A dedicated PagingHandler is a customer extension,
  not part of the runtime library.

---

## Failure modes and gotchas

**Auto-blocking on a poorly-calibrated alpha**: if you ship with
`alpha = 0.1` and auto-suspend on every threshold-exceeded
signal, you will suspend roughly ten percent of legitimate
agents per snapshot under steady-state. Pick `alpha` such that
your tolerated false-positive cost is acceptable, **and** layer
severity tiers between the library's signal and your action.

**Adversarial FPR injection**: an attacker who can shape the
risk-score distribution of one agent can push it into the tail
of the calibration set, raising the chance of false positives
across the fleet. Mitigations: per-agent rate limits on signals
(do not suspend the same agent more than N times per hour),
correlation with request-level identity from your runtime, and
human review of CRITICAL events before irreversible action.

**Enforcement handler is slow**: `FleetMonitor` dispatches
handlers concurrently inside an `anyio.TaskGroup`, so a slow
enforcement handler does not block sibling handlers. But if
your enforcement handler awaits a remote API with no timeout,
its task can stay open indefinitely. Always set a timeout on
the underlying runtime calls (`anyio.fail_after`, `httpx`
timeout, or your runtime's own timeout API).

**Audit trail of who-blocked-whom**: the library audits
`(signal, label)` pairs through `AuditFeedbackHandler`. It does
not audit "this BlockingHandler suspended agent-7 at 12:34".
Your enforcement handler should write its own audit event when
it acts. The cleanest pattern: enforcement handler writes to
the same audit log used for FeedbackPort, with an event type
distinguishing "library-detected" from "enforcement-acted".

---

## See also

- `docs/observability.md` for how to pipe library events into
  Loki / Datadog / Splunk for human review.
- The `OperatorSignal` schema in
  `conformal_margin.ontology.ports.operator_signal` for the
  full set of fields your enforcement handler can read.