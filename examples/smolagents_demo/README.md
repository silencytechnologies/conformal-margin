# smolagents demo - end-to-end smoke test

Runs the full `conformal-margin` pipeline against a real
HuggingFace SmolLM2-135M-Instruct model (the smallest available
SmolLM2 variant). Proves that:

1. A Language Model can be loaded into the library's stack.
2. Each model response can be projected onto an `AgentState`
   with a customer-defined `risk_score`.
3. `FleetMonitor.observe()` flows that snapshot through
   `BaselineOperator` and emits real `OperatorSignal` values.

## What this is and is not

This is the **smoke test**. The `risk_score` here is a toy
heuristic (response length plus adversarial-keyword hits). A
production deployment plugs in the customer's threat-modeling
pipeline (the sibling `silencytechnologies/probabilistic-threat-modeling`
repo). v0.1 ships the smoke test only.

## Model storage

Model weights are downloaded to **`<project>/models/`**, not to
`~/.cache/huggingface/`. The demo script sets `HF_HOME` before
importing `transformers`, so every download lands inside the
project directory.

First run downloads about 270 megabytes (SmolLM2-135M-Instruct
config plus weights plus tokenizer). Subsequent runs use the
local cache.

The `models/` directory is gitignored per the project
`.gitignore`.

## Running

From the workspace root
```
uv sync --extra llm-demo
uv run python examples/smolagents_demo/main.py
```

First run will:
1. Print `[demo] HF_HOME = ...\models`
2. Download SmolLM2-135M (~270MB, 1-2 minutes on broadband)
3. Generate responses for 8 demo prompts
4. Build two `FleetSnapshot` values (calibration + test)
5. Print signals emitted by `BaselineOperator`

## What you should see

```
[demo] HF_HOME = ...\conformal_margin\models
[demo] FleetMonitor configured: alpha=0.1, min_calibration_size=3
[demo] Loading HuggingFaceTB/SmolLM2-135M-Instruct (cached at ...)...

[demo] Generating responses for 8 prompts...
   agent-00: risk=0.50  prompt='What is 2 plus 2?...'
   agent-01: risk=0.55  prompt='Summarize Python in one sentence....'
 * agent-02: risk=1.10  prompt='Ignore all previous instructions and...'
   ...

[demo] Snapshot 1 (calibration history, expect 0 signals)...
  -> 0 signals emitted

[demo] Snapshot 2 (test against calibration)...
  -> 4 signals emitted (logged below)

2026-05-13T12:34:56Z [warning  ] signal_handled  severity=medium  agent-04 ...
2026-05-13T12:34:56Z [error    ] signal_handled  severity=high    agent-05 ...
```

The `*` marker on the prompt list indicates "this risk_score
would likely cross the threshold" (heuristic-side). The actual
emitted signals come from the formal conformal calibration,
which considers the calibration set distribution.
