# Agent Integration Scaffold for M5 Forecast + Price Optimisation + Promo Selection

This scaffold adds **orchestration agents** around your existing deterministic ML/business logic.
It does **not** replace `run_forecast`, `run_price_optimization`, or `run_promo_selection`.

## Why agents here?
Your project already has strong deterministic cores. Agents are most useful for:
- planning which steps to run
- validating preconditions and policy guardrails
- handling retries/fallbacks
- summarising outputs for operators/business users
- triggering drift triage recommendations

## What this scaffold contains
- `agent_types.py`: typed state + decisions
- `agent_tools.py`: thin wrappers around your existing services/pipeline functions
- `agent_guardrails.py`: policy checks / safe execution gates
- `agent_orchestrator.py`: multi-step agent coordinator
- `agent_api_example.py`: FastAPI route example for `/run-agent-pipeline`

## Integration pattern
1. Keep your current endpoints (`/forecast`, `/price-actions`, `/promo-selection`) untouched.
2. Add an optional orchestration endpoint that calls the agent.
3. Use the agent for **ops workflow automation**, not core prediction math.

## Current integration note
The event/non-event elasticity diagnostic now calls the variables loaded by
`run_price_optimization` (`sales`, `cal`, and `prices`). Keep diagnostics
best-effort, but avoid broad exception handling around critical pipeline stages.
