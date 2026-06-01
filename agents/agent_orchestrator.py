from __future__ import annotations

from typing import Any, Dict

from .agent_types import AgentPolicy, PipelineAgentState, StageResult
from .agent_guardrails import assess_forecast_gate, assess_promo_constraints, summarise_profit
from . import agent_tools


class RetailPipelineOpsAgent:
    """Production-friendly orchestration agent for your existing M5 pipeline.

    IMPORTANT: This is an *orchestrator agent*, not a model-training agent.
    It wraps deterministic ML/business modules and adds planning/guardrails/fallback logic.
    """

    def __init__(self, policy: AgentPolicy | None = None):
        self.policy = policy or AgentPolicy()

    def run(self, *, data_dir: str, params: Dict[str, Any]) -> Dict[str, Any]:
        state = PipelineAgentState(data_dir=data_dir, params=params, policy=self.policy)

        # 0) Validation
        if self.policy.require_validation:
            try:
                vrep, dt = agent_tools.run_validate(data_dir)
                state.add_result(
                    StageResult(
                        stage="validate_inputs", ok=True, data={"validation": vrep}, duration_ms=dt
                    )
                )
                state.decisions.append("Validation passed; proceeding.")
            except Exception as e:
                state.add_result(StageResult(stage="validate_inputs", ok=False, errors=[str(e)]))
                return self._finalise(state, ok=False, reason="input_validation_failed")

        # 1) Forecast
        try:
            fres, dt = agent_tools.run_forecast(data_dir, params)
            state.add_result(StageResult(stage="forecast", ok=True, data=fres, duration_ms=dt))
        except Exception as e:
            state.add_result(StageResult(stage="forecast", ok=False, errors=[str(e)]))
            return self._finalise(state, ok=False, reason="forecast_failed")

        # Gate to pricing/promo
        f_ok, f_msg = assess_forecast_gate(fres)
        state.decisions.append(f_msg)
        if not f_ok:
            return self._finalise(state, ok=False, reason="forecast_gate_failed")

        if self.policy.stop_on_forecast_baseline_loss and str(fres.get("winner")) != "lgbm":
            return self._finalise(state, ok=False, reason="baseline_won_forecast")

        # 2) Price optimisation
        try:
            pres, dt = agent_tools.run_price_actions(data_dir, params)
            state.add_result(StageResult(stage="price_opt", ok=True, data=pres, duration_ms=dt))
        except Exception as e:
            state.add_result(StageResult(stage="price_opt", ok=False, errors=[str(e)]))
            return self._finalise(state, ok=False, reason="price_opt_failed")

        # 3) Promo selection
        try:
            sres, dt = agent_tools.run_promo_selection(data_dir, params)
            state.add_result(
                StageResult(stage="promo_selection", ok=True, data=sres, duration_ms=dt)
            )
        except Exception as e:
            state.add_result(StageResult(stage="promo_selection", ok=False, errors=[str(e)]))
            return self._finalise(state, ok=False, reason="promo_selection_failed")

        p_ok, p_msg = assess_promo_constraints(sres)
        state.decisions.append(p_msg)
        if self.policy.require_no_constraint_violation and not p_ok:
            return self._finalise(state, ok=False, reason="promo_constraint_violation")

        # 4) Optional uplift backtest
        if self.policy.run_uplift_backtest:
            try:
                bres, dt = agent_tools.run_uplift_backtest(data_dir, params)
                state.add_result(
                    StageResult(stage="uplift_backtest", ok=True, data=bres, duration_ms=dt)
                )
                state.decisions.append("Uplift backtest completed.")
            except Exception as e:
                state.add_result(StageResult(stage="uplift_backtest", ok=False, errors=[str(e)]))
                state.decisions.append(
                    "Uplift backtest failed; continuing because core outputs exist."
                )

        # 5) Summarise for API / operator dashboard
        state.final_summary = {
            "profit": summarise_profit(pres, sres),
            "forecast_winner": fres.get("winner"),
            "forecast_artifacts": fres.get("artifacts", {}),
            "price_reports": pres.get("reports", {}),
            "promo_reports": sres.get("reports", {}),
        }
        state.add_result(StageResult(stage="summarise", ok=True, data=state.final_summary))

        return self._finalise(state, ok=True, reason="completed")

    def _finalise(self, state: PipelineAgentState, *, ok: bool, reason: str) -> Dict[str, Any]:
        return {
            "ok": ok,
            "reason": reason,
            "decisions": state.decisions,
            "stages": [
                {
                    "stage": r.stage,
                    "ok": r.ok,
                    "duration_ms": r.duration_ms,
                    "errors": r.errors,
                    "warnings": r.warnings,
                    "data": r.data,
                }
                for r in state.stage_results
            ],
            "summary": state.final_summary,
        }
