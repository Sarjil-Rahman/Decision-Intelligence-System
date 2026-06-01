from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

StageName = Literal[
    "validate_inputs",
    "forecast",
    "price_opt",
    "promo_selection",
    "uplift_backtest",
    "drift_check",
    "summarise",
]


@dataclass
class AgentPolicy:
    require_validation: bool = True
    stop_on_forecast_baseline_loss: bool = False
    allow_price_optimisation_if_forecast_baseline_wins: bool = True
    max_retry_per_stage: int = 1
    require_no_constraint_violation: bool = True
    require_reports: bool = True
    run_uplift_backtest: bool = True
    run_drift_check: bool = False


@dataclass
class StageResult:
    stage: StageName
    ok: bool
    data: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    duration_ms: Optional[int] = None


@dataclass
class PipelineAgentState:
    data_dir: str
    params: Dict[str, Any]
    policy: AgentPolicy = field(default_factory=AgentPolicy)
    stage_results: List[StageResult] = field(default_factory=list)
    decisions: List[str] = field(default_factory=list)
    final_summary: Dict[str, Any] = field(default_factory=dict)

    def add_result(self, res: StageResult) -> None:
        self.stage_results.append(res)

    def latest(self, stage: StageName) -> Optional[StageResult]:
        for r in reversed(self.stage_results):
            if r.stage == stage:
                return r
        return None
