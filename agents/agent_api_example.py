"""Example FastAPI integration for your existing `api/main.py`.

You can either:
1) import and mount this router in api/main.py, OR
2) copy the endpoint into api/main.py directly.
"""

from __future__ import annotations

from typing import Any, Dict, List
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException

from agents.agent_orchestrator import RetailPipelineOpsAgent
from agents.agent_types import AgentPolicy

router = APIRouter(tags=["agent-orchestration"])


class AgentRunRequest(BaseModel):
    data_dir: str = Field(default="data")
    params: Dict[str, Any] = Field(default_factory=dict)

    # policy overrides
    require_validation: bool = True
    stop_on_forecast_baseline_loss: bool = False
    require_no_constraint_violation: bool = True
    run_uplift_backtest: bool = True


class AgentRunResponse(BaseModel):
    ok: bool
    reason: str
    decisions: List[str]
    stages: List[Dict[str, Any]]
    summary: Dict[str, Any]


@router.post("/run-agent-pipeline", response_model=AgentRunResponse)
def run_agent_pipeline(req: AgentRunRequest):
    try:
        agent = RetailPipelineOpsAgent(
            policy=AgentPolicy(
                require_validation=req.require_validation,
                stop_on_forecast_baseline_loss=req.stop_on_forecast_baseline_loss,
                require_no_constraint_violation=req.require_no_constraint_violation,
                run_uplift_backtest=req.run_uplift_backtest,
            )
        )
        result = agent.run(data_dir=req.data_dir, params=req.params)
        return AgentRunResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
