import math

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from api.schemas import (
    ABTestSimRequest,
    ABTestSimResponse,
    ForecastRequest,
    ForecastResponse,
    PriceActionsRequest,
    PriceActionsResponse,
    PromoSelectionRequest,
    PromoSelectionResponse,
    BusinessPackRequest,
    BusinessPackResponse,
)
from api.services import (
    forecast_point_or_quantiles,
    generate_business_reporting_pack,
    price_actions,
    promo_selection,
    simulate_ab_test,
)

try:
    from agents.agent_orchestrator import RetailPipelineOpsAgent
except Exception:
    RetailPipelineOpsAgent = None  # type: ignore
from monitoring.metrics import metrics_middleware, metrics_endpoint
from m5_pipeline.utils import get_logger

logger = get_logger("api")

app = FastAPI(title="M5 Forecasting + Price Optimisation API", version="1.1.0")

# Metrics middleware (latency + error rate)
app.middleware("http")(metrics_middleware)


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/metrics")
def metrics():
    return metrics_endpoint()


@app.post("/forecast", response_model=ForecastResponse)
def post_forecast(req: ForecastRequest):
    try:
        fres = forecast_point_or_quantiles(
            data_dir=req.data_dir,
            max_series=req.max_series,
            start_d=req.start_d,
            last_train_d=req.last_train_d,
            horizon=req.horizon,
            objective=req.objective,
            tweedie_variance_power=req.tweedie_variance_power,
            two_stage=req.two_stage,
            split_strategy=req.split_strategy,
            n_backtests=req.n_backtests,
            backtest_stride=req.backtest_stride,
            validate_inputs=req.validate_inputs,
            save_artifacts=req.save_artifacts,
        )
        return ForecastResponse(
            submission_path=fres["submission_path"],
            winner=fres["winner"],
            backtests=fres["backtests"],
            residual_q10=fres["residual_quantiles"]["q10"],
            residual_q50=fres["residual_quantiles"]["q50"],
            residual_q90=fres["residual_quantiles"]["q90"],
            artifacts=fres.get("artifacts", {}),
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Forecast failed")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/price-actions", response_model=PriceActionsResponse)
def post_price_actions(req: PriceActionsRequest):
    try:
        res = price_actions(
            data_dir=req.data_dir,
            submission_path=req.submission_path,
            last_train_d=req.last_train_d,
            margin=req.margin,
            unit_econ_path=req.unit_econ_path,
            max_series=req.max_series,
            lookback_days=req.lookback_days,
            elasticity_clip_low=req.elasticity_clip_low,
            elasticity_clip_high=req.elasticity_clip_high,
            max_abs_price_change_pct=req.max_abs_price_change_pct,
            max_demand_mult=req.max_demand_mult,
            suspicious_profit_gain_pct=req.suspicious_profit_gain_pct,
            suspicious_demand_gain_pct=req.suspicious_demand_gain_pct,
        )
        return PriceActionsResponse(**res)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Price-actions failed")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/promo-selection", response_model=PromoSelectionResponse)
def post_promo_selection(req: PromoSelectionRequest):
    try:
        res = promo_selection(
            data_dir=req.data_dir,
            input_path=req.input_path,
            max_price_changes_total=req.max_price_changes_total,
            max_price_changes_per_store=req.max_price_changes_per_store,
            max_price_changes_per_cat=req.max_price_changes_per_cat,
            budget=req.budget,
            forbid_price_increase=req.forbid_price_increase,
            max_abs_price_change_pct=req.max_abs_price_change_pct,
            max_demand_mult=req.max_demand_mult,
            objective=req.objective,
            require_price_change=req.require_price_change,
            promo_discount_grid=req.promo_discount_grid,
        )
        return PromoSelectionResponse(**res)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Promo-selection failed")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/business-pack", response_model=BusinessPackResponse)
def post_business_pack(req: BusinessPackRequest):
    try:
        res = generate_business_reporting_pack(data_dir=req.data_dir)
        return BusinessPackResponse(**res)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Business-pack failed")
        raise HTTPException(status_code=500, detail=str(e)) from e


def _sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [_sanitize_for_json(v) for v in obj]

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, np.ndarray):
        return [_sanitize_for_json(v) for v in obj.tolist()]

    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None

    return obj


@app.post("/run-agent-pipeline")
def post_run_agent_pipeline(req: dict):
    if RetailPipelineOpsAgent is None:
        raise HTTPException(status_code=501, detail="Agent orchestrator not installed/configured")
    try:
        data_dir = req.get("data_dir")
        if not data_dir:
            raise HTTPException(status_code=400, detail="data_dir is required")

        agent = RetailPipelineOpsAgent()
        result = agent.run(
            data_dir=data_dir,
            params=req.get("params", {}),
        )

        return JSONResponse(content=_sanitize_for_json(result))

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Agent pipeline failed")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.exception_handler(HTTPException)
def http_exception_handler(_, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


@app.post("/ab-test/simulate", response_model=ABTestSimResponse)
def ab_test_simulate(req: ABTestSimRequest) -> ABTestSimResponse:
    report = simulate_ab_test(
        price_actions_csv=req.price_actions_csv,
        out_report_json=req.out_report_json,
        treatment_share=req.treatment_share,
        noise_sigma=req.noise_sigma,
        elasticity_col=req.elasticity_col,
        n_boot=req.n_boot,
        seed=req.seed,
    )
    return ABTestSimResponse(report=report)
