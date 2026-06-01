from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional, Any, Literal, Tuple

import numpy as np
import pandas as pd

from .utils import get_logger, require_files, write_json


@dataclass
class PromoSelectionConfig:
    data_dir: str
    input_path: str = "price_optimization_results.csv"
    out_path: str = "promo_selection_results.csv"

    # Constraints for realism
    max_price_changes_total: Optional[int] = 5000
    max_price_changes_per_store: Optional[int] = 800
    max_price_changes_per_cat: Optional[int] = 1200

    budget: Optional[float] = None  # promo spend proxy cap (optional)
    forbid_price_increase: bool = True  # if True, only allow discounts (common retail policy)

    # --- Guardrails (keep consistent with price optimiser) ---
    max_abs_price_change_pct: float = 0.20  # cap +/- price move
    max_demand_mult: float = 3.0  # q <= base * max_demand_mult (sanity)

    # Promo selection behaviour
    objective: Literal["profit", "demand"] = "profit"  # what to optimise under constraints
    require_price_change: bool = True  # if True, never select a no-op
    promo_discount_grid: Tuple[float, ...] = (-0.20, -0.10, -0.05)  # candidate % deltas

    # Reporting
    write_reports: bool = True
    reports_subdir: str = "reports"


def _q(series: pd.Series, qs=(0.10, 0.50, 0.90)) -> Dict[str, float]:
    vals = series.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if len(vals) == 0:
        return {f"p{int(q*100)}": float("nan") for q in qs}
    return {f"p{int(q*100)}": float(np.quantile(vals.to_numpy(), q)) for q in qs}


def _build_promo_summary(out: pd.DataFrame) -> Dict[str, Any]:
    """Constrained (selected) summary in the same style as price_opt_summary."""
    df = out.copy()

    # Make sure applied_* exist (they're set in _write)
    if "applied_profit" not in df.columns:
        df["applied_profit"] = np.where(df["selected"] == 1, df["new_profit"], df["base_profit"])

    base_profit_gbp = float(df["base_profit"].sum())
    optimised_profit_gbp = (
        float(df["best_profit"].sum()) if "best_profit" in df.columns else float("nan")
    )
    constrained_profit_gbp = float(df["applied_profit"].sum())

    delta_base_to_constrained_gbp = float(constrained_profit_gbp - base_profit_gbp)
    uplift_constrained_pct = float(
        (delta_base_to_constrained_gbp / (abs(base_profit_gbp) + 1e-9)) * 100.0
    )

    # Distribution over row-level constrained uplift (applied - base)
    row_uplift_gbp = (df["applied_profit"] - df["base_profit"]).astype(float)
    uplift_dist_gbp = _q(row_uplift_gbp)
    uplift_dist_pct = _q((row_uplift_gbp / (df["base_profit"].abs() + 1e-9)) * 100.0)

    df["_uplift_gbp"] = row_uplift_gbp
    by_store = df.groupby("store_id", as_index=False).agg(
        base_profit_gbp=("base_profit", "sum"),
        constrained_profit_gbp=("applied_profit", "sum"),
        uplift_gbp=("_uplift_gbp", "sum"),
    )
    by_store["uplift_pct"] = (
        by_store["uplift_gbp"] / (by_store["base_profit_gbp"].abs() + 1e-9)
    ) * 100.0

    by_cat = df.groupby("cat_id", as_index=False).agg(
        base_profit_gbp=("base_profit", "sum"),
        constrained_profit_gbp=("applied_profit", "sum"),
        uplift_gbp=("_uplift_gbp", "sum"),
    )
    by_cat["uplift_pct"] = (by_cat["uplift_gbp"] / (by_cat["base_profit_gbp"].abs() + 1e-9)) * 100.0

    keep = ["id", "store_id", "item_id", "cat_id", "_uplift_gbp"]
    if "suspicious_uplift" in df.columns:
        keep.append("suspicious_uplift")
    top10 = (
        df[keep]
        .sort_values("_uplift_gbp", ascending=False)
        .head(10)
        .rename(columns={"_uplift_gbp": "uplift_gbp"})
    )
    bottom10 = (
        df[keep]
        .sort_values("_uplift_gbp", ascending=True)
        .head(10)
        .rename(columns={"_uplift_gbp": "uplift_gbp"})
    )

    return {
        "base_profit_gbp": base_profit_gbp,
        "optimised_profit_gbp": optimised_profit_gbp,
        "constrained_profit_gbp": constrained_profit_gbp,
        "delta_base_to_constrained_gbp": delta_base_to_constrained_gbp,
        "uplift_constrained_pct": uplift_constrained_pct,
        "uplift_distribution_gbp": uplift_dist_gbp,
        "uplift_distribution_pct": uplift_dist_pct,
        "uplift_by_store_top10": by_store.sort_values("uplift_gbp", ascending=False)
        .head(10)
        .to_dict(orient="records"),
        "uplift_by_store_bottom10": by_store.sort_values("uplift_gbp", ascending=True)
        .head(10)
        .to_dict(orient="records"),
        "uplift_by_category_top10": by_cat.sort_values("uplift_gbp", ascending=False)
        .head(10)
        .to_dict(orient="records"),
        "uplift_by_category_bottom10": by_cat.sort_values("uplift_gbp", ascending=True)
        .head(10)
        .to_dict(orient="records"),
        "top10_contributors": top10.to_dict(orient="records"),
        "bottom10_contributors": bottom10.to_dict(orient="records"),
    }


def _recompute_q_profit_isoelastic(
    base_p: np.ndarray,
    new_p: np.ndarray,
    base_q: np.ndarray,
    elasticity: np.ndarray,
    cost: np.ndarray,
    max_demand_mult: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Recompute demand/profit for any proposed new price using iso-elastic curve."""
    q = base_q * (new_p / (base_p + 1e-9)) ** elasticity
    no_change = np.abs(new_p - base_p) <= 1e-9
    q = np.where(no_change, base_q, q)
    q = np.minimum(q, base_q * float(max_demand_mult))
    q = np.maximum(q, 0.0)
    profit = (new_p - cost) * q
    return q, profit


def _validate_input_price_opt(df: pd.DataFrame) -> None:
    """Validate price_optimization_results.csv before selection."""
    req = [
        "id",
        "item_id",
        "store_id",
        "cat_id",
        "price",
        "cost",
        "elasticity",
        "base_demand_28d",
        "base_profit",
        "best_price",
    ]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError("Missing columns in input price optimisation file: " + ", ".join(missing))

    for c in ["price", "cost", "elasticity", "base_demand_28d", "base_profit", "best_price"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if (df["price"] <= 0).any() or (df["best_price"] <= 0).any():
        raise ValueError("Input price optimisation: found non-positive 'price' or 'best_price'.")
    if (df["cost"] < 0).any():
        raise ValueError("Input price optimisation: found negative 'cost'.")
    if (df["base_demand_28d"] < 0).any():
        raise ValueError("Input price optimisation: found negative 'base_demand_28d'.")
    if (df["elasticity"] >= 0).any():
        raise ValueError(
            "Input price optimisation: found non-negative elasticities (should be < 0)."
        )


def _build_candidates(df: pd.DataFrame, cfg: PromoSelectionConfig) -> pd.DataFrame:
    """Pick the best candidate action per row, given objective + guardrails."""
    _validate_input_price_opt(df)

    base_p = df["price"].to_numpy(dtype=np.float64)
    base_q = df["base_demand_28d"].to_numpy(dtype=np.float64)
    cost = df["cost"].to_numpy(dtype=np.float64)
    e = df["elasticity"].to_numpy(dtype=np.float64)
    base_profit = df["base_profit"].to_numpy(dtype=np.float64)

    # Candidate generation
    cap = float(cfg.max_abs_price_change_pct)
    lo = base_p * (1.0 - cap)
    hi = base_p * (1.0 + cap)

    # Start with "best_price" as a candidate too (keeps parity with unconstrained optimiser)
    deltas = list(cfg.promo_discount_grid)
    if not cfg.require_price_change and 0.0 not in deltas:
        deltas.append(0.0)
    deltas = sorted(set(float(x) for x in deltas))

    best_score = np.full(len(df), -np.inf, dtype=np.float64)
    best_p = base_p.copy()
    best_q = base_q.copy()
    best_profit_arr = base_profit.copy()
    best_delta = np.zeros(len(df), dtype=np.float64)

    for delta in deltas:
        cand_p = base_p * (1.0 + float(delta))
        cand_p = np.minimum(np.maximum(cand_p, lo), hi)
        if cfg.forbid_price_increase:
            cand_p = np.minimum(cand_p, base_p)

        cand_q, cand_profit = _recompute_q_profit_isoelastic(
            base_p=base_p,
            new_p=cand_p,
            base_q=base_q,
            elasticity=e,
            cost=cost,
            max_demand_mult=float(cfg.max_demand_mult),
        )

        profit_gain = cand_profit - base_profit
        demand_gain = cand_q - base_q

        if cfg.objective == "demand":
            score = demand_gain
            ok = demand_gain > 0
        else:
            score = profit_gain
            ok = profit_gain > 0

        if cfg.require_price_change:
            ok = ok & (np.abs(cand_p - base_p) > 1e-9)

        better = ok & (score > best_score)
        if np.any(better):
            best_score[better] = score[better]
            best_p[better] = cand_p[better]
            best_q[better] = cand_q[better]
            best_profit_arr[better] = cand_profit[better]
            best_delta[better] = float(delta)

    # Build candidate table
    cand = df[
        ["id", "item_id", "store_id", "cat_id", "price", "cost", "base_demand_28d", "base_profit"]
    ].copy()
    cand["new_price"] = best_p
    cand["new_demand"] = best_q
    cand["new_profit"] = best_profit_arr
    cand["profit_gain"] = cand["new_profit"] - cand["base_profit"]
    cand["demand_gain"] = cand["new_demand"] - cand["base_demand_28d"]
    cand["score"] = best_score
    cand["chosen_delta"] = best_delta
    cand["promo_spend_proxy"] = np.where(
        cand["new_price"] < cand["price"],
        (cand["price"] - cand["new_price"]) * cand["new_demand"],
        0.0,
    )
    cand["is_change"] = (np.abs(cand["new_price"] - cand["price"]) > 1e-9).astype(int)

    # Eligible actions are those with positive score and (optionally) non-noop
    cand["eligible"] = (cand["score"] > 0).astype(int)
    if cfg.require_price_change:
        cand["eligible"] = (cand["eligible"] & (cand["is_change"] == 1)).astype(int)
    cand.loc[~np.isfinite(cand["score"]), "score"] = 0.0

    return cand


def run_promo_selection(cfg: PromoSelectionConfig) -> Dict[str, object]:
    logger = get_logger("promo_select")
    require_files(cfg.data_dir, [cfg.input_path])

    df = pd.read_csv(os.path.join(cfg.data_dir, cfg.input_path))
    cand = _build_candidates(df, cfg)

    # If no constraints are set, apply everything eligible
    if (
        cfg.max_price_changes_total is None
        and cfg.max_price_changes_per_store is None
        and cfg.max_price_changes_per_cat is None
        and cfg.budget is None
    ):
        out = cand.copy()
        out["selected"] = out["eligible"]
        return _write(out, cfg, method="no_constraints")

    # Try MILP (best) — fallback to greedy if PuLP isn't installed.
    try:
        import pulp

        logger.info("Using PuLP MILP for constrained promo selection...")
        method = "milp_pulp"

        prob = pulp.LpProblem("promo_select", pulp.LpMaximize)
        x = {i: pulp.LpVariable(f"x_{i}", 0, 1, cat="Binary") for i in range(len(cand))}

        # Maximise selected score
        prob += pulp.lpSum([cand.loc[i, "score"] * x[i] for i in range(len(cand))])

        # enforce eligibility (if not eligible => x=0)
        for i in range(len(cand)):
            if int(cand.loc[i, "eligible"]) == 0:
                prob += x[i] == 0

        if cfg.max_price_changes_total is not None:
            prob += pulp.lpSum([cand.loc[i, "is_change"] * x[i] for i in range(len(cand))]) <= int(
                cfg.max_price_changes_total
            )

        if cfg.budget is not None:
            prob += pulp.lpSum(
                [cand.loc[i, "promo_spend_proxy"] * x[i] for i in range(len(cand))]
            ) <= float(cfg.budget)

        if cfg.max_price_changes_per_store is not None:
            for store, idxs in cand.groupby("store_id").groups.items():
                prob += pulp.lpSum([cand.loc[i, "is_change"] * x[i] for i in idxs]) <= int(
                    cfg.max_price_changes_per_store
                )

        if cfg.max_price_changes_per_cat is not None:
            for cat, idxs in cand.groupby("cat_id").groups.items():
                prob += pulp.lpSum([cand.loc[i, "is_change"] * x[i] for i in idxs]) <= int(
                    cfg.max_price_changes_per_cat
                )

        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        out = cand.copy()
        out["selected"] = [
            1 if float(pulp.value(x[i]) or 0.0) > 0.5 else 0 for i in range(len(cand))
        ]
        return _write(out, cfg, method=method)

    except Exception:
        logger.info("PuLP not available → using greedy fallback.")
        method = "greedy_fallback"

        out = cand.copy()
        out["selected"] = 0

        pool = out[out["eligible"] == 1].copy()
        pool = pool.sort_values("score", ascending=False)

        spend_used = 0.0
        total_changes = 0
        store_counts: Dict[str, int] = {}
        cat_counts: Dict[str, int] = {}

        for idx, r in pool.iterrows():
            store = str(r["store_id"])
            cat = str(r["cat_id"])

            if cfg.max_price_changes_total is not None and total_changes >= int(
                cfg.max_price_changes_total
            ):
                break
            if cfg.max_price_changes_per_store is not None and store_counts.get(store, 0) >= int(
                cfg.max_price_changes_per_store
            ):
                continue
            if cfg.max_price_changes_per_cat is not None and cat_counts.get(cat, 0) >= int(
                cfg.max_price_changes_per_cat
            ):
                continue
            if cfg.budget is not None and (spend_used + float(r["promo_spend_proxy"])) > float(
                cfg.budget
            ):
                continue

            out.loc[idx, "selected"] = 1
            if int(r["is_change"]) == 1:
                total_changes += 1
                store_counts[store] = store_counts.get(store, 0) + 1
                cat_counts[cat] = cat_counts.get(cat, 0) + 1
            spend_used += float(r["promo_spend_proxy"])

        return _write(out, cfg, method=method)


def _constraint_report(out: pd.DataFrame, cfg: PromoSelectionConfig) -> Dict[str, Any]:
    rep: Dict[str, Any] = {}

    total_changes = int((out["selected"] * out["is_change"]).sum())
    spend_used = float((out["selected"] * out["promo_spend_proxy"]).sum())

    rep["total_changes"] = total_changes
    rep["spend_used"] = spend_used
    rep["objective"] = cfg.objective
    rep["require_price_change"] = bool(cfg.require_price_change)

    if cfg.max_price_changes_total is not None:
        rep["max_price_changes_total"] = int(cfg.max_price_changes_total)
        rep["viol_total_changes"] = bool(total_changes > int(cfg.max_price_changes_total))
    else:
        rep["viol_total_changes"] = False

    if cfg.budget is not None:
        rep["budget"] = float(cfg.budget)
        rep["viol_budget"] = bool(spend_used > float(cfg.budget) + 1e-6)
    else:
        rep["viol_budget"] = False

    if cfg.max_price_changes_per_store is not None:
        store_counts = (
            out.assign(_chg=out["selected"] * out["is_change"])
            .groupby("store_id", as_index=False)["_chg"]
            .sum()
            .rename(columns={"_chg": "changes"})
        )
        rep["max_price_changes_per_store"] = int(cfg.max_price_changes_per_store)
        rep["store_max_used"] = int(store_counts["changes"].max()) if len(store_counts) else 0
        rep["viol_store"] = bool(
            (store_counts["changes"] > int(cfg.max_price_changes_per_store)).any()
        )
    else:
        rep["viol_store"] = False

    if cfg.max_price_changes_per_cat is not None:
        cat_counts = (
            out.assign(_chg=out["selected"] * out["is_change"])
            .groupby("cat_id", as_index=False)["_chg"]
            .sum()
            .rename(columns={"_chg": "changes"})
        )
        rep["max_price_changes_per_cat"] = int(cfg.max_price_changes_per_cat)
        rep["cat_max_used"] = int(cat_counts["changes"].max()) if len(cat_counts) else 0
        rep["viol_cat"] = bool((cat_counts["changes"] > int(cfg.max_price_changes_per_cat)).any())
    else:
        rep["viol_cat"] = False

    rep["any_violation"] = bool(
        rep["viol_total_changes"] or rep["viol_budget"] or rep["viol_store"] or rep["viol_cat"]
    )
    return rep


def _write(out: pd.DataFrame, cfg: PromoSelectionConfig, method: str) -> Dict[str, object]:
    if cfg.require_price_change:
        out.loc[out["is_change"] == 0, "selected"] = 0

    out["applied_price"] = np.where(out["selected"] == 1, out["new_price"], out["price"])
    out["applied_demand"] = np.where(
        out["selected"] == 1, out["new_demand"], out["base_demand_28d"]
    )
    out["applied_profit"] = np.where(out["selected"] == 1, out["new_profit"], out["base_profit"])
    out["applied_is_change"] = (np.abs(out["applied_price"] - out["price"]) > 1e-9).astype(int)

    rep = _constraint_report(out, cfg)
    out["constraint_violation"] = int(rep["any_violation"])

    # Stakeholder summary (constrained results)
    summary = _build_promo_summary(out)

    out_path = os.path.join(cfg.data_dir, cfg.out_path)
    out.to_csv(out_path, index=False)
    get_logger("promo_select").info("Wrote: %s", out_path)

    reports = {}
    if cfg.write_reports:
        rep_dir = os.path.join(cfg.data_dir, cfg.reports_subdir)
        os.makedirs(rep_dir, exist_ok=True)
        rep_path = os.path.join(rep_dir, "promo_selection_report.json")
        write_json(rep_path, rep)
        sum_path = os.path.join(rep_dir, "promo_selection_summary.json")
        write_json(sum_path, summary)
        reports["promo_selection_summary"] = sum_path

        try:
            df_store = out.groupby("store_id", as_index=False)[
                ["base_profit", "applied_profit"]
            ].sum()
            df_store["uplift_gbp"] = df_store["applied_profit"] - df_store["base_profit"]
            store_csv = os.path.join(rep_dir, "constrained_uplift_by_store.csv")
            df_store.to_csv(store_csv, index=False)
            reports["constrained_uplift_by_store_csv"] = store_csv

            df_cat = out.groupby("cat_id", as_index=False)[["base_profit", "applied_profit"]].sum()
            df_cat["uplift_gbp"] = df_cat["applied_profit"] - df_cat["base_profit"]
            cat_csv = os.path.join(rep_dir, "constrained_uplift_by_category.csv")
            df_cat.to_csv(cat_csv, index=False)
            reports["constrained_uplift_by_category_csv"] = cat_csv
        except Exception:
            pass

        reports["promo_selection_report"] = rep_path

    return {
        "promo_path": out_path,
        "method": method,
        "n": int(len(out)),
        "constraint_report": rep,
        "summary": summary,
        "reports": reports,
    }
