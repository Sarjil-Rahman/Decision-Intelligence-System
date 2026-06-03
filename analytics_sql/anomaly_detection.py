from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pandas as pd

ANOMALY_COLUMNS = [
    "run_id",
    "grain",
    "entity_id",
    "d",
    "date",
    "metric_name",
    "metric_value",
    "expected_value",
    "anomaly_score",
    "severity",
    "reason",
]


def _severity(score: float, threshold: float) -> str:
    if score >= threshold * 2:
        return "high"
    if score >= threshold * 1.5:
        return "medium"
    return "low"


def _direction(value: float, expected: float) -> str:
    return "spike" if value > expected else "drop"


def _detect_metric_anomalies(
    df: pd.DataFrame,
    *,
    run_id: str,
    grain: str,
    entity_col: str,
    metric_name: str,
    threshold: float,
    window: int,
    min_periods: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    work = df[[entity_col, "d", "date", metric_name]].copy()
    work = work.sort_values([entity_col, "date", "d"])

    for entity_id, group in work.groupby(entity_col, dropna=False):
        metric = pd.to_numeric(group[metric_name], errors="coerce")
        expected = metric.rolling(window=window, min_periods=min_periods).median().shift(1)
        abs_dev = (metric - expected).abs()
        mad = abs_dev.rolling(window=window, min_periods=min_periods).median().shift(1)
        scale = (mad * 1.4826).where(mad > 1e-9)
        fallback_scale = metric.rolling(window=window, min_periods=min_periods).std().shift(1)
        scale = scale.fillna(fallback_scale).fillna(0.0)
        diff = (metric - expected).abs()
        score = (diff / scale.where(scale > 1e-9)).fillna(0.0)
        score = score.mask((scale <= 1e-9) & expected.notna() & (diff > 1e-9), threshold * 2)

        for idx, anomaly_score in score[score >= threshold].items():
            value = float(metric.loc[idx])
            expected_value = float(expected.loc[idx])
            record = group.loc[idx]
            rows.append(
                {
                    "run_id": run_id,
                    "grain": grain,
                    "entity_id": str(entity_id),
                    "d": str(record["d"]),
                    "date": record["date"],
                    "metric_name": metric_name,
                    "metric_value": value,
                    "expected_value": expected_value,
                    "anomaly_score": float(anomaly_score),
                    "severity": _severity(float(anomaly_score), threshold),
                    "reason": (
                        f"{metric_name} {_direction(value, expected_value)} versus rolling "
                        f"median at {grain} grain"
                    ),
                }
            )
    return rows


def detect_kpi_anomalies(
    con: duckdb.DuckDBPyConnection,
    *,
    run_id: str,
    threshold: float = 3.5,
    window: int = 7,
    min_periods: int = 3,
) -> pd.DataFrame:
    """Detect explainable KPI anomalies from fact_retail_daily_kpis."""
    base = con.execute(
        """
        SELECT
          run_id,
          d,
          date,
          store_id,
          cat_id,
          units_sold,
          revenue_gbp,
          gross_margin_proxy_gbp,
          CASE
            WHEN active_items = 0 THEN NULL
            ELSE zero_sales_items * 1.0 / active_items
          END AS zero_sales_item_share
        FROM fact_retail_daily_kpis
        WHERE run_id = ?
        ORDER BY date, d
        """,
        [run_id],
    ).fetchdf()
    if base.empty:
        return pd.DataFrame(columns=ANOMALY_COLUMNS)

    metrics = [
        "units_sold",
        "revenue_gbp",
        "gross_margin_proxy_gbp",
        "zero_sales_item_share",
    ]
    anomaly_rows: list[dict[str, object]] = []

    store_daily = (
        base.groupby(["store_id", "d", "date"], as_index=False)[metrics]
        .sum(min_count=1)
        .sort_values(["store_id", "date", "d"])
    )
    for metric_name in metrics:
        anomaly_rows.extend(
            _detect_metric_anomalies(
                store_daily,
                run_id=run_id,
                grain="store_daily",
                entity_col="store_id",
                metric_name=metric_name,
                threshold=threshold,
                window=window,
                min_periods=min_periods,
            )
        )

    category_daily = (
        base.groupby(["cat_id", "d", "date"], as_index=False)[metrics]
        .sum(min_count=1)
        .sort_values(["cat_id", "date", "d"])
    )
    for metric_name in metrics:
        anomaly_rows.extend(
            _detect_metric_anomalies(
                category_daily,
                run_id=run_id,
                grain="category_daily",
                entity_col="cat_id",
                metric_name=metric_name,
                threshold=threshold,
                window=window,
                min_periods=min_periods,
            )
        )

    anomalies = pd.DataFrame(anomaly_rows, columns=ANOMALY_COLUMNS)
    if not anomalies.empty:
        anomalies = anomalies.sort_values(
            ["date", "severity", "anomaly_score"], ascending=[True, True, False]
        ).reset_index(drop=True)
    return anomalies


def write_kpi_anomalies(
    con: duckdb.DuckDBPyConnection,
    *,
    run_id: str,
    threshold: float = 3.5,
    window: int = 7,
    min_periods: int = 3,
) -> pd.DataFrame:
    anomalies = detect_kpi_anomalies(
        con,
        run_id=run_id,
        threshold=threshold,
        window=window,
        min_periods=min_periods,
    )
    con.execute("DELETE FROM fact_kpi_anomalies WHERE run_id = ?", [run_id])
    if not anomalies.empty:
        con.register("tmp_anomalies", anomalies)
        con.execute("INSERT INTO fact_kpi_anomalies SELECT * FROM tmp_anomalies")
        con.unregister("tmp_anomalies")
    return anomalies


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="analytics.duckdb")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--threshold", type=float, default=3.5)
    parser.add_argument("--window", type=int, default=7)
    parser.add_argument("--min-periods", type=int, default=3)
    args = parser.parse_args()

    con = duckdb.connect(str(Path(args.db)))
    rows = write_kpi_anomalies(
        con,
        run_id=args.run_id,
        threshold=args.threshold,
        window=args.window,
        min_periods=args.min_periods,
    )
    print(f"Wrote {len(rows)} anomalies for run_id={args.run_id}")


if __name__ == "__main__":
    main()
