from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from analytics_sql.anomaly_detection import write_kpi_anomalies


def test_anomaly_detection_flags_obvious_revenue_spike(tmp_path: Path) -> None:
    db = tmp_path / "analytics.duckdb"
    schema = Path("analytics_sql/schema.sql").read_text(encoding="utf-8")
    kpis = pd.DataFrame(
        {
            "run_id": ["spike_run"] * 6,
            "d": [f"d_{i}" for i in range(1, 7)],
            "date": pd.date_range("2011-01-01", periods=6).date,
            "store_id": ["CA_1"] * 6,
            "cat_id": ["FOODS"] * 6,
            "units_sold": [10.0, 10.0, 10.0, 10.0, 10.0, 100.0],
            "revenue_gbp": [20.0, 20.0, 20.0, 20.0, 20.0, 200.0],
            "gross_margin_proxy_gbp": [6.0, 6.0, 6.0, 6.0, 6.0, 60.0],
            "avg_selling_price": [2.0] * 6,
            "active_items": [1] * 6,
            "zero_sales_items": [0] * 6,
        }
    )

    with duckdb.connect(str(db)) as con:
        con.execute(schema)
        con.register("tmp_kpis", kpis)
        con.execute("INSERT INTO fact_retail_daily_kpis SELECT * FROM tmp_kpis")
        anomalies = write_kpi_anomalies(
            con,
            run_id="spike_run",
            threshold=3.0,
            window=4,
            min_periods=3,
        )

    assert not anomalies.empty
    assert "revenue_gbp" in set(anomalies["metric_name"])
    assert (anomalies["severity"] == "high").any()
