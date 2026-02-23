# blackroad-analytics-dashboard

> Production Python analytics engine — part of [BlackRoad Foundation](https://github.com/BlackRoad-Foundation).

## Features

- **Metric Ingestion** — Bulk insert with dimensions, source, unit metadata
- **Flexible Querying** — Time-range + dimension filters + group-by aggregation
- **KPI Engine** — sum/avg/max/min/last/count with target & threshold evaluation
- **Chart Data** — line, bar, scatter, pie — bucket-aggregated and chart-ready
- **CSV Export** — Serialize any metric resultset to CSV
- **Anomaly Detection** — Rolling z-score (|z| > 3 flagged) with window control
- **Rollup Aggregations** — Hourly/daily rollups persisted to aggregations table

## Quick Start

```python
from datetime import datetime, timedelta
from src.analytics_dashboard import AnalyticsDashboard, Metric, KPIDefinition

dash = AnalyticsDashboard("metrics.db")

# Ingest
dash.ingest_one("cpu_pct", 72.5, dimensions={"host": "web-01"})

# Query
start = datetime.utcnow() - timedelta(hours=1)
end = datetime.utcnow()
pts = dash.query_metrics("cpu_pct", start, end)

# KPI
defn = KPIDefinition("CPU Usage", "cpu_pct", "avg",
                     target=80.0, threshold_warn=85.0, threshold_crit=95.0,
                     higher_is_better=False)
results = dash.calculate_kpis(pts_as_metrics, [defn])

# Anomaly detection
series = [m.value for m in metrics]
anomalies = dash.anomaly_score(series, window=20)

# Chart
chart = dash.generate_chart_data(pts, chart_type="line")
```

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

## License

© BlackRoad OS, Inc. All rights reserved.
