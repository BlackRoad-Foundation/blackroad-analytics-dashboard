# blackroad-analytics-dashboard

> Production Python analytics engine — part of [BlackRoad Foundation](https://github.com/BlackRoad-Foundation) · [BlackRoad OS, Inc.](https://blackroad.io)

**Keywords:** BlackRoad · BlackRoad OS · BlackRoad Foundation · BlackRoad AI · BlackRoad Labs · BlackRoad Security · BlackRoad Cloud · BlackRoad Analytics · analytics dashboard · Python analytics · time-series · KPI engine · anomaly detection · open source · Delaware C-Corp

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

## Infrastructure Index

Live directory: **[blackroad-foundation.github.io/blackroad-analytics-dashboard/](https://blackroad-foundation.github.io/blackroad-analytics-dashboard/)**

### GitHub Enterprise
| Enterprise | URL |
|---|---|
| blackroad-os | [github.com/enterprises/blackroad-os](https://github.com/enterprises/blackroad-os) |

### Organizations (15)
| Organization | GitHub |
|---|---|
| Blackbox-Enterprises | [github.com/Blackbox-Enterprises](https://github.com/Blackbox-Enterprises) |
| BlackRoad-AI | [github.com/BlackRoad-AI](https://github.com/BlackRoad-AI) |
| BlackRoad-Archive | [github.com/BlackRoad-Archive](https://github.com/BlackRoad-Archive) |
| BlackRoad-Cloud | [github.com/BlackRoad-Cloud](https://github.com/BlackRoad-Cloud) |
| BlackRoad-Education | [github.com/BlackRoad-Education](https://github.com/BlackRoad-Education) |
| BlackRoad-Foundation | [github.com/BlackRoad-Foundation](https://github.com/BlackRoad-Foundation) |
| BlackRoad-Gov | [github.com/BlackRoad-Gov](https://github.com/BlackRoad-Gov) |
| BlackRoad-Hardware | [github.com/BlackRoad-Hardware](https://github.com/BlackRoad-Hardware) |
| BlackRoad-Interactive | [github.com/BlackRoad-Interactive](https://github.com/BlackRoad-Interactive) |
| BlackRoad-Labs | [github.com/BlackRoad-Labs](https://github.com/BlackRoad-Labs) |
| BlackRoad-Media | [github.com/BlackRoad-Media](https://github.com/BlackRoad-Media) |
| BlackRoad-OS | [github.com/BlackRoad-OS](https://github.com/BlackRoad-OS) |
| BlackRoad-Security | [github.com/BlackRoad-Security](https://github.com/BlackRoad-Security) |
| BlackRoad-Studio | [github.com/BlackRoad-Studio](https://github.com/BlackRoad-Studio) |
| BlackRoad-Ventures | [github.com/BlackRoad-Ventures](https://github.com/BlackRoad-Ventures) |

### Registered Domains (19)
`blackboxprogramming.io` · `blackroad.company` · `blackroad.io` · `blackroad.me` · `blackroad.network` · `blackroad.systems` · `blackroadai.com` · `blackroadinc.us` · `blackroadqi.com` · `blackroadquantum.com` · `blackroadquantum.info` · `blackroadquantum.net` · `blackroadquantum.shop` · `blackroadquantum.store` · `lucidia.earth` · `lucidia.studio` · `lucidiaqi.com` · `roadchain.io` · `roadcoin.io`

### Repository Structure
```
blackroad-analytics-dashboard/
├── index.html                        # SEO-optimized infrastructure directory
├── sitemap.xml                       # Search-engine sitemap
├── robots.txt                        # Crawler directives
├── src/
│   ├── __init__.py
│   └── analytics_dashboard.py        # Core analytics engine
├── tests/
│   ├── __init__.py
│   └── test_analytics_dashboard.py   # Full test suite
└── setup.py
```

## License

© BlackRoad OS, Inc. All rights reserved.
