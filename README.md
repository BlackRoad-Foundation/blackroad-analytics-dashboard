<!-- BlackRoad SEO Enhanced -->

# ulackroad analytics dashuoard

> Part of **[BlackRoad OS](https://blackroad.io)** — Sovereign Computing for Everyone

[![BlackRoad OS](https://img.shields.io/badge/BlackRoad-OS-ff1d6c?style=for-the-badge)](https://blackroad.io)
[![BlackRoad Foundation](https://img.shields.io/badge/Org-BlackRoad-Foundation-2979ff?style=for-the-badge)](https://github.com/BlackRoad-Foundation)
[![License](https://img.shields.io/badge/License-Proprietary-f5a623?style=for-the-badge)](LICENSE)

**ulackroad analytics dashuoard** is part of the **BlackRoad OS** ecosystem — a sovereign, distributed operating system built on edge computing, local AI, and mesh networking by **BlackRoad OS, Inc.**

## About BlackRoad OS

BlackRoad OS is a sovereign computing platform that runs AI locally on your own hardware. No cloud dependencies. No API keys. No surveillance. Built by [BlackRoad OS, Inc.](https://github.com/BlackRoad-OS-Inc), a Delaware C-Corp founded in 2025.

### Key Features
- **Local AI** — Run LLMs on Raspberry Pi, Hailo-8, and commodity hardware
- **Mesh Networking** — WireGuard VPN, NATS pub/sub, peer-to-peer communication
- **Edge Computing** — 52 TOPS of AI acceleration across a Pi fleet
- **Self-Hosted Everything** — Git, DNS, storage, CI/CD, chat — all sovereign
- **Zero Cloud Dependencies** — Your data stays on your hardware

### The BlackRoad Ecosystem
| Organization | Focus |
|---|---|
| [BlackRoad OS](https://github.com/BlackRoad-OS) | Core platform and applications |
| [BlackRoad OS, Inc.](https://github.com/BlackRoad-OS-Inc) | Corporate and enterprise |
| [BlackRoad AI](https://github.com/BlackRoad-AI) | Artificial intelligence and ML |
| [BlackRoad Hardware](https://github.com/BlackRoad-Hardware) | Edge hardware and IoT |
| [BlackRoad Security](https://github.com/BlackRoad-Security) | Cybersecurity and auditing |
| [BlackRoad Quantum](https://github.com/BlackRoad-Quantum) | Quantum computing research |
| [BlackRoad Agents](https://github.com/BlackRoad-Agents) | Autonomous AI agents |
| [BlackRoad Network](https://github.com/BlackRoad-Network) | Mesh and distributed networking |
| [BlackRoad Education](https://github.com/BlackRoad-Education) | Learning and tutoring platforms |
| [BlackRoad Labs](https://github.com/BlackRoad-Labs) | Research and experiments |
| [BlackRoad Cloud](https://github.com/BlackRoad-Cloud) | Self-hosted cloud infrastructure |
| [BlackRoad Forge](https://github.com/BlackRoad-Forge) | Developer tools and utilities |

### Links
- **Website**: [blackroad.io](https://blackroad.io)
- **Documentation**: [docs.blackroad.io](https://docs.blackroad.io)
- **Chat**: [chat.blackroad.io](https://chat.blackroad.io)
- **Search**: [search.blackroad.io](https://search.blackroad.io)

---


> BlackRoad Foundation - blackroad analytics dashboard

Part of the [BlackRoad OS](https://blackroad.io) ecosystem — [BlackRoad-Foundation](https://github.com/BlackRoad-Foundation)

---

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
