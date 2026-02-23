"""
BlackRoad Analytics Dashboard — production implementation.
Time-series metric ingestion, KPI engine, chart data, anomaly detection.
"""
from __future__ import annotations

import csv
import io
import math
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from statistics import mean, stdev
from typing import Any, Callable, Dict, List, Optional, Tuple
import uuid


# ─────────────────────────── data models ────────────────────────────────────

@dataclass
class Metric:
    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    dimensions: Dict[str, str] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source: str = ""
    unit: str = ""


@dataclass
class KPIDefinition:
    name: str
    metric_name: str
    aggregation: str           # sum | avg | max | min | last | count
    target: Optional[float] = None
    threshold_warn: Optional[float] = None
    threshold_crit: Optional[float] = None
    higher_is_better: bool = True


@dataclass
class DataSource:
    name: str
    source_type: str           # push | pull | stream
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tags: Dict[str, str] = field(default_factory=dict)
    last_seen: Optional[datetime] = None


# ──────────────────────────── database layer ────────────────────────────────

DDL = """
CREATE TABLE IF NOT EXISTS metrics (
    id         TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    value      REAL NOT NULL,
    source     TEXT DEFAULT '',
    unit       TEXT DEFAULT '',
    dimensions TEXT DEFAULT '',
    ts         TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS aggregations (
    id       TEXT PRIMARY KEY,
    name     TEXT NOT NULL,
    period   TEXT NOT NULL,
    agg_type TEXT NOT NULL,
    value    REAL NOT NULL,
    ts       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS kpi_snapshots (
    id         TEXT PRIMARY KEY,
    kpi_name   TEXT NOT NULL,
    value      REAL NOT NULL,
    target     REAL,
    status     TEXT NOT NULL DEFAULT 'ok',
    captured_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics(name);
CREATE INDEX IF NOT EXISTS idx_metrics_ts   ON metrics(ts);
CREATE INDEX IF NOT EXISTS idx_kpi_name     ON kpi_snapshots(kpi_name);
"""


def _encode_dims(dims: Dict[str, str]) -> str:
    return ",".join(f"{k}={v}" for k, v in sorted(dims.items()))


def _decode_dims(raw: str) -> Dict[str, str]:
    if not raw:
        return {}
    result: Dict[str, str] = {}
    for part in raw.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            result[k] = v
    return result


class AnalyticsDashboard:
    """
    Full analytics engine backed by SQLite.
    Supports metric ingestion, flexible querying, KPI calculation,
    chart data generation, CSV export, and z-score anomaly detection.
    """

    def __init__(self, db_path: str = ":memory:"):
        self.conn = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(DDL)
        self.conn.commit()

    # ── ingestion ────────────────────────────────────────────────────────────

    def ingest_metrics(self, metrics: List[Metric]) -> int:
        """Bulk-insert metrics.  Returns count inserted."""
        rows = [
            (m.id, m.name, m.value, m.source, m.unit,
             _encode_dims(m.dimensions), m.timestamp.isoformat())
            for m in metrics
        ]
        self.conn.executemany(
            "INSERT OR IGNORE INTO metrics VALUES (?,?,?,?,?,?,?)", rows
        )
        self.conn.commit()
        return len(rows)

    def ingest_one(self, name: str, value: float,
                   dimensions: Optional[Dict[str, str]] = None,
                   source: str = "", unit: str = "") -> Metric:
        m = Metric(name=name, value=value, dimensions=dimensions or {},
                   source=source, unit=unit)
        self.ingest_metrics([m])
        return m

    # ── querying ─────────────────────────────────────────────────────────────

    def query_metrics(
        self,
        name: str,
        start: datetime,
        end: datetime,
        group_by: Optional[str] = None,
        filters: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query time-series data for *name* between *start* and *end*.
        Optional *group_by* dimension key aggregates (avg) per group.
        Optional *filters* restricts to metrics whose dimensions match.
        """
        rows = self.conn.execute(
            "SELECT * FROM metrics WHERE name=? AND ts >= ? AND ts <= ? "
            "ORDER BY ts",
            (name, start.isoformat(), end.isoformat()),
        ).fetchall()

        result = []
        for r in rows:
            dims = _decode_dims(r["dimensions"])
            # Apply dimension filters
            if filters and not all(dims.get(k) == v for k, v in filters.items()):
                continue
            result.append({
                "id": r["id"],
                "name": r["name"],
                "value": r["value"],
                "ts": r["ts"],
                "dimensions": dims,
                "source": r["source"] or "",
                "unit": r["unit"] or "",
            })

        if not group_by or not result:
            return result

        # Group by dimension key and return averaged series
        groups: Dict[str, List[float]] = defaultdict(list)
        for pt in result:
            key = pt["dimensions"].get(group_by, "__unset__")
            groups[key].append(pt["value"])
        return [
            {"group": k, "count": len(vals),
             "avg": round(mean(vals), 4),
             "min": min(vals), "max": max(vals),
             "sum": round(sum(vals), 4)}
            for k, vals in sorted(groups.items())
        ]

    def latest(self, name: str,
               filters: Optional[Dict[str, str]] = None) -> Optional[float]:
        """Return the most recent value for *name*."""
        row = self.conn.execute(
            "SELECT value FROM metrics WHERE name=? ORDER BY ts DESC LIMIT 1",
            (name,),
        ).fetchone()
        return row["value"] if row else None

    # ── KPI calculation ──────────────────────────────────────────────────────

    def calculate_kpis(
        self,
        metrics: List[Metric],
        definitions: List[KPIDefinition],
    ) -> List[Dict[str, Any]]:
        """
        Calculate KPIs from an in-memory list of metrics, persist snapshots,
        and return results with status evaluation.
        """
        by_name: Dict[str, List[float]] = defaultdict(list)
        for m in metrics:
            by_name[m.name].append(m.value)

        results = []
        for defn in definitions:
            vals = by_name.get(defn.metric_name, [])
            if not vals:
                continue

            agg_fn: Callable[[List[float]], float] = {
                "sum":   sum,
                "avg":   mean,
                "max":   max,
                "min":   min,
                "last":  lambda v: v[-1],
                "count": lambda v: float(len(v)),
            }.get(defn.aggregation, mean)
            value = round(agg_fn(vals), 4)

            # Determine status
            status = "ok"
            if defn.threshold_crit is not None:
                if defn.higher_is_better and value < defn.threshold_crit:
                    status = "critical"
                elif not defn.higher_is_better and value > defn.threshold_crit:
                    status = "critical"
            if status == "ok" and defn.threshold_warn is not None:
                if defn.higher_is_better and value < defn.threshold_warn:
                    status = "warn"
                elif not defn.higher_is_better and value > defn.threshold_warn:
                    status = "warn"

            pct_of_target = None
            if defn.target is not None and defn.target != 0:
                pct_of_target = round(value / defn.target * 100, 1)

            snap_id = str(uuid.uuid4())
            self.conn.execute(
                "INSERT INTO kpi_snapshots VALUES (?,?,?,?,?,?)",
                (snap_id, defn.name, value, defn.target, status,
                 datetime.utcnow().isoformat()),
            )
            results.append({
                "kpi": defn.name,
                "metric": defn.metric_name,
                "value": value,
                "aggregation": defn.aggregation,
                "target": defn.target,
                "pct_of_target": pct_of_target,
                "status": status,
                "higher_is_better": defn.higher_is_better,
            })
        self.conn.commit()
        return results

    # ── chart data generation ─────────────────────────────────────────────────

    def generate_chart_data(
        self,
        metrics: List[Dict[str, Any]],
        chart_type: str = "line",
        bucket_minutes: int = 60,
    ) -> Dict[str, Any]:
        """
        Transform raw metric rows (from query_metrics) into chart-ready
        structures.  Supports line, bar, scatter, and pie.
        """
        if not metrics:
            return {"chart_type": chart_type, "series": [], "labels": []}

        if chart_type in ("line", "bar"):
            # Time-bucket aggregation
            buckets: Dict[str, List[float]] = defaultdict(list)
            for pt in metrics:
                ts = datetime.fromisoformat(pt["ts"])
                bucket_ts = ts.replace(
                    minute=(ts.minute // bucket_minutes) * bucket_minutes,
                    second=0, microsecond=0,
                ).isoformat()
                buckets[bucket_ts].append(pt["value"])
            labels = sorted(buckets)
            values = [round(mean(buckets[t]), 4) for t in labels]
            return {
                "chart_type": chart_type,
                "labels": labels,
                "series": [{"name": metrics[0]["name"], "data": values}],
                "bucket_minutes": bucket_minutes,
            }

        if chart_type == "scatter":
            return {
                "chart_type": "scatter",
                "points": [{"x": pt["ts"], "y": pt["value"]} for pt in metrics],
            }

        if chart_type == "pie":
            totals: Dict[str, float] = defaultdict(float)
            for pt in metrics:
                dim_label = next(iter(pt.get("dimensions", {}).values()), "default")
                totals[dim_label] += pt["value"]
            total = sum(totals.values()) or 1
            return {
                "chart_type": "pie",
                "slices": [
                    {"label": k, "value": round(v, 4),
                     "pct": round(v / total * 100, 1)}
                    for k, v in sorted(totals.items(), key=lambda x: -x[1])
                ],
            }

        raise ValueError(f"Unsupported chart_type {chart_type!r}")

    # ── CSV export ────────────────────────────────────────────────────────────

    def export_csv(self, metrics: List[Dict[str, Any]]) -> str:
        """Serialize metric rows (from query_metrics) to CSV string."""
        if not metrics:
            return ""
        buf = io.StringIO()
        writer = csv.DictWriter(
            buf,
            fieldnames=["name", "value", "ts", "source", "unit", "dimensions"],
            extrasaction="ignore",
        )
        writer.writeheader()
        for pt in metrics:
            row = dict(pt)
            if isinstance(row.get("dimensions"), dict):
                row["dimensions"] = _encode_dims(row["dimensions"])
            writer.writerow(row)
        return buf.getvalue()

    # ── anomaly detection ─────────────────────────────────────────────────────

    def anomaly_score(
        self,
        metric_series: List[float],
        window: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Compute z-score anomaly scores using a rolling moving-average window.
        Returns a list of {index, value, z_score, is_anomaly} dicts.
        A point is flagged as anomalous when |z| > 3.
        """
        if len(metric_series) < 3:
            return [{"index": i, "value": v, "z_score": 0.0, "is_anomaly": False}
                    for i, v in enumerate(metric_series)]

        results = []
        for i, value in enumerate(metric_series):
            lo = max(0, i - window)
            window_vals = metric_series[lo:i + 1]
            if len(window_vals) < 2:
                results.append(
                    {"index": i, "value": value, "z_score": 0.0, "is_anomaly": False}
                )
                continue
            mu = mean(window_vals)
            sigma = stdev(window_vals)
            z = (value - mu) / sigma if sigma > 0 else 0.0
            results.append({
                "index": i,
                "value": value,
                "z_score": round(z, 4),
                "is_anomaly": abs(z) > 3.0,
                "mean": round(mu, 4),
                "stddev": round(sigma, 4),
            })
        return results

    def detect_anomalies_db(
        self, name: str, start: datetime, end: datetime, window: int = 20
    ) -> List[Dict[str, Any]]:
        """Convenience: query DB → anomaly_score pipeline."""
        pts = self.query_metrics(name, start, end)
        if not pts:
            return []
        series = [p["value"] for p in pts]
        scores = self.anomaly_score(series, window)
        for i, score in enumerate(scores):
            score["ts"] = pts[i]["ts"]
        return scores

    # ── rollup aggregations ───────────────────────────────────────────────────

    def rollup(self, name: str, period: str = "hour",
               agg_type: str = "avg") -> List[Dict[str, Any]]:
        """
        Aggregate raw metrics to hourly/daily periods and persist to
        aggregations table.  Returns the rolled-up series.
        """
        fmt_map = {"hour": "%Y-%m-%dT%H:00:00", "day": "%Y-%m-%d"}
        if period not in fmt_map:
            raise ValueError(f"period must be one of {list(fmt_map)}")
        fmt = fmt_map[period]

        agg_sql = {"avg": "AVG(value)", "sum": "SUM(value)",
                   "max": "MAX(value)", "min": "MIN(value)"}
        if agg_type not in agg_sql:
            raise ValueError(f"agg_type must be one of {list(agg_sql)}")

        rows = self.conn.execute(
            f"SELECT strftime('{fmt}', ts) as bucket, {agg_sql[agg_type]} as val "
            f"FROM metrics WHERE name=? GROUP BY bucket ORDER BY bucket",
            (name,),
        ).fetchall()

        result = []
        for r in rows:
            snap_id = str(uuid.uuid4())
            self.conn.execute(
                "INSERT OR IGNORE INTO aggregations VALUES (?,?,?,?,?,?)",
                (snap_id, name, r["bucket"], agg_type,
                 round(r["val"], 6), r["bucket"]),
            )
            result.append({"ts": r["bucket"], "value": round(r["val"], 6)})
        self.conn.commit()
        return result

    def close(self) -> None:
        self.conn.close()
