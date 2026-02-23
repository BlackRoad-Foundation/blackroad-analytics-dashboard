"""Tests for BlackRoad Analytics Dashboard."""
import pytest
from datetime import datetime, timedelta
from analytics_dashboard import (
    AnalyticsDashboard, Metric, KPIDefinition, _encode_dims, _decode_dims
)


@pytest.fixture
def dash():
    d = AnalyticsDashboard(":memory:")
    yield d
    d.close()


def make_series(name: str, values, start: datetime = None):
    start = start or datetime(2024, 1, 1, 0, 0)
    return [
        Metric(name=name, value=v,
               timestamp=start + timedelta(hours=i))
        for i, v in enumerate(values)
    ]


# ── test 1: ingest and query metrics ────────────────────────────────────────
def test_ingest_and_query(dash):
    series = make_series("cpu", [10, 20, 30, 40])
    dash.ingest_metrics(series)
    start = datetime(2024, 1, 1)
    end = datetime(2024, 1, 2)
    results = dash.query_metrics("cpu", start, end)
    assert len(results) == 4
    assert results[0]["value"] == 10


# ── test 2: query with dimension filter ──────────────────────────────────────
def test_query_with_filter(dash):
    m1 = Metric(name="req", value=100, dimensions={"region": "us"},
                timestamp=datetime(2024, 1, 1, 1))
    m2 = Metric(name="req", value=200, dimensions={"region": "eu"},
                timestamp=datetime(2024, 1, 1, 2))
    dash.ingest_metrics([m1, m2])
    results = dash.query_metrics(
        "req",
        datetime(2024, 1, 1),
        datetime(2024, 1, 2),
        filters={"region": "us"},
    )
    assert len(results) == 1
    assert results[0]["value"] == 100


# ── test 3: KPI calculation ───────────────────────────────────────────────────
def test_calculate_kpis(dash):
    metrics = make_series("uptime_pct", [99.9, 99.5, 100.0, 98.0])
    defn = KPIDefinition(
        name="Uptime", metric_name="uptime_pct",
        aggregation="avg", target=99.0,
        threshold_warn=98.5, threshold_crit=95.0,
    )
    results = dash.calculate_kpis(metrics, [defn])
    assert len(results) == 1
    assert results[0]["kpi"] == "Uptime"
    assert results[0]["value"] == pytest.approx(99.35, 0.01)
    assert results[0]["status"] == "ok"
    assert results[0]["pct_of_target"] == pytest.approx(100.35, 0.1)


# ── test 4: anomaly detection ─────────────────────────────────────────────────
def test_anomaly_score(dash):
    # Mostly flat series with one spike
    series = [1.0] * 30 + [100.0] + [1.0] * 10
    scores = dash.anomaly_score(series, window=20)
    spike_score = scores[30]
    assert spike_score["is_anomaly"] is True
    assert abs(spike_score["z_score"]) > 3


# ── test 5: generate line chart data ─────────────────────────────────────────
def test_generate_line_chart(dash):
    start = datetime(2024, 1, 1)
    metrics = []
    for i in range(10):
        metrics.append({
            "name": "req_rate", "value": float(i * 10),
            "ts": (start + timedelta(hours=i * 2)).isoformat(),
            "dimensions": {}, "source": "", "unit": ""
        })
    chart = dash.generate_chart_data(metrics, chart_type="line", bucket_minutes=120)
    assert chart["chart_type"] == "line"
    assert len(chart["series"]) == 1
    assert len(chart["labels"]) > 0


# ── test 6: CSV export ────────────────────────────────────────────────────────
def test_export_csv(dash):
    series = make_series("temp", [22.5, 23.0, 21.8])
    dash.ingest_metrics(series)
    pts = dash.query_metrics("temp", datetime(2024, 1, 1), datetime(2024, 1, 2))
    csv_out = dash.export_csv(pts)
    assert "temp" in csv_out
    assert "22.5" in csv_out


# ── test 7: rollup aggregation ────────────────────────────────────────────────
def test_rollup(dash):
    base = datetime(2024, 3, 1, 0, 0)
    metrics = [
        Metric(name="hits", value=float(i), timestamp=base + timedelta(minutes=i * 10))
        for i in range(12)
    ]
    dash.ingest_metrics(metrics)
    rolled = dash.rollup("hits", period="hour", agg_type="sum")
    assert len(rolled) >= 1
    assert rolled[0]["value"] >= 0


# ── test 8: pie chart ──────────────────────────────────────────────────────────
def test_generate_pie_chart(dash):
    pts = [
        {"name": "sales", "value": 400.0, "ts": "2024-01-01T00:00:00",
         "dimensions": {"region": "us"}, "source": "", "unit": ""},
        {"name": "sales", "value": 300.0, "ts": "2024-01-01T01:00:00",
         "dimensions": {"region": "eu"}, "source": "", "unit": ""},
        {"name": "sales", "value": 100.0, "ts": "2024-01-01T02:00:00",
         "dimensions": {"region": "apac"}, "source": "", "unit": ""},
    ]
    chart = dash.generate_chart_data(pts, chart_type="pie")
    assert chart["chart_type"] == "pie"
    slices = {s["label"]: s["pct"] for s in chart["slices"]}
    assert slices["us"] == pytest.approx(50.0, 0.1)
