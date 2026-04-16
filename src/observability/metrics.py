import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict

METRICS_FILE = Path("metrics/metrics_log.json")


def ensure_metrics_file():
    """Create metrics file and folder if they don't exist."""
    METRICS_FILE.parent.mkdir(exist_ok=True)
    if not METRICS_FILE.exists():
        METRICS_FILE.write_text("[]")


def log_request_metric(
    question: str,
    latency_seconds: float,
    input_tokens: int,
    output_tokens: int,
    estimated_cost_usd: float,
    citation_supported: bool,
    status: str,
    trace_id: str
):
    """Append a single request metric to the log file."""
    ensure_metrics_file()

    entry = {
        "timestamp": datetime.now().isoformat(),
        "trace_id": trace_id,
        "question_preview": question[:60],
        "latency_seconds": latency_seconds,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "estimated_cost_usd": estimated_cost_usd,
        "citation_supported": citation_supported,
        "status": status
    }

    entries = json.loads(METRICS_FILE.read_text())
    entries.append(entry)
    METRICS_FILE.write_text(json.dumps(entries, indent=2))


def compute_summary(entries: List[Dict]) -> Dict:
    """
    Compute P50, P95 latency, average cost, citation coverage
    and failure rate from a list of metric entries.
    """
    if not entries:
        return {}

    latencies = [e["latency_seconds"] for e in entries]
    costs = [e["estimated_cost_usd"] for e in entries]
    citations = [e["citation_supported"] for e in entries]
    statuses = [e["status"] for e in entries]

    return {
        "total_requests": len(entries),
        "latency_p50_seconds": round(float(np.percentile(latencies, 50)), 3),
        "latency_p95_seconds": round(float(np.percentile(latencies, 95)), 3),
        "latency_avg_seconds": round(float(np.mean(latencies)), 3),
        "cost_per_request_avg_usd": round(float(np.mean(costs)), 6),
        "cost_total_usd": round(float(sum(costs)), 6),
        "citation_coverage_pct": round(sum(citations) / len(citations) * 100, 1),
        "failure_rate_pct": round(
            sum(1 for s in statuses if s == "error") / len(statuses) * 100, 1
        ),
        "decline_rate_pct": round(
            sum(1 for e in entries if not e["citation_supported"]) / len(entries) * 100, 1
        )
    }


def get_metrics_summary() -> Dict:
    """Load all metrics and return computed summary."""
    ensure_metrics_file()
    entries = json.loads(METRICS_FILE.read_text())
    return compute_summary(entries)


def print_dashboard():
    """Print a terminal metrics dashboard."""
    ensure_metrics_file()
    entries = json.loads(METRICS_FILE.read_text())

    if not entries:
        print("No metrics recorded yet.")
        return

    summary = compute_summary(entries)

    print("\n" + "=" * 50)
    print("  RAG SYSTEM — METRICS DASHBOARD")
    print("=" * 50)
    print(f"  Total requests:       {summary['total_requests']}")
    print(f"  Latency P50:          {summary['latency_p50_seconds']}s")
    print(f"  Latency P95:          {summary['latency_p95_seconds']}s")
    print(f"  Latency avg:          {summary['latency_avg_seconds']}s")
    print(f"  Avg cost/request:     ${summary['cost_per_request_avg_usd']}")
    print(f"  Total cost:           ${summary['cost_total_usd']}")
    print(f"  Citation coverage:    {summary['citation_coverage_pct']}%")
    print(f"  Failure rate:         {summary['failure_rate_pct']}%")
    print(f"  Decline rate:         {summary['decline_rate_pct']}%")
    print("=" * 50)

    print("\n  Last 5 requests:")
    for e in entries[-5:]:
        status = "OK" if e["citation_supported"] else "DECLINED"
        print(f"  [{e['timestamp'][:19]}] {e['latency_seconds']}s "
              f"${e['estimated_cost_usd']} {status}")
        print(f"    Q: {e['question_preview']}")