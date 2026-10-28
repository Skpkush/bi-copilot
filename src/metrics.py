"""Semantic metric layer: one definition per business metric.

Every SQL path (the agent's generated SQL, the offline query planner, the
dashboard, and the analysis library) reads metric expressions from here, so
"revenue" or "gross margin" means the same thing on every screen.

Expressions assume the aliases ``o`` (orders) and ``p`` (products).
"""

from __future__ import annotations

from dataclasses import dataclass

from src.config import LATEST_QUARTER_END, LATEST_QUARTER_START

PRIOR_QUARTER_START = "2026-04-01"
PRIOR_QUARTER_END = "2026-06-30"


@dataclass(frozen=True)
class Metric:
    name: str
    sql: str
    description: str
    unit: str  # "currency" | "percent" | "count"
    needs_products: bool = False


NET_LINE = "o.quantity * o.unit_price * (1 - o.discount)"
COGS_LINE = "o.quantity * p.cost_price"

METRICS: dict[str, Metric] = {
    "revenue": Metric(
        "revenue",
        f"ROUND(SUM({NET_LINE}), 2)",
        "Net booked revenue: quantity x unit_price x (1 - discount).",
        "currency",
    ),
    "orders": Metric("orders", "COUNT(DISTINCT o.order_id)", "Distinct booked orders.", "count"),
    "units": Metric("units", "SUM(o.quantity)", "Units sold.", "count"),
    "avg_order_value": Metric(
        "avg_order_value",
        f"ROUND(SUM({NET_LINE}) / NULLIF(COUNT(DISTINCT o.order_id), 0), 2)",
        "Revenue per booked order.",
        "currency",
    ),
    "avg_discount_pct": Metric(
        "avg_discount_pct",
        "ROUND(100.0 * AVG(o.discount), 1)",
        "Average line discount, in percent.",
        "percent",
    ),
    "cogs": Metric(
        "cogs",
        f"ROUND(SUM({COGS_LINE}), 2)",
        "Product cost of goods sold (excludes freight).",
        "currency",
        needs_products=True,
    ),
    "gross_profit": Metric(
        "gross_profit",
        f"ROUND(SUM({NET_LINE} - {COGS_LINE}), 2)",
        "Revenue minus product COGS (excludes freight).",
        "currency",
        needs_products=True,
    ),
    "gross_margin_pct": Metric(
        "gross_margin_pct",
        f"ROUND(100.0 * SUM({NET_LINE} - {COGS_LINE}) / NULLIF(SUM({NET_LINE}), 0), 1)",
        "Gross profit as a percent of revenue (product-level, excludes freight).",
        "percent",
        needs_products=True,
    ),
    "active_customers": Metric(
        "active_customers",
        "COUNT(DISTINCT o.customer_id)",
        "Customers with at least one booked order.",
        "count",
    ),
}

# Booked revenue excludes cancelled/returned orders and lines with no price.
BOOKED_FILTER = "o.status NOT IN ('Cancelled', 'Returned') AND o.unit_price IS NOT NULL"


def period_filter(start: str, end: str) -> str:
    return f"o.order_date BETWEEN '{start}' AND '{end}'"


LATEST_QUARTER_FILTER = period_filter(LATEST_QUARTER_START, LATEST_QUARTER_END)
PRIOR_QUARTER_FILTER = period_filter(PRIOR_QUARTER_START, PRIOR_QUARTER_END)


def select_metrics(*names: str) -> str:
    """Render ``expr AS name`` select items for the requested metrics."""
    return ",\n  ".join(f"{METRICS[name].sql} AS {name}" for name in names)


def metric_glossary() -> str:
    """Plain-text metric definitions for the LLM prompt and the README."""
    lines = [f"- {m.name} = {m.sql}  -- {m.description}" for m in METRICS.values()]
    lines.append(f"- Booked filter (always apply for revenue metrics): {BOOKED_FILTER}")
    return "\n".join(lines)
