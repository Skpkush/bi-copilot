"""Analyst SQL library runner and insights report.

Each file in ``sql/analysis`` answers one business question. This module runs
them through the same governed read-only path as the agent, powers the
dashboard tab, and writes ``reports/insights.md`` with headline findings
computed from the results (no hand-typed numbers).

Usage:
    python -m src.analytics            # print findings and write the report
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT
from src.sql_guard import execute_readonly_query, rows_to_markdown

ANALYSIS_DIR = PROJECT_ROOT / "sql" / "analysis"
REPORT_PATH = PROJECT_ROOT / "reports" / "insights.md"


@dataclass(frozen=True)
class Analysis:
    key: str
    title: str
    question: str
    sql: str
    path: Path


def _parse_header(path: Path) -> tuple[str, str]:
    """The first comment line is the title; the second is the business framing."""
    comments = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("--"):
            break
        comments.append(line.lstrip("- ").strip())
    title = comments[0] if comments else path.stem
    question = " ".join(comments[1:])
    return title, question


@lru_cache(maxsize=1)
def list_analyses() -> tuple[Analysis, ...]:
    analyses = []
    for path in sorted(ANALYSIS_DIR.glob("*.sql")):
        title, question = _parse_header(path)
        key = path.stem.split("_", 1)[1] if "_" in path.stem else path.stem
        analyses.append(
            Analysis(key, title, question, path.read_text(encoding="utf-8"), path)
        )
    return tuple(analyses)


def get_analysis(key: str) -> Analysis:
    for analysis in list_analyses():
        if analysis.key == key:
            return analysis
    raise KeyError(f"Unknown analysis: {key}")


def run_analysis(key: str) -> pd.DataFrame:
    result = execute_readonly_query(get_analysis(key).sql, max_rows=500)
    return pd.DataFrame(result.rows, columns=result.columns)


def _pct_change(current: float, prior: float) -> float | None:
    if not prior:
        return None
    return 100.0 * (current - prior) / prior


def kpi_summary() -> dict[str, dict[str, float | None]]:
    """Current vs prior quarter KPIs with percent / point deltas."""
    df = run_analysis("kpi_quarter_over_quarter").set_index("period")
    current, prior = df.loc["current"], df.loc["prior"]
    summary: dict[str, dict[str, float | None]] = {}
    for column in df.columns:
        cur, prv = float(current[column]), float(prior[column])
        if column.endswith("_pct"):
            delta = cur - prv  # percentage points
        else:
            delta = _pct_change(cur, prv)
        summary[column] = {"current": cur, "prior": prv, "delta": delta}
    return summary


def headline_findings() -> list[str]:
    """Plain-English findings derived from the analysis results."""
    findings: list[str] = []

    kpi = kpi_summary()
    rev, gm = kpi["revenue"], kpi["gross_margin_pct"]
    findings.append(
        f"Q3 revenue was ${rev['current']:,.0f}, {rev['delta']:+.1f}% vs Q2, on "
        f"{kpi['orders']['current']:.0f} orders ({kpi['orders']['delta']:+.1f}%). "
        f"AOV moved {kpi['avg_order_value']['delta']:+.1f}%."
    )
    findings.append(
        f"Product gross margin was {gm['current']:.1f}% in Q3 vs {gm['prior']:.1f}% in Q2 "
        f"({gm['delta']:+.1f} pts). The Q3 summary's '8% margin drop' is therefore not "
        "visible in product economics; it can only come from freight, which the "
        "warehouse does not store. Recommendation: load carrier invoices before "
        "acting on that figure."
    )

    cat = run_analysis("category_margin")
    top_cat = cat.iloc[0]
    low_margin = cat.sort_values("gross_margin_pct").iloc[0]
    lead = (
        f"{top_cat['category']} is the largest category ({top_cat['revenue_share_pct']:.1f}% "
        "of Q3 revenue)"
    )
    if low_margin["category"] == top_cat["category"]:
        findings.append(
            f"{lead} and also has the thinnest margin ({low_margin['gross_margin_pct']:.1f}%): "
            f"the biggest revenue line is the least profitable per dollar, so "
            f"{top_cat['category'].lower()} pricing and supplier cost are the "
            "highest-leverage margin levers."
        )
    else:
        findings.append(
            f"{lead}; {low_margin['category']} has the thinnest margin "
            f"({low_margin['gross_margin_pct']:.1f}%)."
        )

    disc = run_analysis("discount_band_impact")
    no_disc, deep = disc.iloc[0], disc.iloc[-1]
    findings.append(
        f"Orders discounted {deep['discount_band']} earn {deep['gross_margin_pct']:.1f}% margin vs "
        f"{no_disc['gross_margin_pct']:.1f}% at full price, with "
        f"{deep['avg_units_per_order']:.2f} vs {no_disc['avg_units_per_order']:.2f} units per order: "
        + (
            "deep discounts are not buying extra volume."
            if deep["avg_units_per_order"] <= no_disc["avg_units_per_order"] * 1.1
            else "deep discounts do lift basket size; weigh that against the margin cost."
        )
    )

    pareto = run_analysis("customer_pareto")
    top10 = pareto.iloc[0]
    findings.append(
        f"The top 10% of customers ({top10['customers']:.0f}) generate "
        f"{top10['revenue_share_pct']:.1f}% of revenue."
    )

    rfm = run_analysis("rfm_segments")
    at_risk = rfm[rfm["segment"] == "At Risk (high value)"]
    if not at_risk.empty:
        row = at_risk.iloc[0]
        findings.append(
            f"{row['customers']:.0f} high-value customers are 'At Risk' (no recent order), "
            f"holding {row['revenue_share_pct']:.1f}% of lifetime revenue: the first "
            "win-back campaign target."
        )

    cohorts = run_analysis("cohort_retention")
    mature = cohorts.iloc[:-3] if len(cohorts) > 3 else cohorts
    if not mature.empty:
        findings.append(
            f"On average {mature['repeat_within_3m_pct'].mean():.1f}% of a monthly cohort "
            "orders again within 3 months (cohorts with a full 3-month window)."
        )
    return findings


def write_report(path: Path = REPORT_PATH) -> Path:
    lines = [
        "# Summit Outdoor Co. - Analyst Insights Report",
        "",
        "_Generated by `python -m src.analytics` from `sql/analysis/*.sql`. "
        "Every number below is computed from the warehouse, not typed by hand._",
        "",
        "## Headline findings",
        "",
    ]
    lines += [f"{i}. {finding}" for i, finding in enumerate(headline_findings(), 1)]
    for analysis in list_analyses():
        df = run_analysis(analysis.key)
        lines += [
            "",
            f"## {analysis.title}",
            "",
            f"_{analysis.question}_" if analysis.question else "",
            "",
            f"Source: [`{analysis.path.name}`](../sql/analysis/{analysis.path.name})",
            "",
            rows_to_markdown(list(df.columns), df.to_dict("records")),
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    for i, finding in enumerate(headline_findings(), 1):
        print(f"{i}. {finding}")
    print(f"\nReport written to {write_report()}")


if __name__ == "__main__":
    main()
