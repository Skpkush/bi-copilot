"""Checks for the metric layer, SQL analysis library, quality report, and evals."""

from __future__ import annotations

import json
import os
import unittest

os.environ["BI_COPILOT_OFFLINE"] = "1"

from src.analytics import headline_findings, kpi_summary, list_analyses, run_analysis  # noqa: E402
from src.build_rag import read_index_meta  # noqa: E402
from src.config import DB_PATH, QUALITY_REPORT_PATH  # noqa: E402
from src.metrics import BOOKED_FILTER, LATEST_QUARTER_FILTER, select_metrics  # noqa: E402
from src.sql_guard import execute_readonly_query  # noqa: E402


def _requires_data():
    if not DB_PATH.exists():
        raise unittest.SkipTest("Run `python -m src.pipeline` first.")


class MetricLayerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _requires_data()

    def test_library_kpis_match_the_metric_layer(self):
        """The hand-written SQL library and the metric layer must agree."""
        sql = f"""
SELECT {select_metrics("revenue", "orders", "gross_margin_pct")}
FROM orders o JOIN products p ON p.product_id = o.product_id
WHERE {LATEST_QUARTER_FILTER} AND {BOOKED_FILTER}
"""
        governed = execute_readonly_query(sql).rows[0]
        kpi = kpi_summary()
        self.assertAlmostEqual(governed["revenue"], kpi["revenue"]["current"], places=2)
        self.assertEqual(governed["orders"], kpi["orders"]["current"])
        self.assertAlmostEqual(
            governed["gross_margin_pct"], kpi["gross_margin_pct"]["current"], places=1
        )

    def test_every_library_query_runs_through_the_guard(self):
        analyses = list_analyses()
        self.assertGreaterEqual(len(analyses), 8)
        for analysis in analyses:
            with self.subTest(analysis=analysis.key):
                df = run_analysis(analysis.key)
                self.assertFalse(df.empty)

    def test_shares_sum_to_100(self):
        for key, column in (
            ("category_margin", "revenue_share_pct"),
            ("customer_pareto", "revenue_share_pct"),
            ("rfm_segments", "revenue_share_pct"),
        ):
            with self.subTest(analysis=key):
                self.assertAlmostEqual(run_analysis(key)[column].sum(), 100.0, delta=0.5)

    def test_monthly_growth_first_month_has_no_baseline(self):
        df = run_analysis("monthly_growth")
        self.assertTrue(df["mom_growth_pct"].isna().iloc[0])
        self.assertEqual(len(df), 9)

    def test_findings_do_not_claim_unverified_margin_drop(self):
        findings = " ".join(headline_findings())
        self.assertIn("not visible in product economics", findings)


class QualityReportTests(unittest.TestCase):
    def test_all_quality_checks_pass(self):
        if not QUALITY_REPORT_PATH.exists():
            raise unittest.SkipTest("Run `python -m src.load_data` first.")
        report = json.loads(QUALITY_REPORT_PATH.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(report["checks"]), 8)
        failed = [c["check"] for c in report["checks"] if not c["passed"]]
        self.assertEqual(failed, [])
        self.assertEqual(report["tables"]["orders"]["clean_rows"], 490)


class EvaluationTests(unittest.TestCase):
    def test_golden_set_passes(self):
        _requires_data()
        if not read_index_meta():
            raise unittest.SkipTest("Run `python -m src.build_rag` first.")
        from src.evaluate import evaluate

        results, scores = evaluate()
        failures = {r.id: r.failures for r in results if not r.passed}
        self.assertEqual(failures, {})
        self.assertEqual(scores["route_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
