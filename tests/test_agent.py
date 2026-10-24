"""Integration checks for the offline LangGraph demo path."""

from __future__ import annotations

import os
import unittest

from src.build_rag import read_index_meta
from src.config import DB_PATH

os.environ["BI_COPILOT_OFFLINE"] = "1"

from src.agent import answer_question  # noqa: E402


class AgentIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not DB_PATH.exists() or not read_index_meta():
            raise unittest.SkipTest("Run the data/index setup before integration tests.")
        cls.answers = {
            "top": answer_question(
                "What were the top 3 products by revenue in the Northeast last quarter?"
            ),
            "policy": answer_question("What is the discount policy for orders above $1000?"),
            "margin": answer_question(
                "Why did profit margins drop in Q3, and which region was most affected?"
            ),
        }

    def test_sql_route_returns_table_and_chart_rows(self):
        result = self.answers["top"]
        self.assertEqual(result.route, "sql")
        self.assertEqual(len(result.rows), 3)
        self.assertIn("revenue", result.columns)
        self.assertIn("Velocity Trail Shoe", result.answer)
        self.assertIn("SQL query result", result.answer)

    def test_document_route_cites_filename(self):
        result = self.answers["policy"]
        self.assertEqual(result.route, "docs")
        self.assertIn("$1,000", result.answer)
        self.assertIn("Sales Manager approval", result.answer)
        self.assertIn("Sources:", result.answer)
        self.assertIn("sales_policy.txt", result.answer)

    def test_combined_route_uses_both_sources(self):
        result = self.answers["margin"]
        self.assertEqual(result.route, "both")
        self.assertGreater(len(result.rows), 0)
        self.assertIn("Northeast", result.answer)
        self.assertIn("8%", result.answer)
        self.assertIn("SQL query result", result.answer)
        self.assertIn("q3_summary.txt", result.answer)

    def test_every_answer_ends_with_sources(self):
        for name, result in self.answers.items():
            with self.subTest(question=name):
                tail = result.answer[-400:]
                self.assertIn("**Sources:**", tail)
                self.assertGreaterEqual(tail.count("\n- "), 1)


if __name__ == "__main__":
    unittest.main()
