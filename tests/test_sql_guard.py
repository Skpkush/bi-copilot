"""Unit tests for the governed SQL layer (no API key required)."""

from __future__ import annotations

import unittest

from src.sql_guard import SQLValidationError, execute_readonly_query, validate_sql


class SQLGuardTests(unittest.TestCase):
    def test_allows_single_select(self):
        self.assertEqual(
            validate_sql("SELECT status, COUNT(*) FROM orders GROUP BY status;"),
            "SELECT status, COUNT(*) FROM orders GROUP BY status",
        )

    def test_allows_cte(self):
        sql = "WITH x AS (SELECT 1 AS n) SELECT n FROM x"
        self.assertEqual(validate_sql(sql), sql)

    def test_blocks_dml(self):
        for sql in (
            "DELETE FROM orders",
            "UPDATE orders SET status = 'Cancelled'",
            "DROP TABLE orders",
            "INSERT INTO orders(order_id) VALUES ('x')",
        ):
            with self.subTest(sql=sql), self.assertRaises(SQLValidationError):
                validate_sql(sql)

    def test_blocks_stacked_statements(self):
        with self.assertRaises(SQLValidationError):
            validate_sql("SELECT * FROM orders; DROP TABLE orders;")

    def test_comments_cannot_hide_stacked_statement(self):
        with self.assertRaises(SQLValidationError):
            validate_sql("SELECT 1; /* comment */ DELETE FROM orders")

    def test_keywords_inside_strings_are_allowed(self):
        sql = "SELECT 'DROP TABLE' AS note"
        self.assertEqual(validate_sql(sql), sql)

    def test_allows_semicolon_in_string(self):
        sql = "SELECT 'hello; world' AS greeting"
        self.assertEqual(validate_sql(sql), sql)

    def test_blocks_keyword_hidden_in_cte(self):
        with self.assertRaises(SQLValidationError):
            validate_sql("WITH x AS (DELETE FROM orders RETURNING *) SELECT * FROM x")

    def test_executes_read_only_query(self):
        result = execute_readonly_query("SELECT COUNT(*) AS n FROM orders", max_rows=5)
        self.assertEqual(result.columns, ["n"])
        self.assertGreater(result.row_count, 0)

    def test_executes_cte_and_enforces_row_cap(self):
        result = execute_readonly_query(
            "WITH x AS (SELECT order_id FROM orders ORDER BY order_id) SELECT * FROM x",
            max_rows=2,
        )
        self.assertEqual(len(result.rows), 2)
        self.assertTrue(result.truncated)

    def test_preserves_ordering(self):
        result = execute_readonly_query(
            "SELECT order_id FROM orders ORDER BY order_id LIMIT 3", max_rows=10
        )
        self.assertEqual(
            [row["order_id"] for row in result.rows],
            ["ORD-00001", "ORD-00002", "ORD-00003"],
        )

    def test_execution_rejects_writes_even_if_validation_is_bypassed(self):
        # Defense in depth: the SQLite connection itself is opened read-only.
        with self.assertRaises(Exception):
            from sqlalchemy import text

            from src.sql_guard import get_readonly_engine

            with get_readonly_engine().connect() as connection:
                connection.execute(text("DELETE FROM orders"))


if __name__ == "__main__":
    unittest.main()
