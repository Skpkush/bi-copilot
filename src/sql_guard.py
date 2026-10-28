"""SQL safety, read-only execution, and schema helpers.

The guard is intentionally conservative: one SELECT/CTE statement, no DDL or
DML keywords, no stacked statements, and a hard row cap. Every query also runs
through a SQLite connection opened with ``mode=ro`` and ``PRAGMA query_only``.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from functools import lru_cache
from typing import Any
from urllib.parse import quote

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.pool import NullPool

from src.config import DB_PATH, MAX_SQL_ROWS, require_db

TABLE_NAMES = ("orders", "products", "customers")

FORBIDDEN_KEYWORDS = re.compile(
    r"\b("
    r"insert|update|delete|drop|alter|create|truncate|attach|detach|pragma|"
    r"vacuum|reindex|grant|revoke|merge|call|copy|begin|commit|rollback|"
    r"savepoint|replace\s+into|into|load_extension|writefile|readfile"
    r")\b",
    re.IGNORECASE,
)


class SQLValidationError(ValueError):
    """Raised when a user or model query is not a safe read-only SELECT."""


@dataclass
class SQLResult:
    columns: list[str]
    rows: list[dict[str, Any]]
    sql: str
    row_count: int
    truncated: bool = False

    def to_markdown(self, max_rows: int | None = None) -> str:
        rows = self.rows if max_rows is None else self.rows[:max_rows]
        return rows_to_markdown(self.columns, rows, truncated=self.truncated)


def _extract_code_fence(sql: str) -> str:
    value = sql.strip()
    fenced = re.fullmatch(r"```(?:sql)?\s*(.*?)\s*```", value, flags=re.IGNORECASE | re.DOTALL)
    return fenced.group(1).strip() if fenced else value


def _mask_literals_and_comments(sql: str) -> str:
    """Blank out strings/comments so keyword checks cannot be fooled by them."""
    output: list[str] = []
    idx = 0
    length = len(sql)
    while idx < length:
        char = sql[idx]
        if char == "-" and idx + 1 < length and sql[idx + 1] == "-":
            newline = sql.find("\n", idx + 2)
            if newline == -1:
                output.extend(" " * (length - idx))
                break
            output.extend(" " * (newline - idx))
            idx = newline
            continue
        if char == "/" and idx + 1 < length and sql[idx + 1] == "*":
            end = sql.find("*/", idx + 2)
            if end == -1:
                raise SQLValidationError("Unterminated block comment.")
            output.extend(" " * (end + 2 - idx))
            idx = end + 2
            continue
        if char in {"'", '"', "`", "["}:
            closing = "]" if char == "[" else char
            output.append(" ")
            idx += 1
            closed = False
            while idx < length:
                current = sql[idx]
                if current == closing:
                    if closing != "]" and idx + 1 < length and sql[idx + 1] == closing:
                        output.extend("  ")
                        idx += 2
                        continue
                    output.append(" ")
                    idx += 1
                    closed = True
                    break
                output.append("\n" if current == "\n" else " ")
                idx += 1
            if not closed:
                raise SQLValidationError("Unterminated quoted value in SQL.")
            continue
        output.append(char)
        idx += 1
    return "".join(output)


def validate_sql(sql: str) -> str:
    """Return a cleaned single SELECT statement or raise SQLValidationError."""
    if not sql or not sql.strip():
        raise SQLValidationError("SQL is empty.")

    cleaned = _extract_code_fence(sql)
    masked = _mask_literals_and_comments(cleaned)
    core = masked.strip()
    if core.endswith(";"):
        core = core[:-1].rstrip()
    if ";" in core:
        raise SQLValidationError("Only one statement is allowed; multiple statements were detected.")
    if not core:
        raise SQLValidationError("SQL is empty.")

    if not re.match(r"^(select|with)\b", core, flags=re.IGNORECASE):
        raise SQLValidationError("Only read-only SELECT statements are allowed.")
    if core.upper().startswith("WITH") and not re.search(r"\bselect\b", core, re.IGNORECASE):
        raise SQLValidationError("A WITH query must contain a SELECT.")

    match = FORBIDDEN_KEYWORDS.search(core)
    if match:
        raise SQLValidationError(
            f"Blocked SQL keyword: {match.group(0).upper()}. Only read-only queries are allowed."
        )

    # Keep the trailing semicolon for display but execute without it.
    return cleaned.strip().rstrip(";").strip()


@lru_cache(maxsize=1)
def get_readonly_engine() -> Engine:
    """A SQLAlchemy engine whose connections physically cannot write."""
    require_db()
    db_path = DB_PATH.resolve().as_posix()
    uri = f"file:{quote(db_path, safe='/:')}?mode=ro"

    def _connect() -> sqlite3.Connection:
        connection = sqlite3.connect(uri, uri=True, timeout=10)
        connection.execute("PRAGMA query_only = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    return create_engine("sqlite://", creator=_connect, poolclass=NullPool)


@lru_cache(maxsize=1)
def get_langchain_database():
    """LangChain SQLDatabase wrapper, used for schema inspection and SQL execution."""
    from langchain_community.utilities import SQLDatabase

    return SQLDatabase(
        engine=get_readonly_engine(),
        include_tables=list(TABLE_NAMES),
        sample_rows_in_table_info=3,
        view_support=False,
    )


def get_schema_description() -> str:
    """Return a compact schema with types, samples, and relationships."""
    db = get_langchain_database()
    schema = db.get_table_info()
    return (
        f"{schema}\n\n"
        "Relationships:\n"
        "- orders.product_id -> products.product_id (many orders per product)\n"
        "- orders.customer_id -> customers.customer_id (many orders per customer)\n\n"
        "Business definitions:\n"
        "- discount is stored as a decimal fraction (0.10 means 10%).\n"
        "- Booked revenue excludes status IN ('Cancelled', 'Returned').\n"
        "- product cost_price is per unit and excludes freight; shipping cost is "
        "not stored anywhere in the warehouse.\n"
        "- Rows with NULL unit_price or NULL region cannot contribute to the "
        "corresponding revenue/region aggregation.\n"
        "- Orders span 2026-01-01 through 2026-09-30. 2026-Q3 is "
        "2026-07-01 through 2026-09-30.\n"
    )


def execute_readonly_query(sql: str, max_rows: int = MAX_SQL_ROWS) -> SQLResult:
    """Validate and execute one read-only SQL query with a hard row cap.

    Execution goes through LangChain's ``SQLDatabase`` (using its structured
    ``_execute`` path because the public ``run`` method stringifies rows). The
    query is wrapped in an outer LIMIT so the cap is enforced by SQLite itself,
    and the underlying engine is still opened with ``mode=ro``.
    """
    safe_sql = validate_sql(sql)
    database = get_langchain_database()
    wanted = max(1, max_rows)
    # A newline before the closing parenthesis also neutralizes a trailing
    # line comment in the user's query.
    limited_sql = (
        f"SELECT * FROM (\n{safe_sql}\n) AS _bi_copilot_result LIMIT {wanted + 1}"
    )
    records = database._execute(limited_sql, fetch="all")
    raw_rows = list(records or [])
    truncated = len(raw_rows) > wanted
    rows = [dict(row) for row in raw_rows[:wanted]]
    columns = list(rows[0].keys()) if rows else []

    return SQLResult(
        columns=columns,
        rows=rows,
        sql=safe_sql,
        row_count=len(rows),
        truncated=truncated,
    )


def format_sql_value(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if value.is_integer():
            return f"{int(value):,}"
        return f"{value:,.2f}"
    return str(value)


def rows_to_markdown(
    columns: list[str], rows: list[dict[str, Any]], truncated: bool = False
) -> str:
    """Render query rows as a compact, safe Markdown table."""
    if not columns:
        return "_No columns returned._"
    if not rows:
        return "_No rows matched the query._"

    def clean(value: Any) -> str:
        text_value = format_sql_value(value).replace("|", "\\|").replace("\n", "<br>")
        return text_value

    header = "| " + " | ".join(column.replace("|", "\\|") for column in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, divider]
    for row in rows:
        lines.append("| " + " | ".join(clean(row.get(column)) for column in columns) + " |")
    if truncated:
        lines.append(f"\n_Showing the first {len(rows)} rows (result truncated)._")
    return "\n".join(lines)


def preview_table(table: str, limit: int = 50):
    """Read a small preview from one of the known warehouse tables."""
    if table not in TABLE_NAMES:
        raise SQLValidationError(f"Unknown table: {table}")
    return execute_readonly_query(
        f'SELECT * FROM "{table}" LIMIT {max(1, min(int(limit), 500))}',
        max_rows=limit,
    )
