"""LangGraph agent that routes business questions to SQL and documents.

The graph is a small supervisor workflow:

    START -> supervisor -> (SQL | documents | both) -> synthesize -> END

`ask(question)` returns the cited final answer. `answer_question(question)`
returns the answer plus structured artifacts for the Streamlit chart layer.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from src import audit
from src.build_rag import get_vectorstore
from src.config import (
    LATEST_QUARTER,
    LATEST_QUARTER_END,
    LATEST_QUARTER_START,
    llm_model,
    offline_mode,
    openai_api_key,
)
from src.metrics import (
    BOOKED_FILTER,
    COGS_LINE,
    LATEST_QUARTER_FILTER,
    NET_LINE,
    PRIOR_QUARTER_END,
    PRIOR_QUARTER_FILTER,
    PRIOR_QUARTER_START,
    metric_glossary,
    select_metrics,
)
from src.sql_guard import (
    SQLResult,
    execute_readonly_query,
    get_schema_description,
    validate_sql,
)

REGIONS = ("Northeast", "Southeast", "Midwest", "Southwest", "West")

AGENT_INSTRUCTIONS = """You are BI Copilot, a careful senior business analyst for Summit Outdoor Co.
Choose the smallest set of sources that can answer the question:
- If the question is about numbers, trends, revenue, orders, products, regions, or
  customers, use the sql_query specialist.
- If the question is about policy, rules, qualitative summaries, product
  discontinuations, or launches, use the document_search specialist.
- If both are needed, use both and combine the evidence.
- Never invent a number or a policy. If the warehouse cannot answer a qualitative
  cause (for example, shipping invoices are not in the warehouse), say so.
- Always make sure the final answer presented to the user ends with a "Sources:"
  section citing either "SQL query result" or the exact document filename(s), such
  as q3_summary.txt.
- Cite document evidence inline with the filename in square brackets.
"""


class AgentState(TypedDict, total=False):
    question: str
    route: Literal["sql", "docs", "both", "none"]
    route_reason: str
    sql: str
    sql_columns: list[str]
    sql_rows: list[dict[str, Any]]
    sql_markdown: str
    sql_error: str
    sql_executed: bool
    documents: list[dict[str, Any]]
    doc_error: str
    answer: str
    mode: str
    degraded: bool


class RouteDecision(BaseModel):
    route: Literal["sql", "docs", "both", "none"] = Field(
        description="The specialist(s) needed: sql, docs, both, or none."
    )
    rationale: str = Field(description="One short sentence explaining the route.")


class SQLPlan(BaseModel):
    sql: str = Field(description="A single read-only SQLite SELECT statement.")
    explanation: str = Field(default="", description="Short explanation of the query.")


@dataclass
class AgentResult:
    answer: str
    route: str
    sql: str | None = None
    columns: list[str] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)
    documents: list[dict[str, Any]] = field(default_factory=list)
    mode: str = "offline-demo"
    error: str | None = None
    latency_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "route": self.route,
            "sql": self.sql,
            "columns": self.columns,
            "rows": self.rows,
            "documents": self.documents,
            "mode": self.mode,
            "error": self.error,
            "latency_ms": self.latency_ms,
        }


@lru_cache(maxsize=1)
def _chat_model():
    if not openai_api_key():
        raise RuntimeError("OPENAI_API_KEY is not configured.")
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=llm_model(),
        temperature=0,
        api_key=openai_api_key(),
        timeout=60,
        max_retries=2,
        max_tokens=1000,
    )


def _tokenize(text: str) -> set[str]:
    stop = {
        "the",
        "and",
        "for",
        "was",
        "were",
        "what",
        "which",
        "who",
        "did",
        "does",
        "with",
        "from",
        "that",
        "this",
        "are",
        "our",
        "you",
        "how",
        "why",
        "into",
        "over",
        "under",
        "than",
        "last",
        "next",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9$%]+", text.lower())
        if len(token) > 1 and token not in stop
    }


def heuristic_route(question: str) -> tuple[str, str]:
    """Deterministic fallback router used offline or if the LLM call fails."""
    q = question.lower().strip()
    if re.match(r"^(hi|hello|hey|thanks|thank you|good (morning|afternoon|evening))\b", q):
        return "none", "Greeting or meta question."

    analytics = bool(
        re.search(
            r"\b(revenue|sales|profit|margin|margins|units?|quantity|aov|average|"
            r"trend|growth|top|bottom|rank|total|count|how many|highest|lowest|"
            r"compare|quarter|q[1-4]|region|regions|customers?|products?|orders?)\b",
            q,
        )
    )
    document = bool(
        re.search(
            r"\b(policy|policies|discount|approval|return|returns|rule|rules|"
            r"guideline|guidelines|summary|why|reason|explain|phase[ds]? out|"
            r"discontinu\w*|launch\w*|notes?|documents?|shipping|qualitative|"
            r"strategy|process)\b",
            q,
        )
    )
    strong_document = bool(
        re.search(
            r"\b(policy|policies|approval|return|rule|guideline|summary|"
            r"phase[ds]? out|discontinu\w*|launch\w*|notes?|why|reason)\b",
            q,
        )
    )
    strong_analytics = bool(
        re.search(
            r"\b(revenue|sales|profit|margin|margins|units?|quantity|aov|"
            r"average|trend|growth|top|bottom|rank|total|count|how many|"
            r"highest|lowest|compare)\b",
            q,
        )
    )

    if strong_document and not strong_analytics:
        return "docs", "The question asks for policy or qualitative guidance."
    if document and analytics:
        return "both", "The question needs policy context and warehouse evidence."
    if analytics:
        return "sql", "The question asks for a metric, trend, or segment comparison."
    if document:
        return "docs", "The question is qualitative or document-oriented."
    return "none", "No structured-data or document intent was detected."


def _supervisor_node(state: AgentState) -> dict[str, Any]:
    question = state["question"]
    if offline_mode():
        route, reason = heuristic_route(question)
        return {"route": route, "route_reason": reason}

    prompt = (
        "You are acting as the supervisor. Choose exactly one route for the "
        "business question.\n\n"
        "Routes:\n"
        "- sql: numbers, revenue, orders, units, trends, rankings, regions, "
        "customers, product performance, comparisons.\n"
        "- docs: policy, discount rules, returns, qualitative summaries, reasons, "
        "product discontinuations, launches, and other document content.\n"
        "- both: the answer needs warehouse numbers plus document context.\n"
        "- none: greeting, thanks, or unrelated meta question.\n\n"
        "Important: 'What is the discount policy for orders above $1,000?' is docs, "
        "because it asks for a written policy rather than an order metric. "
        "'Why did margins drop, and which region was affected?' is both.\n"
    )
    try:
        decision = _chat_model().with_structured_output(RouteDecision).invoke(
            [
                SystemMessage(content=f"{AGENT_INSTRUCTIONS}\n\n{prompt}"),
                HumanMessage(content=question),
            ]
        )
        return {"route": decision.route, "route_reason": decision.rationale}
    except Exception:
        route, reason = heuristic_route(question)
        return {
            "route": route,
            "route_reason": reason,
            "degraded": True,
        }


def _region_from_question(question: str) -> str | None:
    q = question.lower()
    for region in REGIONS:
        if region.lower() in q:
            return region
    return None


def _heuristic_sql_plan(question: str) -> SQLPlan:
    """Build a conservative SQL plan for common questions in offline mode."""
    q = question.lower()
    region = _region_from_question(question)
    region_clause = f" AND o.region = '{region}'" if region else ""
    top_match = re.search(r"\btop\s+(\d+)\b", q)
    limit = min(int(top_match.group(1)), 25) if top_match else 5

    common_filter = f"{LATEST_QUARTER_FILTER} AND {BOOKED_FILTER}{region_clause}"

    if re.search(r"\bmargin|profit", q) and re.search(r"\bregion|affected|which", q):
        # Compare product-level gross margin in the latest quarter with the prior
        # quarter, per region. Freight is not in the warehouse, so this tells us
        # whether the decline is visible in product economics at all.
        cur = LATEST_QUARTER_FILTER
        prior = PRIOR_QUARTER_FILTER
        sql = f"""
WITH by_region AS (
  SELECT
    o.region,
    SUM(CASE WHEN {cur} THEN {NET_LINE} END) AS rev_cur,
    SUM(CASE WHEN {cur} THEN {NET_LINE} - {COGS_LINE} END) AS gp_cur,
    SUM(CASE WHEN {prior} THEN {NET_LINE} END) AS rev_prior,
    SUM(CASE WHEN {prior} THEN {NET_LINE} - {COGS_LINE} END) AS gp_prior
  FROM orders o
  JOIN products p ON p.product_id = o.product_id
  WHERE {BOOKED_FILTER}
    AND o.region IS NOT NULL
    AND o.order_date BETWEEN '{PRIOR_QUARTER_START}' AND '{LATEST_QUARTER_END}'
  GROUP BY o.region
)
SELECT
  region,
  ROUND(rev_cur, 2) AS revenue,
  ROUND(100.0 * gp_cur / NULLIF(rev_cur, 0), 1) AS gross_margin_pct,
  ROUND(100.0 * gp_prior / NULLIF(rev_prior, 0), 1) AS prior_gross_margin_pct,
  ROUND(100.0 * gp_cur / NULLIF(rev_cur, 0) - 100.0 * gp_prior / NULLIF(rev_prior, 0), 1)
    AS margin_change_pts
FROM by_region
ORDER BY revenue DESC
"""
        return SQLPlan(
            sql=sql.strip(),
            explanation="Q3 vs Q2 product gross margin by region.",
        )

    if "category" in q or "categories" in q:
        sql = f"""
SELECT p.category,
  {select_metrics("revenue", "gross_profit", "gross_margin_pct")}
FROM orders o
JOIN products p ON p.product_id = o.product_id
WHERE {common_filter}
GROUP BY p.category
ORDER BY revenue DESC
"""
        return SQLPlan(sql=sql.strip(), explanation="Revenue and margin by category.")

    if "product" in q and re.search(r"\btop\b|\bbest\b|\bhighest\b", q):
        metric = "units" if "unit" in q and "revenue" not in q else "revenue"
        sql = f"""
SELECT p.product_name,
  {select_metrics(metric)}
FROM orders o
JOIN products p ON p.product_id = o.product_id
WHERE {common_filter}
GROUP BY p.product_id, p.product_name
ORDER BY {metric} DESC
LIMIT {limit}
"""
        return SQLPlan(sql=sql.strip(), explanation=f"Top {limit} products in scope.")

    if "customer" in q and re.search(r"\btop\b|\bbest\b|\bhighest\b", q):
        sql = f"""
SELECT c.customer_name,
  {select_metrics("revenue", "orders")}
FROM orders o
JOIN customers c ON c.customer_id = o.customer_id
WHERE {common_filter}
GROUP BY c.customer_id, c.customer_name
ORDER BY revenue DESC
LIMIT {limit}
"""
        return SQLPlan(sql=sql.strip(), explanation=f"Top {limit} customers in scope.")

    if "month" in q or "trend" in q:
        # Trends need history, so look at the full dataset rather than one quarter.
        sql = f"""
SELECT substr(o.order_date, 1, 7) AS month,
  {select_metrics("revenue", "orders")}
FROM orders o
WHERE {BOOKED_FILTER}{region_clause}
GROUP BY month
ORDER BY month
"""
        return SQLPlan(sql=sql.strip(), explanation="Monthly revenue trend.")

    if ("region" in q or "regional" in q) and re.search(r"revenue|sales", q):
        sql = f"""
SELECT o.region,
  {select_metrics("revenue", "orders")}
FROM orders o
WHERE {common_filter}
  AND o.region IS NOT NULL
GROUP BY o.region
ORDER BY revenue DESC
"""
        return SQLPlan(sql=sql.strip(), explanation="Revenue by region.")

    sql = f"""
SELECT
  {select_metrics("orders", "revenue", "avg_order_value")}
FROM orders o
WHERE {common_filter}
"""
    return SQLPlan(sql=sql.strip(), explanation="Summary KPIs for the latest quarter.")


def _generate_sql_online(question: str, feedback: str | None = None) -> SQLPlan:
    schema = get_schema_description()
    prompt = (
        "You are a senior analytics engineer writing SQLite SQL. Write exactly one "
        "read-only SELECT query that answers the business question.\n\n"
        "Rules:\n"
        "- Never use DDL or DML. No INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, "
        "PRAGMA, or multiple statements.\n"
        "- Use the schema exactly. Join products on product_id and customers on "
        "customer_id when needed.\n"
        "- Use these governed metric definitions exactly:\n"
        f"{metric_glossary()}\n"
        "- Shipping cost is not in the warehouse. For a margin question, compute "
        "product gross_margin_pct (join products) and compare it with the prior "
        f"quarter ({PRIOR_QUARTER_START} through {PRIOR_QUARTER_END}); never claim "
        "product cost data explains freight.\n"
        f"- The latest complete quarter in the data is {LATEST_QUARTER} "
        f"({LATEST_QUARTER_START} through {LATEST_QUARTER_END}). Treat "
        "'last quarter' as that period.\n"
        "- Use NULLIF for division to avoid divide-by-zero.\n"
        "- Round currency to 2 decimals and percentages to 1 decimal.\n"
        "- Return only the query, no markdown fences.\n\n"
        f"Schema:\n{schema}\n"
    )
    if feedback:
        prompt += f"\nThe previous attempt failed with this error; fix it:\n{feedback}\n"
    return _chat_model().with_structured_output(SQLPlan).invoke(
        [
            SystemMessage(content=prompt),
            HumanMessage(content=f"Business question: {question}"),
        ]
    )


def _sql_query_impl(question: str) -> dict[str, Any]:
    """Internal SQL specialist returning structured rows for the state."""
    degraded = False
    if offline_mode():
        plan = _heuristic_sql_plan(question)
    else:
        try:
            plan = _generate_sql_online(question)
        except Exception:
            plan = _heuristic_sql_plan(question)
            degraded = True

    last_error: Exception | None = None
    for _ in range(2):
        try:
            safe_sql = validate_sql(plan.sql)
            result: SQLResult = execute_readonly_query(safe_sql, max_rows=200)
            return {
                "sql": result.sql,
                "sql_columns": result.columns,
                "sql_rows": result.rows,
                "sql_markdown": result.to_markdown(max_rows=20),
                "sql_executed": True,
                "sql_error": "",
                "degraded": degraded,
            }
        except Exception as exc:
            last_error = exc
            if offline_mode():
                break
            try:
                plan = _generate_sql_online(question, feedback=str(exc))
            except Exception:
                break

    return {
        "sql": getattr(plan, "sql", ""),
        "sql_columns": [],
        "sql_rows": [],
        "sql_markdown": "",
        "sql_executed": False,
        "sql_error": str(last_error or "SQL execution failed."),
        "degraded": degraded,
    }


def sql_query(question: str) -> str:
    """Tool 1: turn a business question into governed SQL and a Markdown table."""
    result = _sql_query_impl(question)
    if result["sql_error"]:
        raise RuntimeError(result["sql_error"])
    return result["sql_markdown"]


def _stems(text: str) -> set[str]:
    """Crude 6-character stems so 'discontinued' matches 'discontinuations'."""
    return {token[:6] for token in _tokenize(text) if len(token) > 2}


def _hybrid_rerank(question: str, matches: list, k: int) -> list:
    """Re-rank vector candidates by semantic similarity plus keyword overlap.

    Pure vector search misses exact business terms (policy names, product IDs)
    often enough to matter; a lexical boost on a wider candidate pool fixes that
    without a second index.
    """
    query = _stems(question)
    if not query:
        return matches[:k]

    def score(item) -> float:
        document, distance = item
        similarity = 1.0 - float(distance or 0.0)
        overlap = len(query & _stems(document.page_content)) / len(query)
        return 0.5 * similarity + 0.5 * overlap

    return sorted(matches, key=score, reverse=True)[:k]


def _document_search_impl(question: str, k: int = 4) -> list[dict[str, Any]]:
    vectorstore = get_vectorstore()
    candidates = vectorstore.similarity_search_with_score(question, k=max(12, 3 * k))
    matches = _hybrid_rerank(question, candidates, k)
    hits: list[dict[str, Any]] = []
    for document, distance in matches:
        source = document.metadata.get("source", "unknown")
        hits.append(
            {
                "source": source,
                "content": document.page_content,
                "distance": float(distance) if distance is not None else None,
                "metadata": {
                    key: value
                    for key, value in document.metadata.items()
                    if isinstance(value, (str, int, float, bool))
                },
            }
        )
    return hits


def document_search(question: str, k: int = 4) -> str:
    """Tool 2: retrieve the most relevant document chunks with filenames."""
    hits = _document_search_impl(question, k=k)
    if not hits:
        return "_No matching document content found._"
    blocks = []
    for hit in hits:
        blocks.append(f"[{hit['source']}]\n{hit['content'].strip()}")
    return "\n\n---\n\n".join(blocks)


def _sql_specialist_node(state: AgentState) -> dict[str, Any]:
    return _sql_query_impl(state["question"])


def _document_specialist_node(state: AgentState) -> dict[str, Any]:
    try:
        hits = _document_search_impl(state["question"], k=4)
        if not hits:
            return {"documents": [], "doc_error": "No matching document content found."}
        return {"documents": hits, "doc_error": ""}
    except Exception as exc:
        return {"documents": [], "doc_error": str(exc)}


def _source_list(state: AgentState, cited_documents: list[str] | None = None) -> list[str]:
    sources: list[str] = []
    if state.get("sql_executed"):
        sources.append("SQL query result")
    for hit in state.get("documents", []):
        source = hit.get("source")
        if not source or source in sources:
            continue
        if cited_documents is not None and source not in cited_documents:
            continue
        sources.append(source)
    return sources


def _sources_section(
    state: AgentState, cited_documents: list[str] | None = None
) -> str:
    sources = _source_list(state, cited_documents=cited_documents)
    if not sources:
        sources = ["No source available"]
    return "\n\n**Sources:**\n" + "\n".join(f"- {source}" for source in sources)


def _format_number(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and not value.is_integer():
            return f"{value:,.2f}"
        return f"{int(value):,}"
    return str(value)


def _sql_takeaway(columns: list[str], rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "The query returned no rows."
    label_column = next(
        (
            column
            for column in columns
            if not all(isinstance(row.get(column), (int, float)) for row in rows)
        ),
        columns[0],
    )
    numeric_columns = [
        column
        for column in columns
        if column != label_column
        and all(isinstance(row.get(column), (int, float)) for row in rows)
    ]
    if not numeric_columns:
        return f"The query returned {len(rows)} rows."

    margin_col = next((column for column in columns if "margin" in column.lower()), None)
    revenue_col = next((column for column in columns if "revenue" in column.lower()), None)
    if margin_col and revenue_col and len(rows) > 1:
        low_margin = min(rows, key=lambda row: row.get(margin_col) or float("inf"))
        high_revenue = max(rows, key=lambda row: row.get(revenue_col) or float("-inf"))
        return (
            f"{low_margin.get(label_column)} has the lowest gross margin "
            f"({_format_number(low_margin.get(margin_col))}%), while "
            f"{high_revenue.get(label_column)} carries the most revenue "
            f"(${_format_number(high_revenue.get(revenue_col))})."
        )

    metric = numeric_columns[0]
    for candidate in ("revenue", "units", "gross_profit", "gross_margin_pct"):
        if candidate in numeric_columns:
            metric = candidate
            break
    leader = max(rows, key=lambda row: row.get(metric) or float("-inf"))
    if len(rows) == 1:
        return f"{label_column} for this query is {_format_number(leader.get(metric))}."
    if "revenue" in metric.lower() or "sales" in metric.lower():
        metric_label = "revenue"
        value = f"${_format_number(leader.get(metric))}"
    elif "percent" in metric.lower() or "margin" in metric.lower():
        metric_label = metric.replace("_", " ")
        value = f"{_format_number(leader.get(metric))}%"
    else:
        metric_label = metric.replace("_", " ")
        value = _format_number(leader.get(metric))
    return f"{leader.get(label_column)} leads on {metric_label} with {value}."


def _clean_sentence(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _document_evidence(
    hits: list[dict[str, Any]], question: str, max_blocks: int = 1
) -> tuple[str, list[str]]:
    """Select the most query-relevant document paragraphs for a grounded answer."""
    query_tokens = _tokenize(question)
    candidates: list[tuple[float, str, str]] = []
    query_stems = _stems(question)
    for rank, hit in enumerate(hits):
        source = hit["source"]
        content = hit["content"]
        for block in re.split(r"\n\s*\n", content):
            block = block.strip()
            if len(block) < 35:
                continue
            block_lower = block.lower()
            block_tokens = _tokenize(block)
            overlap = len(query_tokens & block_tokens)
            # Stem overlap catches inflections; retrieval rank breaks ties.
            score = overlap * 2.0 + len(query_stems & _stems(block)) + (len(hits) - rank) * 0.5
            for phrase in (
                "$1,000",
                "discount",
                "approval",
                "shipping",
                "margin",
                "most affected",
                "northeast",
                "phase",
            ):
                if phrase in block_lower and phrase in question.lower():
                    score += 4.0
            if "margin" in question.lower() and "margin" in block_lower:
                score += 3.0
            if "discount" in question.lower() and "discount" in block_lower:
                score += 3.0
            if "1000" in question.lower() and ("$1,000" in block_lower or "1,000" in block_lower):
                score += 8.0
            candidates.append((score, source, block))

    candidates.sort(key=lambda item: item[0], reverse=True)
    chosen: list[tuple[str, str]] = []
    used_sources: list[str] = []
    for _, source, block in candidates:
        if len(chosen) >= max_blocks:
            break
        if any(block == existing[1] for existing in chosen):
            continue
        chosen.append((source, block))
        if source not in used_sources:
            used_sources.append(source)

    rendered = []
    for source, block in chosen:
        if len(block) > 1300:
            block = block[:1300].rsplit(" ", 1)[0] + " …"
        quote = "\n".join(f"> {line}" for line in block.splitlines())
        rendered.append(f"**{source}**\n\n{quote}")
    return "\n\n".join(rendered), used_sources


def _margin_reconciliation(rows: list[dict[str, Any]]) -> str:
    """Compare document claims against product-level margin, without assuming agreement."""
    valid = [
        row
        for row in rows
        if isinstance(row.get("margin_change_pts"), (int, float))
        and isinstance(row.get("revenue"), (int, float))
    ]
    if not valid:
        return "The warehouse query returned no comparable regional margins."

    total_rev = sum(row["revenue"] for row in valid)
    # Revenue-weighted margin change across regions.
    weighted = sum(row["margin_change_pts"] * row["revenue"] for row in valid) / total_rev
    worst = min(valid, key=lambda row: row["margin_change_pts"])
    biggest = max(valid, key=lambda row: row["revenue"])

    direction = "fell" if weighted < 0 else "rose"
    text = (
        f"Product-level gross margin (revenue minus product cost) {direction} "
        f"{abs(weighted):.1f} pts quarter-over-quarter, revenue-weighted. "
        f"{biggest['region']} carries the most Q3 revenue (${_format_number(biggest['revenue'])}); "
        f"{worst['region']} had the weakest margin change ({worst['margin_change_pts']:+.1f} pts)."
    )
    if weighted >= 0:
        text += (
            " So the decline is **not** visible in product economics: if margins "
            "dropped, the cause sits in costs the warehouse does not hold (for example, "
            "freight invoices). Treat the documented figure as unverified until "
            "shipping cost data is loaded."
        )
    else:
        text += (
            " The warehouse confirms a product-level decline; freight is not stored, "
            "so the shipping share of it cannot be quantified here."
        )
    return text


def _combined_lead(
    question: str,
    hits: list[dict[str, Any]],
    columns: list[str],
    rows: list[dict[str, Any]],
) -> str:
    """Compose a short, grounded lead when both specialists contribute."""
    q = question.lower()
    evidence, _ = _document_evidence(hits, question, max_blocks=1)
    evidence_plain = re.sub(r"^>\s*", "", evidence, flags=re.MULTILINE)
    evidence_plain = re.sub(r"\*\*.*?\*\*", "", evidence_plain).strip()
    evidence_plain = _clean_sentence(evidence_plain)
    if len(evidence_plain) > 420:
        evidence_plain = evidence_plain[:420].rsplit(" ", 1)[0] + " …"
    sql_takeaway = _sql_takeaway(columns, rows)

    if ("margin" in q or "profit" in q) and "margin_change_pts" in columns:
        check = _margin_reconciliation(rows)
        if evidence_plain:
            return f"**What the documents say:** {evidence_plain}\n\n**What the warehouse shows:** {check}"
        return f"**What the warehouse shows:** {check}"
    if evidence_plain:
        return f"{evidence_plain}\n\nFrom the warehouse: {sql_takeaway}"
    return f"Combining the available sources: {sql_takeaway}"


def _offline_answer(state: AgentState) -> str:
    question = state["question"]
    route = state.get("route", "none")
    sql_error = state.get("sql_error")
    doc_error = state.get("doc_error")
    columns = state.get("sql_columns", [])
    rows = state.get("sql_rows", [])
    hits = state.get("documents", [])
    parts: list[str] = []

    if route == "none" and not sql_error and not doc_error:
        return (
            "I can answer questions about sales performance, products, customers, "
            "regions, discounts, returns, and the Q3 business summary. Try asking "
            "for a ranking, a trend, or a policy."
        )

    if route == "both" and hits and not sql_error:
        parts.append("**Answer**\n")
        parts.append(_combined_lead(question, hits, columns, rows))
        parts.append("\n\n**Warehouse result**\n")
        parts.append(state.get("sql_markdown", ""))
        evidence, _ = _document_evidence(hits, question)
        if evidence:
            parts.append("\n\n**Document evidence**\n")
            parts.append(evidence)
    elif route == "sql" or (not hits and rows):
        parts.append("**Answer**\n")
        if rows:
            parts.append(state.get("sql_markdown", ""))
            parts.append("\n\n**Takeaway:** " + _sql_takeaway(columns, rows))
        else:
            parts.append(
                "I could not complete the warehouse query. "
                + (f"Details: {sql_error}" if sql_error else "")
            )
    else:
        parts.append("**Answer**\n")
        if hits:
            evidence, _ = _document_evidence(hits, question, max_blocks=2)
            parts.append(evidence or "I found relevant documents but could not extract a passage.")
            parts.append(
                "\n\nThe policy and lifecycle documents are the authoritative source "
                "for this question."
            )
        elif doc_error:
            parts.append(f"I could not search the documents: {doc_error}")
        elif sql_error:
            parts.append(f"I could not run the query: {sql_error}")
        else:
            parts.append("I could not find a grounded answer in the available sources.")

    if route == "both" and sql_error and hits:
        parts.append(f"\n\n_The warehouse query failed: {sql_error}_")
    if route == "both" and doc_error:
        parts.append(f"\n\n_The document search failed: {doc_error}_")
    return "".join(parts)


def _online_answer(state: AgentState) -> tuple[str, bool]:
    sql_table = state.get("sql_markdown", "")
    hits = state.get("documents", [])
    doc_context = "\n\n".join(
        f"[{hit['source']}]\n{hit['content']}" for hit in hits
    )
    prompt = (
        "Answer the user's question using only the evidence below. Do not invent "
        "numbers or policies.\n\n"
        f"Question: {state['question']}\n"
        f"Route: {state.get('route')}\n\n"
        f"SQL result table:\n{sql_table or 'No SQL result.'}\n\n"
        f"SQL error:\n{state.get('sql_error') or 'None.'}\n\n"
        f"Business document excerpts:\n{doc_context or 'No document excerpts.'}\n\n"
        "Requirements:\n"
        "- Lead with the direct answer in 1–3 sentences, then add supporting detail.\n"
        "- If SQL returned rows, include a compact Markdown table of the key results.\n"
        "- Cite document evidence inline with the filename in square brackets, e.g. "
        "[q3_summary.txt].\n"
        "- Mention the limitation when shipping costs or qualitative causes are not "
        "present in the warehouse.\n"
        "- You may include a Sources section; the application canonicalizes it.\n"
        "- Use clear Markdown with short headings. No code fences."
    )
    try:
        response = _chat_model().invoke(
            [
                SystemMessage(content=AGENT_INSTRUCTIONS),
                HumanMessage(content=prompt),
            ]
        )
        body = str(response.content).strip()
        body = re.split(
            r"\n\s*(?:\*\*)?Sources:(?:\*\*)?",
            body,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].rstrip()
        if body:
            return body, True
    except Exception:
        pass
    return _offline_answer(state), False


def _synthesize_node(state: AgentState) -> dict[str, Any]:
    online_completed = False
    if state.get("route") == "none":
        body = (
            "I can answer questions about sales performance, products, customers, "
            "regions, discounts, returns, and the Q3 business summary. Try asking "
            "for a ranking, a trend, or a policy."
        )
    elif offline_mode() or state.get("degraded"):
        body = _offline_answer(state)
    else:
        body, online_completed = _online_answer(state)

    available_sources = [
        str(hit.get("source")) for hit in state.get("documents", []) if hit.get("source")
    ]
    cited_sources = [source for source in available_sources if source in body]
    if not cited_sources:
        cited_sources = available_sources
    answer = body.rstrip() + _sources_section(state, cited_documents=cited_sources)
    if state.get("degraded") or (not offline_mode() and not online_completed and state.get("route") != "none"):
        mode = "offline-fallback"
    elif offline_mode():
        mode = "offline-demo"
    else:
        mode = "openai"
    return {"answer": answer, "mode": mode}


def _route_after_sql(state: AgentState) -> str:
    return "documents" if state.get("route") == "both" else "synthesize"


@lru_cache(maxsize=1)
def _build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("supervisor", _supervisor_node)
    builder.add_node("sql", _sql_specialist_node)
    builder.add_node("documents", _document_specialist_node)
    builder.add_node("synthesize", _synthesize_node)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        lambda state: state.get("route", "none"),
        {
            "sql": "sql",
            "docs": "documents",
            "both": "sql",
            "none": "synthesize",
        },
    )
    builder.add_conditional_edges(
        "sql",
        _route_after_sql,
        {"documents": "documents", "synthesize": "synthesize"},
    )
    builder.add_edge("documents", "synthesize")
    builder.add_edge("synthesize", END)
    return builder.compile()


def answer_question(question: str) -> AgentResult:
    """Run the full agent and return the answer plus chart-ready artifacts."""
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    state: AgentState = {"question": question.strip(), "degraded": False}
    started = time.perf_counter()
    final = _build_graph().invoke(state)
    latency_ms = round(1000 * (time.perf_counter() - started))
    audit.record(
        {
            "question": state["question"],
            "route": final.get("route"),
            "route_reason": final.get("route_reason"),
            "mode": final.get("mode"),
            "sql": final.get("sql"),
            "sql_rows": len(final.get("sql_rows", [])),
            "documents": sorted({hit.get("source") for hit in final.get("documents", [])}),
            "error": final.get("sql_error") or final.get("doc_error") or None,
            "latency_ms": latency_ms,
        }
    )
    return AgentResult(
        answer=final.get("answer", ""),
        route=final.get("route", "none"),
        sql=final.get("sql"),
        columns=final.get("sql_columns", []),
        rows=final.get("sql_rows", []),
        documents=final.get("documents", []),
        mode=final.get("mode", "offline-demo"),
        error=final.get("sql_error") or final.get("doc_error") or None,
        latency_ms=latency_ms,
    )


def ask(question: str) -> str:
    """Public entry point required by the app: return a cited final answer."""
    return answer_question(question).answer
