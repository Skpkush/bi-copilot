"""Streamlit front end for BI Copilot."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(
    page_title="AI-Powered BI Copilot",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.config import DB_PATH, MAX_PREVIEW_ROWS, offline_mode  # noqa: E402
from src.sql_guard import (  # noqa: E402
    SQLValidationError,
    execute_readonly_query,
    validate_sql,
)

APP_CSS = """
<style>
    .stApp { background: #F8FAFC; }
    [data-testid="stHeader"], [data-testid="stToolbar"] { display: none !important; }
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1240px; }
    h1, h2, h3 { letter-spacing: -0.02em; color: #0F172A; }
    h1 { font-size: 2.05rem !important; margin-bottom: 0.15rem; }
    [data-testid="stSidebar"] { background: #0F172A; }
    [data-testid="stSidebar"] * { color: #E2E8F0; }
    [data-testid="stSidebar"] .stButton button {
        background: #0F766E; color: white; border: 0; font-weight: 600;
    }
    [data-testid="stSidebar"] .stButton button:hover { background: #115E59; color: white; }
    [data-testid="stSidebar"] [data-testid="stExpander"] {
        border: 1px solid #334155; background: #111C30; border-radius: 10px;
    }
    [data-testid="stChatMessage"] {
        background: #FFFFFF; border: 1px solid #E2E8F0;
        border-radius: 12px; padding: 0.55rem 0.8rem;
    }
    [data-testid="stChatMessage"] p { line-height: 1.55; }
    .hero-caption { color: #475569; font-size: 1.02rem; margin-bottom: 1.1rem; }
    .pill {
        display: inline-block; background: #CCFBF1; color: #115E59;
        border-radius: 999px; padding: 0.2rem 0.65rem; font-size: 0.76rem;
        font-weight: 700; letter-spacing: 0.02em; margin-right: 0.35rem;
    }
    .section-intro { color: #475569; margin-bottom: 1rem; }
    div[data-testid="stDataFrame"] { border: 1px solid #E2E8F0; border-radius: 10px; }
    .stTabs [data-baseweb="tab-list"] { gap: 0.35rem; border-bottom: 1px solid #E2E8F0; }
    .stTabs [data-baseweb="tab"] { font-weight: 600; }
    [data-testid="stTabPanel"] [data-testid="stElementContainer"]:has([data-testid="stChatInput"]) {
        order: 99;
        margin-top: 0.75rem;
    }
</style>
"""
st.markdown(APP_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_agent_runner():
    """Cache the compiled LangGraph workflow across reruns."""
    from src.agent import answer_question

    return answer_question


@st.cache_resource(show_spinner=False)
def load_database_engine():
    """Cache the read-only SQLAlchemy connection pool."""
    from src.sql_guard import get_readonly_engine

    return get_readonly_engine()


@st.cache_data(ttl=60, show_spinner=False)
def load_table_preview(table: str) -> pd.DataFrame:
    engine = load_database_engine()
    query = f'SELECT * FROM "{table}" LIMIT {MAX_PREVIEW_ROWS}'
    with engine.connect() as connection:
        return pd.read_sql_query(text(query), connection)


@st.cache_data(ttl=60, show_spinner=False)
def load_table_counts() -> dict[str, int]:
    engine = load_database_engine()
    with engine.connect() as connection:
        rows = connection.execute(
            text(
                "SELECT 'orders' AS table_name, COUNT(*) AS n FROM orders "
                "UNION ALL SELECT 'products', COUNT(*) FROM products "
                "UNION ALL SELECT 'customers', COUNT(*) FROM customers"
            )
        ).all()
    return {str(row[0]): int(row[1]) for row in rows}


@st.cache_data(ttl=30, show_spinner=False)
def get_index_meta() -> dict:
    from src.build_rag import read_index_meta

    return read_index_meta()


def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chart_seq" not in st.session_state:
        st.session_state.chart_seq = 0


def render_chart(chart: dict, key: str) -> None:
    """Render a Plotly bar chart from a stored SQL result."""
    rows = chart.get("rows") or []
    columns = chart.get("columns") or []
    if not rows or not columns:
        return
    try:
        df = pd.DataFrame(rows)
        if df.empty:
            return
        numeric_columns = [
            column
            for column in df.columns
            if pd.api.types.is_numeric_dtype(df[column]) and df[column].notna().any()
        ]
        if not numeric_columns:
            return
        label_columns = [column for column in df.columns if column not in numeric_columns]
        label_column = label_columns[0] if label_columns else "index"
        chart_df = df.copy()
        if label_column == "index":
            chart_df["index"] = [f"Row {index + 1}" for index in range(len(chart_df))]
        chart_df[label_column] = chart_df[label_column].fillna("(blank)").astype(str)
        chart_df = chart_df.head(25)

        preferred = [
            column
            for column in ("revenue", "gross_profit", "units", "gross_margin_pct")
            if column in numeric_columns
        ]
        metric = preferred[0] if preferred else numeric_columns[0]
        chart_df = chart_df.dropna(subset=[metric])

        title = chart.get("title") or (
            f"{metric.replace('_', ' ').title()} by {label_column.replace('_', ' ').title()}"
        )
        fig = px.bar(
            chart_df,
            x=metric,
            y=label_column,
            orientation="h",
            title=title,
            color_discrete_sequence=["#0F766E"],
        )
        fig.update_layout(
            template="plotly_white",
            height=min(500, max(300, 34 * len(chart_df) + 130)),
            margin=dict(l=10, r=18, t=48, b=35),
            xaxis_title=metric.replace("_", " ").title(),
            yaxis_title=None,
            yaxis={"categoryorder": "total ascending"},
            title_font_size=16,
        )
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, width="stretch", key=key)
    except Exception as exc:
        st.caption(f"Chart could not be rendered: {exc}")


def chart_payload(columns: list[str], rows: list[dict], title: str) -> dict:
    return {"columns": columns, "rows": rows[:50], "title": title}


def render_sidebar() -> None:
    index_meta = get_index_meta()
    with st.sidebar:
        st.markdown("## 🧭 AI-Powered BI Copilot")
        st.caption("Natural-language analytics with governed SQL and cited documents.")
        st.divider()

        with st.expander("How to use", expanded=False):
            st.markdown(
                "1. Ask in plain English — no SQL required.\n"
                "2. The supervisor routes to **SQL**, **documents**, or **both**.\n"
                "3. Every answer ends with a **Sources** section.\n\n"
                "**Try:**\n"
                "- Top 3 products by revenue in the Northeast last quarter\n"
                "- What is the discount policy for orders above $1,000?\n"
                "- Why did profit margins drop in Q3?"
            )

        if st.button("Clear conversation", width="stretch"):
            st.session_state.messages = []
            st.session_state.chart_seq = 0
            st.rerun()

        st.divider()
        st.markdown("**Workspace status**")
        try:
            counts = load_table_counts()
            st.success(
                f"SQLite connected · {counts.get('orders', 0):,} orders · "
                f"{counts.get('products', 0)} products"
            )
        except Exception:
            st.error("SQLite database not available. Run the data setup commands.")

        if index_meta.get("backend"):
            backend = str(index_meta["backend"])
            label = "local demo embeddings" if backend.startswith("local") else "OpenAI embeddings"
            st.success(f"RAG index ready · {index_meta.get('chunks', 0)} chunks · {label}")
            if not offline_mode() and backend.startswith("local"):
                st.warning(
                    "The index uses local demo embeddings. Run `python -m src.build_rag` "
                    "to rebuild it with OpenAI embeddings."
                )
            if offline_mode() and backend.startswith("openai:"):
                st.error(
                    "This index needs OpenAI embeddings, but no API key is configured. "
                    "Set OPENAI_API_KEY or rebuild with `python -m src.build_rag --local`."
                )
        else:
            st.warning("RAG index missing. Run `python -m src.build_rag`.")

        if offline_mode():
            st.info(
                "Demo mode: no OpenAI key found. Uses deterministic local embeddings "
                "and an offline answer fallback."
            )
        else:
            st.info("OpenAI mode: GPT-4o-mini with hosted embeddings.")

        st.divider()
        st.caption("Built with LangGraph, SQLite, ChromaDB, and Streamlit.")


def render_chat_tab() -> None:
    st.markdown(
        '<p class="section-intro">Ask a business question. The copilot chooses the '
        "right source and shows its evidence.</p>",
        unsafe_allow_html=True,
    )

    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("meta"):
                st.caption(message["meta"])
            if message.get("chart"):
                render_chart(message["chart"], key=f"history_chart_{idx}")

    prompt = st.chat_input("Ask about revenue, products, regions, or policy…")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Routing question, querying data, and checking documents…"):
            try:
                runner = load_agent_runner()
                result = runner(prompt)
                answer = result.answer
                meta = f"Route: {result.route} · Mode: {result.mode}"
                if result.rows:
                    meta += f" · {len(result.rows)} rows"
                if result.documents:
                    meta += f" · {len(result.documents)} document chunks"
                chart = None
                if result.rows and result.columns:
                    chart = chart_payload(
                        result.columns,
                        result.rows,
                        "",
                    )
            except Exception as exc:
                answer = (
                    "I hit an error while processing that question. Check the sidebar "
                    f"status and try again.\n\n_Details: {exc}_"
                )
                meta = None
                chart = None

        st.markdown(answer)
        if meta:
            st.caption(meta)
        if chart:
            render_chart(chart, key=f"live_chart_{st.session_state.chart_seq}")
            st.session_state.chart_seq += 1

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "meta": meta,
            "chart": chart,
        }
    )


def render_data_explorer_tab() -> None:
    st.markdown(
        '<p class="section-intro">Inspect the cleaned warehouse tables that power '
        "the SQL specialist.</p>",
        unsafe_allow_html=True,
    )
    table = st.selectbox(
        "Choose a table",
        options=["orders", "products", "customers"],
        format_func=lambda value: value.title(),
    )
    descriptions = {
        "orders": "Order lines with product, customer, date, discount, region, and status.",
        "products": "Product master with category, cost, and list price.",
        "customers": "Customer master with signup date and loyalty tier.",
    }
    try:
        df = load_table_preview(table)
        counts = load_table_counts()
        st.caption(
            f"{descriptions[table]} Showing the first {len(df)} of "
            f"{counts.get(table, 0):,} rows."
        )
        st.dataframe(df, width="stretch", hide_index=True)
        st.download_button(
            "Download preview as CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"{table}_preview.csv",
            mime="text/csv",
        )
    except Exception as exc:
        st.error(f"Could not load {table}: {exc}")


def render_sql_runner_tab() -> None:
    st.markdown(
        '<p class="section-intro">Run governed, read-only SQL against the warehouse. '
        "DDL and DML are blocked.</p>",
        unsafe_allow_html=True,
    )
    default_sql = """SELECT
  o.region,
  COUNT(*) AS orders,
  ROUND(SUM(o.quantity * o.unit_price * (1 - o.discount)), 2) AS revenue
FROM orders o
WHERE o.order_date BETWEEN '2026-07-01' AND '2026-09-30'
  AND o.status NOT IN ('Cancelled', 'Returned')
  AND o.region IS NOT NULL
GROUP BY o.region
ORDER BY revenue DESC;"""
    sql = st.text_area("SQL query", value=default_sql, height=230)
    run = st.button("Run query", type="primary")
    if not run:
        return
    try:
        validate_sql(sql)
        result = execute_readonly_query(sql, max_rows=500)
        df = pd.DataFrame(result.rows)
        if df.empty:
            st.info("Query ran successfully but returned no rows.")
            return
        st.caption(f"{result.row_count} rows returned" + (" (truncated)" if result.truncated else ""))
        st.dataframe(df, width="stretch", hide_index=True)
        st.download_button(
            "Download results as CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="query_results.csv",
            mime="text/csv",
        )
    except SQLValidationError as exc:
        st.error(f"Query blocked: {exc}")
    except Exception as exc:
        st.error(f"Query failed: {exc}")


def main() -> None:
    init_state()
    render_sidebar()

    st.markdown('<span class="pill">AGENTIC BI</span>', unsafe_allow_html=True)
    st.title("AI-Powered BI Copilot")
    st.markdown(
        '<p class="hero-caption">Ask questions about sales data and business documents '
        "in plain English. Get one answer, with every source cited.</p>",
        unsafe_allow_html=True,
    )

    if not DB_PATH.exists():
        st.error(
            "The SQLite database is missing. From the project folder, run:\n\n"
            "```bash\npython -m src.generate_data\npython -m src.load_data\n"
            "python -m src.build_rag\n```"
        )
        st.stop()

    chat_tab, explorer_tab, sql_tab = st.tabs(["💬 Chat", "🗂️ Data Explorer", "🛠️ SQL Runner"])
    with chat_tab:
        render_chat_tab()
    with explorer_tab:
        render_data_explorer_tab()
    with sql_tab:
        render_sql_runner_tab()


if __name__ == "__main__":
    main()
