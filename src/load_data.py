"""Clean the raw CSV fixtures and load them into SQLite.

Cleaning rules:
- normalize mixed date formats to YYYY-MM-DD
- trim whitespace from string columns
- replace the impossible negative cost with the median valid cost
- normalize loyalty-tier casing
- keep only the latest record for each duplicated order_id
"""

from __future__ import annotations

import json
from datetime import datetime

import pandas as pd
from sqlalchemy import create_engine, text

from src.config import DATA_DIR, DB_PATH, QUALITY_REPORT_PATH

DATE_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%m/%d/%Y %H:%M:%S",
    "%Y-%m-%d",
    "%m/%d/%Y",
    "%Y/%m/%d",
)


def parse_mixed_datetime(value: object) -> pd.Timestamp:
    """Parse the two (plus a couple of) date formats present in the raw files."""
    if pd.isna(value):
        return pd.NaT
    raw = str(value).strip()
    if not raw:
        return pd.NaT
    for fmt in DATE_FORMATS:
        try:
            return pd.Timestamp(datetime.strptime(raw, fmt))
        except ValueError:
            continue
    return pd.to_datetime(raw, errors="coerce")


def parse_date_column(series: pd.Series) -> pd.Series:
    parsed = series.map(parse_mixed_datetime)
    bad = parsed.isna() & series.notna()
    if bad.any():
        examples = series[bad].head(3).tolist()
        raise ValueError(f"Could not parse {bad.sum()} dates, examples: {examples}")
    return parsed


def strip_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in df.select_dtypes(include=["object", "string"]).columns:
        df[column] = df[column].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )
    return df


def clean_orders(raw: pd.DataFrame) -> pd.DataFrame:
    orders = strip_string_columns(raw)
    orders["_parsed_dt"] = parse_date_column(orders["order_date"])

    before = len(orders)
    # Sort newest last, then retain the final record for each order ID.
    orders = orders.sort_values(["order_id", "_parsed_dt"], kind="stable")
    orders = orders.drop_duplicates(subset=["order_id"], keep="last")
    removed = before - len(orders)
    print(f"  duplicate orders removed: {removed} (kept latest order_date)")

    orders["order_date"] = orders["_parsed_dt"].dt.strftime("%Y-%m-%d")
    orders = orders.drop(columns=["_parsed_dt"])

    orders["quantity"] = pd.to_numeric(orders["quantity"], errors="coerce").astype("Int64")
    for column in ("unit_price", "discount"):
        orders[column] = pd.to_numeric(orders[column], errors="coerce")
    orders["discount"] = orders["discount"].fillna(0.0)
    return orders.reset_index(drop=True)


def clean_products(raw: pd.DataFrame) -> pd.DataFrame:
    products = strip_string_columns(raw)
    for column in ("cost_price", "list_price"):
        products[column] = pd.to_numeric(products[column], errors="coerce")

    negative = products["cost_price"] < 0
    if negative.any():
        replacement = products.loc[~negative, "cost_price"].median()
        bad_values = products.loc[negative, "cost_price"].tolist()
        products.loc[negative, "cost_price"] = replacement
        print(f"  negative cost_price {bad_values} replaced with median {replacement:.2f}")
    return products.reset_index(drop=True)


def clean_customers(raw: pd.DataFrame) -> pd.DataFrame:
    customers = strip_string_columns(raw)
    customers["signup_date"] = parse_date_column(customers["signup_date"]).dt.strftime(
        "%Y-%m-%d"
    )
    customers["loyalty_tier"] = customers["loyalty_tier"].str.title()
    return customers.reset_index(drop=True)


def print_summary(tables: dict[str, pd.DataFrame]) -> None:
    print("\n" + "=" * 72)
    print("CLEAN DATA SUMMARY")
    print("=" * 72)
    for name, df in tables.items():
        null_cells = int(df.isna().sum().sum())
        print(f"\n{name}: {len(df):,} rows x {len(df.columns)} columns | null cells: {null_cells}")
        null_counts = df.isna().sum()
        null_counts = null_counts[null_counts > 0]
        if null_counts.empty:
            print("  null counts: none")
        else:
            print("  null counts: " + ", ".join(f"{k}={v}" for k, v in null_counts.items()))
        print("  sample:")
        print(df.head(3).to_string(index=False))

    orders = tables["orders"]
    print("\nOrders date range:", orders["order_date"].min(), "to", orders["order_date"].max())
    print("Distinct order IDs:", orders["order_id"].nunique())


def build_quality_report(
    raw: dict[str, pd.DataFrame], cleaned: dict[str, pd.DataFrame]
) -> dict:
    """Before/after data-quality profile, persisted for the Data Quality tab."""
    report: dict = {"tables": {}, "checks": []}
    for name, before in raw.items():
        after = cleaned[name]
        report["tables"][name] = {
            "raw_rows": int(len(before)),
            "clean_rows": int(len(after)),
            "columns": int(len(after.columns)),
            "nulls_after": {
                column: int(count) for column, count in after.isna().sum().items() if count
            },
        }

    raw_orders, raw_products, raw_customers = raw["orders"], raw["products"], raw["customers"]
    orders, products = cleaned["orders"], cleaned["products"]

    def check(name: str, found: int, action: str, passed: bool) -> None:
        report["checks"].append(
            {"check": name, "issues_found": int(found), "action": action, "passed": bool(passed)}
        )

    check(
        "Duplicate order_id",
        raw_orders["order_id"].duplicated().sum(),
        "Kept latest record per order_id",
        orders["order_id"].is_unique,
    )
    check(
        "Non-ISO order_date values",
        (~raw_orders["order_date"].astype(str).str.match(r"^\d{4}-\d{2}-\d{2}$")).sum(),
        "Parsed and normalized to YYYY-MM-DD",
        orders["order_date"].str.match(r"^\d{4}-\d{2}-\d{2}$").all(),
    )
    check(
        "Null unit_price",
        raw_orders["unit_price"].isna().sum(),
        "Kept; excluded from revenue metrics by the booked filter",
        True,
    )
    check(
        "Null region",
        raw_orders["region"].isna().sum(),
        "Kept; excluded from regional breakdowns only",
        True,
    )
    check(
        "Negative cost_price",
        (pd.to_numeric(raw_products["cost_price"], errors="coerce") < 0).sum(),
        "Replaced with median valid cost",
        (products["cost_price"] >= 0).all(),
    )
    check(
        "Untrimmed product names",
        (raw_products["product_name"] != raw_products["product_name"].str.strip()).sum(),
        "Trimmed whitespace",
        (products["product_name"] == products["product_name"].str.strip()).all(),
    )
    check(
        "Inconsistent loyalty_tier casing",
        (raw_customers["loyalty_tier"] != raw_customers["loyalty_tier"].str.strip().str.title()).sum(),
        "Title-cased",
        cleaned["customers"]["loyalty_tier"].isin(["Bronze", "Silver", "Gold", "Platinum"]).all(),
    )
    check(
        "Orphan product_id in orders",
        (~orders["product_id"].isin(products["product_id"])).sum(),
        "Referential integrity check (should be 0)",
        orders["product_id"].isin(products["product_id"]).all(),
    )
    check(
        "Orphan customer_id in orders",
        (~orders["customer_id"].isin(cleaned["customers"]["customer_id"])).sum(),
        "Referential integrity check (should be 0)",
        orders["customer_id"].isin(cleaned["customers"]["customer_id"]).all(),
    )
    return report


def load_to_sqlite(tables: dict[str, pd.DataFrame]) -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{DB_PATH}")
    with engine.begin() as connection:
        for table_name, df in tables.items():
            df.to_sql(table_name, connection, if_exists="replace", index=False)
        for statement in (
            "CREATE INDEX IF NOT EXISTS idx_orders_date ON orders(order_date)",
            "CREATE INDEX IF NOT EXISTS idx_orders_region ON orders(region)",
            "CREATE INDEX IF NOT EXISTS idx_orders_product ON orders(product_id)",
            "CREATE INDEX IF NOT EXISTS idx_orders_customer ON orders(customer_id)",
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_orders_id ON orders(order_id)",
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_products_id ON products(product_id)",
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_customers_id ON customers(customer_id)",
        ):
            connection.execute(text(statement))
    engine.dispose()
    print(f"\nSQLite database written to {DB_PATH}")


def main() -> None:
    required = [DATA_DIR / "orders.csv", DATA_DIR / "products.csv", DATA_DIR / "customers.csv"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Raw CSVs missing. Run `python -m src.generate_data`. Missing: " + ", ".join(missing)
        )

    print("Reading raw CSVs ...")
    orders = pd.read_csv(DATA_DIR / "orders.csv")
    products = pd.read_csv(DATA_DIR / "products.csv")
    customers = pd.read_csv(DATA_DIR / "customers.csv")

    print("Cleaning ...")
    cleaned = {
        "orders": clean_orders(orders),
        "products": clean_products(products),
        "customers": clean_customers(customers),
    }
    print_summary(cleaned)
    load_to_sqlite(cleaned)

    report = build_quality_report(
        {"orders": orders, "products": products, "customers": customers}, cleaned
    )
    QUALITY_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    failed = [item["check"] for item in report["checks"] if not item["passed"]]
    print(f"Data quality report written to {QUALITY_REPORT_PATH}")
    if failed:
        raise SystemExit(f"Data quality checks failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
