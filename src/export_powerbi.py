"""Export the cleaned warehouse tables for the Power BI report.

Power BI Desktop has no built-in SQLite connector, so the report reads these
CSVs (the same cleaned rows the copilot queries). Re-run after the pipeline:

    python -m src.export_powerbi
"""

from __future__ import annotations

import sqlite3

import pandas as pd

from src.config import DB_PATH, PROJECT_ROOT, require_db

POWERBI_DATA_DIR = PROJECT_ROOT / "powerbi" / "data"
TABLES = ("orders", "products", "customers")


def export() -> list[str]:
    require_db()
    POWERBI_DATA_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    with sqlite3.connect(DB_PATH) as connection:
        for table in TABLES:
            df = pd.read_sql_query(f'SELECT * FROM "{table}"', connection)
            path = POWERBI_DATA_DIR / f"{table}.csv"
            df.to_csv(path, index=False, encoding="utf-8")
            written.append(f"{path.name} ({len(df)} rows)")
    return written


def main() -> None:
    for item in export():
        print(f"  wrote {item}")
    print(f"Power BI data folder: {POWERBI_DATA_DIR}")


if __name__ == "__main__":
    main()
