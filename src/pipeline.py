"""End-to-end build: raw data -> cleaned warehouse -> RAG index -> insights report.

    python -m src.pipeline            # full rebuild
    python -m src.pipeline --local    # force offline embeddings

The Streamlit app calls ``ensure_ready()`` on start-up, so a fresh clone or a
Streamlit Cloud deploy works without running setup commands by hand.
"""

from __future__ import annotations

import argparse
import logging

from src import analytics, build_rag, export_powerbi, generate_data, load_data
from src.build_rag import read_index_meta
from src.config import DB_PATH, QUALITY_REPORT_PATH

logger = logging.getLogger("bi_copilot")


def run(force_local: bool = False) -> None:
    logger.info("1/5 generating raw data")
    generate_data.main()
    logger.info("2/5 cleaning and loading SQLite")
    load_data.main()
    logger.info("3/5 building the document index")
    build_rag.build_index(force_local=force_local)
    logger.info("4/5 writing the insights report")
    analytics.write_report()
    logger.info("5/5 exporting Power BI data")
    export_powerbi.export()


def ensure_ready() -> bool:
    """Build whatever is missing. Returns True if anything was built."""
    built = False
    if not DB_PATH.exists() or not QUALITY_REPORT_PATH.exists():
        generate_data.main()
        load_data.main()
        built = True
    if not read_index_meta():
        build_rag.build_index()
        built = True
    return built


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Rebuild the full BI Copilot pipeline.")
    parser.add_argument("--local", action="store_true", help="Force offline embeddings.")
    run(force_local=parser.parse_args().local)


if __name__ == "__main__":
    main()
