"""Generate the Power BI report (PBIR format) for BI_Copilot.pbip.

Report-as-code: every page, visual, field binding and filter is declared here,
so the report can be reviewed in a diff and rebuilt identically.

    python powerbi/tools/build_report.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

POWERBI_DIR = Path(__file__).resolve().parents[1]
REPORT_DIR = POWERBI_DIR / "BI_Copilot.Report"
DEFINITION = REPORT_DIR / "definition"
THEME = "CY26SU04"
THEME_SOURCE = Path(
    r"C:\Program Files\Microsoft Power BI Desktop\bin\WebView2Resources"
    rf"\minerva\sharedresources\BaseThemes\{THEME}.json"
)
SCHEMA = "https://developer.microsoft.com/json-schemas/fabric"
TEAL = "#0F766E"
AMBER = "#F59E0B"


def write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


# --- field and literal helpers ------------------------------------------------

def measure(name: str, table: str = "Orders") -> dict:
    return {"Measure": {"Expression": {"SourceRef": {"Entity": table}}, "Property": name}}


def column(table: str, name: str) -> dict:
    return {"Column": {"Expression": {"SourceRef": {"Entity": table}}, "Property": name}}


def projection(field: dict) -> dict:
    kind = "Measure" if "Measure" in field else "Column"
    entity = field[kind]["Expression"]["SourceRef"]["Entity"]
    prop = field[kind]["Property"]
    return {"field": field, "queryRef": f"{entity}.{prop}", "nativeQueryRef": prop}


def lit(value) -> dict:
    if isinstance(value, bool):
        text = "true" if value else "false"
    elif isinstance(value, (int, float)):
        text = f"{value}D"
    else:
        text = f"'{value}'"
    return {"expr": {"Literal": {"Value": text}}}


def color(hex_value: str) -> dict:
    return {"solid": {"color": lit(hex_value)}}


def quarter_filter(name: str, quarter: str = "Q3 2026") -> dict:
    """Categorical filter Date[Quarter] = quarter (page- or visual-level)."""
    return {
        "name": name,
        "field": column("Date", "Quarter"),
        "type": "Categorical",
        "filter": {
            "Version": 2,
            "From": [{"Name": "d", "Entity": "Date", "Type": 0}],
            "Where": [
                {
                    "Condition": {
                        "In": {
                            "Expressions": [
                                {"Column": {"Expression": {"SourceRef": {"Source": "d"}}, "Property": "Quarter"}}
                            ],
                            "Values": [[{"Literal": {"Value": f"'{quarter}'"}}]],
                        }
                    }
                }
            ],
        },
    }


def not_blank_filter(name: str, table: str, field: str) -> dict:
    """Advanced filter: table[field] is not blank (matches SQL `IS NOT NULL`)."""
    return {
        "name": name,
        "field": column(table, field),
        "type": "Advanced",
        "filter": {
            "Version": 2,
            "From": [{"Name": "t", "Entity": table, "Type": 0}],
            "Where": [
                {
                    "Condition": {
                        "Not": {
                            "Expression": {
                                "Comparison": {
                                    "ComparisonKind": 0,
                                    "Left": {"Column": {"Expression": {"SourceRef": {"Source": "t"}}, "Property": field}},
                                    "Right": {"Literal": {"Value": "null"}},
                                }
                            }
                        }
                    }
                }
            ],
        },
    }


# --- visual builders ----------------------------------------------------------

class Page:
    def __init__(self, name: str, display: str, filters: list[dict] | None = None):
        self.name, self.display, self.filters = name, display, filters or []
        self.visuals: list[dict] = []

    def add(self, name: str, x, y, w, h, visual: dict, filters: list[dict] | None = None):
        container = {
            "$schema": f"{SCHEMA}/item/report/definition/visualContainer/1.0.0/schema.json",
            "name": name,
            "position": {"x": x, "y": y, "z": len(self.visuals) * 1000, "width": w, "height": h,
                         "tabOrder": len(self.visuals) * 1000},
            "visual": visual,
        }
        if filters:
            container["filterConfig"] = {"filters": filters}
        self.visuals.append(container)

    def save(self) -> None:
        page_dir = DEFINITION / "pages" / self.name
        page = {
            "$schema": f"{SCHEMA}/item/report/definition/page/1.0.0/schema.json",
            "name": self.name,
            "displayName": self.display,
            "displayOption": "FitToPage",
            "height": 720,
            "width": 1280,
        }
        if self.filters:
            page["filterConfig"] = {"filters": self.filters}
        write(page_dir / "page.json", page)
        for visual in self.visuals:
            write(page_dir / "visuals" / visual["name"] / "visual.json", visual)


def title_objects(text: str) -> dict:
    return {"title": [{"properties": {"show": lit(True), "text": lit(text), "fontSize": lit(12)}}]}


def textbox(text: str, size: str = "20pt", bold: bool = True) -> dict:
    style = {"fontSize": size, "color": "#0F172A"}
    if bold:
        style["fontWeight"] = "bold"
    return {
        "visualType": "textbox",
        "objects": {"general": [{"properties": {"paragraphs": [{"textRuns": [{"value": text, "textStyle": style}]}]}}]},
    }


def card(field: dict, label: str, font: int = 26, wrap: bool = False) -> dict:
    objects = {
        "labels": [{"properties": {"fontSize": lit(font), "color": color(TEAL)}}],
        "categoryLabels": [{"properties": {"show": lit(True)}}],
        **title_objects(label),
    }
    if wrap:
        objects["wordWrap"] = [{"properties": {"show": lit(True)}}]
        objects["categoryLabels"] = [{"properties": {"show": lit(False)}}]
    return {
        "visualType": "card",
        "query": {"queryState": {"Values": {"projections": [projection(field)]}}},
        "objects": objects,
        "drillFilterOtherVisuals": True,
    }


def chart(kind: str, category: dict, values: list[dict], title: str,
          line_values: list[dict] | None = None, sort_by: dict | None = None,
          descending: bool = True) -> dict:
    state = {
        "Category": {"projections": [projection(category)]},
        "Y": {"projections": [projection(v) for v in values]},
    }
    if line_values:
        state["Y2"] = {"projections": [projection(v) for v in line_values]}
    query: dict = {"queryState": state}
    if sort_by:
        query["sortDefinition"] = {
            "sort": [{"field": sort_by, "direction": "Descending" if descending else "Ascending"}],
            "isDefaultSort": False,
        }
    return {
        "visualType": kind,
        "query": query,
        "objects": {
            "dataPoint": [{"properties": {"fill": color(TEAL)}}],
            "labels": [{"properties": {"show": lit(True)}}],
            **title_objects(title),
        },
        "drillFilterOtherVisuals": True,
    }


def table(fields: list[dict], title: str, sort_by: dict | None = None,
          descending: bool = True) -> dict:
    query: dict = {"queryState": {"Values": {"projections": [projection(f) for f in fields]}}}
    if sort_by:
        query["sortDefinition"] = {
            "sort": [{"field": sort_by, "direction": "Descending" if descending else "Ascending"}],
            "isDefaultSort": False,
        }
    return {"visualType": "tableEx", "query": query, "objects": title_objects(title),
            "drillFilterOtherVisuals": True}


def slicer(field: dict, title: str) -> dict:
    return {
        "visualType": "slicer",
        "query": {"queryState": {"Values": {"projections": [projection(field)]}}},
        "objects": {"data": [{"properties": {"mode": lit("Dropdown")}}], **title_objects(title)},
    }


# --- pages --------------------------------------------------------------------

def build_pages() -> list[Page]:
    q3 = quarter_filter("PageQuarter")

    overview = Page("executive_overview", "Executive Overview", [q3])
    overview.add("title", 20, 10, 900, 50, textbox("Summit Outdoor Co. | Q3 2026 business review"))
    overview.add("subtitle", 20, 55, 1100, 30, textbox(
        "Quarter is set in the Filters pane (default Q3 2026). Measures mirror src/metrics.py.",
        size="10pt", bold=False))
    kpis = [
        ("Revenue", "Revenue"), ("Booked Orders", "Orders"), ("AOV", "Avg order value"),
        ("Gross Margin %", "Gross margin"), ("Active Customers", "Active customers"),
    ]
    for i, (name, label) in enumerate(kpis):
        overview.add(f"kpi_{i}", 20 + i * 250, 95, 235, 105, card(measure(name), label))
    deltas = [
        ("Revenue QoQ %", "Revenue vs prior quarter"), ("Orders QoQ %", "Orders vs prior quarter"),
        ("AOV QoQ %", "AOV vs prior quarter"), ("Margin Change pts", "Margin vs prior quarter"),
    ]
    for i, (name, label) in enumerate(deltas):
        overview.add(f"delta_{i}", 20 + i * 250, 210, 235, 85, card(measure(name), label, font=18))
    overview.add("margin_insight", 1020, 210, 240, 85,
                 card(measure("Margin Insight"), "Margin check", font=9, wrap=True))
    overview.add("rev_by_category", 20, 310, 410, 390, chart(
        "clusteredColumnChart", column("Products", "category"), [measure("Revenue")],
        "Revenue by category", sort_by=measure("Revenue")))
    # Orders with no region are excluded from regional views, as in the SQL library.
    known_region = not_blank_filter("KnownRegion", "Orders", "region")
    overview.add("rev_by_region", 445, 310, 410, 390, chart(
        "clusteredBarChart", column("Orders", "region"), [measure("Revenue")],
        "Revenue by region", sort_by=measure("Revenue")), filters=[known_region])
    overview.add("category_table", 870, 310, 390, 390, table(
        [column("Products", "category"), measure("Revenue"), measure("Gross Profit"),
         measure("Gross Margin %")], "Category profitability", sort_by=measure("Revenue")))

    trends = Page("trends", "Trends")
    trends.add("title", 20, 10, 900, 50, textbox("Monthly trends, Jan to Sep 2026"))
    trends.add("rev_trend", 20, 70, 820, 320, chart(
        "lineClusteredColumnComboChart", column("Date", "Month"), [measure("Revenue")],
        "Monthly revenue and 3-month rolling average",
        line_values=[measure("Revenue 3M Rolling Avg")],
        sort_by=column("Date", "Month"), descending=False))
    trends.add("margin_trend", 20, 400, 820, 300, chart(
        "lineChart", column("Date", "Month"), [measure("Gross Margin %")],
        "Product gross margin % by month", sort_by=column("Date", "Month"), descending=False))
    trends.add("month_table", 855, 70, 405, 630, table(
        [column("Date", "Month"), measure("Revenue"), measure("Revenue MoM %"),
         measure("Booked Orders"), measure("Gross Margin %")],
        "Month detail", sort_by=column("Date", "Month"), descending=False))

    margin = Page("margin_discounts", "Margin & Discounts")
    margin.add("title", 20, 10, 1000, 50, textbox("Is the Q3 margin drop real? Do discounts pay?"))
    margin.add("insight", 20, 65, 1240, 60, card(measure("Margin Insight"), "Margin check", font=12, wrap=True),
               filters=[quarter_filter("VisualQuarter")])
    margin.add("region_margin", 20, 135, 620, 270, table(
        [column("Orders", "region"), measure("Revenue"), measure("Gross Margin %"),
         measure("Gross Margin % PQ"), measure("Margin Change pts")],
        "Q3 vs Q2 gross margin by region", sort_by=measure("Revenue")),
        filters=[quarter_filter("VisualQuarter"), not_blank_filter("KnownRegion", "Orders", "region")])
    margin.add("margin_by_band", 655, 135, 605, 270, chart(
        "clusteredColumnChart", column("Orders", "Discount Band"), [measure("Gross Margin %")],
        "Gross margin % by discount band (all booked orders)",
        sort_by=column("Orders", "Discount Band"), descending=False))
    margin.add("band_table", 20, 420, 1240, 280, table(
        [column("Orders", "Discount Band"), measure("Booked Orders"), measure("Avg Units per Order"),
         measure("AOV"), measure("Gross Margin %")],
        "Discount bands: deeper discounts lower margin without lifting basket size",
        sort_by=column("Orders", "Discount Band"), descending=False))

    customers = Page("customers", "Customers")
    customers.add("title", 20, 10, 900, 50, textbox("Customer concentration and loyalty"))
    customers.add("quarter_slicer", 1020, 15, 240, 60, slicer(column("Date", "Quarter"), "Quarter"))
    for i, (name, label) in enumerate([
        ("Active Customers", "Active customers"),
        ("Top 10% Customer Revenue Share", "Revenue from top 10% of customers"),
        ("Repeat Customer Rate", "Customers with 2+ orders"),
    ]):
        customers.add(f"kpi_{i}", 20 + i * 330, 85, 315, 110, card(measure(name), label))
    customers.add("tier_revenue", 20, 210, 620, 490, chart(
        "clusteredColumnChart", column("Customers", "loyalty_tier"), [measure("Revenue")],
        "Revenue by loyalty tier", sort_by=measure("Revenue")))
    customers.add("top_customers", 655, 210, 605, 490, table(
        [column("Customers", "customer_name"), column("Customers", "loyalty_tier"),
         measure("Revenue"), measure("Booked Orders")],
        "Customers by revenue", sort_by=measure("Revenue")))

    return [overview, trends, margin, customers]


def main() -> None:
    # Clear generated content (not the folder itself, which an editor may hold open).
    if DEFINITION.exists():
        for child in DEFINITION.iterdir():
            shutil.rmtree(child) if child.is_dir() else child.unlink()

    write(REPORT_DIR / "definition.pbir", {
        "$schema": f"{SCHEMA}/item/report/definitionProperties/2.0.0/schema.json",
        "version": "4.0",
        "datasetReference": {"byPath": {"path": "../BI_Copilot.SemanticModel"}},
    })
    write(DEFINITION / "version.json", {
        "$schema": f"{SCHEMA}/item/report/definition/versionMetadata/1.0.0/schema.json",
        "version": "2.0.0",
    })
    write(DEFINITION / "report.json", {
        "$schema": f"{SCHEMA}/item/report/definition/report/1.0.0/schema.json",
        "themeCollection": {
            "baseTheme": {
                "name": THEME,
                "reportVersionAtImport": "5.55",
                "type": "SharedResources",
            }
        },
        "layoutOptimization": "None",
        "resourcePackages": [
            {"name": "SharedResources", "type": "SharedResources",
             "items": [{"name": THEME, "path": f"BaseThemes/{THEME}.json", "type": "BaseTheme"}]}
        ],
        "settings": {"useStylableVisualContainerHeader": True, "defaultDrillFilterOtherVisuals": True},
    })

    theme_target = REPORT_DIR / "StaticResources" / "SharedResources" / "BaseThemes" / f"{THEME}.json"
    if not theme_target.exists():
        theme_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(THEME_SOURCE, theme_target)

    pages = build_pages()
    for page in pages:
        page.save()
    write(DEFINITION / "pages" / "pages.json", {
        "$schema": f"{SCHEMA}/item/report/definition/pagesMetadata/1.0.0/schema.json",
        "pageOrder": [page.name for page in pages],
        "activePageName": pages[0].name,
    })
    count = sum(len(page.visuals) for page in pages)
    print(f"Wrote {len(pages)} pages, {count} visuals to {DEFINITION}")


if __name__ == "__main__":
    main()
