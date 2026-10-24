"""Generate deliberately messy sample CSVs for the BI Copilot demo.

This script is the raw-data fixture. It intentionally leaves nulls, duplicate
order IDs, inconsistent date formats, inconsistent casing, and one impossible
cost price so the cleaning stage can be demonstrated.
"""

from __future__ import annotations

import random
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from src.config import DATA_DIR

SEED = 42
ORDER_ROWS = 500
CUSTOMER_ROWS = 200
PRODUCT_ROWS = 30

RNG = random.Random(SEED)
np.random.seed(SEED)

REGIONS = ["Northeast", "Southeast", "Midwest", "Southwest", "West"]
REGION_WEIGHTS = [0.26, 0.22, 0.20, 0.16, 0.16]
STATUSES = ["Completed", "Shipped", "Pending", "Cancelled", "Returned"]
STATUS_WEIGHTS = [0.70, 0.10, 0.08, 0.07, 0.05]
DISCOUNTS = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25]
DISCOUNT_WEIGHTS = [0.35, 0.22, 0.18, 0.11, 0.09, 0.05]

PRODUCT_CATALOG = [
    ("Summit Shell Jacket", "Apparel", 68.00, 139.00),
    ("TrailForge Daypack", "Accessories", 34.50, 79.00),
    ("Ridgeline Tent 2P", "Camping", 142.00, 289.00),
    ("BaseCamp Stove", "Camping", 41.25, 94.50),
    ("Cascade Hiking Boots", "Footwear", 64.00, 154.00),
    ("Merino Base Layer", "Apparel", 29.00, 72.00),
    ("AeroPress Bottle", "Accessories", 12.75, 34.00),
    ("Glacier Sleeping Bag", "Camping", 86.00, 189.00),
    ("Talus Trekking Poles", "Accessories", 31.00, 84.00),
    ("Northwind Fleece", "Apparel", 37.50, 99.00),
    ("Summit Rain Pants", "Apparel", 42.00, 108.00),
    ("PowerCore Kettlebell", "Fitness", -18.50, 74.00),
    ("FlexBand Set", "Fitness", 11.00, 39.00),
    ("Tempo Jump Rope", "Fitness", 5.40, 24.00),
    ("Summit Yoga Mat", "Fitness", 18.00, 58.00),
    ("Cinder Camp Chair", "Camping", 24.00, 69.00),
    ("AeroLite Running Shoe", "Footwear", 58.00, 132.00),
    ("Velocity Trail Shoe", "Footwear", 71.00, 159.00),
    ("Cirrus Windbreaker", "Apparel", 33.00, 89.00),
    ("Granite Duffel 40L", "Accessories", 49.00, 119.00),
    ("Ember Hand Warmer", "Accessories", 7.20, 26.00),
    ("Summit Headlamp", "Accessories", 15.00, 44.00),
    ("Ridge Compression Socks", "Apparel", 8.00, 28.00),
    ("Northstar Beanie", "Apparel", 9.50, 32.00),
    ("Alpine Trek Poles", "Accessories", 39.00, 99.00),
    ("Tundra Parka", "Apparel", 118.00, 269.00),
    ("Creek Sandal", "Footwear", 26.00, 69.00),
    ("Summit Hydration Pack", "Accessories", 44.00, 112.00),
    ("BaseCamp Cook Set", "Camping", 36.00, 89.00),
    ("Trail Mix Kit", "Accessories", 6.50, 21.00),
]

# Some products sell far more often than others (a realistic long tail).
PRODUCT_WEIGHTS = np.array(
    [
        1.4,
        1.3,
        0.7,
        1.0,
        1.2,
        1.2,
        2.0,
        0.6,
        0.8,
        1.1,
        0.7,
        0.6,
        1.5,
        1.3,
        1.4,
        0.8,
        1.6,
        1.1,
        0.9,
        1.0,
        1.7,
        1.2,
        1.8,
        1.9,
        0.7,
        0.5,
        1.0,
        0.9,
        0.7,
        1.9,
    ],
    dtype=float,
)
PRODUCT_WEIGHTS /= PRODUCT_WEIGHTS.sum()

FIRST_NAMES = [
    "Avery",
    "Jordan",
    "Riley",
    "Maya",
    "Noah",
    "Priya",
    "Luis",
    "Sofia",
    "Ethan",
    "Amara",
    "Kai",
    "Zoe",
    "Marcus",
    "Nina",
    "Owen",
    "Leila",
    "Diego",
    "Hannah",
    "Theo",
    "Imani",
    "Caleb",
    "Grace",
    "Arjun",
    "Elena",
    "Miles",
    "Fatima",
    "Jasper",
    "Chloe",
    "Ravi",
    "Nora",
]
LAST_NAMES = [
    "Chen",
    "Patel",
    "Rivera",
    "Brooks",
    "Kim",
    "Okafor",
    "Nguyen",
    "Silva",
    "Reed",
    "Hassan",
    "Moreau",
    "Park",
    "Bennett",
    "Singh",
    "Alvarez",
    "Foster",
    "Ibrahim",
    "Wright",
    "Tanaka",
    "Mensah",
]


def _order_window() -> tuple[date, date]:
    """Return the sample order window: the first three quarters of 2026."""
    return date(2026, 1, 1), date(2026, 9, 30)


def _random_order_dt() -> datetime:
    """Pick a date with more weight on the most recent quarters."""
    quarter = RNG.choices([1, 2, 3], weights=[0.25, 0.35, 0.40], k=1)[0]
    starts = {1: date(2026, 1, 1), 2: date(2026, 4, 1), 3: date(2026, 7, 1)}
    ends = {1: date(2026, 3, 31), 2: date(2026, 6, 30), 3: date(2026, 9, 30)}
    span = (ends[quarter] - starts[quarter]).days
    day = starts[quarter] + timedelta(days=RNG.randint(0, span))
    # Most source rows are date-only; some carry a timestamp.
    hour = RNG.randint(8, 20) if RNG.random() < 0.35 else 0
    minute = RNG.randint(0, 59) if hour else 0
    return datetime(day.year, day.month, day.day, hour, minute)


def _format_mixed_date(value: datetime, force_datetime: bool = False) -> str:
    """Render a date in one of two source formats to create messy input."""
    if force_datetime or value.hour or value.minute:
        if RNG.random() < 0.32:
            return value.strftime("%m/%d/%Y %H:%M:%S")
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if RNG.random() < 0.32:
        return value.strftime("%m/%d/%Y")
    return value.strftime("%Y-%m-%d")


def build_products() -> pd.DataFrame:
    rows = []
    for idx, (name, category, cost, list_price) in enumerate(PRODUCT_CATALOG, start=1):
        # Dirty the product names: trailing whitespace is a classic join hazard.
        dirty_name = f"{name}  " if idx in {1, 5, 9, 13, 18, 23, 28} else name
        rows.append(
            {
                "product_id": f"P{idx:03d}",
                "product_name": dirty_name,
                "category": category,
                "cost_price": cost,
                "list_price": list_price,
            }
        )
    return pd.DataFrame(rows)


def build_customers() -> pd.DataFrame:
    rows = []
    tier_variants = {
        "Bronze": ["Bronze", "bronze", "BRONZE"],
        "Silver": ["Silver", "silver", "SILVER"],
        "Gold": ["Gold", "gold", "GOLD"],
        "Platinum": ["Platinum", "platinum", "PLATINUM"],
    }
    for idx in range(1, CUSTOMER_ROWS + 1):
        first = RNG.choice(FIRST_NAMES)
        last = RNG.choice(LAST_NAMES)
        name = f"{first} {last}"
        tier = RNG.choices(
            list(tier_variants),
            weights=[0.36, 0.34, 0.22, 0.08],
            k=1,
        )[0]
        email = f"{first.lower()}.{last.lower()}{idx:03d}@example.com"
        signup = date(2023, 6, 1) + timedelta(days=RNG.randint(0, 1100))
        signup_text = (
            signup.strftime("%m/%d/%Y")
            if RNG.random() < 0.25
            else signup.strftime("%Y-%m-%d")
        )
        rows.append(
            {
                "customer_id": f"C{idx:04d}",
                "customer_name": name,
                "email": email,
                "signup_date": signup_text,
                "loyalty_tier": RNG.choice(tier_variants[tier]),
            }
        )
    customers = pd.DataFrame(rows)
    # Make the 3% null-email rate exact and reproducible.
    for idx in RNG.sample(range(CUSTOMER_ROWS), k=6):
        customers.loc[idx, "email"] = np.nan
    return customers


def build_orders(products: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    product_ids = products["product_id"].tolist()
    list_prices = dict(zip(products["product_id"], products["list_price"]))
    customer_ids = customers["customer_id"].tolist()

    # 490 base orders + 10 later duplicates = 500 rows but only 490 unique IDs.
    base_rows = ORDER_ROWS - 10
    rows: list[dict] = []
    for idx in range(1, base_rows + 1):
        product_id = RNG.choices(product_ids, weights=PRODUCT_WEIGHTS, k=1)[0]
        order_dt = _random_order_dt()
        list_price = float(list_prices[product_id])
        unit_price = round(list_price * RNG.uniform(0.93, 1.06), 2)
        rows.append(
            {
                "order_id": f"ORD-{idx:05d}",
                "customer_id": RNG.choice(customer_ids),
                "order_date": _format_mixed_date(order_dt),
                "product_id": product_id,
                "quantity": RNG.choices([1, 2, 3, 4, 5, 6], weights=[30, 28, 20, 12, 6, 4])[0],
                "unit_price": unit_price,
                "discount": RNG.choices(DISCOUNTS, weights=DISCOUNT_WEIGHTS, k=1)[0],
                "region": RNG.choices(REGIONS, weights=REGION_WEIGHTS, k=1)[0],
                "status": RNG.choices(STATUSES, weights=STATUS_WEIGHTS, k=1)[0],
            }
        )

    orders = pd.DataFrame(rows)

    # Duplicate 10 order IDs with a later timestamp. The cleaning job must keep
    # the newest version of each duplicated order.
    duplicate_candidates = orders[
        pd.to_datetime(orders["order_date"], format="mixed", errors="coerce")
        <= datetime(2026, 9, 20)
    ].index
    duplicate_indices = RNG.sample(list(duplicate_candidates), 10)
    duplicates = orders.loc[duplicate_indices].copy()
    new_dates = []
    for value in duplicates["order_date"]:
        parsed = None
        for fmt in ("%Y-%m-%d %H:%M:%S", "%m/%d/%Y %H:%M:%S", "%Y-%m-%d", "%m/%d/%Y"):
            try:
                parsed = datetime.strptime(value, fmt)
                break
            except ValueError:
                continue
        if parsed is None:
            raise ValueError(f"Unrecognized generated date: {value}")
        later = parsed + timedelta(days=RNG.randint(1, 5), hours=RNG.randint(1, 8))
        new_dates.append(_format_mixed_date(later, force_datetime=True))
    duplicates["order_date"] = new_dates
    orders = pd.concat([orders, duplicates], ignore_index=True)

    # Exactly 5% nulls in each of two columns.
    for column in ("unit_price", "region"):
        null_indices = RNG.sample(list(orders.index), k=round(ORDER_ROWS * 0.05))
        orders.loc[null_indices, column] = np.nan

    # Shuffle so duplicate pairs are not adjacent in the source file.
    orders = orders.sample(frac=1, random_state=SEED).reset_index(drop=True)
    return orders[
        [
            "order_id",
            "customer_id",
            "order_date",
            "product_id",
            "quantity",
            "unit_price",
            "discount",
            "region",
            "status",
        ]
    ]


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    products = build_products()
    customers = build_customers()
    orders = build_orders(products, customers)

    products.to_csv(DATA_DIR / "products.csv", index=False)
    customers.to_csv(DATA_DIR / "customers.csv", index=False)
    orders.to_csv(DATA_DIR / "orders.csv", index=False)

    duplicate_ids = int(orders["order_id"].duplicated().sum())
    print("Raw sample data written to", DATA_DIR)
    print(f"  orders.csv    : {len(orders):>3} rows | unit_price nulls={orders['unit_price'].isna().sum()} | "
          f"region nulls={orders['region'].isna().sum()} | duplicate order_ids={duplicate_ids}")
    print(f"  products.csv  : {len(products):>3} rows | negative costs={(products['cost_price'] < 0).sum()} | "
          f"names with trailing spaces={products['product_name'].str.endswith(' ').sum()}")
    print(f"  customers.csv : {len(customers):>3} rows | null emails={customers['email'].isna().sum()} | "
          f"tier spellings={customers['loyalty_tier'].nunique()}")


if __name__ == "__main__":
    main()
