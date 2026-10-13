import numpy as np
import pandas as pd

CATEGORIES = {
    "electronics": (180.0, 0.18),
    "home": (60.0, 0.30),
    "clothing": (45.0, 0.28),
    "beauty": (25.0, 0.14),
    "sports": (70.0, 0.10),
}
CHANNELS = ["organic", "paid", "email", "referral"]
REGIONS = ["north", "south", "east", "west"]


def generate_orders(n_customers=2500, days=730, seed=42):
    """Synthetic order lines with trend, yearly seasonality and a weekend effect."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2023-01-01")
    dates = pd.date_range(start, periods=days, freq="D")

    t = np.arange(days)
    season = 1 + 0.25 * np.sin(2 * np.pi * (t - 60) / 365) + 0.5 * (dates.month == 12)
    weekend = np.where(dates.dayofweek >= 5, 1.2, 1.0)
    daily_orders = rng.poisson(120 * (1 + t / 1500) * season * weekend)

    day_idx = np.repeat(np.arange(days), daily_orders)
    n = len(day_idx)
    cats = list(CATEGORIES)
    cat_p = np.array([CATEGORIES[c][1] for c in cats])
    cat_p = cat_p / cat_p.sum()
    category = rng.choice(cats, size=n, p=cat_p)

    base = np.array([CATEGORIES[c][0] for c in category])
    unit_price = np.round(base * rng.lognormal(0, 0.25, n), 2)
    discount = rng.choice([0.0, 0.05, 0.1, 0.2], size=n, p=[0.6, 0.15, 0.15, 0.1])
    quantity = 1 + rng.poisson(np.where(discount > 0.1, 0.9, 0.5), n)

    # a small share of customers generates most of the orders
    weights = rng.pareto(1.5, n_customers) + 1
    customer_id = rng.choice(n_customers, size=n, p=weights / weights.sum())

    df = pd.DataFrame(
        {
            "order_id": np.arange(n),
            "order_date": dates[day_idx],
            "customer_id": customer_id,
            "category": category,
            "product_id": [f"{c[:3]}-{i}" for c, i in zip(category, rng.integers(1, 40, n))],
            "quantity": quantity,
            "unit_price": unit_price,
            "discount": discount,
            "channel": rng.choice(CHANNELS, size=n, p=[0.4, 0.3, 0.2, 0.1]),
            "region": rng.choice(REGIONS, size=n),
        }
    )
    return df
