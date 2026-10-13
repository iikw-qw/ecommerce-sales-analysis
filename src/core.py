import pandas as pd


def add_revenue(df):
    out = df.copy()
    out["revenue"] = out["quantity"] * out["unit_price"] * (1 - out["discount"])
    return out


def monthly_kpis(df):
    """Revenue, orders, customers, average order value and month-over-month growth."""
    d = df.copy()
    d["month"] = d["order_date"].dt.to_period("M")
    g = d.groupby("month")
    kpi = pd.DataFrame(
        {
            "revenue": g["revenue"].sum(),
            "orders": g["order_id"].nunique(),
            "customers": g["customer_id"].nunique(),
        }
    )
    kpi["aov"] = kpi["revenue"] / kpi["orders"]
    kpi["mom_growth"] = kpi["revenue"].pct_change()
    return kpi


def category_summary(df):
    g = df.groupby("category")
    out = pd.DataFrame(
        {
            "revenue": g["revenue"].sum(),
            "units": g["quantity"].sum(),
            "avg_discount": g["discount"].mean(),
        }
    )
    out["share"] = out["revenue"] / out["revenue"].sum()
    return out.sort_values("revenue", ascending=False)


def pareto_share(df, by="product_id", top_fraction=0.2):
    """Share of revenue produced by the top fraction of products (or customers)."""
    rev = df.groupby(by)["revenue"].sum().sort_values(ascending=False)
    k = max(1, int(round(len(rev) * top_fraction)))
    return float(rev.iloc[:k].sum() / rev.sum())


def weekday_profile(df):
    d = df.copy()
    d["weekday"] = d["order_date"].dt.day_name()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    daily = d.groupby(["order_date", "weekday"])["revenue"].sum().reset_index()
    return daily.groupby("weekday")["revenue"].mean().reindex(order)


def discount_effect(df):
    """Average units per order line and revenue per line by discount level."""
    g = df.groupby("discount")
    return pd.DataFrame(
        {
            "lines": g.size(),
            "avg_quantity": g["quantity"].mean(),
            "avg_revenue": g["revenue"].mean(),
        }
    )


def channel_summary(df):
    g = df.groupby("channel")
    out = pd.DataFrame(
        {
            "revenue": g["revenue"].sum(),
            "orders": g["order_id"].nunique(),
            "customers": g["customer_id"].nunique(),
        }
    )
    out["revenue_per_customer"] = out["revenue"] / out["customers"]
    return out.sort_values("revenue", ascending=False)
