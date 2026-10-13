import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import core, data

OUT = Path("reports")


def main():
    OUT.mkdir(exist_ok=True)
    df = core.add_revenue(data.generate_orders())
    kpi = core.monthly_kpis(df)
    cats = core.category_summary(df)
    channels = core.channel_summary(df)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    kpi["revenue"].plot(ax=axes[0], color="black")
    axes[0].set_title("Monthly revenue")
    axes[0].set_ylabel("revenue")
    cats["revenue"].plot.barh(ax=axes[1], color="grey")
    axes[1].set_title("Revenue by category")
    axes[1].invert_yaxis()
    fig.tight_layout()
    fig.savefig(OUT / "overview.png", dpi=130)

    summary = {
        "orders": int(df["order_id"].nunique()),
        "total_revenue": round(float(df["revenue"].sum()), 2),
        "avg_order_value": round(float(kpi["aov"].mean()), 2),
        "best_month": str(kpi["revenue"].idxmax()),
        "top_category": cats.index[0],
        "top_category_share": round(float(cats["share"].iloc[0]), 3),
        "top20_products_revenue_share": round(core.pareto_share(df, "product_id"), 3),
        "top20_customers_revenue_share": round(core.pareto_share(df, "customer_id"), 3),
        "best_channel": channels.index[0],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    kpi.to_csv(OUT / "monthly_kpis.csv")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
