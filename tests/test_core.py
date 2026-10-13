import pandas as pd
import pytest

from src import core, data


@pytest.fixture(scope="module")
def orders():
    return core.add_revenue(data.generate_orders(n_customers=300, days=120, seed=1))


def test_revenue_formula():
    df = pd.DataFrame(
        {"quantity": [2], "unit_price": [10.0], "discount": [0.1]}
    )
    assert core.add_revenue(df)["revenue"].iloc[0] == pytest.approx(18.0)


def test_monthly_revenue_adds_up(orders):
    kpi = core.monthly_kpis(orders)
    assert kpi["revenue"].sum() == pytest.approx(orders["revenue"].sum())


def test_category_shares_sum_to_one(orders):
    assert core.category_summary(orders)["share"].sum() == pytest.approx(1.0)


def test_pareto_share_in_range(orders):
    share = core.pareto_share(orders, "product_id", 0.2)
    assert 0.2 <= share <= 1.0


def test_weekday_profile_has_seven_days(orders):
    assert len(core.weekday_profile(orders)) == 7
