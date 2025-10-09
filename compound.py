# app.py — Lifetime wealth simulator with UI
# UI inputs: monthly income, compounding start/end ages, % income spent
# Notes: realistic life events; mortgage starts after down payment is saved;
#        two legends placed below plot; full event legend (can be long).

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.lines import Line2D

# ---------- Core model ----------

def annuity_payment(principal, annual_rate, years):
    r = annual_rate / 12.0
    n = int(years * 12)
    if r == 0:
        return principal / n
    return principal * (r * (1 + r)**n) / ((1 + r)**n - 1)

def sample_life_events(start_age, end_age, earn_start_age=25, rng_seed=137):
    """
    Life-like events after earn_start_age. Returns list of (age, cost, label).
    """
    rng = np.random.default_rng(rng_seed)
    events = []

    # 1) Wedding/Honeymoon 25–35, 60% once
    if rng.random() < 0.6 and end_age >= max(earn_start_age, 25):
        age = int(rng.integers(max(earn_start_age, 25), min(35, end_age) + 1))
        events.append((age, rng.uniform(5_000, 20_000), "Wedding/Honeymoon"))
        if rng.random() < 0.3:
            events.append((age, rng.uniform(3_000, 10_000), "Relocation after wedding"))

    # 2) Children 0–3 between 26–38
    n_kids = int(rng.choice([0,1,2,3], p=[0.35, 0.35, 0.22, 0.08]))
    kid_ages = []
    for _ in range(n_kids):
        if end_age >= max(earn_start_age, 26):
            a = int(rng.integers(max(earn_start_age, 26), min(38, end_age) + 1))
            kid_ages.append(a)
            events.append((a, rng.uniform(4_000, 12_000), "Child birth/setup"))

    # 3) Car purchase/major repair cycles
    if end_age >= max(earn_start_age, 25):
        a = int(rng.integers(max(earn_start_age, 25), min(35, end_age) + 1))
        while a <= min(75, end_age):
            events.append((a, rng.uniform(8_000, 25_000), "Car purchase/major repair"))
            a += int(rng.integers(8, 13))

    # 4) Home renovation: 0–2 times, 30–60
    renos = int(rng.choice([0,1,2], p=[0.3,0.5,0.2]))
    for _ in range(renos):
        if end_age >= max(earn_start_age, 30):
            a = int(rng.integers(max(earn_start_age, 30), min(60, end_age) + 1))
            events.append((a, rng.uniform(10_000, 40_000), "Home renovation"))

    # 5) Parental support: 0–2 times, 45–65
    supports = int(rng.choice([0,1,2], p=[0.5,0.35,0.15]))
    for _ in range(supports):
        if end_age >= max(earn_start_age, 45):
            a = int(rng.integers(max(earn_start_age, 45), min(65, end_age) + 1))
            events.append((a, rng.uniform(5_000, 20_000), "Parental support"))

    # 6) College contributions (kid +18)
    for ba in kid_ages:
        ca = ba + 18
        if earn_start_age <= ca <= end_age:
            events.append((ca, rng.uniform(10_000, 40_000), "College contribution"))

    # 7) Medical procedures: 45+, probability rises with age
    for a in range(max(earn_start_age, 45), end_age + 1):
        # 2% at 45, +0.3% per year, capped at 20%
        p = min(0.02 + 0.003 * max(0, a - 45), 0.20)
        if rng.random() < p:
            scale = 8_000 + 400 * (a - 45)
            mu, sigma = np.log(max(scale, 1.0)), 0.6  # log-normal-ish
            cost = float(np.exp(rng.normal(mu, sigma)))
            cost = float(np.clip(cost, 5_000, 120_000))
            events.append((a, cost, "Medical procedure"))

    # 8) Long-term care: 75+, at most once, 12%/yr
    ltc_done = False
    for a in range(max(earn_start_age, 75), end_age + 1):
        if not ltc_done and np.random.default_rng(rng.integers(0, 10**9)).random() < 0.12:
            cost = np.random.default_rng(rng.integers(0, 10**9)).uniform(50_000, 200_000)
            events.append((a, cost, "Long-term care"))
            ltc_done = True

    # Filter & combine per age
    events = [(a, c, lab) for (a, c, lab) in events if a >= earn_start_age]
    by_age, labels = {}, {}
    for a, c, lab in events:
        by_age[a] = by_age.get(a, 0.0) + c
        labels.setdefault(a, []).append(lab)
    return [(a, by_age[a], ", ".join(labels[a])) for a in sorted(by_age.keys())]

def build_baseline_expenses(earn_start_age, earn_end_age, end_age=90):
    """Annual fixed expenses by age (very simple, tweak as needed)."""
    e = {}
    for a in range(0, 18): e[a] = 0.0
    for a in range(18, int(earn_start_age)): e[a] = 8_000.0
    for a in range(int(earn_start_age), 35): e[a] = 24_000.0
    for a in range(35, 50): e[a] = 30_000.0
    for a in range(50, int(earn_end_age)): e[a] = 26_000.0
    for a in range(int(earn_end_age), 75): e[a] = 24_000.0
    for a in range(75, end_age + 1): e[a] = 21_600.0
    return e

def run_simulation(
    monthly_income=4000,
    earn_start_age=25,
    earn_end_age=65,
    percent_income_spent=50.0,
    growth_rate=0.05,
    start_age=0, end_age=90,
    rng_seed=137,
    house_price=250_000.0, down_pct=0.20, mortgage_rate=0.04, mortgage_years=30,
):
    """Return dict with ages, values, events, mortgage_start_age, df_events."""
    expenses_by_age = build_baseline_expenses(earn_start_age, earn_end_age, end_age)

    events = sample_life_events(start_age, end_age, earn_start_age=earn_start_age, rng_seed=rng_seed)
    event_costs = {a: cost for a, cost, _ in events}

    total_value = 0.0
    saved_for_down = 0.0
    ages, values = [], []
    mortgage_start_age = None
    mortgage_end_age = None
    monthly_mortgage = None

    for age in range(start_age, end_age + 1):
        earning = (earn_start_age <= age < earn_end_age)
        yearly_income = (monthly_income * 12.0) if earning else 0.0

        pct_spent = (percent_income_spent / 100.0) if earning else 0.0
        pct_expense_amount = yearly_income * pct_spent
        base_expense = expenses_by_age.get(age, 0.0)

        # Mortgage start when down payment saved
        down_needed = house_price * down_pct
        if mortgage_start_age is None and earning and saved_for_down >= down_needed:
            mortgage_start_age = age
            mortgage_end_age = min(end_age + 1, age + mortgage_years)
            principal = house_price * (1.0 - down_pct)
            monthly_mortgage = annuity_payment(principal, mortgage_rate, mortgage_years)

        mortgage_expense_annual = 0.0
        if mortgage_start_age is not None and mortgage_start_age <= age < mortgage_end_age:
            mortgage_expense_annual = monthly_mortgage * 12.0

        life_event_expense = event_costs.get(age, 0.0)

        yearly_expenses = base_expense + pct_expense_amount + mortgage_expense_annual + life_event_expense
        net_annual = yearly_income - yearly_expenses

        if mortgage_start_age is None and net_annual > 0 and earning:
            saved_for_down += net_annual

        total_value += net_annual
        if earning:
            total_value *= (1.0 + growth_rate)

        ages.append(age)
        values.append(total_value)

    df_events = pd.DataFrame(
        [{"Age": a, "Event": lab, "Cost (€)": round(cost, 2)} for a, cost, lab in events]
    )
    return {
        "ages": ages,
        "values": values,
        "events": events,
        "mortgage_start_age": mortgage_start_age,
        "df_events": df_events,
    }

def make_figure(res, monthly_income, earn_start_age, earn_end_age, percent_income_spent):
    ages, values = res["ages"], res["values"]
    events = res["events"]
    mortgage_start_age = res["mortgage_start_age"]

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.plot(ages, values, label=f"€{monthly_income}/mo")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Total Value (€)")
    ax.set_title("Wealth over Life: Mortgage + %Income Expenses + Realistic Events")
    ax.grid(True)

    # Mortgage marker
    event_handles, event_labels = [], []
    if mortgage_start_age is not None:
        mlabel = f"Mortgage start @ {mortgage_start_age}"
        h = ax.axvline(mortgage_start_age, linestyle="--", linewidth=1.6, label=mlabel)
        event_handles.append(h); event_labels.append(mlabel)

    # Life events (full legend)
    for (age, cost, lab) in events:
        elabel = f"{lab} €{int(round(cost,0))} @ age {age}"
        h = ax.axvline(age, linestyle=":", linewidth=1.0, label=elabel)
        event_handles.append(h); event_labels.append(elabel)

    # Legend A (income setup) — below
    income_handles = [Line2D([0],[0], linestyle='-')]
    income_labels = [f"Income €{monthly_income}/mo; % spent={percent_income_spent:.0f}%; comp {earn_start_age}–{earn_end_age}"]
    leg1 = ax.legend(income_handles, income_labels, title="Income setup",
                     loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1, frameon=True)
    ax.add_artist(leg1)

    # Legend B (events) — below (can be long)
    if event_handles:
        ax.legend(event_handles, event_labels, title="Events",
                  loc="upper center", bbox_to_anchor=(0.5, -0.38), ncol=2, frameon=True)

    plt.tight_layout()
    return fig

# ---------- UI ----------

st.set_page_config(page_title="Lifetime Wealth Simulator", layout="wide")

st.title("Lifetime Wealth Simulator")
st.caption("Mortgage after down payment, % income spent during earning years, age-realistic life events. Full event legend placed below the chart.")

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    monthly_income = st.number_input("Monthly income (€)", min_value=500, max_value=50000, value=4000, step=100)
with col2:
    comp_start = st.number_input("Compounds from age", min_value=18, max_value=50, value=25, step=1)
with col3:
    comp_end = st.number_input("Compounds until age", min_value=50, max_value=85, value=65, step=1)
with col4:
    pct_spent = st.slider("% income spent (earning years)", min_value=0, max_value=95, value=50, step=1)
with col5:
    rng_seed = st.number_input("Random seed", min_value=1, max_value=999999, value=137, step=1)

# Optional advanced mortgage params (collapsed)
with st.expander("Mortgage parameters (optional)"):
    house_price = st.number_input("House price (€)", min_value=50_000, max_value=2_000_000, value=250_000, step=5_000)
    down_pct = st.slider("Down payment (%)", min_value=0, max_value=90, value=20, step=1) / 100.0
    mortgage_rate = st.number_input("Mortgage APR (%)", min_value=0.0, max_value=20.0, value=4.0, step=0.1) / 100.0
    mortgage_years = st.number_input("Mortgage term (years)", min_value=5, max_value=40, value=30, step=1)
    growth_rate = st.number_input("Portfolio growth during earning years (%)", min_value=0.0, max_value=20.0, value=5.0, step=0.1) / 100.0

run = st.button("Run simulation", type="primary")

if run:
    res = run_simulation(
        monthly_income=monthly_income,
        earn_start_age=int(comp_start),
        earn_end_age=int(comp_end),
        percent_income_spent=float(pct_spent),
        growth_rate=growth_rate,
        rng_seed=int(rng_seed),
        house_price=float(house_price),
        down_pct=float(down_pct),
        mortgage_rate=float(mortgage_rate),
        mortgage_years=int(mortgage_years),
    )
    fig = make_figure(res, monthly_income, int(comp_start), int(comp_end), float(pct_spent))
    st.pyplot(fig, clear_figure=True)

    # Events table + download
    st.subheader("Generated life events")
    st.dataframe(res["df_events"], use_container_width=True)
    csv = res["df_events"].to_csv(index=False).encode("utf-8")
    st.download_button("Download events CSV", data=csv, file_name="lifelike_events.csv", mime="text/csv")

else:
    st.info("Set your inputs and click **Run simulation**.")

