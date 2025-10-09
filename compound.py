# app.py — Lifetime wealth simulator (Streamlit UI)
# Features:
# - Flexible compounding ages (0–110)
# - % of income spent (whole living cost) OR optional essentials+discretionary mode
# - Mortgage starts after down payment, mortgage params exposed
# - Random life events, human-plausible by age
# - Borrowing mode: negative wealth accrues debt interest APR
# - Legends placed below the plot so the chart stays visible

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.lines import Line2D

# ---------------- Utilities ----------------

def annuity_payment(principal, annual_rate, years):
    r = annual_rate / 12.0
    n = int(years * 12)
    if r == 0:
        return principal / n
    return principal * (r * (1 + r)**n) / ((1 + r)**n - 1)

def sample_life_events(start_age, end_age, earn_start_age=25, rng_seed=137):
    """Life-like events after earn_start_age; returns [(age, cost, label), ...]."""
    rng = np.random.default_rng(rng_seed)
    events = []

    # 1) Wedding/Honeymoon (25–35, ~60% once)
    if rng.random() < 0.6 and end_age >= max(earn_start_age, 25):
        age = int(rng.integers(max(earn_start_age, 25), min(35, end_age) + 1))
        events.append((age, rng.uniform(5_000, 20_000), "Wedding/Honeymoon"))
        if rng.random() < 0.3:
            events.append((age, rng.uniform(3_000, 10_000), "Relocation after wedding"))

    # 2) Children 0–3 between 26–38
    n_kids = int(rng.choice([0,1,2,3], p=[0.35,0.35,0.22,0.08]))
    kid_ages = []
    for _ in range(n_kids):
        if end_age >= max(earn_start_age, 26):
            a = int(rng.integers(max(earn_start_age, 26), min(38, end_age) + 1))
            kid_ages.append(a)
            events.append((a, rng.uniform(4_000, 12_000), "Child birth/setup"))

    # 3) Car cycles: first 25–35, then every 8–12 yrs until ~75
    if end_age >= max(earn_start_age, 25):
        a = int(rng.integers(max(earn_start_age, 25), min(35, end_age) + 1))
        while a <= min(75, end_age):
            events.append((a, rng.uniform(8_000, 25_000), "Car purchase/major repair"))
            a += int(rng.integers(8, 13))

    # 4) Home renovation: 30–60, 0–2 times
    renos = int(rng.choice([0,1,2], p=[0.3,0.5,0.2]))
    for _ in range(renos):
        if end_age >= max(earn_start_age, 30):
            a = int(rng.integers(max(earn_start_age, 30), min(60, end_age) + 1))
            events.append((a, rng.uniform(10_000, 40_000), "Home renovation"))

    # 5) Parental support: 45–65, 0–2 times
    sup = int(rng.choice([0,1,2], p=[0.5,0.35,0.15]))
    for _ in range(sup):
        if end_age >= max(earn_start_age, 45):
            a = int(rng.integers(max(earn_start_age, 45), min(65, end_age) + 1))
            events.append((a, rng.uniform(5_000, 20_000), "Parental support"))

    # 6) College contributions (kid birth age + 18)
    for ba in kid_ages:
        ca = ba + 18
        if earn_start_age <= ca <= end_age:
            events.append((ca, rng.uniform(10_000, 40_000), "College contribution"))

    # 7) Medical procedures: 45+, probability rises with age (max 20%/yr)
    for a in range(max(earn_start_age, 45), end_age + 1):
        p = min(0.02 + 0.003 * max(0, a - 45), 0.20)
        if rng.random() < p:
            scale = 8_000 + 400 * (a - 45)
            mu, sigma = np.log(max(scale, 1.0)), 0.6
            cost = float(np.exp(rng.normal(mu, sigma)))
            cost = float(np.clip(cost, 5_000, 120_000))
            events.append((a, cost, "Medical procedure"))

    # 8) Long-term care: 75+, at most once, ~12%/yr
    ltc_done = False
    for a in range(max(earn_start_age, 75), end_age + 1):
        if not ltc_done and np.random.default_rng(rng.integers(0, 10**9)).random() < 0.12:
            cost = np.random.default_rng(rng.integers(0, 10**9)).uniform(50_000, 200_000)
            events.append((a, cost, "Long-term care"))
            ltc_done = True

    # Filter + combine same-age events
    events = [(a, c, lab) for (a, c, lab) in events if a >= earn_start_age]
    by_age, labels = {}, {}
    for a, c, lab in events:
        by_age[a] = by_age.get(a, 0.0) + c
        labels.setdefault(a, []).append(lab)
    return [(a, by_age[a], ", ".join(labels[a])) for a in sorted(by_age.keys())]

# ---------------- Simulation core ----------------

def run_sim(
    monthly_income,
    earn_start_age,
    earn_end_age,
    percent_income_spent,               # if essentials_mode=False: total living cost % of income
    growth_rate,                         # portfolio CAGR while compounding ages
    start_age=0,
    end_age=90,
    rng_seed=137,
    # Mortgage
    house_price=250_000.0,
    down_pct=0.20,
    mortgage_rate=0.04,
    mortgage_years=30,
    # Borrowing
    allow_borrowing=False,
    debt_apr=0.10,                       # annual interest applied to negative wealth
    # Optional essentials + discretionary
    essentials_mode=False,
    essentials_per_year=18_000.0,
):
    events = sample_life_events(start_age, end_age, earn_start_age=earn_start_age, rng_seed=rng_seed)
    event_costs = {a: cost for a, cost, _ in events}

    wealth = 0.0
    saved_for_down = 0.0
    mortgage_start_age = None
    mortgage_end_age = None
    monthly_mortgage = None

    ages, values = [], []

    for age in range(start_age, end_age + 1):
        earning = (earn_start_age <= age < earn_end_age)
        compounding = (earn_start_age <= age <= earn_end_age)  # inclusive end if desired

        yearly_income = (monthly_income * 12.0) if earning else 0.0

        # Spending model
        if essentials_mode and earning:
            # Essentials are fixed €/year, then discretionary is % of leftover after essentials + mortgage
            base_essentials = max(0.0, essentials_per_year)
            discretionary_base = max(0.0, yearly_income - base_essentials)
            living_expenses_pre_mortgage = base_essentials + (percent_income_spent/100.0) * discretionary_base
        elif earning:
            # Simple model: the % is the whole living cost
            living_expenses_pre_mortgage = yearly_income * (percent_income_spent/100.0)
        else:
            living_expenses_pre_mortgage = 0.0

        # Decide if mortgage starts this year (after surplus saved for down payment)
        down_needed = house_price * down_pct
        if mortgage_start_age is None and earning and saved_for_down >= down_needed:
            mortgage_start_age = age
            mortgage_end_age = min(end_age + 1, age + mortgage_years)
            principal = house_price * (1.0 - down_pct)
            monthly_mortgage = annuity_payment(principal, mortgage_rate, mortgage_years)

        mortgage_expense = 0.0
        if mortgage_start_age is not None and mortgage_start_age <= age < mortgage_end_age:
            mortgage_expense = monthly_mortgage * 12.0

        life_event_expense = event_costs.get(age, 0.0)

        yearly_expenses = living_expenses_pre_mortgage + mortgage_expense + life_event_expense
        net_annual = yearly_income - yearly_expenses

        # Accumulate down payment savings only before mortgage starts (from positive surplus)
        if mortgage_start_age is None and net_annual > 0 and earning:
            saved_for_down += net_annual

        # Wealth update before interest
        wealth += net_annual

        # Apply interest next: portfolio growth on positive wealth during compounding,
        # debt APR on negative wealth (always, if borrowing allowed). If borrowing is NOT allowed,
        # floor at zero before applying growth.
        if not allow_borrowing and wealth < 0:
            wealth = 0.0

        if wealth >= 0:
            if compounding:
                wealth *= (1.0 + growth_rate)
        else:
            # Negative wealth = debt; apply debt interest
            wealth *= (1.0 + debt_apr)

        ages.append(age)
        values.append(wealth)

    df_events = pd.DataFrame([{"Age": a, "Event": lab, "Cost (€)": round(cost,2)} for a, cost, lab in events])

    return {
        "ages": ages,
        "values": values,
        "events": events,
        "mortgage_start_age": mortgage_start_age,
        "df_events": df_events,
    }

def make_figure(res, monthly_income, earn_start_age, earn_end_age, percent_income_spent,
                essentials_mode, essentials_per_year):
    ages, values = res["ages"], res["values"]
    events = res["events"]
    mortgage_start_age = res["mortgage_start_age"]

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.plot(ages, values)
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Total Value (€)")
    ax.set_title("Wealth over Life: Mortgage + Spending + Events + Borrowing")
    ax.grid(True)

    event_handles, event_labels = [], []
    if mortgage_start_age is not None:
        mlabel = f"Mortgage start @ {mortgage_start_age}"
        h = ax.axvline(mortgage_start_age, linestyle="--", linewidth=1.6, label=mlabel)
        event_handles.append(h); event_labels.append(mlabel)

    for (age, cost, lab) in events:
        elabel = f"{lab} €{int(round(cost,0))} @ age {age}"
        h = ax.axvline(age, linestyle=":", linewidth=1.0, label=elabel)
        event_handles.append(h); event_labels.append(elabel)

    # Legend A (income setup) — below
    income_handles = [Line2D([0],[0], linestyle='-')]
    mode_txt = ("essentials+discretionary" if essentials_mode else "percent = total living cost")
    inc_lbl = f"€{monthly_income}/mo; spend={percent_income_spent:.0f}% ({mode_txt}); comp {earn_start_age}–{earn_end_age}"
    if essentials_mode:
        inc_lbl += f"; essentials≈€{essentials_per_year:,.0f}/yr"
    leg1 = ax.legend(income_handles, [inc_lbl], title="Income setup",
                     loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1, frameon=True)
    ax.add_artist(leg1)

    # Legend B (events) — below
    if event_handles:
        ax.legend(event_handles, event_labels, title="Events",
                  loc="upper center", bbox_to_anchor=(0.5, -0.38), ncol=2, frameon=True)
    plt.tight_layout()
    return fig

# ---------------- UI ----------------

st.set_page_config(page_title="Lifetime Wealth Simulator", layout="wide")

st.title("Lifetime Wealth Simulator")
st.caption("Mortgage after down payment, flexible compounding ages, optional borrowing with debt APR, and life-like events. Legends are below the chart.")

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    monthly_income = st.number_input("Monthly income (€)", min_value=0, max_value=100_000, value=4000, step=100)
with col2:
    comp_start = st.number_input("Compounds from age", min_value=0, max_value=110, value=25, step=1)
with col3:
    comp_end = st.number_input("Compounds until age", min_value=0, max_value=110, value=65, step=1)
with col4:
    pct_spent = st.slider("% income spent (earning years)", min_value=0, max_value=95, value=50, step=1)
with col5:
    rng_seed = st.number_input("Random seed", min_value=1, max_value=999999, value=137, step=1)

st.divider()
st.subheader("Borrowing")
c1, c2 = st.columns(2)
with c1:
    allow_borrowing = st.checkbox("Allow borrowing (wealth can go negative)", value=False)
with c2:
    debt_apr = st.number_input("Debt APR (%)", min_value=0.0, max_value=100.0, value=10.0, step=0.1) / 100.0

st.subheader("Mortgage parameters")
m1, m2, m3, m4 = st.columns(4)
with m1:
    house_price = st.number_input("House price (€)", min_value=0, max_value=5_000_000, value=250_000, step=5_000)
with m2:
    down_pct = st.slider("Down payment (%)", min_value=0, max_value=90, value=20, step=1) / 100.0
with m3:
    mortgage_rate = st.number_input("Mortgage APR (%)", min_value=0.0, max_value=50.0, value=4.0, step=0.1) / 100.0
with m4:
    mortgage_years = st.number_input("Mortgage term (years)", min_value=1, max_value=60, value=30, step=1)

st.subheader("Portfolio growth")
growth_rate = st.number_input("Growth CAGR while compounding (%)", min_value=0.0, max_value=50.0, value=5.0, step=0.1) / 100.0

with st.expander("Optional: essentials + discretionary mode"):
    essentials_mode = st.checkbox("Use essentials + discretionary model", value=False)
    essentials_per_year = st.number_input("Essentials per year (€)", min_value=0, max_value=200_000, value=18_000, step=500)

run = st.button("Run simulation", type="primary")

if run:
    res = run_sim(
        monthly_income=monthly_income,
        earn_start_age=int(comp_start),
        earn_end_age=int(comp_end),
        percent_income_spent=float(pct_spent),
        growth_rate=growth_rate,
        start_age=0, end_age=90,
        rng_seed=int(rng_seed),
        house_price=float(house_price),
        down_pct=float(down_pct),
        mortgage_rate=float(mortgage_rate),
        mortgage_years=int(mortgage_years),
        allow_borrowing=allow_borrowing,
        debt_apr=float(debt_apr),
        essentials_mode=essentials_mode,
        essentials_per_year=float(essentials_per_year),
    )
    fig = make_figure(res, monthly_income, int(comp_start), int(comp_end),
                      float(pct_spent), essentials_mode, float(essentials_per_year))
    st.pyplot(fig, clear_figure=True)

    st.subheader("Generated life events")
    st.dataframe(res["df_events"], use_container_width=True)
    st.download_button(
        "Download events CSV",
        data=res["df_events"].to_csv(index=False).encode("utf-8"),
        file_name="lifelike_events.csv",
        mime="text/csv",
    )
else:
    st.info("Set inputs and click **Run simulation**.")

