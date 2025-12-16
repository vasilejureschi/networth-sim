# app.py — Wealth simulator (5-year incomes, essentials+discretionary, optional mortgage with balance-sheet)
# Net worth = cash/investments + house_value - remaining_mortgage
# - Mortgage optional; earliest start age enforced + must have down payment cash
# - When mortgage starts: cash -= down_payment; add house asset; set mortgage liability; amortize monthly
# - Borrowing toggle + Debt APR; positive wealth compounds only between comp ages
# - Life events realistic by age; none before IncomeStart
# - Legends placed below plot

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.lines import Line2D

# ---------- Helpers ----------

def annuity_payment(principal, annual_rate, years):
    r = annual_rate / 12.0
    n = int(years * 12)
    if r == 0:
        return principal / n
    return principal * (r * (1 + r)**n) / ((1 + r)**n - 1)

def sample_life_events(start_age, end_age, earn_start_age=25, rng_seed=137):
    """Life-like events after earn_start_age; returns [(age, cost, label), ...]."""
    rng = np.random.default_rng(rng_seed)
    ev = []

    # 1) Wedding/Honeymoon (25–35, ~60% once)
    if rng.random() < 0.6 and end_age >= max(earn_start_age, 25):
        a = int(rng.integers(max(earn_start_age, 25), min(35, end_age) + 1))
        ev.append((a, rng.uniform(5_000, 20_000), "Wedding/Honeymoon"))
        if rng.random() < 0.3:
            ev.append((a, rng.uniform(3_000, 10_000), "Relocation after wedding"))

    # 2) Children (0–3) 26–38
    n_kids = int(rng.choice([0,1,2,3], p=[0.35,0.35,0.22,0.08]))
    kid_ages = []
    for _ in range(n_kids):
        if end_age >= max(earn_start_age, 26):
            a = int(rng.integers(max(earn_start_age, 26), min(38, end_age) + 1))
            kid_ages.append(a)
            ev.append((a, rng.uniform(4_000, 12_000), "Child birth/setup"))

    # 3) Car cycles: first 25–35, then every 8–12 yrs until ~75
    if end_age >= max(earn_start_age, 25):
        a = int(rng.integers(max(earn_start_age, 25), min(35, end_age) + 1))
        while a <= min(75, end_age):
            ev.append((a, rng.uniform(8_000, 25_000), "Car purchase/major repair"))
            a += int(rng.integers(8, 13))

    # 4) Home renovation: 30–60, 0–2 times
    renos = int(rng.choice([0,1,2], p=[0.3,0.5,0.2]))
    for _ in range(renos):
        if end_age >= max(earn_start_age, 30):
            a = int(rng.integers(max(earn_start_age, 30), min(60, end_age) + 1))
            ev.append((a, rng.uniform(10_000, 40_000), "Home renovation"))

    # 5) Parental support: 45–65, 0–2 times
    sup = int(rng.choice([0,1,2], p=[0.5,0.35,0.15]))
    for _ in range(sup):
        if end_age >= max(earn_start_age, 45):
            a = int(rng.integers(max(earn_start_age, 45), min(65, end_age) + 1))
            ev.append((a, rng.uniform(5_000, 20_000), "Parental support"))

    # 6) College contributions (kid birth age + 18)
    for ba in kid_ages:
        ca = ba + 18
        if earn_start_age <= ca <= end_age:
            ev.append((ca, rng.uniform(10_000, 40_000), "College contribution"))

    # 7) Medical procedures: 45+, probability rises with age (max 20%/yr)
    for a in range(max(earn_start_age, 45), end_age + 1):
        p = min(0.02 + 0.003 * max(0, a - 45), 0.20)
        if rng.random() < p:
            scale = 8_000 + 400 * (a - 45)
            mu, sigma = np.log(max(scale, 1.0)), 0.6
            cost = float(np.exp(rng.normal(mu, sigma)))
            cost = float(np.clip(cost, 5_000, 120_000))
            ev.append((a, cost, "Medical procedure"))

    # 8) Long-term care: 75+, ~12%/yr, at most once
    ltc = False
    for a in range(max(earn_start_age, 75), end_age + 1):
        if not ltc and np.random.default_rng(rng.integers(0, 10**9)).random() < 0.12:
            ev.append((a, np.random.default_rng(rng.integers(0, 10**9)).uniform(50_000, 200_000), "Long-term care"))
            ltc = True

    # Filter + combine same-age events
    ev = [(a, c, label) for (a, c, label) in ev if a >= earn_start_age]
    by_age, labels = {}, {}
    for a, c, label in ev:
        by_age[a] = by_age.get(a, 0.0) + c
        labels.setdefault(a, []).append(label)
    return [(a, by_age[a], ", ".join(labels[a])) for a in sorted(by_age.keys())]

def monthly_income_from_schedule(age, schedule):
    for b in schedule:
        if b["start"] <= age < b["end"]:
            return float(b["monthly"])
    return 0.0

# ---------- Simulation ----------

def run_sim(
    income_schedule,           # list of buckets [{'start','end','monthly'}]
    income_start_age,          # used to gate life events
    comp_start_age, comp_end_age,
    # Spending model (essentials + discretionary %)
    essentials_per_year,
    discretionary_pct,
    apply_disc_after_mortgage,
    # Growth & horizon
    growth_rate,
    start_age=0, end_age=90,
    rng_seed=137,
    # Mortgage
    enable_mortgage=True,
    earliest_mortgage_age=25,
    house_price=250_000.0,
    down_pct=0.20,
    mortgage_rate=0.04,
    mortgage_years=30,
    # Borrowing
    allow_borrowing=False,
    debt_apr=0.10,
    # Starting wealth
    starting_wealth=0.0,
):
    events = sample_life_events(start_age, end_age, earn_start_age=income_start_age, rng_seed=rng_seed)
    event_costs = {a: cost for a, cost, _ in events}

    # Balance-sheet components
    cash_wealth = float(starting_wealth)   # investable cash account
    house_value = 0.0                      # asset, added at mortgage start
    mort_balance = 0.0                     # liability, amortized monthly
    monthly_payment = 0.0

    ages, net_worth_series = [], []
    saved_for_down = 0.0                   # track positive surplus pre-mortgage (informative)
    mortgage_started = False

    for age in range(start_age, end_age + 1):
        monthly_income = monthly_income_from_schedule(age, income_schedule)
        earning = monthly_income > 0.0
        compounding = (comp_start_age <= age <= comp_end_age)

        yearly_income = monthly_income * 12.0 if earning else 0.0

        # Essentials + discretionary (discretionary applied to leftover after essentials and optionally mortgage)
        essentials = essentials_per_year if earning else 0.0

        # If mortgage active, we know the yearly payment; for "disc after mortgage" base use that
        current_year_mortgage_payment = 12.0 * monthly_payment if mortgage_started else 0.0

        # Check start conditions for mortgage at the *start* of the year
        if enable_mortgage and not mortgage_started and earning:
            down_needed = house_price * down_pct
            if (age >= earliest_mortgage_age) and (cash_wealth >= down_needed):
                # Start mortgage:
                # - pay down payment from cash
                # - add house asset
                # - set mortgage balance
                cash_wealth -= down_needed
                house_value = house_price
                mort_balance = house_price * (1.0 - down_pct)
                monthly_payment = annuity_payment(mort_balance, mortgage_rate, mortgage_years)
                mortgage_started = True
                current_year_mortgage_payment = 12.0 * monthly_payment

        # Discretionary base
        if earning:
            if apply_disc_after_mortgage:
                disc_base = max(0.0, yearly_income - essentials - current_year_mortgage_payment)
            else:
                disc_base = max(0.0, yearly_income - essentials)
            discretionary = (discretionary_pct / 100.0) * disc_base
        else:
            discretionary = 0.0

        # Life event one-off
        life_event_expense = event_costs.get(age, 0.0)

        # Total non-mortgage outflows this year
        non_mortgage_outflow = essentials + discretionary + life_event_expense

        # Cash flow pre-mortgage-payment
        net_cash_flow = yearly_income - non_mortgage_outflow
        if not mortgage_started:
            # track surplus pre-mortgage for reference
            if net_cash_flow > 0 and earning:
                saved_for_down += net_cash_flow

        # Apply non-mortgage cash flow to cash
        cash_wealth += net_cash_flow

        # Apply mortgage payments (monthly loop for interest/principal split)
        if mortgage_started and mort_balance > 1e-6:
            m_rate = mortgage_rate / 12.0
            for _ in range(12):
                if mort_balance <= 1e-6:
                    break
                interest = mort_balance * m_rate
                principal = monthly_payment - interest
                if principal > mort_balance:
                    principal = mort_balance
                    monthly_out = interest + principal
                else:
                    monthly_out = monthly_payment
                # Pay from cash
                cash_wealth -= monthly_out
                # Reduce liability by principal component
                mort_balance -= principal

        # Apply interest/growth on cash_wealth
        if cash_wealth >= 0:
            if compounding:
                cash_wealth *= (1.0 + growth_rate)
        else:
            if allow_borrowing:
                cash_wealth *= (1.0 + debt_apr)
            else:
                cash_wealth = 0.0

        # Net worth this year
        net_worth = cash_wealth + house_value - mort_balance
        ages.append(age)
        net_worth_series.append(net_worth)

    df_events = pd.DataFrame([{"Age": a, "Event": label, "Cost (€)": round(cost,2)} for a, cost, label in events])
    return {
        "ages": ages,
        "values": net_worth_series,    # plot NET WORTH
        "events": events,
        "mortgage_start_age": (None if not mortgage_started else next(a for a in ages if a >= start_age and True)),  # simple marker if started
        "df_events": df_events,
    }

def make_figure(res, comp_start_age, comp_end_age, essentials_per_year,
                discretionary_pct, apply_disc_after_mortgage, income_desc,
                starting_wealth, enable_mortgage):
    ages, values = res["ages"], res["values"]
    events = res["events"]
    mort_marker_age = res["mortgage_start_age"]

    fig, ax = plt.subplots(figsize=(11,7))
    ax.plot(ages, values)
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Net Worth (€)")
    ax.set_title("Net Worth over Life: 5-year Income • Essentials+Discretionary • Optional Mortgage • Borrowing • Life Events")
    ax.grid(True)

    ev_handles, ev_labels = [], []
    if enable_mortgage and mort_marker_age is not None:
        lbl = f"Mortgage starts @ age {mort_marker_age}"
        h = ax.axvline(mort_marker_age, linestyle="--", linewidth=1.6, label=lbl)
        ev_handles.append(h); ev_labels.append(lbl)

    for (a, c, lab) in events:
        lbl = f"{lab} €{int(round(c,0))} @ age {a}"
        h = ax.axvline(a, linestyle=":", linewidth=1.0, label=lbl)
        ev_handles.append(h); ev_labels.append(lbl)

    inc_line = Line2D([0],[0], linestyle='-')
    mode = "disc. after mortgage" if apply_disc_after_mortgage else "disc. before mortgage"
    setup = (f"{income_desc}; start wealth=€{starting_wealth:,.0f}; "
             f"essentials≈€{essentials_per_year:,.0f}/yr; disc={discretionary_pct:.0f}% ({mode}); "
             f"comp {comp_start_age}–{comp_end_age}")
    leg1 = ax.legend([inc_line], [setup], title="Setup",
                     loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1, frameon=True)
    ax.add_artist(leg1)

    if ev_handles:
        ax.legend(ev_handles, ev_labels, title="Events",
                  loc="upper center", bbox_to_anchor=(0.5, -0.38), ncol=2, frameon=True)
    plt.tight_layout()
    return fig

# ---------- UI ----------

st.set_page_config(page_title="Wealth Simulator (5-year incomes)", layout="wide")
st.title("Wealth Simulator")
st.caption("Net worth with 5-year income buckets, essentials+discretionary spending, optional mortgage (asset & liability), borrowing APR, and realistic life events. Legends are below the chart.")

top1, top2, top3, top4 = st.columns(4)
with top1:
    income_start_age = st.number_input("Income start age", 0, 110, 25, 1)
with top2:
    income_end_age   = st.number_input("Income end age",   0, 110, 65, 1)
with top3:
    starting_wealth  = st.number_input("Starting wealth (€)", -10_000_000, 10_000_000, 0, 1_000)
with top4:
    rng_seed = st.number_input("Random seed", 1, 999999, 137, 1)

if income_end_age <= income_start_age:
    st.error("Income end age must be > income start age.")
    st.stop()

st.subheader("Income schedule (5-year buckets) — monthly amounts (€)")
buckets = []
cols_per_row = 5
bucket_starts = list(range(income_start_age, income_end_age, 5))
rows = [bucket_starts[i:i+cols_per_row] for i in range(0, len(bucket_starts), cols_per_row)]
defaults = [4000 + 200*i for i in range(len(bucket_starts))]
for r_i, row in enumerate(rows):
    cols = st.columns(len(row))
    for j, start in enumerate(row):
        end = min(start+5, income_end_age)
        label = f"{start}–{end}"
        default_val = defaults[r_i*cols_per_row + j]
        val = cols[j].number_input(f"{label}", min_value=0, max_value=100_000, value=default_val, step=100)
        buckets.append({"start": int(start), "end": int(end), "monthly": float(val)})

st.subheader("Compounding")
c1, c2, c3 = st.columns(3)
with c1:
    comp_start_age = st.number_input("Compound starts at age", 0, 110, 25, 1)
with c2:
    comp_end_age   = st.number_input("Compound ends at age",   0, 110, 65, 1)
with c3:
    growth_rate = st.number_input("Growth CAGR while compounding (%)", 0.0, 50.0, 5.0, 0.1) / 100.0

st.subheader("Spending — Essentials + Discretionary %")
s1, s2, s3 = st.columns(3)
with s1:
    essentials_per_year = st.number_input("Essentials per year (€)", 0, 200_000, 18_000, 500)
with s2:
    discretionary_pct = st.slider("Discretionary % (of leftover)", 0, 100, 30, 1)
with s3:
    apply_disc_after_mortgage = st.checkbox("Apply discretionary after mortgage too", value=False)

st.subheader("Borrowing")
b1, b2 = st.columns(2)
with b1:
    allow_borrowing = st.checkbox("Allow borrowing (wealth can go negative)", value=False)
with b2:
    debt_apr = st.number_input("Debt APR (%)", 0.0, 100.0, 10.0, 0.1) / 100.0

st.subheader("Mortgage (optional)")
m0, m1, m2, m3, m4, m5 = st.columns(6)
with m0:
    enable_mortgage = st.checkbox("Enable mortgage", value=True)
with m1:
    earliest_mortgage_age = st.number_input("Earliest mortgage start age", 0, 110, max(25, income_start_age), 1)
with m2:
    house_price = st.number_input("House price (€)", 0, 5_000_000, 250_000, 5_000)
with m3:
    down_pct = st.slider("Down payment (%)", 0, 90, 20, 1) / 100.0
with m4:
    mortgage_rate = st.number_input("Mortgage APR (%)", 0.0, 50.0, 4.0, 0.1) / 100.0
with m5:
    mortgage_years = st.number_input("Mortgage term (years)", 1, 60, 30, 1)

run = st.button("Run simulation", type="primary")

if run:
    res = run_sim(
        income_schedule=buckets,
        income_start_age=int(income_start_age),
        comp_start_age=int(comp_start_age),
        comp_end_age=int(comp_end_age),
        essentials_per_year=float(essentials_per_year),
        discretionary_pct=float(discretionary_pct),
        apply_disc_after_mortgage=bool(apply_disc_after_mortgage),
        growth_rate=float(growth_rate),
        start_age=0, end_age=90, rng_seed=int(rng_seed),
        enable_mortgage=bool(enable_mortgage),
        earliest_mortgage_age=int(earliest_mortgage_age),
        house_price=float(house_price), down_pct=float(down_pct),
        mortgage_rate=float(mortgage_rate), mortgage_years=int(mortgage_years),
        allow_borrowing=bool(allow_borrowing), debt_apr=float(debt_apr),
        starting_wealth=float(starting_wealth),
    )
    income_desc = "; ".join([f"{b['start']}–{b['end']}: €{int(b['monthly'])}/mo" for b in buckets])
    fig = make_figure(res, int(comp_start_age), int(comp_end_age),
                      float(essentials_per_year), float(discretionary_pct),
                      bool(apply_disc_after_mortgage), income_desc,
                      float(starting_wealth), bool(enable_mortgage))
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
    st.info("Fill the 5-year income buckets and click **Run simulation**.")

