# app.py — Wealth simulator (5-year incomes, essentials+discretionary, optional mortgage with balance-sheet)
# Net worth = cash/investments + house_value - remaining_mortgage
# - Mortgage optional; earliest start age enforced + must have down payment cash
# - When mortgage starts: cash -= down_payment; add house asset; set mortgage liability; amortize monthly
# - Borrowing toggle + Debt APR; positive wealth compounds only between comp ages
# - Life events realistic by age; none before IncomeStart
# - Legends placed below plot
#
# + Monte Carlo:
#   - Runs N simulations with randomized life events (via rng_seed+i)
#   - Optional stochastic annual returns during compounding (Normal around mean CAGR)
#   - Plots percentile bands + shows final-net-worth distribution

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.lines import Line2D

# ---------- Helpers ----------

def annuity_payment(principal, annual_rate, years):
    r = annual_rate / 12.0
    n = int(years * 12)
    if n <= 0:
        return 0.0
    if r == 0:
        return principal / n
    return principal * (r * (1 + r) ** n) / ((1 + r) ** n - 1)

def _sample_age_inclusive(rng: np.random.Generator, low: int, high: int):
    """Return int in [low, high] or None if invalid range."""
    if high < low:
        return None
    return int(rng.integers(low, high + 1))

def sample_life_events(start_age, end_age, earn_start_age=25, rng_seed=137):
    """Life-like events after earn_start_age; returns [(age, cost, label), ...]."""
    rng = np.random.default_rng(rng_seed)
    ev = []

    # 1) Wedding/Honeymoon (25–35, ~60% once)
    if rng.random() < 0.6:
        a = _sample_age_inclusive(rng, max(earn_start_age, 25), min(35, end_age))
        if a is not None:
            ev.append((a, rng.uniform(5_000, 20_000), "Wedding/Honeymoon"))
            if rng.random() < 0.3:
                ev.append((a, rng.uniform(3_000, 10_000), "Relocation after wedding"))

    # 2) Children (0–3) 26–38
    n_kids = int(rng.choice([0, 1, 2, 3], p=[0.35, 0.35, 0.22, 0.08]))
    kid_ages = []
    for _ in range(n_kids):
        a = _sample_age_inclusive(rng, max(earn_start_age, 26), min(38, end_age))
        if a is not None:
            kid_ages.append(a)
            ev.append((a, rng.uniform(4_000, 12_000), "Child birth/setup"))

    # 3) Car cycles: first 25–35, then every 8–12 yrs until ~75
    a = _sample_age_inclusive(rng, max(earn_start_age, 25), min(35, end_age))
    if a is not None:
        while a <= min(75, end_age):
            ev.append((a, rng.uniform(8_000, 25_000), "Car purchase/major repair"))
            a += int(rng.integers(8, 13))

    # 4) Home renovation: 30–60, 0–2 times
    renos = int(rng.choice([0, 1, 2], p=[0.3, 0.5, 0.2]))
    for _ in range(renos):
        a = _sample_age_inclusive(rng, max(earn_start_age, 30), min(60, end_age))
        if a is not None:
            ev.append((a, rng.uniform(10_000, 40_000), "Home renovation"))

    # 5) Parental support: 45–65, 0–2 times
    sup = int(rng.choice([0, 1, 2], p=[0.5, 0.35, 0.15]))
    for _ in range(sup):
        a = _sample_age_inclusive(rng, max(earn_start_age, 45), min(65, end_age))
        if a is not None:
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
        if (not ltc) and (rng.random() < 0.12):
            ev.append((a, rng.uniform(50_000, 200_000), "Long-term care"))
            ltc = True

    # Filter + combine same-age events
    ev = [(a, c, label) for (a, c, label) in ev if a >= earn_start_age]
    by_age, labels = {}, {}
    for a, c, label in ev:
        by_age[a] = by_age.get(a, 0.0) + float(c)
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
    # Monte Carlo / stochastic returns
    stochastic_returns=False,
    return_volatility=0.0,      # annual std-dev of returns during compounding (e.g., 0.15 for 15%)
    clamp_return_min=-0.95,     # avoid "return <= -100%" numerics
    clamp_return_max=2.0,
):
    # Use ONE rng stream per run for returns; life events get their own deterministic seed (derived from rng_seed)
    rng = np.random.default_rng(int(rng_seed))
    life_seed = int(rng.integers(0, 2**31 - 1))
    events = sample_life_events(start_age, end_age, earn_start_age=income_start_age, rng_seed=life_seed)
    event_costs = {a: cost for a, cost, _ in events}

    # Balance-sheet components
    cash_wealth = float(starting_wealth)   # investable cash account
    house_value = 0.0                      # asset, added at mortgage start
    mort_balance = 0.0                     # liability, amortized monthly
    monthly_payment = 0.0

    ages, net_worth_series = [], []
    mortgage_started = False
    mortgage_start_age = None

    for age in range(start_age, end_age + 1):
        monthly_income = monthly_income_from_schedule(age, income_schedule)
        earning = monthly_income > 0.0
        compounding = (comp_start_age <= age <= comp_end_age)

        yearly_income = monthly_income * 12.0 if earning else 0.0

        essentials = float(essentials_per_year) if earning else 0.0
        mortgage_active = mortgage_started and (mort_balance > 1e-6)
        current_year_mortgage_payment = 12.0 * monthly_payment if mortgage_active else 0.0

        # Check start conditions for mortgage at the *start* of the year
        if enable_mortgage and (not mortgage_started) and earning:
            down_needed = float(house_price) * float(down_pct)
            if (age >= int(earliest_mortgage_age)) and (cash_wealth >= down_needed):
                cash_wealth -= down_needed
                house_value = float(house_price)
                mort_balance = float(house_price) * (1.0 - float(down_pct))
                monthly_payment = float(annuity_payment(mort_balance, float(mortgage_rate), int(mortgage_years)))
                mortgage_started = True
                mortgage_start_age = age
                mortgage_active = mort_balance > 1e-6
                current_year_mortgage_payment = 12.0 * monthly_payment if mortgage_active else 0.0

        # Discretionary base
        if earning:
            if apply_disc_after_mortgage:
                disc_base = max(0.0, yearly_income - essentials - current_year_mortgage_payment)
            else:
                disc_base = max(0.0, yearly_income - essentials)
            discretionary = (float(discretionary_pct) / 100.0) * disc_base
        else:
            discretionary = 0.0

        # Life event one-off
        life_event_expense = float(event_costs.get(age, 0.0))

        # Total non-mortgage outflows this year
        non_mortgage_outflow = essentials + discretionary + life_event_expense

        # Apply non-mortgage cash flow to cash
        cash_wealth += (yearly_income - non_mortgage_outflow)

        # If borrowing is disabled, don't allow negative cash to "fund" mortgage payments.
        if (not allow_borrowing) and (cash_wealth < 0.0):
            cash_wealth = 0.0

        # Apply mortgage payments (monthly). Don't reduce balance unless you actually paid.
        if mortgage_started and mort_balance > 1e-6:
            m_rate = float(mortgage_rate) / 12.0
            for _ in range(12):
                if mort_balance <= 1e-6:
                    break

                # Interest accrues
                interest = mort_balance * m_rate
                mort_balance += interest

                # Pay if possible
                if allow_borrowing or (cash_wealth >= monthly_payment):
                    pay = min(monthly_payment, mort_balance)  # final payoff month
                    cash_wealth -= pay
                    mort_balance -= pay
                else:
                    # Can't pay this month; interest has already accrued.
                    continue

            if mort_balance <= 1e-6:
                mort_balance = 0.0
                monthly_payment = 0.0

        # Apply returns / debt cost
        if cash_wealth >= 0.0:
            if compounding:
                if stochastic_returns and (return_volatility > 0.0):
                    r = float(rng.normal(loc=float(growth_rate), scale=float(return_volatility)))
                    r = float(np.clip(r, clamp_return_min, clamp_return_max))
                    cash_wealth *= (1.0 + r)
                else:
                    cash_wealth *= (1.0 + float(growth_rate))
        else:
            if allow_borrowing:
                cash_wealth *= (1.0 + float(debt_apr))
            else:
                cash_wealth = 0.0

        # Net worth this year
        net_worth = cash_wealth + house_value - mort_balance
        ages.append(age)
        net_worth_series.append(net_worth)

    df_events = pd.DataFrame(
        [{"Age": a, "Event": label, "Cost (€)": round(float(cost), 2)} for a, cost, label in events]
    )

    return {
        "ages": ages,
        "values": net_worth_series,
        "events": events,
        "mortgage_start_age": mortgage_start_age,
        "df_events": df_events,
    }

def run_monte_carlo(
    n_sims: int,
    base_seed: int,
    **sim_kwargs
):
    """Run n_sims independent runs; return percentile bands + final distribution."""
    n_sims = int(n_sims)
    if n_sims <= 0:
        raise ValueError("n_sims must be > 0")

    # Run once to get age axis length
    first = run_sim(rng_seed=base_seed, **sim_kwargs)
    ages = np.asarray(first["ages"], dtype=int)
    t = len(ages)

    paths = np.empty((n_sims, t), dtype=float)
    finals = np.empty(n_sims, dtype=float)

    paths[0, :] = np.asarray(first["values"], dtype=float)
    finals[0] = paths[0, -1]

    for i in range(1, n_sims):
        res = run_sim(rng_seed=base_seed + i, **sim_kwargs)
        v = np.asarray(res["values"], dtype=float)
        if len(v) != t:
            raise RuntimeError("Simulation produced inconsistent horizon length.")
        paths[i, :] = v
        finals[i] = v[-1]

    pct_levels = np.array([5, 25, 50, 75, 95], dtype=float)
    bands = np.percentile(paths, pct_levels, axis=0)

    # Summary stats
    final_pcts = np.percentile(finals, pct_levels)
    out = {
        "ages": ages,
        "pct_levels": pct_levels,
        "bands": bands,          # shape (5, T)
        "finals": finals,
        "final_pcts": final_pcts,
        "paths": paths,          # keep (handy for debugging / downloads)
    }
    return out

def make_figure_single(res, comp_start_age, comp_end_age, essentials_per_year,
                       discretionary_pct, apply_disc_after_mortgage, income_desc,
                       starting_wealth, enable_mortgage):
    ages, values = res["ages"], res["values"]
    events = res["events"]
    mort_marker_age = res["mortgage_start_age"]

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.plot(ages, values)
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Net Worth (€)")
    ax.set_title("Net Worth over Life: 5-year Income • Essentials+Discretionary • Optional Mortgage • Borrowing • Life Events")
    ax.grid(True)

    ev_handles, ev_labels = [], []
    if enable_mortgage and mort_marker_age is not None:
        lbl = f"Mortgage starts @ age {mort_marker_age}"
        h = ax.axvline(mort_marker_age, linestyle="--", linewidth=1.6, label=lbl)
        ev_handles.append(h)
        ev_labels.append(lbl)

    for (a, c, lab) in events:
        lbl = f"{lab} €{int(round(c, 0))} @ age {a}"
        h = ax.axvline(a, linestyle=":", linewidth=1.0, label=lbl)
        ev_handles.append(h)
        ev_labels.append(lbl)

    inc_line = Line2D([0], [0], linestyle="-")
    mode = "disc. after mortgage" if apply_disc_after_mortgage else "disc. before mortgage"
    setup = (
        f"{income_desc}; start wealth=€{starting_wealth:,.0f}; "
        f"essentials≈€{essentials_per_year:,.0f}/yr; disc={discretionary_pct:.0f}% ({mode}); "
        f"comp {comp_start_age}–{comp_end_age}"
    )
    leg1 = ax.legend(
        [inc_line], [setup], title="Setup",
        loc="upper center", bbox_to_anchor=(0.5, -0.12),
        ncol=1, frameon=True
    )
    ax.add_artist(leg1)

    if ev_handles:
        ax.legend(
            ev_handles, ev_labels, title="Events",
            loc="upper center", bbox_to_anchor=(0.5, -0.38),
            ncol=2, frameon=True
        )

    fig.subplots_adjust(bottom=0.42 if ev_handles else 0.26)
    return fig

def make_figure_mc(mc_res, title_suffix=""):
    ages = mc_res["ages"]
    bands = mc_res["bands"]  # rows: 5,25,50,75,95
    p5, p25, p50, p75, p95 = bands

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.fill_between(ages, p5, p95, alpha=0.15, label="5–95%")
    ax.fill_between(ages, p25, p75, alpha=0.25, label="25–75%")
    ax.plot(ages, p50, linewidth=2.0, label="Median (50%)")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Net Worth (€)")
    ax.set_title(f"Monte Carlo Net Worth Bands{title_suffix}")
    ax.grid(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=True)
    fig.subplots_adjust(bottom=0.22)
    return fig

# ---------- UI ----------

st.set_page_config(page_title="Wealth Simulator (5-year incomes)", layout="wide")
st.title("Wealth Simulator")
st.caption(
    "Net worth with 5-year income buckets, essentials+discretionary spending, optional mortgage (asset & liability), "
    "borrowing APR, realistic life events, plus Monte Carlo percentile bands."
)

tab_single, tab_mc = st.tabs(["Single run", "Monte Carlo"])

# Shared inputs (both tabs)
top1, top2, top3, top4 = st.columns(4)
with top1:
    income_start_age = st.number_input("Income start age", 0, 110, 25, 1)
with top2:
    income_end_age = st.number_input("Income end age", 0, 110, 65, 1)
with top3:
    starting_wealth = st.number_input("Starting wealth (€)", -10_000_000, 10_000_000, 0, 1_000)
with top4:
    rng_seed = st.number_input("Random seed (base)", 1, 999999, 137, 1)

if income_end_age <= income_start_age:
    st.error("Income end age must be > income start age.")
    st.stop()

st.subheader("Income schedule (5-year buckets) — monthly amounts (€)")
buckets = []
cols_per_row = 5
bucket_starts = list(range(int(income_start_age), int(income_end_age), 5))
rows = [bucket_starts[i:i + cols_per_row] for i in range(0, len(bucket_starts), cols_per_row)]
defaults = [4000 + 200 * i for i in range(len(bucket_starts))]

for r_i, row in enumerate(rows):
    cols = st.columns(len(row))
    for j, start in enumerate(row):
        end = min(start + 5, int(income_end_age))
        label = f"{start}–{end}"
        default_val = defaults[r_i * cols_per_row + j]
        val = cols[j].number_input(
            f"{label}",
            min_value=0,
            max_value=100_000,
            value=int(default_val),
            step=100
        )
        buckets.append({"start": int(start), "end": int(end), "monthly": float(val)})

st.subheader("Compounding")
c1, c2, c3 = st.columns(3)
with c1:
    comp_start_age = st.number_input("Compound starts at age", 0, 110, 25, 1)
with c2:
    comp_end_age = st.number_input("Compound ends at age", 0, 110, 65, 1)
with c3:
    growth_rate = st.number_input("Mean CAGR while compounding (%)", 0.0, 50.0, 5.0, 0.1) / 100.0

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
    earliest_mortgage_age = st.number_input("Earliest mortgage start age", 0, 110, max(25, int(income_start_age)), 1)
with m2:
    house_price = st.number_input("House price (€)", 0, 5_000_000, 250_000, 5_000)
with m3:
    down_pct = st.slider("Down payment (%)", 0, 90, 20, 1) / 100.0
with m4:
    mortgage_rate = st.number_input("Mortgage APR (%)", 0.0, 50.0, 4.0, 0.1) / 100.0
with m5:
    mortgage_years = st.number_input("Mortgage term (years)", 1, 60, 30, 1)

income_desc = "; ".join([f"{b['start']}–{b['end']}: €{int(b['monthly'])}/mo" for b in buckets])

# ---------------- Single run tab ----------------
with tab_single:
    run_single = st.button("Run single simulation", type="primary", key="run_single")
    if run_single:
        res = run_sim(
            income_schedule=buckets,
            income_start_age=int(income_start_age),
            comp_start_age=int(comp_start_age),
            comp_end_age=int(comp_end_age),
            essentials_per_year=float(essentials_per_year),
            discretionary_pct=float(discretionary_pct),
            apply_disc_after_mortgage=bool(apply_disc_after_mortgage),
            growth_rate=float(growth_rate),
            start_age=0,
            end_age=90,
            rng_seed=int(rng_seed),
            enable_mortgage=bool(enable_mortgage),
            earliest_mortgage_age=int(earliest_mortgage_age),
            house_price=float(house_price),
            down_pct=float(down_pct),
            mortgage_rate=float(mortgage_rate),
            mortgage_years=int(mortgage_years),
            allow_borrowing=bool(allow_borrowing),
            debt_apr=float(debt_apr),
            starting_wealth=float(starting_wealth),
            stochastic_returns=False,
            return_volatility=0.0,
        )

        fig = make_figure_single(
            res,
            int(comp_start_age),
            int(comp_end_age),
            float(essentials_per_year),
            float(discretionary_pct),
            bool(apply_disc_after_mortgage),
            income_desc,
            float(starting_wealth),
            bool(enable_mortgage),
        )
        st.pyplot(fig, clear_figure=True)

        st.subheader("Generated life events (single run)")
        st.dataframe(res["df_events"], use_container_width=True)
        st.download_button(
            "Download events CSV",
            data=res["df_events"].to_csv(index=False).encode("utf-8"),
            file_name="lifelike_events.csv",
            mime="text/csv",
        )
    else:
        st.info("Fill the 5-year income buckets and click **Run single simulation**.")

# ---------------- Monte Carlo tab ----------------
with tab_mc:
    st.subheader("Monte Carlo settings")
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        n_sims = st.number_input("Simulations", min_value=10, max_value=50000, value=300, step=10)
    with mc2:
        stochastic_returns = st.checkbox("Stochastic returns during compounding", value=True)
    with mc3:
        return_vol = st.number_input("Return volatility σ (%/yr)", 0.0, 100.0, 15.0, 0.5) / 100.0
    with mc4:
        target_final = st.number_input("Target final net worth (€)", 0, 50_000_000, 1_000_000, 50_000)

    run_mc = st.button("Run Monte Carlo", type="primary", key="run_mc")

    if run_mc:
        mc_res = run_monte_carlo(
            n_sims=int(n_sims),
            base_seed=int(rng_seed),
            income_schedule=buckets,
            income_start_age=int(income_start_age),
            comp_start_age=int(comp_start_age),
            comp_end_age=int(comp_end_age),
            essentials_per_year=float(essentials_per_year),
            discretionary_pct=float(discretionary_pct),
            apply_disc_after_mortgage=bool(apply_disc_after_mortgage),
            growth_rate=float(growth_rate),
            start_age=0,
            end_age=90,
            enable_mortgage=bool(enable_mortgage),
            earliest_mortgage_age=int(earliest_mortgage_age),
            house_price=float(house_price),
            down_pct=float(down_pct),
            mortgage_rate=float(mortgage_rate),
            mortgage_years=int(mortgage_years),
            allow_borrowing=bool(allow_borrowing),
            debt_apr=float(debt_apr),
            starting_wealth=float(starting_wealth),
            stochastic_returns=bool(stochastic_returns),
            return_volatility=float(return_vol) if stochastic_returns else 0.0,
        )

        title_suffix = f" (N={int(n_sims)}, μ={growth_rate*100:.1f}%"
        if stochastic_returns:
            title_suffix += f", σ={return_vol*100:.1f}%)"
        else:
            title_suffix += ")"

        fig = make_figure_mc(mc_res, title_suffix=title_suffix)
        st.pyplot(fig, clear_figure=True)

        finals = mc_res["finals"]
        p5, p25, p50, p75, p95 = mc_res["final_pcts"]

        cA, cB, cC, cD, cE = st.columns(5)
        cA.metric("Final P5", f"€{p5:,.0f}")
        cB.metric("Final P25", f"€{p25:,.0f}")
        cC.metric("Final Median", f"€{p50:,.0f}")
        cD.metric("Final P75", f"€{p75:,.0f}")
        cE.metric("Final P95", f"€{p95:,.0f}")

        prob_below_zero = float(np.mean(finals < 0.0)) * 100.0
        prob_hit_target = float(np.mean(finals >= float(target_final))) * 100.0
        st.write(
            f"- P(final net worth < 0): **{prob_below_zero:.1f}%**\n"
            f"- P(final net worth ≥ €{float(target_final):,.0f}): **{prob_hit_target:.1f}%**"
        )

        # Downloads
        ages = mc_res["ages"]
        bands = mc_res["bands"]
        df_bands = pd.DataFrame(
            {
                "Age": ages,
                "P5": bands[0],
                "P25": bands[1],
                "P50": bands[2],
                "P75": bands[3],
                "P95": bands[4],
            }
        )
        df_finals = pd.DataFrame({"FinalNetWorth": finals})

        st.download_button(
            "Download MC percentile bands (CSV)",
            data=df_bands.to_csv(index=False).encode("utf-8"),
            file_name="mc_percentile_bands.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download MC final net worth samples (CSV)",
            data=df_finals.to_csv(index=False).encode("utf-8"),
            file_name="mc_final_net_worth.csv",
            mime="text/csv",
        )

        st.subheader("Monte Carlo bands table")
        st.dataframe(df_bands, use_container_width=True)
    else:
        st.info("Configure Monte Carlo settings and click **Run Monte Carlo**.")

