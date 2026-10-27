
"""Interactive Streamlit dashboard for Singapore bank tail-risk analysis."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_pipeline import load_or_build_returns
from src.distributions import fit_distributions, summary_statistics, tail_comparison, tail_probability_ratios
from src.inequalities import chebyshev_table
from src.dependence import dependence_matrices, compute_joint_crashes

st.set_page_config(page_title="Singapore Bank Tail Risk", page_icon="📉", layout="wide")
st.title("📉 Heavy Tails & Extreme Risk in Singapore's Banking Trio")
st.caption("DS603 Probability for Data Science | DBS · OCBC · UOB | Historical data: 2014–2024")
st.warning("Educational research dashboard only. Historical risk analysis is not investment advice. Yahoo Finance data availability and adjustments may change.")

@st.cache_data(ttl=3600, show_spinner="Loading prices and calculating the statistical models…")
def load_analysis(refresh_token: int = 0):
    returns = load_or_build_returns(refresh=bool(refresh_token))
    fits = fit_distributions(returns)
    return returns, fits

with st.sidebar:
    st.header("Controls")
    st.write("Data source: Yahoo Finance via yfinance")
    if st.button("Refresh downloaded data"):
        st.session_state["refresh_data"] = 1
        st.cache_data.clear()
        st.rerun()
    bank_options = ["DBS", "OCBC", "UOB"]
    selected_banks = st.multiselect("Banks to display", bank_options, default=bank_options)
    if not selected_banks: selected_banks = bank_options
    k = st.selectbox("Downside shock threshold (kσ)", [2.0, 2.5, 3.0, 4.0], index=1)
    st.markdown("---")
    st.caption("Daily log returns: ln(adjusted price today / adjusted price yesterday)")

try:
    returns, fits = load_analysis(int(st.session_state.get("refresh_data", 0)))
except Exception as exc:
    st.error(f"Could not load the market data: {exc}")
    st.info("Check internet access, then try Refresh downloaded data. You can also run `python -m src.run_analysis` from the repository root.")
    st.stop()

returns = returns[selected_banks]
fits = {b: fits[b] for b in selected_banks}
summary = summary_statistics(returns)
bounds = chebyshev_table(returns, fits)
tails = tail_comparison(returns, fits)
ratios = tail_probability_ratios(returns, fits)
matrices = dependence_matrices(returns)
crashes = compute_joint_crashes(returns, [2.0, 2.5, 3.0])

latest = returns.index.max().strftime("%d %b %Y")
cols = st.columns(4)
cols[0].metric("Common trading days", f"{len(returns):,}")
cols[1].metric("First observation", returns.index.min().strftime("%d %b %Y"))
cols[2].metric("Last observation", latest)
cols[3].metric("Banks selected", len(selected_banks))

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Distribution & VaR", "Chebyshev bounds", "Dependence & co-crashes", "Data & methodology"])

with tab1:
    st.subheader("Historical performance")
    wealth = np.exp(returns.cumsum()) * 100
    st.line_chart(wealth, y_label="Index value (first point approximately 100)")
    st.caption("The index compounds daily log returns. It is a normalized comparison, not a forecast or a dollar portfolio simulation.")
    st.subheader("Descriptive statistics")
    st.dataframe(summary.style.format("{:.4f}"), use_container_width=True)
    fig, ax = plt.subplots(figsize=(12, 4))
    for b in selected_banks: ax.plot(returns.index, returns[b]*100, linewidth=.65, alpha=.7, label=b)
    ax.set_ylabel("Daily log return (%)"); ax.set_title("Daily log returns"); ax.legend()
    st.pyplot(fig); plt.close(fig)

with tab2:
    st.subheader("Normal vs Student-t distribution fits")
    for b in selected_banks:
        x = returns[b].to_numpy()
        p = fits[b]
        fig, ax = plt.subplots(figsize=(9, 3.4))
        ax.hist(x, bins=65, density=True, alpha=.5, label="Empirical")
        grid = np.linspace(np.quantile(x, .002), np.quantile(x, .998), 400)
        from scipy import stats
        ax.plot(grid, stats.norm.pdf(grid, p["normal_mu"], p["normal_sigma"]), label="Normal MLE")
        ax.plot(grid, stats.t.pdf(grid, p["student_df"], loc=p["student_loc"], scale=p["student_scale"]), label="Student-t MLE")
        ax.set_title(f"{b} — Student-t df = {p['student_df']:.2f}"); ax.set_xlabel("Daily log return"); ax.legend()
        st.pyplot(fig); plt.close(fig)
    st.markdown("**How to interpret:** a lower fitted Student-t degrees-of-freedom parameter generally indicates heavier tails. Q-Q plots and empirical tail tables should be read together; a Student-t fit is not guaranteed to outperform Normal for every bank or threshold.")
    st.subheader("Historical and model-implied left-tail quantiles")
    st.dataframe(tails[tails["Bank"].isin(selected_banks)].style.format({
        "Tail probability (%)":"{:.2f}", "Historical quantile (%)":"{:.3f}",
        "Normal quantile (%)":"{:.3f}", "Student-t quantile (%)":"{:.3f}"
    }), use_container_width=True)
    st.subheader("Extreme loss probability ratios")
    st.dataframe(ratios.style.format(precision=5), use_container_width=True)

with tab3:
    st.subheader("Chebyshev inequality and downside exceedances")
    st.latex(r"P(R-\\mu\\leq-k\\sigma)\\leq P(|R-\\mu|\\geq k\\sigma)\\leq\\frac{1}{k^2}")
    view = bounds[bounds["k"] == k].copy()
    st.dataframe(view.style.format({
        "Threshold return (%)":"{:.3f}", "Empirical downside probability":"{:.6f}",
        "Chebyshev upper bound":"{:.6f}", "Cantelli one-sided bound":"{:.6f}",
        "Normal implied probability":"{:.6f}", "Student-t implied probability":"{:.6f}",
        "Empirical / Normal":"{:.2f}"
    }), use_container_width=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    for b in selected_banks:
        d = bounds[bounds["Bank"] == b]
        ax.plot(d["k"], d["Empirical downside probability"].clip(lower=1e-7), marker="o", label=f"{b} empirical")
        ax.plot(d["k"], d["Normal implied probability"].clip(lower=1e-7), linestyle="--", marker=".", label=f"{b} Normal")
    ax.plot(sorted(bounds["k"].unique()), [1/(v*v) for v in sorted(bounds["k"].unique())], color="black", linestyle=":", marker="s", label="Chebyshev bound")
    ax.set_yscale("log"); ax.set_xlabel("k standard deviations"); ax.set_ylabel("Probability (log scale)")
    ax.set_title("Empirical downside probability vs benchmarks"); ax.legend(ncol=2, fontsize=8)
    st.pyplot(fig); plt.close(fig)
    st.info("Chebyshev's bound is distribution-free but can be loose. The standard bound 1/k² is two-sided and also valid as an upper bound for the one-sided downside event. Cantelli's 1/(1+k²) is a sharper one-sided bound.")

with tab4:
    st.subheader("Correlation matrices")
    c1, c2 = st.columns(2)
    for col, name, title in [(c1, "pearson", "Pearson linear correlation"), (c2, "spearman", "Spearman rank correlation")]:
        with col:
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(matrices[name], annot=True, fmt=".2f", vmin=-1, vmax=1, center=0, cmap="coolwarm", ax=ax)
            ax.set_title(title); st.pyplot(fig); plt.close(fig)
    st.subheader(f"Pairwise extreme-loss dependence (k={k:g}σ)")
    pair = crashes["pairwise"]
    st.dataframe(pair[pair["k"] == min(k, 3.0)].style.format(precision=5), use_container_width=True)
    st.subheader("All-three simultaneous crash probability")
    st.dataframe(crashes["trio"].style.format(precision=6), use_container_width=True)
    st.subheader("Dates when all three banks breached the threshold")
    dates = crashes["dates"]
    if dates.empty:
        st.write("No simultaneous three-bank crash events were identified at k = 2, 2.5, or 3 in the current sample.")
    else:
        st.dataframe(dates, use_container_width=True)

with tab5:
    st.subheader("Data and reproducibility")
    st.markdown("""
- **Source:** Yahoo Finance through `yfinance`; the script downloads auto-adjusted daily prices.
- **Tickers:** DBS `D05.SI`, OCBC `O39.SI`, UOB `U11.SI`.
- **Window:** 2014-01-01 through 2024-12-31 (the API end date is exclusive).
- **Alignment:** common available trading dates only; no forward filling.
- **Returns:** daily log returns, `ln(P_t / P_(t-1))`.
- **Estimation:** `scipy.stats.norm.fit` and `scipy.stats.t.fit` on the full return sample.
- **Risk thresholds:** k = 2, 2.5, 3, 4 for Chebyshev; k = 2, 2.5, 3 for pairwise/trio crashes.
- **Caution:** estimates depend on the downloaded sample and are subject to finite-sample tail uncertainty, changing volatility, and data-provider revisions.
""")
    st.download_button("Download cleaned log-return CSV", returns.to_csv(index_label="Date").encode("utf-8"),
                       file_name="bank_log_returns_2014_2024.csv", mime="text/csv")
    st.download_button("Download summary statistics CSV", summary.to_csv().encode("utf-8"),
                       file_name="summary_statistics.csv", mime="text/csv")
