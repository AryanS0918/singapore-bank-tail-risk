
"""Reusable plotting functions; all figures are saved in outputs/figures/."""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from src.config import FIGURES_DIR, ensure_directories

def make_figures(returns, fits, bounds, matrices, crashes):
    ensure_directories()
    sns.set_theme(style="whitegrid")
    banks = list(returns.columns)
    # Figure 1: full-sample distributions with fitted PDFs
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, bank in zip(axes, banks):
        x = returns[bank].dropna().to_numpy()
        p = fits[bank]
        ax.hist(x, bins=65, density=True, alpha=.55, label="Empirical")
        grid = np.linspace(np.quantile(x, .002), np.quantile(x, .998), 500)
        ax.plot(grid, stats.norm.pdf(grid, p["normal_mu"], p["normal_sigma"]), label="Normal MLE")
        ax.plot(grid, stats.t.pdf(grid, p["student_df"], loc=p["student_loc"], scale=p["student_scale"]), label="Student-t MLE")
        ax.set_title(bank); ax.set_xlabel("Daily log return"); ax.set_ylabel("Density"); ax.legend(fontsize=8)
    fig.suptitle("Daily log returns and fitted distributions"); fig.tight_layout()
    fig.savefig(FIGURES_DIR/"fig1_return_distributions.png", dpi=180, bbox_inches="tight"); plt.close(fig)
    # Figure 2: Q-Q plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for col, bank in enumerate(banks):
        x = np.sort(returns[bank].dropna().to_numpy()); n = len(x)
        probs = (np.arange(1, n+1)-.5)/n
        p = fits[bank]
        for row, (label, theoretical) in enumerate([
            ("Normal", stats.norm.ppf(probs, p["normal_mu"], p["normal_sigma"])),
            ("Student-t", stats.t.ppf(probs, p["student_df"], loc=p["student_loc"], scale=p["student_scale"]))
        ]):
            ax = axes[row, col]
            ax.scatter(theoretical, x, s=5, alpha=.35)
            lo = min(theoretical.min(), x.min()); hi = max(theoretical.max(), x.max())
            ax.plot([lo, hi], [lo, hi], linestyle="--")
            ax.set_title(f"{bank}: {label} Q-Q"); ax.set_xlabel("Theoretical quantiles"); ax.set_ylabel("Empirical quantiles")
    fig.tight_layout(); fig.savefig(FIGURES_DIR/"fig2_qq_plots.png", dpi=180, bbox_inches="tight"); plt.close(fig)
    # Figure 3: probability comparison on log scale
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    for ax, bank in zip(axes, banks):
        d = bounds[bounds["Bank"] == bank]
        for col, label in [("Empirical downside probability","Empirical"),("Chebyshev upper bound","Chebyshev"),("Student-t implied probability","Student-t"),("Normal implied probability","Normal")]:
            ax.plot(d["k"], d[col].clip(lower=1e-7), marker="o", label=label)
        ax.set_yscale("log"); ax.set_title(bank); ax.set_xlabel("k standard deviations"); ax.legend(fontsize=7)
    axes[0].set_ylabel("Downside probability (log scale)")
    fig.suptitle("Empirical losses vs probability benchmarks"); fig.tight_layout()
    fig.savefig(FIGURES_DIR/"fig3_chebyshev_comparison.png", dpi=180, bbox_inches="tight"); plt.close(fig)
    # Figure 4: correlation heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    sns.heatmap(matrices["pearson"], annot=True, fmt=".2f", vmin=-1, vmax=1, center=0, cmap="coolwarm", ax=axes[0])
    axes[0].set_title("Pearson correlation")
    sns.heatmap(matrices["spearman"], annot=True, fmt=".2f", vmin=-1, vmax=1, center=0, cmap="coolwarm", ax=axes[1])
    axes[1].set_title("Spearman rank correlation")
    fig.tight_layout(); fig.savefig(FIGURES_DIR/"fig4_correlation_heatmaps.png", dpi=180, bbox_inches="tight"); plt.close(fig)
    # Figure 5: cumulative returns indexed to 100
    wealth = np.exp(returns.cumsum())*100
    fig, ax = plt.subplots(figsize=(12, 5))
    for bank in banks: ax.plot(wealth.index, wealth[bank], label=bank)
    ax.set_title("Growth of 100 index units using daily log returns"); ax.set_ylabel("Index (start = 100)"); ax.legend()
    fig.tight_layout(); fig.savefig(FIGURES_DIR/"fig5_growth_of_100.png", dpi=180, bbox_inches="tight"); plt.close(fig)
    # Figure 6: daily returns
    fig, ax = plt.subplots(figsize=(12, 5))
    for bank in banks: ax.plot(returns.index, returns[bank]*100, linewidth=.65, alpha=.7, label=bank)
    ax.set_title("Daily log returns (percent)"); ax.set_ylabel("Log return (%)"); ax.legend()
    fig.tight_layout(); fig.savefig(FIGURES_DIR/"fig6_daily_returns.png", dpi=180, bbox_inches="tight"); plt.close(fig)
