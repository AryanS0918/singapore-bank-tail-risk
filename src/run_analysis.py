
"""Run the complete reproducible analysis and export tables and figures."""
import pandas as pd
from src.config import TABLES_DIR, ensure_directories
from src.data_pipeline import load_or_build_returns
from src.distributions import fit_distributions, summary_statistics, tail_comparison, tail_probability_ratios
from src.inequalities import chebyshev_table
from src.dependence import dependence_matrices, compute_joint_crashes
from src.visualize import make_figures

def main(refresh: bool = False):
    ensure_directories()
    returns = load_or_build_returns(refresh=refresh)
    fits = fit_distributions(returns)
    summary = summary_statistics(returns)
    tails = tail_comparison(returns, fits)
    ratios = tail_probability_ratios(returns, fits)
    bounds = chebyshev_table(returns, fits)
    matrices = dependence_matrices(returns)
    crashes = compute_joint_crashes(returns, [2.0, 2.5, 3.0])
    summary.to_csv(TABLES_DIR/"table1_summary_stats.csv")
    tails.to_csv(TABLES_DIR/"table2_tail_comparison.csv", index=False)
    ratios.to_csv(TABLES_DIR/"table2b_tail_probability_ratios.csv", index=False)
    bounds.to_csv(TABLES_DIR/"table3_chebyshev_bounds.csv", index=False)
    for name, frame in matrices.items():
        frame.to_csv(TABLES_DIR/f"table4_{name}.csv")
    crashes["pairwise"].to_csv(TABLES_DIR/"table5_pairwise_crashes.csv", index=False)
    crashes["trio"].to_csv(TABLES_DIR/"table5_trio_crashes.csv", index=False)
    crashes["dates"].to_csv(TABLES_DIR/"table5_joint_crash_dates.csv", index=False)
    # Markdown versions are convenient for README embedding
    summary.to_markdown(TABLES_DIR/"table1_summary_stats.md")
    tails.to_markdown(TABLES_DIR/"table2_tail_comparison.md", index=False)
    bounds.to_markdown(TABLES_DIR/"table3_chebyshev_bounds.md", index=False)
    matrices["pearson"].to_markdown(TABLES_DIR/"table4_pearson.md")
    matrices["spearman"].to_markdown(TABLES_DIR/"table4_spearman.md")
    make_figures(returns, fits, bounds, matrices, crashes)
    print(f"Analysis complete: {len(returns):,} common trading days, {returns.index.min().date()} to {returns.index.max().date()}")
    print("\nDescriptive statistics:\n", summary.round(4).to_string())
    print("\nStudent-t degrees of freedom:")
    for bank, p in fits.items(): print(f"  {bank}: {p['student_df']:.3f}")
    print("\nOutputs saved under outputs/")

if __name__ == "__main__":
    main()
