
"""Normal/Student-t MLE fits and empirical-versus-parametric tail tables."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from src.config import TAIL_QUANTILES

def fit_distributions(returns: pd.DataFrame) -> dict:
    fitted = {}
    for bank in returns.columns:
        x = returns[bank].dropna().to_numpy(dtype=float)
        mu_n, sigma_n = stats.norm.fit(x)
        nu, loc_t, scale_t = stats.t.fit(x)
        fitted[bank] = {
            "normal_mu": float(mu_n), "normal_sigma": float(sigma_n),
            "student_df": float(nu), "student_loc": float(loc_t), "student_scale": float(scale_t),
            "n": int(len(x))
        }
    return fitted

def summary_statistics(returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bank in returns.columns:
        x = returns[bank].dropna()
        rows.append({
            "Bank": bank, "N": len(x), "Mean (%)": x.mean()*100,
            "Std dev (%)": x.std(ddof=1)*100, "Skewness": x.skew(),
            "Excess kurtosis": x.kurtosis(), "Minimum (%)": x.min()*100,
            "Maximum (%)": x.max()*100
        })
    return pd.DataFrame(rows).set_index("Bank")

def tail_comparison(returns: pd.DataFrame, fits: dict) -> pd.DataFrame:
    rows = []
    for bank in returns.columns:
        x = returns[bank].dropna().to_numpy()
        p = fits[bank]
        for q in TAIL_QUANTILES:
            cutoff = float(np.quantile(x, q))
            rows.append({
                "Bank": bank, "Tail probability (%)": q*100,
                "Historical quantile (%)": cutoff*100,
                "Normal quantile (%)": stats.norm.ppf(q, p["normal_mu"], p["normal_sigma"])*100,
                "Student-t quantile (%)": stats.t.ppf(q, p["student_df"], loc=p["student_loc"], scale=p["student_scale"])*100
            })
    return pd.DataFrame(rows)

def tail_probability_ratios(returns: pd.DataFrame, fits: dict, cutoffs=(-0.03, -0.05)) -> pd.DataFrame:
    rows = []
    for bank in returns.columns:
        x = returns[bank].dropna().to_numpy()
        p = fits[bank]
        for cutoff in cutoffs:
            empirical = float(np.mean(x <= cutoff))
            normal = float(stats.norm.cdf(cutoff, p["normal_mu"], p["normal_sigma"]))
            student = float(stats.t.cdf(cutoff, p["student_df"], loc=p["student_loc"], scale=p["student_scale"]))
            rows.append({"Bank": bank, "Loss cutoff (%)": cutoff*100, "Empirical probability": empirical,
                         "Normal probability": normal, "Student-t probability": student,
                         "Empirical / Normal": empirical/normal if normal > 0 else np.nan,
                         "Empirical / Student-t": empirical/student if student > 0 else np.nan})
    return pd.DataFrame(rows)
