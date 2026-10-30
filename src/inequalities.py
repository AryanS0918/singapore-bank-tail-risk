
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from src.config import K_VALUES

def chebyshev_table(returns: pd.DataFrame, fits: dict, k_values=K_VALUES) -> pd.DataFrame:
    rows = []
    for bank in returns.columns:
        x = returns[bank].dropna().to_numpy(dtype=float)
        mu, sigma = float(np.mean(x)), float(np.std(x, ddof=1))
        p = fits[bank]
        for k in k_values:
            cutoff = mu - k*sigma
            empirical = float(np.mean(x <= cutoff))
            normal_prob = float(stats.norm.cdf(cutoff, p["normal_mu"], p["normal_sigma"]))
            student_prob = float(stats.t.cdf(cutoff, p["student_df"], loc=p["student_loc"], scale=p["student_scale"]))
            rows.append({
                "Bank": bank, "k": k, "Threshold return (%)": cutoff*100,
                "Empirical downside probability": empirical,
                "Chebyshev upper bound": 1/(k*k),
                "Cantelli one-sided bound": 1/(1+k*k),
                "Normal implied probability": normal_prob,
                "Student-t implied probability": student_prob,
                "Empirical / Normal": empirical/normal_prob if normal_prob > 0 else np.nan
            })
    return pd.DataFrame(rows)
