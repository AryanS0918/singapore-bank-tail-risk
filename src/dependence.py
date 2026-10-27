"""Covariance, correlation, joint and conditional downside-loss probabilities."""
from __future__ import annotations

import numpy as np
import pandas as pd


def dependence_matrices(returns: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "covariance": returns.cov(),
        "pearson": returns.corr(method="pearson"),
        "spearman": returns.corr(method="spearman"),
    }


def compute_joint_crashes(
    returns: pd.DataFrame,
    k_thresholds: list[float] | None = None,
) -> dict[str, pd.DataFrame]:

    if k_thresholds is None:
        k_thresholds = [2.0, 2.5, 3.0]

    results = []
    pairwise = []
    conditional = []
    trio = []

    banks = list(returns.columns)

    mu = returns.mean()
    sigma = returns.std(ddof=1)

    # Calculate downside-crash masks for every selected bank.
    for k in k_thresholds:
        masks = {
            bank: (returns[bank] - mu[bank]) <= -k * sigma[bank]
            for bank in banks
        }

        # Pairwise and conditional crash probabilities.
        for i, bank_a in enumerate(banks):
            for bank_b in banks[i + 1:]:
                both = masks[bank_a] & masks[bank_b]

                count_both = int(both.sum())
                count_a = int(masks[bank_a].sum())
                count_b = int(masks[bank_b].sum())

                pairwise.append({
                    "k": k,
                    "Bank A": bank_a,
                    "Bank B": bank_b,
                    "Joint probability": float(both.mean()),
                    "P(A crash | B crash)": (
                        count_both / count_b
                        if count_b > 0 else np.nan
                    ),
                    "P(B crash | A crash)": (
                        count_both / count_a
                        if count_a > 0 else np.nan
                    ),
                    "Independent benchmark": float(
                        masks[bank_a].mean() * masks[bank_b].mean()
                    ),
                })

        # Three-bank crashes are only defined when all three banks exist.
        if len(banks) == 3:
            all_three = (
                masks[banks[0]]
                & masks[banks[1]]
                & masks[banks[2]]
            )

            trio.append({
                "k": k,
                "All-three joint probability": float(all_three.mean()),
                "All-three crash days": int(all_three.sum()),
                "Trading days": len(returns),
            })

            for date in returns.index[all_three]:
                results.append({
                    "Date": pd.Timestamp(date).date().isoformat(),
                    "k": k,
                    **{
                        f"{bank} return (%)": float(
                            returns.loc[date, bank] * 100
                        )
                        for bank in banks
                    },
                })

    return {
        "pairwise": pd.DataFrame(
            pairwise,
            columns=[
                "k",
                "Bank A",
                "Bank B",
                "Joint probability",
                "P(A crash | B crash)",
                "P(B crash | A crash)",
                "Independent benchmark",
            ],
        ),
        "trio": pd.DataFrame(
            trio,
            columns=[
                "k",
                "All-three joint probability",
                "All-three crash days",
                "Trading days",
            ],
        ),
        "dates": pd.DataFrame(
            results,
            columns=[
                "Date",
                "k",
                *[f"{bank} return (%)" for bank in banks],
            ],
        ),
    }