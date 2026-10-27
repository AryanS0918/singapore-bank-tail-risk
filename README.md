
# Heavy Tails & Extreme Risk in Singapore's Banking Trio (DBS, OCBC, UOB)

**Course:** DS603 Probability for Data Science | **Team:** Aryan Singh & Nikhil Balaji  
**Period:** 1 January 2014–31 December 2024

## Research question
To what extent do empirical daily log-return loss tails of DBS, OCBC and UOB deviate from fitted Normal and Student-t distributions, how conservatively does Chebyshev's inequality bound observed downside exceedances, and how often do extreme losses occur jointly across the banking sector?

## Features
- Streamlit dashboard with historical performance, descriptive statistics, distribution fits, tail quantiles, Chebyshev comparisons, correlation heatmaps and co-crash analysis.
- Modular, reusable code in `src/`.
- CSV and Markdown tables plus high-resolution PNG figures.
- Downloadable cleaned return series from the dashboard.
- Data pipeline uses adjusted prices, common-date alignment, and daily log returns.
- Fixed project seed `SEED = 603` for reproducibility settings. Distribution fitting itself is deterministic for the same input sample.

## Repository structure
```text
.
├── app.py
├── README.md
├── AI_USAGE.md
├── CONTRIBUTIONS.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── notebooks/
│   ├── 01_data_prep.ipynb
│   ├── 02_univariate_tails_and_bounds.ipynb
│   └── 03_multivariate_dependence.ipynb
├── src/
│   ├── config.py
│   ├── data_pipeline.py
│   ├── distributions.py
│   ├── inequalities.py
│   ├── dependence.py
│   ├── visualize.py
│   └── run_analysis.py
└── outputs/
    ├── figures/
    ├── tables/
    └── presentation/
```

## Run in VS Code / terminal
Use Python 3.11 or 3.12.

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd singapore-bank-tail-risk
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Launch the interactive dashboard:
```bash
streamlit run app.py
```
Run the full batch analysis and export figures/tables:
```bash
python -m src.run_analysis
```
To re-download prices instead of using the cached raw price file, use:
```bash
python -c "from src.run_analysis import main; main(refresh=True)"
```

## Run in Google Colab
1. Upload the repository ZIP to Colab and unzip it, or clone your GitHub repository.
2. Change directory to the repository root.
3. Install packages with `!pip install -r requirements.txt`.
4. Run batch analysis with `!python -m src.run_analysis`.
5. To run the Streamlit UI in Colab, use a tunnel service approved by your course/institution; the easiest and most reliable presentation route is to run Streamlit locally in VS Code. Do not expose private credentials in notebook cells.

Example Colab cells:
```python
!git clone <YOUR_GITHUB_REPOSITORY_URL>
%cd singapore-bank-tail-risk
!pip install -r requirements.txt
!python -m src.run_analysis
```

## Mathematical methods

### 1. Daily log returns
For adjusted close price \(P_t\), compute
\[
R_t = \ln(P_t/P_{t-1}).
\]
Log returns allow price changes to be analyzed on a comparable scale. The first observation is dropped because it has no previous price.

### 2. Descriptive moments
For each bank, report sample size, mean, standard deviation, skewness, excess kurtosis, minimum and maximum. Positive excess kurtosis is consistent with a more heavy-tailed sample, but is not by itself proof that a particular parametric model is correct.

### 3. Normal and Student-t maximum likelihood estimation
Fit Normal \((\hat\mu,\hat\sigma)\) and location-scale Student-t \((\hat\nu,\widehat{loc},\widehat{scale})\) to the **full return sample**, then evaluate the left tail. A lower fitted Student-t degrees-of-freedom estimate indicates heavier model tails. The Student-t fit is a model, not a guarantee that every empirical tail probability will be matched.

### 4. VaR quantile comparison
At tail probability \(q\), compare empirical quantile \(\hat Q(q)\) with Normal and Student-t model quantiles. Negative values are downside returns; e.g. a quantile of -3% represents a daily return cutoff of -3%, not a prediction that a loss will occur.

### 5. Chebyshev inequality
Define \(E_k=\{R-\mu\le-k\sigma\}\). Since \(E_k\subseteq\{|R-\mu|\ge k\sigma\}\),
\[
P(R-\mu\le-k\sigma)\le \frac{1}{k^2}.
\]
The dashboard compares the empirical frequency with \(1/k^2\), Normal model probability and fitted Student-t probability for \(k=2,2.5,3,4\). It also reports Cantelli's sharper one-sided bound \(1/(1+k^2)\) as an optional comparison. Bounds are upper limits, not forecasts.

### 6. Dependence and systemic risk
Estimate sample covariance, Pearson correlation and Spearman rank correlation. Define a bank crash mask \(I_i(k)=1\{R_i-\mu_i\le-k\sigma_i\}\). Estimate pairwise joint crash probability as the mean of \(I_iI_j\), conditional probability as \(P(I_i=1,I_j=1)/P(I_j=1)\), and all-three crash probability as the mean of \(I_1I_2I_3\). Conditional probabilities are undefined when the conditioning event never occurs; the code reports missing values in that case.

## Generated outputs
Running `python -m src.run_analysis` creates:
- `outputs/figures/fig1_return_distributions.png`
- `outputs/figures/fig2_qq_plots.png`
- `outputs/figures/fig3_chebyshev_comparison.png`
- `outputs/figures/fig4_correlation_heatmaps.png`
- `outputs/figures/fig5_growth_of_100.png`
- `outputs/figures/fig6_daily_returns.png`
- CSV and Markdown tables under `outputs/tables/`

## Interpretation guide
- **Normal vs Student-t:** inspect both Q-Q plots and tail quantiles. Strong departures from the Normal reference line in the lower tail suggest Gaussian underestimation may be a concern.
- **Chebyshev:** expect a broad distribution-free bound; its conservativeness should be demonstrated from the actual output, not assumed in advance.
- **Dependence:** high Pearson/Spearman correlation indicates co-movement overall, but does not alone prove tail dependence. Compare conditional crash probabilities with the independent benchmark and report the observed event counts.
- **Historical context:** investigate dates in the generated joint-crash table against documented market events. Do not label a crash as caused by COVID-19, oil stress or rate changes without checking the date and supporting evidence.

## Limitations and data ethics
- Yahoo Finance is a public third-party data source; data availability and adjustments may change. Cite the source and record the date of download in your final presentation.
- Do not commit data obtained from SMU-licensed Datastream/LSEG Workspace if your licence prohibits redistribution. This repository ignores generated CSV data by default.
- Daily returns are treated as an unconditional sample; volatility clustering and time-varying volatility are not modeled (no ARCH/GARCH).
- Extreme-tail estimates are noisy because very few observations occur at very low quantiles.
- The period is historical and excludes data beyond 2024 by design.
- No EVT/GPD, bootstrap KS test, Chernoff bound or copula estimation is included, in line with the agreed project scope.

## Mandatory MDSE group work statement
> We confirm that this project reflects collaborative work by the group members. Contributions are documented in `CONTRIBUTIONS.md`; use of generative AI and verification steps are documented in `AI_USAGE.md`.

## Disclaimer
For coursework and statistical education only. Not investment advice.
