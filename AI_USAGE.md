
# AI Usage Disclosure

Generative AI assistance was used to help scaffold the repository, explain statistical formulas, draft documentation, and identify possible implementation issues.

## Tasks assisted
- Drafted modular Python structure and initial implementation for price ingestion, log returns, MLE distribution fitting, Chebyshev bounds, dependence statistics, visualization and a Streamlit dashboard.
- Helped formulate README explanations of the mathematical methods and limitations.
- Suggested a reproducible setup and output naming scheme.

## Verification required before submission
1. Re-run the full pipeline from a clean environment using `pip install -r requirements.txt`.
2. Inspect downloaded tickers, date bounds, missingness and adjusted-price handling.
3. Independently verify key equations and spot-check probabilities against SciPy.
4. Confirm generated tables and figures are based on the actual downloaded sample.
5. Review all claims and revise them to match measured results; do not claim that Student-t always wins or that a particular crisis caused a crash without evidence.
6. Follow course rules on AI disclosure and retain this file only if its wording matches the actual work completed by the team.
