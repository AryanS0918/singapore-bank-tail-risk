
# Google Colab quick start

1. Upload the repository ZIP to Google Drive or clone the repository from GitHub.
2. In a Colab cell, clone/unzip the repository and change into its root directory.
3. Install dependencies and run the batch analysis.

```python
!git clone <YOUR_GITHUB_REPOSITORY_URL>
%cd singapore-bank-tail-risk
!pip install -r requirements.txt
!python -m src.run_analysis
```

To inspect results in Colab:
```python
import pandas as pd
display(pd.read_csv("outputs/tables/table1_summary_stats.csv"))
```

The Streamlit app is designed to run locally with `streamlit run app.py`. Colab does not expose a local web server automatically; use an institution-approved tunnel if you specifically need the live UI in Colab. For class presentation, VS Code is the recommended route.
