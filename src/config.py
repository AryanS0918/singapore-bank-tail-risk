
from pathlib import Path

SEED = 603
START_DATE = "2014-01-01"
END_DATE = "2025-01-01"  # yfinance end date is exclusive
TICKERS = {"DBS": "D05.SI", "OCBC": "O39.SI", "UOB": "U11.SI"}
K_VALUES = [2.0, 2.5, 3.0, 4.0]
TAIL_QUANTILES = [0.05, 0.025, 0.01, 0.005, 0.001]
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
INTERIM_DIR = ROOT / "data" / "interim"
PROCESSED_DIR = ROOT / "data" / "processed"
FIGURES_DIR = ROOT / "outputs" / "figures"
TABLES_DIR = ROOT / "outputs" / "tables"

def ensure_directories():
    for path in [RAW_DIR, INTERIM_DIR, PROCESSED_DIR, FIGURES_DIR, TABLES_DIR]:
        path.mkdir(parents=True, exist_ok=True)
