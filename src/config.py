from pathlib import Path

# Project root (one level up from src/)
BASE_DIR = Path(__file__).resolve().parent.parent

# Paths
DATA_DIR   = BASE_DIR / "data"
PLOTS_DIR  = BASE_DIR / "plots"
RESULT_DIR = BASE_DIR / "result"

TWEET_DB_PATH = DATA_DIR / "stock_influencer_tweets.db"
STOCK_DB_PATH = DATA_DIR / "StockPrices.db"

# FINTWITBERT
FIN_SENT_MODEL_NAME = "StephanAkkerman/FinTwitBERT-sentiment"
FIN_SENT_BATCH_SIZE = 32


# Output
PLOTS_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

# Analysis
MAX_LAG = 5          # for Granger
SENTIMENT_LAGS = list(range(0, 6))  # 0–5 days
ROLLING_MEAN_WINDOW = 3
