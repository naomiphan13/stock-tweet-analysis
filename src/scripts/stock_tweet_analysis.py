import sqlite3
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from statsmodels.tsa.stattools import grangercausalitytests


# CONFIRGURATION

_BASE_DIR     = Path(__file__).resolve().parent.parent
_DATA_DIR     = _BASE_DIR / "data"
_PLOTS_DIR    = _BASE_DIR / "plots"
_RESULT_DIR   = _BASE_DIR / "result"

_PLOTS_DIR.mkdir(exist_ok=True)
_RESULT_DIR.mkdir(exist_ok=True)

TWEET_DB_PATH = _DATA_DIR / "stock_influencer_tweets.db"
STOCK_DB_PATH = _DATA_DIR / "StockPrices.db"

FINBERT_MODEL_NAME = "ProsusAI/finbert"
FINBERT_BATCH_SIZE = 32


# DATABASE UTIL 

def create_df(query: str, conn: sqlite3.Connection) -> pd.DataFrame:
    """Execute a SQL query and return the result as a DataFrame."""
    return pd.read_sql_query(query, conn)


def get_companies_list(tweet_conn: sqlite3.Connection) -> List[str]:
    """Return a list of unique stock symbols from the `stocks` table."""
    query = "SELECT symbol FROM stocks"
    companies_df = create_df(query, tweet_conn)
    return companies_df["symbol"].dropna().unique().tolist()


# STOCK TIME-SERIES UTIL

def filter_stock(stock_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Filter the stock price DataFrame for a single ticker."""
    return stock_df[stock_df["company"] == ticker].copy()


def reindex_daily(df: pd.DataFrame, fill_value: float | None = np.nan) -> pd.DataFrame:
    """
    Reindex a DataFrame with a 'date' column to a continuous daily index.
    Optionally fill missing values.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    start_date = df.index.min()
    end_date = df.index.max()
    full_index = pd.date_range(start=start_date, end=end_date, freq="D")

    df_ts = df.reindex(full_index)

    if fill_value is not None:
        df_ts = df_ts.fillna(fill_value)

    df_ts.index.name = "date"
    return df_ts


def fill_missing_prices_with_rolling_mean(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Fill missing numeric columns with a rolling mean."""
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            rolling_mean = df[col].rolling(window=window, min_periods=1).mean()
            df[col] = df[col].fillna(rolling_mean)
    return df


def preprocess_stock_prices(stock_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    For a given ticker:
      - filter stock data
      - reindex to daily frequency
      - fill missing prices
      - compute 1-day percentage change
    """
    stock_df = filter_stock(stock_df, ticker)
    if stock_df.empty:
        return stock_df

    stock_df = reindex_daily(stock_df, fill_value=np.nan)
    stock_df = fill_missing_prices_with_rolling_mean(stock_df)

    if "close" not in stock_df.columns:
        raise ValueError(f"Column 'close' is missing for ticker {ticker}.")

    stock_df["oneDayChange"] = stock_df["close"].pct_change(fill_method=None)
    return stock_df


# PLOTTING – basic tweet statistics

def plot_num_tweets_per_stock(tweet_conn: sqlite3.Connection) -> None:
    """Plot total tweet count per stock symbol."""
    query = """
        SELECT stock_symbol, COUNT(tweet_id) AS num_tweets
        FROM tweet_stocks
        GROUP BY stock_symbol;
    """
    df = create_df(query, tweet_conn)

    plt.figure(figsize=(10, 5))
    plt.bar(df["stock_symbol"], df["num_tweets"])
    plt.ylabel("Number of Tweets")
    plt.xlabel("Stock Symbol")
    plt.title("Number of Tweets per Stock")
    plt.tight_layout()
    plt.show()


def plot_num_tweets_by_stock_and_user(tweet_conn: sqlite3.Connection) -> None:
    """Plot tweet counts per stock per influencer as a grouped bar chart."""
    query = """
        SELECT
            stock_symbol AS company,
            SUM(CASE WHEN users.name = 'Cassandra Unchained' THEN 1 ELSE 0 END) AS "Cassandra Unchained",
            SUM(CASE WHEN users.name = 'Cathie Wood' THEN 1 ELSE 0 END) AS "Cathie Wood",
            SUM(CASE WHEN users.name = 'Donald J. Trump' THEN 1 ELSE 0 END) AS "Donald J. Trump",
            SUM(CASE WHEN users.name = 'Elon Musk' THEN 1 ELSE 0 END) AS "Elon Musk",
            SUM(CASE WHEN users.name = 'Jim Cramer' THEN 1 ELSE 0 END) AS "Jim Cramer",
            SUM(CASE WHEN users.name = 'Ray Dalio' THEN 1 ELSE 0 END) AS "Ray Dalio"
        FROM tweet_stocks
        JOIN tweets ON tweet_stocks.tweet_id = tweets.id
        JOIN users ON users.id = tweets.user_id
        GROUP BY stock_symbol
        ORDER BY stock_symbol;
    """
    df = create_df(query, tweet_conn)

    stocks = df["company"]
    influencer_names = df.columns[1:]  # skip "company"

    x = np.arange(len(stocks))
    width = 0.1

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, name in enumerate(influencer_names):
        ax.bar(x + i * width, df[name].values, width=width, label=name)

    ax.set_xticks(x + width * (len(influencer_names) - 1) / 2)
    ax.set_xticklabels(stocks, rotation=45, ha="right")
    ax.set_ylabel("Number of Tweets")
    ax.set_xlabel("Stock Symbol")
    ax.set_title("Number of Tweets per Stock and Influencer")
    ax.legend()
    fig.tight_layout()
    plt.show()


# PLOTTING – tweets vs price change time series

def load_stock_prices(stock_conn: sqlite3.Connection) -> pd.DataFrame:
    """Load all stock prices."""
    query = "SELECT * FROM StockPrices"
    return create_df(query, stock_conn)


def load_daily_tweet_counts(tweet_conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Load daily tweet counts for a set of hard-coded tickers.
    """
    query = """
        SELECT
            DATE(created_at) AS date,
            SUM(CASE WHEN tweet_stocks.stock_symbol = 'NVDA' THEN 1 ELSE 0 END) AS NVDA,
            SUM(CASE WHEN tweet_stocks.stock_symbol = 'AAPL' THEN 1 ELSE 0 END) AS AAPL,
            SUM(CASE WHEN tweet_stocks.stock_symbol = 'MSFT' THEN 1 ELSE 0 END) AS MSFT,
            SUM(CASE WHEN tweet_stocks.stock_symbol = 'GOOG' THEN 1 ELSE 0 END) AS GOOG,
            SUM(CASE WHEN tweet_stocks.stock_symbol = 'AMZN' THEN 1 ELSE 0 END) AS AMZN,
            SUM(CASE WHEN tweet_stocks.stock_symbol = 'AVGO' THEN 1 ELSE 0 END) AS AVGO,
            SUM(CASE WHEN tweet_stocks.stock_symbol = 'META' THEN 1 ELSE 0 END) AS META,
            SUM(CASE WHEN tweet_stocks.stock_symbol = 'TSM' THEN 1 ELSE 0 END) AS TSM,
            SUM(CASE WHEN tweet_stocks.stock_symbol = 'TSLA' THEN 1 ELSE 0 END) AS TSLA,
            SUM(CASE WHEN tweet_stocks.stock_symbol = 'TCEHY' THEN 1 ELSE 0 END) AS TCEHY
        FROM tweets
        JOIN tweet_stocks ON tweets.id = tweet_stocks.tweet_id
        GROUP BY date
        ORDER BY date;
    """
    df = create_df(query, tweet_conn)
    df["date"] = pd.to_datetime(df["date"])
    return df


def plot_tweet_vs_price_time_series(
    tweet_conn: sqlite3.Connection,
    stock_conn: sqlite3.Connection,
    companies: List[str],
) -> None:
    """
    For each ticker:
      - preprocess stock prices
      - align daily tweet counts
      - plot % price change (line) vs tweet count (bars)
    """
    stock_prices_df = load_stock_prices(stock_conn)
    tweet_counts_df = load_daily_tweet_counts(tweet_conn)

    for ticker in companies:
        # --- stock data ---
        stock_df_full = preprocess_stock_prices(stock_prices_df, ticker)
        if stock_df_full.empty:
            print(f"[INFO] No stock data for {ticker}, skipping.")
            continue

        stock_dates = stock_df_full.index

        # --- tweet counts ---
        if ticker not in tweet_counts_df.columns:
            print(f"[INFO] No tweet count column for {ticker}, skipping.")
            continue

        tweet_df = tweet_counts_df[["date", ticker]].copy()
        tweet_df_full = reindex_daily(tweet_df, fill_value=0)
        tweet_dates = tweet_df_full.index

        # --- plotting ---
        fig, ax1 = plt.subplots(figsize=(12, 6))

        ax1.plot(
            stock_dates,
            stock_df_full["oneDayChange"],
            label="1-Day % Change",
            linewidth=2,
        )
        ax1.set_ylabel("Percentage Change (%)", color="red")
        ax1.tick_params(axis="y", labelcolor="red")

        ax2 = ax1.twinx()
        ax2.bar(
            tweet_dates,
            tweet_df_full[ticker],
            alpha=0.3,
            width=2,
            color="green",
            label="Tweet Count",
        )
        ax2.set_ylabel("Tweet Count", color="blue")
        ax2.tick_params(axis="y", labelcolor="blue")

        plt.title(f"{ticker}: Stock % Change vs Number of Tweets Over Time")
        fig.tight_layout()
        plt.savefig(_PLOTS_DIR / f"{ticker}_tweet_price_plot.png", dpi=300)
        plt.close(fig)



# FINBERT SENTIMENT ANALYSIS

def load_finbert(model_name: str = FINBERT_MODEL_NAME) -> Tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    """Load FinBERT tokenizer and model once."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model


def compute_finbert_scores_batched(
    texts: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    batch_size: int = FINBERT_BATCH_SIZE,
) -> List[float]:
    """
    Compute FinBERT sentiment scores (confidence of predicted class) in batches
    for runtime efficiency.
    """
    scores: List[float] = []
    model.eval()

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
        )
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        max_probs, _ = probs.max(dim=-1)
        scores.extend(max_probs.cpu().tolist())

    return scores


def load_tweets_with_stocks(tweet_conn: sqlite3.Connection) -> pd.DataFrame:
    """Load tweets joined with their associated stock symbols."""
    query = """
        SELECT
            stock_symbol AS stock,
            tweets.text AS tweet,
            DATE(created_at) AS date
        FROM tweets, tweet_stocks, users
        WHERE tweets.id = tweet_stocks.tweet_id
            AND tweets.user_id = users.id
        ORDER BY date;
    """
    df = create_df(query, tweet_conn)
    df["date"] = pd.to_datetime(df["date"])
    return df


def compute_daily_average_sentiment(
    tweet_conn: sqlite3.Connection,
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
) -> pd.DataFrame:
    """
    Compute daily average FinBERT sentiment per stock symbol.
    Returns a DataFrame with columns: ['stock', 'date', 'average_sentiment']
    """
    tweets_df = load_tweets_with_stocks(tweet_conn)

    # Compute sentiment scores in batches for efficiency
    scores = compute_finbert_scores_batched(
        tweets_df["tweet"].astype(str).tolist(),
        tokenizer,
        model,
        batch_size=FINBERT_BATCH_SIZE,
    )
    tweets_df["sentiment"] = scores

    daily_avg_sent_df = (
        tweets_df
        .groupby(["stock", "date"], as_index=False)["sentiment"]
        .mean()
        .rename(columns={"sentiment": "average_sentiment"})
    )
    return daily_avg_sent_df


def build_sentiment_return_merged(
    daily_avg_sent_df: pd.DataFrame,
    stock_prices_df: pd.DataFrame,
    ticker: str,
    lags: list[int] = [1]
) -> pd.DataFrame:
    """
    For a given ticker, merge:
      - 1-day stock returns
      - lagged daily average tweet sentiment
    Returns a DataFrame indexed by date with columns:
      ['oneDayChange', 'average_sentiment', 'sent_lag1']
    """
        
    # Stock side
    stock_df_full = preprocess_stock_prices(stock_prices_df, ticker)
    if stock_df_full.empty:
        return pd.DataFrame()

    returns_df = stock_df_full[["oneDayChange"]].copy()

    # Sentiment side
    sent_filtered_df = daily_avg_sent_df[daily_avg_sent_df["stock"] == ticker].copy()
    if sent_filtered_df.empty:
        return pd.DataFrame()

    sent_filtered_df = sent_filtered_df[["date", "average_sentiment"]]
    sent_filtered_df = reindex_daily(sent_filtered_df, fill_value=0)

    for lag in lags:
        sent_filtered_df[f"sent_lag_{lag}"] = sent_filtered_df["average_sentiment"].shift(lag)

    cols = ["average_sentiment"] + [f"sent_lag_{lag}" for lag in lags]

    merged = returns_df.join(
        sent_filtered_df[cols], 
        how="left")
    
    merged = merged.dropna(subset=["oneDayChange"])
    return merged

import pandas as pd

# Granger Causality Test:

# Test for stationarity:
from statsmodels.tsa.stattools import adfuller
def adf_test(series):
    """
    One of the critical assumption of Granger Causality Test is that both time-series must be stationary.
    Stationary means the series' statistical properties (like mean and variance) are constant over time.
    Non-stationary series must first be transformed, typically by differecing.
    """
    result = adfuller(series.dropna())
    print('ADF Statistic: ', result[0])
    print('p-value: ', result[1])
    print('Critical Values: ')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')

    if result[1] <= 0.05:
        print("Reject the null hypothesis, the series is likely stationary.")
    else:
        print("Fail to reject the null hypothesis, the series is likely non-stationary.")

def summarize_granger(result, alpha=0.05):
    """
    Turn statsmodels.grangercausalitytests result into a readable DataFrame.
    """
    rows = []

    for lag, (tests_dict, _) in result.items():
        for test_name, values in tests_dict.items():
            stat = values[0]
            pval = values[1]
            df = values[2] if len(values) > 2 else None

            rows.append({
                "lag": lag,
                "test": test_name,
                "stat": float(stat),
                "p_value": float(pval),
                "df": int(df) if df is not None else None,
                "significant": pval < alpha
            })

    summary_df = pd.DataFrame(rows)
    summary_df = summary_df.sort_values(["lag", "test"]).reset_index(drop=True)
    return summary_df

# Sentiment vs Volatility:
from arch import arch_model

def fit_GARCH(df, ticker):
    model = arch_model(
        y=df["oneDayChange"],
        x=df["average_sentiment"],
        vol="GARCH",
        p = 1,
        q=1
    )

    res = model.fit(disp="off")

    return res

# MAIN

def main() -> None:
    with sqlite3.connect(TWEET_DB_PATH) as tweet_conn, sqlite3.connect(STOCK_DB_PATH) as stock_conn:
        # Basic plots
        plot_num_tweets_per_stock(tweet_conn)
        plot_num_tweets_by_stock_and_user(tweet_conn)

        companies = get_companies_list(tweet_conn)
        stock_prices_df = load_stock_prices(stock_conn)

        # Plot tweet volume vs price change
        plot_tweet_vs_price_time_series(tweet_conn, stock_conn, companies)

        # Sentiment analysis
        tokenizer, model = load_finbert()
        daily_avg_sent_df = compute_daily_average_sentiment(tweet_conn, tokenizer, model)

        correlation_rows = []  # store results here
        granger_rows = []
        garch_rows = []

        for ticker in companies:
            merged = build_sentiment_return_merged(
                    daily_avg_sent_df, stock_prices_df, ticker, lags=list(range(0, 6))
                )

            if merged.empty:
                print(f"[INFO] No merged data for {ticker}, skipping.")
                continue
            
            # Sentiment and Volatility Analysis
            GARCH_result = fit_GARCH(merged, ticker)
            print(f"====== GARCH result summary for: {ticker} ======")
            print(GARCH_result.summary())

            params = GARCH_result.params
            pvals = GARCH_result.pvalues

            garch_rows.append({
                "ticker": ticker,
                "p": 1,
                "q": 1,
                "mu": params.get("mu", None),
                "mu_pval": pvals.get("mu", None),
                "omega": params.get("omega", None),
                "omega_pval": pvals.get("omega", None),
                "alpha1": params.get("alpha[1]", None),
                "alpha1_pval": pvals.get("alpha[1]", None),
                "beta": params.get("beta[1]", None),
                "beta_pval": pvals.get("beta[1]", None),
                "loglik": GARCH_result.loglikelihood,
                "AIC": GARCH_result.aic,
                "BIC": GARCH_result.bic,
            })


            for lag in range(0, 6):
                # Compute correlation value
                corr_val = merged[["oneDayChange", f"sent_lag_{lag}"]].corr().iloc[0, 1]

                # Append to results table
                correlation_rows.append({
                    "ticker": ticker,
                    "lag": lag,
                    "correlation": corr_val,
                    "n_samples": len(merged)
                })

                # Print to screen
                print(f"{ticker}: lag={lag}, corr={corr_val:.4f}, n={len(merged)}")

                if lag == 1:
                    # Stationary test:
                    print(f"Running stationary test for stock price changes for {ticker}")
                    adf_test(merged[["oneDayChange"]])

                    print(f"Running stationary test for sentiment data for {ticker}")
                    adf_test(merged[[f"sent_lag_{lag}"]])

                    # Granger causality test:
                    granger_result = grangercausalitytests(merged[["oneDayChange", f"sent_lag_{lag}"]], maxlag=5, verbose=True)
                    summary = summarize_granger(granger_result)
                    print(f"Granger causality test result: {summary}")

                    granger_rows.append({
                        "ticker": ticker,
                        "lag": summary["lag"],
                        "test": summary["test"],
                        "stat": summary["stat"],
                        "p_value": summary["p_value"],
                        "significant": summary["significant"]
                    })
                

        # Create correlation df
        corr_df = pd.DataFrame(correlation_rows)
        summary_df = pd.DataFrame(granger_rows)
        GARCH_df = pd.DataFrame(garch_rows)

        # Save to csv
        corr_df.to_csv(_RESULT_DIR / "sentiment_price_correlations_v2.csv", index=False)
        summary_df.to_csv(_RESULT_DIR / "granger_causality_test_result_v2.csv", index=False)
        GARCH_df.to_csv(_RESULT_DIR / "GARCH_result.csv", index=False)


if __name__ == "__main__":
    main()
