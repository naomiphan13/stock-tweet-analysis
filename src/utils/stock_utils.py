from typing import Optional, List

import numpy as np
import pandas as pd

from config import ROLLING_MEAN_WINDOW
from .db_utils import load_stock_prices  # optional, if you want


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


def fill_missing_prices_with_rolling_mean(
    df: pd.DataFrame, window: int = ROLLING_MEAN_WINDOW
) -> pd.DataFrame:
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


def build_sentiment_return_merged(
    daily_avg_sent_df: pd.DataFrame,
    stock_prices_df: pd.DataFrame,
    ticker: str,
    lags: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    For a given ticker, merge:
      - 1-day stock returns
      - lagged daily average tweet sentiment

    Returns a DataFrame indexed by date with columns:
      ['oneDayChange', 'average_sentiment', 'sent_lag_0', 'sent_lag_1', ...]
    """
    if lags is None:
        lags = [1]

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

    merged = returns_df.join(sent_filtered_df[cols], how="left")

    # Drop rows with missing returns; allow sentiment lags to be NaN if needed
    merged = merged.dropna(subset=["oneDayChange"])
    return merged
