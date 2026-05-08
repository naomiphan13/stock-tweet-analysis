from typing import List
import sqlite3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from .db_utils import create_df, load_daily_tweet_counts
from .stock_utils import preprocess_stock_prices, reindex_daily
from config import PLOTS_DIR


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


def plot_tweet_vs_price_time_series(
    tweet_conn: sqlite3.Connection,
    stock_prices_df: pd.DataFrame,
    companies: List[str],
) -> None:
    """
    For each ticker:
      - preprocess stock prices
      - align daily tweet counts
      - plot % price change (line) vs tweet count (bars)
    """
    tweet_counts_df = load_daily_tweet_counts(tweet_conn)

    for ticker in companies:
        stock_df_full = preprocess_stock_prices(stock_prices_df, ticker)
        if stock_df_full.empty:
            print(f"[INFO] No stock data for {ticker}, skipping.")
            continue

        stock_dates = stock_df_full.index

        if ticker not in tweet_counts_df.columns:
            print(f"[INFO] No tweet count column for {ticker}, skipping.")
            continue

        tweet_df = tweet_counts_df[["date", ticker]].copy()
        tweet_df_full = reindex_daily(tweet_df, fill_value=0)
        tweet_dates = tweet_df_full.index

        fig, ax1 = plt.subplots(figsize=(12, 6))

        ax1.plot(
            stock_dates,
            stock_df_full["oneDayChange"],
            label="1-Day % Change",
            linewidth=2,
        )
        ax1.set_ylabel("Percentage Change", color="red")
        ax1.tick_params(axis="y", labelcolor="red")

        ax2 = ax1.twinx()
        ax2.bar(
            tweet_dates,
            tweet_df_full[ticker],
            alpha=0.3,
            width=2,
            label="Tweet Count",
        )
        ax2.set_ylabel("Tweet Count", color="blue")
        ax2.tick_params(axis="y", labelcolor="blue")

        plt.title(f"{ticker}: Stock % Change vs Number of Tweets Over Time")
        fig.tight_layout()
        out_path = PLOTS_DIR / f"{ticker}_tweet_price_plot.png"
        plt.savefig(out_path, dpi=300)
        plt.close(fig)
