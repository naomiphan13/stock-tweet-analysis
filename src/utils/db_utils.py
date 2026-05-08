from typing import List
import sqlite3

import pandas as pd


def create_df(query: str, conn: sqlite3.Connection) -> pd.DataFrame:
    """Execute a SQL query and return the result as a DataFrame."""
    return pd.read_sql_query(query, conn)


def get_companies_list(tweet_conn: sqlite3.Connection) -> List[str]:
    """Return a list of unique stock symbols from the `stocks` table."""
    query = "SELECT symbol FROM stocks"
    companies_df = create_df(query, tweet_conn)
    return companies_df["symbol"].dropna().unique().tolist()


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


def load_tweets_with_stocks(tweet_conn: sqlite3.Connection) -> pd.DataFrame:
    """Load tweets joined with their associated stock symbols."""
    query = """
        SELECT
            tweet_stocks.stock_symbol AS stock,
            tweets.text AS tweet,
            DATE(tweets.created_at) AS date
        FROM tweets
        JOIN tweet_stocks ON tweets.id = tweet_stocks.tweet_id
        JOIN users ON tweets.user_id = users.id
        ORDER BY date;
    """
    df = create_df(query, tweet_conn)
    df["date"] = pd.to_datetime(df["date"])
    return df
