import logging
import sqlite3

import pandas as pd

from config import TWEET_DB_PATH, STOCK_DB_PATH, SENTIMENT_LAGS, RESULT_DIR
from utils.db_utils import get_companies_list, load_stock_prices
from utils.plotting_utils import (
    plot_num_tweets_per_stock,
    plot_num_tweets_by_stock_and_user,
    plot_tweet_vs_price_time_series,
)
from utils.sentiment_utils import load_sentiment_model, compute_daily_average_sentiment
from utils.stock_utils import build_sentiment_return_merged
from utils.analysis_utils import analyze_ticker


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def main() -> None:
    with sqlite3.connect(TWEET_DB_PATH) as tweet_conn, sqlite3.connect(STOCK_DB_PATH) as stock_conn:
        logging.info("Loading company list...")
        companies = get_companies_list(tweet_conn)

        logging.info("Loading stock prices...")
        stock_prices_df = load_stock_prices(stock_conn)

        # Basic plots
        logging.info("Plotting tweet statistics...")
        plot_num_tweets_per_stock(tweet_conn)
        plot_num_tweets_by_stock_and_user(tweet_conn)

        logging.info("Plotting tweets vs price time series...")
        plot_tweet_vs_price_time_series(tweet_conn, stock_prices_df, companies)

        # Sentiment analysis
        logging.info("Loading FinTwitBERT...")
        tokenizer, model = load_sentiment_model()

        logging.info("Computing daily average sentiment...")
        daily_avg_sent_df = compute_daily_average_sentiment(tweet_conn, tokenizer, model)

        correlation_rows: list[dict] = []
        granger_rows: list[dict] = []
        garch_rows: list[dict] = []

        for ticker in companies:
            logging.info(f"Processing {ticker}...")
            merged = build_sentiment_return_merged(
                daily_avg_sent_df,
                stock_prices_df,
                ticker,
                lags=SENTIMENT_LAGS,
            )

            if merged.empty:
                logging.info(f"[INFO] No merged data for {ticker}, skipping.")
                continue

            corr_rows, gr_rows, garch_row = analyze_ticker(ticker, merged)

            correlation_rows.extend(corr_rows)
            granger_rows.extend(gr_rows)
            garch_rows.append(garch_row)

        # Create dataframes
        corr_df = pd.DataFrame(correlation_rows)
        granger_df = pd.DataFrame(granger_rows)
        garch_df = pd.DataFrame(garch_rows)

        # Save to CSV
        logging.info("Saving results to CSV...")
        corr_df.to_csv(RESULT_DIR / "sentiment_price_correlations_v2.csv", index=False)
        granger_df.to_csv(RESULT_DIR / "granger_causality_test_result_v2.csv", index=False)
        garch_df.to_csv(RESULT_DIR / "GARCH_result.csv", index=False)

        logging.info("Analysis complete.")


if __name__ == "__main__":
    main()
