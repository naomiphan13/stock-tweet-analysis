from pathlib import Path
from sqlalchemy import create_engine
import sqlite3
import pandas as pd

# Resolve paths relative to this script's location (src/ → project root → data/)
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

stocktr_conn = sqlite3.connect(DATA_DIR / 'StocksTraining.db')
stockte_conn = sqlite3.connect(DATA_DIR / 'StocksTesting.db')

query = """
        SELECT ticker AS company, DATE(date) as date, open, high, low, close, adj_close, volume
        FROM prices JOIN companies ON prices.ticker_id = companies.id
        """

stocktr_df = pd.read_sql_query(query, stocktr_conn)
stockte_df = pd.read_sql_query(query, stockte_conn)

stock_prices_df = pd.concat([stocktr_df, stockte_df])

# For SQLite:
engine = create_engine(f'sqlite:///{DATA_DIR / "StockPrices.db"}')

stock_prices_df.to_sql(name='StockPrices', con=engine, if_exists='replace', index=False)
