from typing import List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import sqlite3

from config import FIN_SENT_MODEL_NAME, FIN_SENT_BATCH_SIZE
from .db_utils import load_tweets_with_stocks


def load_sentiment_model(model_name: str = FIN_SENT_MODEL_NAME) -> Tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    """
    Load a HuggingFace sentiment model (FinTwitBERT) and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model


def compute_fintwit_sentiment_batched(
    texts: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    batch_size: int = FIN_SENT_BATCH_SIZE,
) -> List[float]:
    """
    Compute FinTwitBERT-sentiment scores in batches.

    Assumes a 3-class setup with labels like:
      - "bearish"
      - "neutral"
      - "bullish"

    Returns a scalar sentiment score per tweet in [-1, 1],
    where:
      bearish -> -1, neutral -> 0, bullish -> +1
    using the class probabilities as weights.
    """
    scores: List[float] = []
    model.eval()

    # Build label -> index mapping from model config
    id2label = model.config.id2label
    label2id = {v.lower(): int(k) for k, v in id2label.items()}

    # Try to find the indices for bearish/neutral/bullish robustly
    try:
        idx_bear = label2id["bearish"]
        idx_neu = label2id["neutral"]
        idx_bull = label2id["bullish"]
    except KeyError as e:
        raise ValueError(f"Expected labels 'bearish', 'neutral', 'bullish' in model.config.id2label, got: {id2label}") from e

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

        # Probabilities per class
        bear = probs[:, idx_bear]
        neu = probs[:, idx_neu]
        bull = probs[:, idx_bull]

        # Expected value over {-1, 0, +1}
        batch_scores = (-1.0 * bear + 0.0 * neu + 1.0 * bull).cpu().tolist()
        scores.extend(batch_scores)

    return scores

def compute_daily_average_sentiment(
    tweet_conn: sqlite3.Connection,
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
) -> pd.DataFrame:
    """
    Compute daily average FinTwitBERT-sentiment score per stock symbol.
    Returns a DataFrame with columns: ['stock', 'date', 'average_sentiment'].

    average_sentiment is in [-1, 1] where:
      -1 ~ bearish, 0 ~ neutral, +1 ~ bullish (on average).
    """
    tweets_df = load_tweets_with_stocks(tweet_conn)

    scores = compute_fintwit_sentiment_batched(
        tweets_df["tweet"].astype(str).tolist(),
        tokenizer,
        model,
        batch_size=FIN_SENT_BATCH_SIZE,
    )
    tweets_df["sentiment"] = scores

    daily_avg_sent_df = (
        tweets_df
        .groupby(["stock", "date"], as_index=False)["sentiment"]
        .mean()
        .rename(columns={"sentiment": "average_sentiment"})
    )
    return daily_avg_sent_df