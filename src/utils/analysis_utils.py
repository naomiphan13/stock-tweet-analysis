from typing import Dict, Any, List

import pandas as pd
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from arch import arch_model

from config import MAX_LAG


def adf_test(series: pd.Series) -> Dict[str, Any]:
    """
    Run ADF test on a 1D series and return results as a dict.
    """
    result = adfuller(series.dropna())
    output = {
        "adf_stat": result[0],
        "p_value": result[1],
        "critical_values": result[4],
        "is_stationary": result[1] <= 0.05,
    }
    return output


def summarize_granger(result, alpha: float = 0.05) -> pd.DataFrame:
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
                "significant": pval < alpha,
            })

    summary_df = pd.DataFrame(rows)
    summary_df = summary_df.sort_values(["lag", "test"]).reset_index(drop=True)
    return summary_df


def fit_garch(df: pd.DataFrame):
    """
    Fit a GARCH(1,1) model:
    - mean depends on constant + average_sentiment
    - volatility is GARCH(1,1) on residuals
    """
    df = df.copy()
    df["scaled_return"] = 100* df["oneDayChange"]

    model = arch_model(
        y=df["scaled_return"],
        x=df["average_sentiment"],
        vol="GARCH",
        p=1,
        q=1,
    )
    res = model.fit(disp="off")
    return res


def analyze_ticker(
    ticker: str,
    merged: pd.DataFrame,
) -> tuple[list[dict], list[dict], dict]:
    """
    For a single ticker:
      - fit GARCH(1,1)
      - compute correlations for all lags
      - run ADF + Granger for lag 1
    Returns:
      correlation_rows, granger_rows, garch_row
    """
    correlation_rows: list[dict] = []
    granger_rows: list[dict] = []

    garch_result = fit_garch(merged)
    params = garch_result.params
    pvals = garch_result.pvalues

    garch_row = {
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
        "loglik": garch_result.loglikelihood,
        "AIC": garch_result.aic,
        "BIC": garch_result.bic,
    }

    # --- Correlations + Granger ---
    for col in merged.columns:
        if col.startswith("sent_lag_"):
            lag = int(col.split("_")[-1])

            corr_df = merged[["oneDayChange", col]].dropna()
            if corr_df.empty:
                continue

            corr_val = corr_df.corr().iloc[0, 1]

            correlation_rows.append({
                "ticker": ticker,
                "lag": lag,
                "correlation": corr_val,
                "n_samples": len(corr_df),
            })

            if lag == 1:
                yx = corr_df

                # Quick sample-size guard for Granger
                if len(yx) <= MAX_LAG + 1:
                    # Not enough data for this ticker/lag, skip Granger
                    continue

                # Stationarity checks on the cleaned series
                adf_price = adf_test(yx["oneDayChange"])
                adf_sent = adf_test(yx[col])

                # Granger (sentiment -> returns)
                granger_result = grangercausalitytests(
                    yx,            
                    maxlag=MAX_LAG,
                    verbose=False,
                )
                summary = summarize_granger(granger_result)

                for row in summary.itertuples(index=False):
                    granger_rows.append({
                        "ticker": ticker,
                        "lag": row.lag,
                        "test": row.test,
                        "stat": row.stat,
                        "p_value": row.p_value,
                        "significant": row.significant,
                        "adf_price_p": adf_price["p_value"],
                        "adf_price_stationary": adf_price["is_stationary"],
                        "adf_sent_p": adf_sent["p_value"],
                        "adf_sent_stationary": adf_sent["is_stationary"],
                    })

    return correlation_rows, granger_rows, garch_row
