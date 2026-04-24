from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


HORIZONS = [1, 7, 30, 90, 180]

HALVINGS = [
    pd.Timestamp("2012-11-28"),
    pd.Timestamp("2016-07-09"),
    pd.Timestamp("2020-05-11"),
    pd.Timestamp("2024-04-20"),
]

FACTOR_GROUPS: Dict[str, str] = {
    "trend": "Technical trend",
    "momentum": "Momentum",
    "volatility": "Volatility/risk",
    "volume": "Volume/liquidity",
    "macro": "Macro/risk appetite",
    "onchain": "On-chain/network",
    "sentiment": "Sentiment",
    "cycle": "Supply/cycle",
    "cross_asset": "Cross-asset",
}

FACTOR_EXPLANATIONS = [
    {
        "factor": "Trend and momentum",
        "positive": "Price above medium/long moving averages, positive 7-90 day returns, constructive MACD.",
        "negative": "Breaks below moving averages, large drawdowns, negative multi-week momentum.",
    },
    {
        "factor": "Volatility and stress",
        "positive": "Falling realized volatility after capitulation, lower VIX, stable drawdowns.",
        "negative": "Rising realized volatility, deeper rolling drawdowns, VIX spikes.",
    },
    {
        "factor": "Macro liquidity",
        "positive": "Falling yields/dollar pressure, improving equities, expanding broad liquidity.",
        "negative": "Higher real/nominal yields, stronger dollar, weak equities, tighter liquidity.",
    },
    {
        "factor": "On-chain demand",
        "positive": "Growing active addresses, transaction count, hash rate, and exchange outflows.",
        "negative": "Weakening network activity, exchange inflows, elevated exchange supply.",
    },
    {
        "factor": "Valuation",
        "positive": "Lower MVRV and less stretched ROI tend to improve forward asymmetry.",
        "negative": "Elevated MVRV, overheated 30d/1y ROI, and parabolic extensions often reduce reward/risk.",
    },
    {
        "factor": "Sentiment",
        "positive": "Recovering sentiment from fear can confirm demand; panic can be contrarian on longer horizons.",
        "negative": "Extreme greed, especially with overbought trend and high volatility, can mark crowded positioning.",
    },
]


@dataclass
class TrainingData:
    frame: pd.DataFrame
    feature_cols: List[str]
    target_col: str
    horizon: int


def _safe_log(series: pd.Series) -> pd.Series:
    return np.log(series.where(series > 0))


def _zscore(series: pd.Series, window: int = 365, min_periods: int = 90) -> pd.Series:
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std()
    return (series - mean) / std.replace(0, np.nan)


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    x = np.arange(window)

    def slope(values: np.ndarray) -> float:
        if np.isnan(values).any():
            return np.nan
        coef = np.polyfit(x, values, 1)[0]
        return coef

    return series.rolling(window, min_periods=window).apply(slope, raw=True)


def _last_halving_date(date: pd.Timestamp) -> pd.Timestamp:
    last = HALVINGS[0]
    for halving in HALVINGS:
        if date >= halving:
            last = halving
    return last


def add_technical_features(raw: pd.DataFrame, out: pd.DataFrame) -> None:
    price = raw["btc_close"]
    log_price = _safe_log(price)
    returns = log_price.diff()

    out["trend_log_price"] = log_price
    for window in [7, 14, 30, 60, 90, 180, 365]:
        out[f"momentum_ret_{window}d"] = log_price.diff(window)

    for window in [10, 20, 50, 100, 200]:
        ma = price.rolling(window, min_periods=max(5, window // 2)).mean()
        out[f"trend_dist_sma_{window}d"] = price / ma - 1

    ema_12 = price.ewm(span=12, adjust=False).mean()
    ema_26 = price.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    out["trend_macd_pct"] = macd / price
    out["trend_macd_signal_pct"] = macd.ewm(span=9, adjust=False).mean() / price
    out["momentum_rsi_14d"] = _rsi(price, 14) / 100

    for window in [7, 14, 30, 90]:
        out[f"volatility_realized_{window}d"] = returns.rolling(window, min_periods=max(5, window // 2)).std() * np.sqrt(365)

    rolling_high_90 = price.rolling(90, min_periods=30).max()
    rolling_high_365 = price.rolling(365, min_periods=120).max()
    out["volatility_drawdown_90d"] = price / rolling_high_90 - 1
    out["volatility_drawdown_365d"] = price / rolling_high_365 - 1
    out["trend_dist_ath"] = price / price.cummax() - 1

    if {"btc_high", "btc_low"}.issubset(raw.columns):
        out["volatility_intraday_range"] = (raw["btc_high"] - raw["btc_low"]) / price

    if "btc_volume" in raw:
        volume = raw["btc_volume"].replace(0, np.nan)
        out["volume_log"] = _safe_log(volume)
        out["volume_z_30d"] = _zscore(volume, 30, 15)
        out["volume_z_90d"] = _zscore(volume, 90, 30)
        out["volume_trend_30d"] = _rolling_slope(_safe_log(volume), 30)


def add_cross_asset_features(raw: pd.DataFrame, out: pd.DataFrame) -> None:
    close_cols = {
        "eth_close": "cross_asset_eth",
        "spx_close": "cross_asset_spx",
        "nasdaq_close": "cross_asset_nasdaq",
        "gold_close": "cross_asset_gold",
        "tlt_close": "cross_asset_tlt",
        "dxy_close": "macro_dxy",
        "vix_close": "macro_vix",
    }
    for column, prefix in close_cols.items():
        if column not in raw:
            continue
        series = raw[column].shift(1).replace(0, np.nan)
        log_series = _safe_log(series)
        out[f"{prefix}_ret_1d"] = log_series.diff(1)
        out[f"{prefix}_ret_7d"] = log_series.diff(7)
        out[f"{prefix}_ret_30d"] = log_series.diff(30)
        out[f"{prefix}_z_365d"] = _zscore(series, 365, 120)


def add_macro_features(raw: pd.DataFrame, out: pd.DataFrame) -> None:
    level_cols = [
        "us_10y_yield",
        "us_10y_2y_spread",
        "us_10y_breakeven",
        "fed_funds_rate",
        "reverse_repo",
        "trade_weighted_usd",
    ]
    for column in level_cols:
        if column not in raw:
            continue
        series = raw[column].shift(1)
        prefix = f"macro_{column}"
        out[prefix] = series
        out[f"{prefix}_chg_7d"] = series.diff(7)
        out[f"{prefix}_chg_30d"] = series.diff(30)

    for column in ["fed_balance_sheet", "m2_money_supply"]:
        if column not in raw:
            continue
        lag_days = 30 if column == "m2_money_supply" else 7
        series = raw[column].shift(lag_days).replace(0, np.nan)
        log_series = _safe_log(series)
        out[f"macro_{column}_ret_30d"] = log_series.diff(30)
        out[f"macro_{column}_ret_90d"] = log_series.diff(90)
        out[f"macro_{column}_z_365d"] = _zscore(series, 365, 120)


def add_onchain_features(raw: pd.DataFrame, out: pd.DataFrame) -> None:
    for column in [
        "cm_active_addresses",
        "cm_tx_count",
        "cm_hash_rate",
        "cm_spot_volume_usd",
        "cm_fees_btc",
        "cm_supply",
    ]:
        if column not in raw:
            continue
        series = raw[column].shift(1).replace(0, np.nan)
        prefix = f"onchain_{column.replace('cm_', '')}"
        out[f"{prefix}_log"] = _safe_log(series)
        out[f"{prefix}_chg_30d"] = _safe_log(series).diff(30)
        out[f"{prefix}_z_365d"] = _zscore(series, 365, 120)

    if {"cm_exchange_outflow_usd", "cm_exchange_inflow_usd"}.issubset(raw.columns):
        net_outflow = raw["cm_exchange_outflow_usd"].shift(1) - raw["cm_exchange_inflow_usd"].shift(1)
        out["onchain_exchange_net_outflow_usd"] = net_outflow
        out["onchain_exchange_net_outflow_z_365d"] = _zscore(net_outflow, 365, 120)
        if "cm_spot_volume_usd" in raw:
            out["onchain_exchange_net_outflow_to_volume"] = net_outflow / raw["cm_spot_volume_usd"].replace(0, np.nan)

    if {"cm_exchange_supply_usd", "cm_market_cap_usd"}.issubset(raw.columns):
        out["onchain_exchange_supply_ratio"] = raw["cm_exchange_supply_usd"].shift(1) / raw["cm_market_cap_usd"].shift(1).replace(0, np.nan)

    if "cm_mvrv" in raw:
        mvrv = raw["cm_mvrv"].shift(1)
        out["onchain_mvrv"] = mvrv
        out["onchain_mvrv_z_365d"] = _zscore(mvrv, 365, 120)

    for column in ["cm_roi_30d", "cm_roi_1y"]:
        if column in raw:
            out[f"onchain_{column.replace('cm_', '')}"] = raw[column].shift(1) / 100


def add_sentiment_features(raw: pd.DataFrame, out: pd.DataFrame) -> None:
    if "fear_greed_value" not in raw:
        return
    fg = raw["fear_greed_value"].shift(1)
    out["sentiment_fear_greed"] = fg / 100
    out["sentiment_fear_greed_chg_7d"] = fg.diff(7) / 100
    out["sentiment_fear_greed_chg_30d"] = fg.diff(30) / 100
    out["sentiment_extreme_greed"] = ((fg - 75).clip(lower=0)) / 25
    out["sentiment_extreme_fear"] = ((25 - fg).clip(lower=0)) / 25
    out["sentiment_fear_greed_z_365d"] = _zscore(fg, 365, 120)


def add_cycle_features(raw: pd.DataFrame, out: pd.DataFrame) -> None:
    index = pd.DatetimeIndex(raw.index)
    days_since = []
    for date in index:
        days_since.append((date - _last_halving_date(date)).days)
    days_since_series = pd.Series(days_since, index=index, dtype=float)
    cycle = days_since_series / 1458.0
    out["cycle_days_since_halving"] = days_since_series
    out["cycle_progress"] = cycle.clip(lower=0, upper=1.5)
    out["cycle_sin_4y"] = np.sin(2 * np.pi * cycle)
    out["cycle_cos_4y"] = np.cos(2 * np.pi * cycle)
    out["cycle_month_sin"] = np.sin(2 * np.pi * index.month / 12)
    out["cycle_month_cos"] = np.cos(2 * np.pi * index.month / 12)


def make_features(raw: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=raw.index)
    add_technical_features(raw, out)
    add_cross_asset_features(raw, out)
    add_macro_features(raw, out)
    add_onchain_features(raw, out)
    add_sentiment_features(raw, out)
    add_cycle_features(raw, out)

    out = out.replace([np.inf, -np.inf], np.nan)
    return out.sort_index().copy()


def feature_group(feature: str) -> str:
    for prefix, group in FACTOR_GROUPS.items():
        if feature.startswith(prefix):
            return group
    return "Other"


def feature_columns(features: pd.DataFrame) -> List[str]:
    cols = []
    for col in features.columns:
        if pd.api.types.is_numeric_dtype(features[col]):
            valid_ratio = features[col].notna().mean()
            if valid_ratio >= 0.55:
                cols.append(col)
    return cols


def build_training_data(raw: pd.DataFrame, horizon: int) -> TrainingData:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    features = make_features(raw)
    price = raw["btc_close"]
    target = np.log(price.shift(-horizon) / price)

    frame = features.copy()
    frame["btc_close"] = price
    frame["future_close"] = price.shift(-horizon)
    frame["target_log_return"] = target
    frame["target_up"] = (target > 0).astype(float)

    cols = feature_columns(features)
    frame = frame.dropna(subset=["btc_close"])
    return TrainingData(frame=frame, feature_cols=cols, target_col="target_log_return", horizon=horizon)


def factor_information_coefficients(training: TrainingData) -> pd.DataFrame:
    rows = []
    for col in training.feature_cols:
        pair = training.frame[[col, training.target_col]].dropna()
        if len(pair) < 180:
            continue
        ranked = pair.rank(method="average")
        corr = ranked[col].corr(ranked[training.target_col])
        if pd.isna(corr):
            continue
        rows.append(
            {
                "feature": col,
                "group": feature_group(col),
                "spearman_ic": corr,
                "abs_ic": abs(corr),
                "valid_obs": int(len(pair)),
            }
        )
    return pd.DataFrame(rows).sort_values("abs_ic", ascending=False).reset_index(drop=True)


def current_factor_pressure(training: TrainingData, lookback: int = 365) -> pd.DataFrame:
    ic = factor_information_coefficients(training)
    if ic.empty:
        return ic
    latest = training.frame.iloc[-1]
    rows = []
    for row in ic.to_dict("records"):
        col = row["feature"]
        history = training.frame[col].dropna().tail(lookback)
        if len(history) < 60 or history.std() == 0 or pd.isna(latest.get(col)):
            continue
        z = (latest[col] - history.mean()) / history.std()
        pressure = z * np.sign(row["spearman_ic"]) * row["abs_ic"]
        rows.append(
            {
                "feature": col,
                "group": row["group"],
                "latest_z": z,
                "ic_direction": "positive" if row["spearman_ic"] > 0 else "negative",
                "pressure": pressure,
                "spearman_ic": row["spearman_ic"],
            }
        )
    return pd.DataFrame(rows).sort_values("pressure", ascending=False).reset_index(drop=True)


def grouped_pressure(pressure: pd.DataFrame) -> pd.DataFrame:
    if pressure.empty:
        return pressure
    grouped = (
        pressure.groupby("group", as_index=False)
        .agg(pressure=("pressure", "sum"), avg_abs_ic=("spearman_ic", lambda s: float(np.mean(np.abs(s)))))
        .sort_values("pressure", ascending=False)
    )
    return grouped


def train_test_dates(training: TrainingData) -> Tuple[pd.Timestamp, pd.Timestamp]:
    valid = training.frame.dropna(subset=[training.target_col])
    return valid.index.min(), valid.index.max()
