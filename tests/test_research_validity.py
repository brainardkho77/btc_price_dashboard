import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from research_config import ResearchConfig
from research_pipeline import (
    apply_release_lags,
    build_features,
    build_target_frame,
    build_walk_forward_windows,
    compute_metrics,
    select_feature_columns,
    window_fingerprint,
)


def synthetic_research_raw(rows=1500):
    dates = pd.date_range("2018-01-01", periods=rows, freq="D")
    base = 10_000 * np.exp(np.linspace(0, 1.1, rows) + np.sin(np.arange(rows) / 35) * 0.08)
    price = pd.Series(base, index=dates)
    raw = pd.DataFrame(
        {
            "btc_open": price * 0.99,
            "btc_high": price * 1.02,
            "btc_low": price * 0.98,
            "btc_close": price,
            "btc_volume": 1_000_000 + np.arange(rows) * 100,
            "eth_close": price / 10,
            "spx_close": 3000 + np.arange(rows),
            "nasdaq_close": 8000 + np.arange(rows),
            "vix_close": 20 + np.sin(np.arange(rows) / 20),
            "dxy_close": 100 + np.cos(np.arange(rows) / 40),
            "gold_close": 1500 + np.arange(rows) * 0.2,
            "tlt_close": 100 + np.sin(np.arange(rows) / 50),
            "cm_active_addresses": 700_000 + np.arange(rows),
            "cm_tx_count": 300_000 + np.arange(rows),
            "cm_hash_rate": 100_000_000 + np.arange(rows) * 1000,
            "cm_mvrv": 2.0 + np.sin(np.arange(rows) / 90) * 0.2,
            "cm_exchange_inflow_usd": 1_000_000 + np.arange(rows) * 10,
            "cm_exchange_outflow_usd": 1_100_000 + np.arange(rows) * 11,
            "cm_market_cap_usd": price * 19_000_000,
            "cm_exchange_supply_usd": price * 1_000_000,
            "cm_spot_volume_usd": 5_000_000 + np.arange(rows) * 100,
            "fear_greed_value": 50 + np.sin(np.arange(rows) / 30) * 20,
            "us_10y_yield": 4.0 + np.sin(np.arange(rows) / 100) * 0.5,
            "trade_weighted_usd": 120 + np.cos(np.arange(rows) / 70),
            "m2_money_supply": 20_000 + np.arange(rows),
            "stablecoin_total_circulating_usd": 100_000_000_000 + np.arange(rows) * 10_000_000,
            "binance_funding_rate": 0.0001 + np.sin(np.arange(rows) / 10) * 0.00005,
        },
        index=dates,
    )
    return raw


def test_target_generation_uses_future_prices_only():
    config = ResearchConfig()
    raw = synthetic_research_raw()
    feature_result = build_features(raw, config)
    frame = build_target_frame(raw, feature_result.features, feature_result.feature_cols, 30)
    row_date = raw.index[100]
    expected = np.log(raw.loc[raw.index[130], "btc_close"] / raw.loc[row_date, "btc_close"])
    assert np.isclose(frame.loc[row_date, "target_log_return"], expected)
    assert pd.isna(frame["target_log_return"].iloc[-1])


def test_future_and_target_columns_do_not_enter_feature_cols():
    config = ResearchConfig()
    dates = pd.date_range("2020-01-01", periods=500, freq="D")
    features = pd.DataFrame(
        {
            "momentum_ret_30d": np.random.default_rng(1).normal(size=500),
            "future_return_30d": np.random.default_rng(2).normal(size=500),
            "target_up": np.random.default_rng(3).normal(size=500),
            "shifted_target_30d": np.random.default_rng(4).normal(size=500),
            "forward_log_return": np.random.default_rng(5).normal(size=500),
        },
        index=dates,
    )
    cols = select_feature_columns(features, config)
    assert "momentum_ret_30d" in cols
    assert "future_return_30d" not in cols
    assert "target_up" not in cols
    assert "shifted_target_30d" not in cols
    assert "forward_log_return" not in cols


def test_non_price_external_features_are_lagged_conservatively():
    config = ResearchConfig()
    raw = synthetic_research_raw(rows=80)
    released = apply_release_lags(raw, config)
    assert released["btc_close"].iloc[10] == raw["btc_close"].iloc[10]
    assert released["binance_funding_rate"].iloc[10] == raw["binance_funding_rate"].iloc[9]
    assert released["stablecoin_total_circulating_usd"].iloc[10] == raw["stablecoin_total_circulating_usd"].iloc[9]
    assert released["us_10y_yield"].iloc[10] == raw["us_10y_yield"].iloc[8]
    assert released["m2_money_supply"].iloc[40] == raw["m2_money_supply"].iloc[10]
    assert released["cm_active_addresses"].iloc[10] == raw["cm_active_addresses"].iloc[8]


def test_monthly_date_alignment_for_30d_official_windows():
    config = ResearchConfig(quick_initial_train_days=365)
    raw = synthetic_research_raw(rows=900)
    feature_result = build_features(raw, config)
    frame = build_target_frame(raw, feature_result.features, feature_result.feature_cols, 30)
    windows = build_walk_forward_windows(frame, 30, "official_monthly", config, quick=True)
    months = [w.test_date.to_period("M") for w in windows]
    assert len(months) == len(set(months))
    assert all((b.test_date - a.test_date).days >= 25 for a, b in zip(windows, windows[1:]))


def test_identical_model_windows_are_enforced_by_shared_fingerprint():
    config = ResearchConfig(quick_initial_train_days=365)
    raw = synthetic_research_raw(rows=900)
    feature_result = build_features(raw, config)
    frame = build_target_frame(raw, feature_result.features, feature_result.feature_cols, 90)
    windows_a = build_walk_forward_windows(frame, 90, "official_quarterly", config, quick=True)
    windows_b = build_walk_forward_windows(frame, 90, "official_quarterly", config, quick=True)
    assert window_fingerprint(windows_a) == window_fingerprint(windows_b)


def test_transaction_costs_reduce_returns_when_turnover_exists():
    dates = pd.date_range("2021-01-01", periods=6, freq="ME")
    preds = pd.DataFrame(
        {
            "probability_up": [0.9, 0.1, 0.9, 0.1, 0.9, 0.1],
            "expected_log_return": [0.05, -0.03, 0.04, -0.02, 0.03, -0.01],
            "actual_log_return": [0.04, -0.02, 0.03, -0.01, 0.02, -0.01],
            "actual_return": np.exp([0.04, -0.02, 0.03, -0.01, 0.02, -0.01]) - 1,
            "actual_up": [1, 0, 1, 0, 1, 0],
            "predicted_up": [1, 0, 1, 0, 1, 0],
            "position": [1, 0, 1, 0, 1, 0],
        },
        index=dates,
    )
    metrics, returns_frame = compute_metrics(preds, 30, "official_monthly", transaction_cost_bps=25)
    assert returns_frame["tc_adjusted_strategy_return"].sum() < returns_frame["gross_strategy_return"].sum()
    assert metrics["tc_adjusted_return"] < metrics["gross_return"]
