import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from features import build_training_data, make_features


def synthetic_raw(rows=500):
    dates = pd.date_range("2020-01-01", periods=rows, freq="D")
    price = pd.Series(10000 * np.exp(np.linspace(0, 0.8, rows)), index=dates)
    raw = pd.DataFrame(
        {
            "btc_open": price * 0.99,
            "btc_high": price * 1.02,
            "btc_low": price * 0.98,
            "btc_close": price,
            "btc_volume": 1_000_000 + np.arange(rows) * 100,
            "eth_close": price / 10,
            "spx_close": 3000 + np.arange(rows),
            "vix_close": 20,
            "cm_active_addresses": 700_000 + np.arange(rows),
            "cm_tx_count": 300_000 + np.arange(rows),
            "cm_hash_rate": 100_000_000 + np.arange(rows) * 1000,
            "cm_mvrv": 2.0,
            "cm_exchange_inflow_usd": 1_000_000,
            "cm_exchange_outflow_usd": 1_100_000,
            "cm_market_cap_usd": price * 19_000_000,
            "cm_exchange_supply_usd": price * 1_000_000,
            "cm_spot_volume_usd": 5_000_000,
            "fear_greed_value": 50,
            "us_10y_yield": 4.0,
            "trade_weighted_usd": 120,
        },
        index=dates,
    )
    return raw


def test_make_features_produces_numeric_columns():
    features = make_features(synthetic_raw())
    assert "momentum_ret_30d" in features
    assert "trend_dist_sma_200d" in features
    assert features.select_dtypes(include="number").shape[1] > 20


def test_training_target_is_future_return():
    raw = synthetic_raw()
    training = build_training_data(raw, horizon=7)
    row_date = raw.index[100]
    expected = np.log(raw.loc[raw.index[107], "btc_close"] / raw.loc[row_date, "btc_close"])
    assert np.isclose(training.frame.loc[row_date, "target_log_return"], expected)
    assert training.horizon == 7
    assert "target_log_return" not in training.feature_cols
