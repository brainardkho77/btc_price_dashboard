# BTC Factor Forecast Dashboard

Local research dashboard for BTC price forecasting across 1, 7, 30, 90, and 180 day horizons.

The dashboard uses free/keyless data sources:

- Yahoo chart endpoint for BTC OHLCV and market proxies.
- Coin Metrics Community API for BTC on-chain metrics.
- FRED graph CSV downloads for macro/liquidity series.
- Alternative.me public Fear & Greed endpoint for sentiment.

This is not financial advice. The forecasts are model outputs with uncertainty, and BTC can move outside historical model ranges.

## Run

From the repository root:

```bash
streamlit run app.py
```

Run walk-forward backtests and save outputs:

```bash
python run_backtest.py --horizons all --model ensemble
```

## Streamlit Community Cloud

Use `app.py` as the main file path. No API keys or secrets are required.

The app ships with cached public data, precomputed latest forecasts, and precomputed backtest outputs so it can boot quickly on Community Cloud. Use the sidebar buttons only when you intentionally want to refresh data or retrain forecasts.

Live forecast recomputation and live backtesting require `scikit-learn`. It is intentionally not included in the default Community Cloud requirements because the deployed dashboard uses precomputed forecasts/backtests and should boot quickly.

Use `--force-refresh` to refresh cached free API data.

## What It Models

Factors are grouped into:

- Technical trend and momentum: moving-average distance, multi-period returns, MACD, RSI.
- Volatility/risk: realized volatility, intraday range, rolling drawdown, distance from ATH.
- Cross-asset risk appetite: ETH, S&P 500, Nasdaq, VIX, gold, TLT, DXY.
- Macro liquidity: Treasury yields, curve spread, breakevens, fed funds, Fed balance sheet, reverse repo, M2, broad dollar index.
- On-chain/network: active addresses, transactions, hash rate, fees, exchange flows, exchange supply ratio, MVRV, reported spot volume.
- Sentiment: fear/greed level, changes, and extremes.
- Supply cycle: days since halving and 4-year cycle phase.

Non-BTC external factors are lagged before modeling: market proxies, on-chain data, and sentiment by one day; Fed balance sheet by seven days; M2 by thirty days. This is intentionally conservative because historical data timestamps often differ from public release time.

## Backtesting

Backtests are walk-forward:

1. Train only on data before the forecast date.
2. Predict the selected horizon.
3. Record actual future BTC return.
4. Repeat through history.

The default strategy metric is long/cash: hold BTC when the model's predicted log return is above the threshold, otherwise hold cash for that forecast window. It is compared to BTC buy-and-hold over the same forecast windows.
