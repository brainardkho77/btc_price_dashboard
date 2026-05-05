# Full Refresh Diagnostics

## Summary
- Full refresh run ID: `btc_research_20260505T154327Z`
- Selected model: `no_valid_edge`
- Signal: `neutral`
- Reliability: `Low confidence`
- no_valid_edge triggered correctly: `True`

## Why Quick Mode Looked Better
Quick mode used `2018-01-01` to `2026-05-05` with `75` 30d official samples and selected `random_forest`. Full refresh used `2015-01-01` to `2026-05-05` with `100` 30d official samples and selected `no_valid_edge`. The quick edge disappeared because the broader 2015+ sample added regimes where the ML models did not beat the 90d momentum baseline or the full baseline set after costs.

## 30d Official Model Comparison
- Best ML model: `random_forest` with directional accuracy `55.0%`
- Best baseline: `momentum_90d` with directional accuracy `55.0%`
- Baselines that beat the best ML model: `none`

Validation gates failed by ML model:
- `logistic_linear`: did_not_beat_buy_hold; did_not_beat_30d_momentum; did_not_beat_90d_momentum; failed_transaction_cost_check; low_reliability
- `hgb`: did_not_beat_buy_hold; did_not_beat_30d_momentum; did_not_beat_90d_momentum; low_reliability
- `random_forest`: did_not_beat_90d_momentum; low_reliability

## Feature Group Findings
- `derivatives_only` / `momentum_90d`: samples `45`, accuracy `60.0%`, net return `249.3%`, reliability `Low confidence`
- `dollar_rates_only` / `logistic_linear`: samples `100`, accuracy `58.0%`, net return `1832.3%`, reliability `Medium confidence`
- `derivatives_only` / `buy_hold_direction`: samples `45`, accuracy `55.6%`, net return `357.3%`, reliability `Low confidence`
- `macro_liquidity_only` / `momentum_90d`: samples `100`, accuracy `55.0%`, net return `840.4%`, reliability `Low confidence`
- `stablecoins_only` / `momentum_90d`: samples `100`, accuracy `55.0%`, net return `840.4%`, reliability `Low confidence`
- `risk_assets_only` / `momentum_90d`: samples `100`, accuracy `55.0%`, net return `840.4%`, reliability `Low confidence`
- `onchain_only` / `momentum_90d`: samples `100`, accuracy `55.0%`, net return `840.4%`, reliability `Low confidence`
- `dollar_rates_only` / `momentum_90d`: samples `100`, accuracy `55.0%`, net return `840.4%`, reliability `Low confidence`
- `prediction_markets_only` / `momentum_90d`: samples `100`, accuracy `55.0%`, net return `840.4%`, reliability `Low confidence`
- `all_features` / `random_forest`: samples `100`, accuracy `55.0%`, net return `993.9%`, reliability `Low confidence`

## Feature Group Watchlist
- `dollar_rates_only` / `logistic_linear`: samples `100`, accuracy `58.0%`, Sharpe `0.88`, max drawdown `-55.0%`, Brier `0.236`, calibration error `0.107`, beats buy-hold `True`, beats momentum `True`.
- It remains diagnostic only because feature-group results are not official model-selection inputs and must prove stability across regimes first.

## Regime Findings
- `logistic_linear` in `post_halving`: samples `24`, accuracy `66.7%`, net return `515.1%`, reliability `Low confidence`
- `momentum_90d` in `low_volatility`: samples `66`, accuracy `65.2%`, net return `2371.9%`, reliability `Low confidence`
- `hgb` in `post_halving`: samples `24`, accuracy `62.5%`, net return `500.2%`, reliability `Low confidence`
- `buy_hold_direction` in `post_halving`: samples `24`, accuracy `62.5%`, net return `662.4%`, reliability `Low confidence`
- `momentum_90d` in `high_rate`: samples `66`, accuracy `62.1%`, net return `267.2%`, reliability `Low confidence`
- `random_forest` in `bear_market`: samples `45`, accuracy `60.0%`, net return `154.8%`, reliability `Low confidence`
- `logistic_linear` in `post_etf`: samples `27`, accuracy `59.3%`, net return `104.7%`, reliability `Low confidence`
- `random_forest` in `post_etf`: samples `27`, accuracy `59.3%`, net return `101.8%`, reliability `Low confidence`
- `random_forest` in `post_halving`: samples `24`, accuracy `58.3%`, net return `636.5%`, reliability `Low confidence`
- `momentum_90d` in `bear_market`: samples `45`, accuracy `57.8%`, net return `64.0%`, reliability `Low confidence`

## Binance Derivatives
- Binance derivatives recovered: `True`
- Manual derivatives used: `True`
- Coverage `Binance USD-M Futures` / `failed`: `5` metrics
- Coverage `manual_csv` / `skipped`: `2` metrics
- Coverage `manual_csv` / `worked`: `3` metrics
- Best derivatives impact row: `random_forest`, derivatives included `False`, accuracy `63.6%`.
- Stable feature groups with positive 30d slices: `all_features, dollar_rates_only, price_momentum_only`

## Conclusion
- The full refresh did not validate a 30d BTC directional edge.
- `no_valid_edge` remains the correct conclusion unless future data reliability or feature diagnostics materially improve.
- Recommended next step: improve Binance/manual derivatives coverage, then rerun this diagnostic sprint before adding more models.
