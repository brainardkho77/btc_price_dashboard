# Full Refresh Diagnostics

## Summary
- Full refresh run ID: `spx_research_20260507T190435Z`
- Selected model: `no_valid_edge`
- Signal: `neutral`
- Reliability: `Low confidence`
- no_valid_edge triggered correctly: `True`

## Why Quick Mode Looked Better
Quick mode used `2018-01-02` to `2026-05-07` with `75` 30d official samples and selected `random_forest`. Full refresh used `2015-01-02` to `2026-05-07` with `99` 30d official samples and selected `no_valid_edge`. The quick edge disappeared because the broader 2015+ sample added regimes where the ML models did not beat the 90d momentum baseline or the full baseline set after costs.

## 30d Official Model Comparison
- Best ML model: `logistic_linear` with directional accuracy `65.7%`
- Best baseline: `buy_hold_direction` with directional accuracy `67.7%`
- Baselines that beat the best ML model: `buy_hold_direction`

Validation gates failed by ML model:
- `logistic_linear`: did_not_beat_buy_hold; low_reliability
- `hgb`: did_not_beat_buy_hold; failed_transaction_cost_check; low_reliability
- `random_forest`: did_not_beat_buy_hold; failed_transaction_cost_check; low_reliability

## Feature Group Findings
- `dollar_rates_only` / `logistic_linear`: samples `99`, accuracy `68.7%`, net return `212.0%`, reliability `Medium confidence`
- `price_momentum_only` / `buy_hold_direction`: samples `99`, accuracy `67.7%`, net return `198.9%`, reliability `Low confidence`
- `all_features` / `buy_hold_direction`: samples `99`, accuracy `67.7%`, net return `198.9%`, reliability `Low confidence`
- `prediction_markets_only` / `buy_hold_direction`: samples `99`, accuracy `67.7%`, net return `198.9%`, reliability `Low confidence`
- `derivatives_only` / `buy_hold_direction`: samples `99`, accuracy `67.7%`, net return `198.9%`, reliability `Low confidence`
- `macro_liquidity_only` / `buy_hold_direction`: samples `99`, accuracy `67.7%`, net return `198.9%`, reliability `Low confidence`
- `onchain_only` / `buy_hold_direction`: samples `99`, accuracy `67.7%`, net return `198.9%`, reliability `Low confidence`
- `sol_ecosystem_only` / `buy_hold_direction`: samples `99`, accuracy `67.7%`, net return `198.9%`, reliability `Low confidence`
- `stablecoins_only` / `buy_hold_direction`: samples `99`, accuracy `67.7%`, net return `198.9%`, reliability `Low confidence`
- `risk_assets_only` / `logistic_linear`: samples `99`, accuracy `67.7%`, net return `216.9%`, reliability `Low confidence`

## Feature Group Watchlist
- `dollar_rates_only` / `logistic_linear`: samples `99`, accuracy `68.7%`, Sharpe `1.01`, max drawdown `-18.2%`, Brier `0.215`, calibration error `0.097`, beats buy-hold `True`, beats momentum `True`.
- It remains diagnostic only because feature-group results are not official model-selection inputs and must prove stability across regimes first.

## Regime Findings
- `logistic_linear` in `bear_market`: samples `23`, accuracy `73.9%`, net return `70.7%`, reliability `Low confidence`
- `buy_hold_direction` in `pre_halving`: samples `24`, accuracy `70.8%`, net return `48.0%`, reliability `Low confidence`
- `momentum_30d` in `pre_halving`: samples `24`, accuracy `70.8%`, net return `44.6%`, reliability `Low confidence`
- `momentum_90d` in `post_halving`: samples `24`, accuracy `70.8%`, net return `65.4%`, reliability `Low confidence`
- `random_permutation` in `pre_halving`: samples `24`, accuracy `70.8%`, net return `60.3%`, reliability `Low confidence`
- `buy_hold_direction` in `high_volatility`: samples `58`, accuracy `70.7%`, net return `128.5%`, reliability `Low confidence`
- `buy_hold_direction` in `low_rate`: samples `34`, accuracy `70.6%`, net return `81.5%`, reliability `Low confidence`
- `random_forest` in `post_etf`: samples `27`, accuracy `70.4%`, net return `50.8%`, reliability `Low confidence`
- `hgb` in `post_etf`: samples `27`, accuracy `70.4%`, net return `50.8%`, reliability `Low confidence`
- `buy_hold_direction` in `post_etf`: samples `27`, accuracy `70.4%`, net return `50.8%`, reliability `Low confidence`

## Binance Derivatives
- Binance derivatives recovered: `False`
- Manual derivatives used: `False`
- Coverage `Binance USD-M Futures` / `skipped`: `7` metrics
- Derivatives impact test was not run because no derivative features were available.
- Stable feature groups with positive 30d slices: `all_features, dollar_rates_only, price_momentum_only`

## Conclusion
- The full refresh did not validate a 30d S&P 500 / SPX directional edge.
- `no_valid_edge` remains the correct conclusion unless future data reliability or feature diagnostics materially improve.
- Recommended next step: improve Binance/manual derivatives coverage, then rerun this diagnostic sprint before adding more models.
