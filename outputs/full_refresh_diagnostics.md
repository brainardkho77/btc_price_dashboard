# Full Refresh Diagnostics

## Summary
- Full refresh run ID: `btc_research_20260425T073548Z`
- Selected model: `no_valid_edge`
- Signal: `neutral`
- Reliability: `Low confidence`
- no_valid_edge triggered correctly: `True`

## Why Quick Mode Looked Better
Quick mode used `2018-01-01` to `2026-04-25` with `74` 30d official samples and selected `hgb`. Full refresh used `2015-01-01` to `2026-04-25` with `99` 30d official samples and selected `no_valid_edge`. The quick edge disappeared because the broader 2015+ sample added regimes where the ML models did not beat the 90d momentum baseline or the full baseline set after costs.

## 30d Official Model Comparison
- Best ML model: `random_forest` with directional accuracy `51.5%`
- Best baseline: `momentum_90d` with directional accuracy `55.6%`
- Baselines that beat the best ML model: `momentum_90d`

Validation gates failed by ML model:
- `logistic_linear`: did_not_beat_buy_hold; did_not_beat_30d_momentum; did_not_beat_90d_momentum; failed_transaction_cost_check; low_reliability
- `hgb`: did_not_beat_buy_hold; did_not_beat_30d_momentum; did_not_beat_90d_momentum; low_reliability
- `random_forest`: did_not_beat_30d_momentum; did_not_beat_90d_momentum; failed_transaction_cost_check; low_reliability

## Feature Group Findings
- `dollar_rates_only` / `logistic_linear`: samples `99`, accuracy `57.6%`, net return `1627.6%`, reliability `Medium confidence`
- `dollar_rates_only` / `momentum_90d`: samples `99`, accuracy `55.6%`, net return `840.4%`, reliability `Low confidence`
- `price_momentum_only` / `momentum_90d`: samples `99`, accuracy `55.6%`, net return `840.4%`, reliability `Low confidence`
- `all_features` / `momentum_90d`: samples `99`, accuracy `55.6%`, net return `840.4%`, reliability `Low confidence`
- `stablecoins_only` / `momentum_90d`: samples `99`, accuracy `55.6%`, net return `840.4%`, reliability `Low confidence`
- `risk_assets_only` / `momentum_90d`: samples `99`, accuracy `55.6%`, net return `840.4%`, reliability `Low confidence`
- `macro_liquidity_only` / `momentum_90d`: samples `99`, accuracy `55.6%`, net return `840.4%`, reliability `Low confidence`
- `derivatives_only` / `momentum_90d`: samples `99`, accuracy `55.6%`, net return `840.4%`, reliability `Low confidence`
- `onchain_only` / `momentum_90d`: samples `99`, accuracy `55.6%`, net return `840.4%`, reliability `Low confidence`
- `price_momentum_only` / `random_forest`: samples `99`, accuracy `52.5%`, net return `770.0%`, reliability `Low confidence`

## Regime Findings
- `momentum_90d` in `low_volatility`: samples `65`, accuracy `66.2%`, net return `2371.9%`, reliability `Low confidence`
- `momentum_90d` in `high_rate`: samples `65`, accuracy `63.1%`, net return `267.2%`, reliability `Low confidence`
- `logistic_linear` in `post_halving`: samples `24`, accuracy `62.5%`, net return `549.0%`, reliability `Low confidence`
- `buy_hold_direction` in `post_halving`: samples `24`, accuracy `62.5%`, net return `662.4%`, reliability `Low confidence`
- `momentum_90d` in `bear_market`: samples `44`, accuracy `59.1%`, net return `64.0%`, reliability `Low confidence`
- `random_forest` in `post_halving`: samples `24`, accuracy `58.3%`, net return `637.2%`, reliability `Low confidence`
- `momentum_90d` in `post_etf`: samples `26`, accuracy `57.7%`, net return `122.8%`, reliability `Low confidence`
- `random_forest` in `low_rate`: samples `34`, accuracy `55.9%`, net return `638.1%`, reliability `Low confidence`
- `momentum_30d` in `low_rate`: samples `34`, accuracy `55.9%`, net return `361.2%`, reliability `Low confidence`
- `momentum_90d` in `pre_etf`: samples `73`, accuracy `54.8%`, net return `322.1%`, reliability `Low confidence`

## Binance Derivatives
- Binance derivatives recovered: `False`
- Derivatives impact test was not run because no derivative features were available.

## Conclusion
- The full refresh did not validate a 30d BTC directional edge.
- `no_valid_edge` remains the correct conclusion unless future data reliability or feature diagnostics materially improve.
- Recommended next step: improve Binance/manual derivatives coverage, then rerun this diagnostic sprint before adding more models.
