# Full Refresh Diagnostics

## Summary
- Full refresh run ID: `btc_research_20260512T122205Z`
- Selected model: `no_valid_edge`
- Signal: `neutral`
- Reliability: `Low confidence`
- no_valid_edge triggered correctly: `True`

## Why Quick Mode Looked Better
Quick mode used `2018-01-01` to `2026-05-11` with `75` 30d official samples and selected `random_forest`. Full refresh used `2015-01-01` to `2026-05-12` with `100` 30d official samples and selected `no_valid_edge`. The quick edge disappeared because the broader 2015+ sample added regimes where the ML models did not beat the 90d momentum baseline or the full baseline set after costs.

## 30d Official Model Comparison
- Best ML model: `hgb` with directional accuracy `49.0%`
- Best baseline: `momentum_90d` with directional accuracy `55.0%`
- Baselines that beat the best ML model: `momentum_90d, momentum_30d, buy_hold_direction`

Validation gates failed by ML model:
- `logistic_linear`: did_not_beat_buy_hold; did_not_beat_30d_momentum; did_not_beat_90d_momentum; failed_transaction_cost_check; low_reliability
- `hgb`: did_not_beat_buy_hold; did_not_beat_30d_momentum; did_not_beat_90d_momentum; failed_transaction_cost_check; low_reliability
- `random_forest`: did_not_beat_buy_hold; did_not_beat_30d_momentum; did_not_beat_90d_momentum; failed_transaction_cost_check; low_reliability

## Feature Group Findings
- `derivatives_only` / `momentum_90d`: samples `45`, accuracy `60.0%`, net return `249.3%`, reliability `Low confidence`
- `btc_dominance_regime` / `hgb`: samples `100`, accuracy `57.0%`, net return `2505.7%`, reliability `Medium confidence`
- `btc_dominance_regime` / `random_forest`: samples `100`, accuracy `57.0%`, net return `1585.4%`, reliability `Medium confidence`
- `stablecoin_liquidity_v2` / `random_forest`: samples `84`, accuracy `56.0%`, net return `2791.1%`, reliability `Medium confidence`
- `derivatives_only` / `buy_hold_direction`: samples `45`, accuracy `55.6%`, net return `357.3%`, reliability `Low confidence`
- `btc_dominance_regime` / `momentum_90d`: samples `100`, accuracy `55.0%`, net return `840.4%`, reliability `Low confidence`
- `btc_liquidity_interactions` / `momentum_90d`: samples `100`, accuracy `55.0%`, net return `840.4%`, reliability `Low confidence`
- `sol_ecosystem_only` / `momentum_90d`: samples `100`, accuracy `55.0%`, net return `840.4%`, reliability `Low confidence`
- `dollar_rates_only` / `logistic_linear`: samples `100`, accuracy `55.0%`, net return `2219.8%`, reliability `Low confidence`
- `dollar_rates_only` / `momentum_90d`: samples `100`, accuracy `55.0%`, net return `840.4%`, reliability `Low confidence`

## Feature Group Watchlist
- `dollar_rates_only` / `logistic_linear`: samples `100`, accuracy `55.0%`, Sharpe `0.91`, max drawdown `-42.9%`, Brier `0.245`, calibration error `0.108`, beats buy-hold `True`, beats momentum `True`.
- It remains diagnostic only because feature-group results are not official model-selection inputs and must prove stability across regimes first.

## Regime Findings
- `momentum_90d` in `low_volatility`: samples `66`, accuracy `65.2%`, net return `2371.9%`, reliability `Low confidence`
- `random_forest` in `post_etf`: samples `27`, accuracy `63.0%`, net return `196.2%`, reliability `Low confidence`
- `buy_hold_direction` in `post_halving`: samples `24`, accuracy `62.5%`, net return `662.4%`, reliability `Low confidence`
- `momentum_90d` in `high_rate`: samples `66`, accuracy `62.1%`, net return `267.2%`, reliability `Low confidence`
- `logistic_linear` in `post_etf`: samples `27`, accuracy `59.3%`, net return `175.1%`, reliability `Low confidence`
- `momentum_90d` in `bear_market`: samples `45`, accuracy `57.8%`, net return `64.0%`, reliability `Low confidence`
- `random_forest` in `bear_market`: samples `45`, accuracy `57.8%`, net return `133.4%`, reliability `Low confidence`
- `momentum_30d` in `low_rate`: samples `34`, accuracy `55.9%`, net return `361.2%`, reliability `Low confidence`
- `momentum_90d` in `post_etf`: samples `27`, accuracy `55.6%`, net return `122.8%`, reliability `Low confidence`
- `hgb` in `bear_market`: samples `45`, accuracy `55.6%`, net return `238.2%`, reliability `Low confidence`

## Binance Derivatives
- Binance derivatives recovered: `True`
- Manual derivatives used: `True`
- Coverage `Binance USD-M Futures` / `failed`: `7` metrics
- Coverage `manual_csv` / `skipped`: `4` metrics
- Coverage `manual_csv` / `worked`: `3` metrics
- Best derivatives impact row: `logistic_linear`, derivatives included `True`, accuracy `66.7%`.
- Stable feature groups with positive 30d slices: `all_features, dollar_rates_only, price_momentum_only`

## Conclusion
- The full refresh did not validate a 30d BTC directional edge.
- `no_valid_edge` remains the correct conclusion unless future data reliability or feature diagnostics materially improve.
- Recommended next step: improve Binance/manual derivatives coverage, then rerun this diagnostic sprint before adding more models.
