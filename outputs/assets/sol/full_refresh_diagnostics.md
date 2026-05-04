# Full Refresh Diagnostics

## Summary
- Full refresh run ID: `sol_research_20260504T045552Z`
- Selected model: `random_forest`
- Signal: `neutral`
- Reliability: `Medium confidence`
- no_valid_edge triggered correctly: `False`

## Why Quick Mode Looked Better
Quick mode used `2020-04-10` to `2026-05-04` with `48` 30d official samples and selected `random_forest`. Full refresh used `2020-04-10` to `2026-05-04` with `36` 30d official samples and selected `random_forest`. The quick edge disappeared because the broader 2015+ sample added regimes where the ML models did not beat the 90d momentum baseline or the full baseline set after costs.

## 30d Official Model Comparison
- Best ML model: `random_forest` with directional accuracy `69.4%`
- Best baseline: `random_permutation` with directional accuracy `55.6%`
- Baselines that beat the best ML model: `none`

Validation gates failed by ML model:
- `logistic_linear`: not_selected_by_primary_30d_rules
- `hgb`: failed_transaction_cost_check; low_reliability
- `random_forest`: selected_model

## Feature Group Findings
- `all_features` / `random_forest`: samples `36`, accuracy `69.4%`, net return `1114.7%`, reliability `Medium confidence`
- `risk_assets_only` / `random_forest`: samples `36`, accuracy `69.4%`, net return `944.1%`, reliability `Medium confidence`
- `dollar_rates_only` / `hgb`: samples `36`, accuracy `69.4%`, net return `1123.6%`, reliability `Medium confidence`
- `macro_liquidity_only` / `logistic_linear`: samples `36`, accuracy `66.7%`, net return `719.2%`, reliability `Medium confidence`
- `dollar_rates_only` / `random_forest`: samples `36`, accuracy `66.7%`, net return `1082.4%`, reliability `Medium confidence`
- `price_momentum_only` / `logistic_linear`: samples `36`, accuracy `66.7%`, net return `544.6%`, reliability `Medium confidence`
- `price_momentum_only` / `hgb`: samples `36`, accuracy `66.7%`, net return `553.8%`, reliability `Medium confidence`
- `price_momentum_only` / `random_forest`: samples `36`, accuracy `66.7%`, net return `613.6%`, reliability `Medium confidence`
- `macro_liquidity_only` / `random_forest`: samples `36`, accuracy `66.7%`, net return `501.7%`, reliability `Low confidence`
- `macro_liquidity_only` / `hgb`: samples `36`, accuracy `63.9%`, net return `714.9%`, reliability `Medium confidence`

## Feature Group Watchlist
- `dollar_rates_only` / `logistic_linear`: samples `36`, accuracy `55.6%`, Sharpe `1.19`, max drawdown `-37.4%`, Brier `0.270`, calibration error `0.108`, beats buy-hold `True`, beats momentum `True`.
- It remains diagnostic only because feature-group results are not official model-selection inputs and must prove stability across regimes first.

## Regime Findings
- `logistic_linear` in `high_volatility`: samples `1`, accuracy `100.0%`, net return `18.5%`, reliability `Medium confidence`
- `random_permutation` in `high_volatility`: samples `1`, accuracy `100.0%`, net return `18.5%`, reliability `Low confidence`
- `buy_hold_direction` in `high_volatility`: samples `1`, accuracy `100.0%`, net return `18.5%`, reliability `Low confidence`
- `random_forest` in `high_volatility`: samples `1`, accuracy `100.0%`, net return `18.5%`, reliability `Medium confidence`
- `hgb` in `high_volatility`: samples `1`, accuracy `100.0%`, net return `18.5%`, reliability `Low confidence`
- `random_forest` in `pre_etf`: samples `9`, accuracy `77.8%`, net return `455.3%`, reliability `Medium confidence`
- `random_forest` in `pre_halving`: samples `12`, accuracy `75.0%`, net return `617.8%`, reliability `Medium confidence`
- `random_forest` in `bull_market`: samples `22`, accuracy `72.7%`, net return `763.7%`, reliability `Medium confidence`
- `random_forest` in `high_rate`: samples `36`, accuracy `69.4%`, net return `1114.7%`, reliability `Medium confidence`
- `random_forest` in `low_volatility`: samples `35`, accuracy `68.6%`, net return `925.2%`, reliability `Medium confidence`

## Binance Derivatives
- Binance derivatives recovered: `False`
- Manual derivatives used: `False`
- Coverage `Binance USD-M Futures` / `skipped`: `5` metrics
- Derivatives impact test was not run because no derivative features were available.
- Stable feature groups with positive 30d slices: `all_features, dollar_rates_only, price_momentum_only`

## Conclusion
- The full refresh did not validate a 30d SOL directional edge.
- `no_valid_edge` remains the correct conclusion unless future data reliability or feature diagnostics materially improve.
- Recommended next step: improve Binance/manual derivatives coverage, then rerun this diagnostic sprint before adding more models.
