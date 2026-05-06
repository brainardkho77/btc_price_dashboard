# Derivatives Recovery Report

## Run Summary
- Run ID: `btc_research_20260506T165426Z`
- Selected model: `no_valid_edge`
- Signal: `neutral`
- Reliability: `Low confidence`
- no_valid_edge remains correct: `True`

## Derivatives Coverage
- Binance derivatives recovered: `False`
- Manual CSV derivatives used: `True`
- `Binance USD-M Futures` / `funding_rate`: `failed`, rows `0`, dates `nan` to `nan`, missing `100.0%`, used `False`
- `manual_csv` / `funding_rate`: `skipped`, rows `0`, dates `nan` to `nan`, missing `100.0%`, used `False`
- `Binance USD-M Futures` / `open_interest`: `failed`, rows `0`, dates `nan` to `nan`, missing `100.0%`, used `False`
- `manual_csv` / `open_interest`: `worked`, rows `2074`, dates `2020-09-01` to `2026-05-06`, missing `50.0%`, used `True`
- `Binance USD-M Futures` / `long_short_ratio`: `failed`, rows `0`, dates `nan` to `nan`, missing `100.0%`, used `False`
- `manual_csv` / `long_short_ratio`: `worked`, rows `2055`, dates `2020-09-01` to `2026-05-06`, missing `50.0%`, used `True`
- `Binance USD-M Futures` / `top_trader_account_ratio`: `failed`, rows `0`, dates `nan` to `nan`, missing `100.0%`, used `False`
- `manual_csv` / `top_trader_account_ratio`: `skipped`, rows `0`, dates `nan` to `nan`, missing `100.0%`, used `False`
- `Binance USD-M Futures` / `top_trader_position_ratio`: `failed`, rows `0`, dates `nan` to `nan`, missing `100.0%`, used `False`
- `manual_csv` / `top_trader_position_ratio`: `skipped`, rows `0`, dates `nan` to `nan`, missing `100.0%`, used `False`
- `Binance USD-M Futures` / `taker_buy_sell_ratio`: `failed`, rows `0`, dates `nan` to `nan`, missing `100.0%`, used `False`
- `manual_csv` / `taker_buy_sell_ratio`: `worked`, rows `1946`, dates `2020-09-01` to `2026-05-06`, missing `50.0%`, used `True`
- `Binance USD-M Futures` / `basis`: `failed`, rows `0`, dates `nan` to `nan`, missing `100.0%`, used `False`
- `manual_csv` / `basis`: `skipped`, rows `0`, dates `nan` to `nan`, missing `100.0%`, used `False`

## Derivatives Impact
- Best with derivatives: `logistic_linear` at `48.0%`. Best without derivatives: `hgb` at `51.0%`. Improved directional accuracy: `False`.

## Feature Group Watchlist
- `dollar_rates_only` / `logistic_linear`: samples `100`, accuracy `55.0%`, Sharpe `0.91`, drawdown `-42.9%`, Brier `0.245`, calibration error `0.108`, reliability `Low confidence`.
- Remains strongest diagnostic group: `False`
- It is not selected unless it passes the official 30d selection rules in the main leaderboard.

## Feature Group Stability
- `price_momentum_only` / `momentum_90d`: `7` stable slices
- `dollar_rates_only` / `logistic_linear`: `6` stable slices
- `all_features` / `random_forest`: `2` stable slices
- `price_momentum_only` / `momentum_30d`: `2` stable slices
- `price_momentum_only` / `random_forest`: `2` stable slices

## Conclusion
- Derivatives improved 30d official diagnostics: `False`
- `no_valid_edge` remains correct: `True`
- Recommended next step: Feature-group leadership changed; inspect stability before pruning or expanding features.
