# Derivatives Recovery Report

## Run Summary
- Run ID: `btc_research_20260427T104950Z`
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
- `manual_csv` / `open_interest`: `worked`, rows `2065`, dates `2020-09-01` to `2026-04-27`, missing `50.1%`, used `True`
- `Binance USD-M Futures` / `long_short_ratio`: `failed`, rows `0`, dates `nan` to `nan`, missing `100.0%`, used `False`
- `manual_csv` / `long_short_ratio`: `worked`, rows `2046`, dates `2020-09-01` to `2026-04-27`, missing `50.1%`, used `True`
- `Binance USD-M Futures` / `taker_buy_sell_ratio`: `failed`, rows `0`, dates `nan` to `nan`, missing `100.0%`, used `False`
- `manual_csv` / `taker_buy_sell_ratio`: `worked`, rows `1937`, dates `2020-09-01` to `2026-04-27`, missing `50.1%`, used `True`
- `Binance USD-M Futures` / `basis`: `failed`, rows `0`, dates `nan` to `nan`, missing `100.0%`, used `False`
- `manual_csv` / `basis`: `skipped`, rows `0`, dates `nan` to `nan`, missing `100.0%`, used `False`

## Derivatives Impact
- Best with derivatives: `random_forest` at `53.5%`. Best without derivatives: `random_forest` at `51.5%`. Improved directional accuracy: `True`.

## Feature Group Watchlist
- `dollar_rates_only` / `logistic_linear`: samples `99`, accuracy `57.6%`, Sharpe `0.86`, drawdown `-55.0%`, Brier `0.236`, calibration error `0.103`, reliability `Medium confidence`.
- Remains strongest diagnostic group: `False`
- It is not selected unless it passes the official 30d selection rules in the main leaderboard.

## Feature Group Stability
- `price_momentum_only` / `momentum_90d`: `6` stable slices
- `dollar_rates_only` / `logistic_linear`: `5` stable slices
- `all_features` / `hgb`: `4` stable slices
- `all_features` / `random_forest`: `4` stable slices
- `price_momentum_only` / `random_forest`: `3` stable slices

## Conclusion
- Derivatives improved 30d official diagnostics: `True`
- `no_valid_edge` remains correct: `True`
- Recommended next step: Derivatives improved diagnostics but did not pass official selection; extend historical coverage before adding models.
