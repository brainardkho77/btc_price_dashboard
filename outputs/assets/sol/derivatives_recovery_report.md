# Derivatives Recovery Report

## Run Summary
- Run ID: `sol_research_20260427T221648Z`
- Selected model: `random_forest`
- Signal: `neutral`
- Reliability: `Medium confidence`
- no_valid_edge remains correct: `False`

## Derivatives Coverage
- Binance derivatives recovered: `False`
- Manual CSV derivatives used: `False`
- `Binance USD-M Futures` / `funding_rate`: `skipped`, rows `0`, dates `nan` to `nan`, missing `100.0%`, used `False`
- `Binance USD-M Futures` / `open_interest`: `skipped`, rows `0`, dates `nan` to `nan`, missing `100.0%`, used `False`
- `Binance USD-M Futures` / `long_short_ratio`: `skipped`, rows `0`, dates `nan` to `nan`, missing `100.0%`, used `False`
- `Binance USD-M Futures` / `taker_buy_sell_ratio`: `skipped`, rows `0`, dates `nan` to `nan`, missing `100.0%`, used `False`
- `Binance USD-M Futures` / `basis`: `skipped`, rows `0`, dates `nan` to `nan`, missing `100.0%`, used `False`

## Derivatives Impact
- No derivatives impact comparison ran because no valid derivative feature entered the model feature set.

## Feature Group Watchlist
- `dollar_rates_only` / `logistic_linear`: samples `35`, accuracy `57.1%`, Sharpe `1.21`, drawdown `-37.4%`, Brier `0.255`, calibration error `0.104`, reliability `Low confidence`.
- Remains strongest diagnostic group: `False`
- It is not selected unless it passes the official 30d selection rules in the main leaderboard.

## Feature Group Stability
- `all_features` / `logistic_linear`: `6` stable slices
- `all_features` / `random_forest`: `6` stable slices
- `dollar_rates_only` / `logistic_linear`: `6` stable slices
- `price_momentum_only` / `random_forest`: `6` stable slices
- `price_momentum_only` / `momentum_90d`: `3` stable slices

## Conclusion
- Derivatives improved 30d official diagnostics: `False`
- `no_valid_edge` remains correct: `False`
- Recommended next step: A model passed official 30d rules; review the full leaderboard and calibration before deployment.
