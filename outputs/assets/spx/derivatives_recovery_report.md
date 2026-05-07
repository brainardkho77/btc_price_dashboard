# Derivatives Recovery Report

## Run Summary
- Run ID: `spx_research_20260507T130913Z`
- Selected model: `no_valid_edge`
- Signal: `neutral`
- Reliability: `Low confidence`
- no_valid_edge remains correct: `True`

## Derivatives Coverage
- Binance derivatives recovered: `False`
- Manual CSV derivatives used: `False`
- `Binance USD-M Futures` / `funding_rate`: `skipped`, rows `0`, dates `nan` to `nan`, missing `100.0%`, used `False`
- `Binance USD-M Futures` / `open_interest`: `skipped`, rows `0`, dates `nan` to `nan`, missing `100.0%`, used `False`
- `Binance USD-M Futures` / `long_short_ratio`: `skipped`, rows `0`, dates `nan` to `nan`, missing `100.0%`, used `False`
- `Binance USD-M Futures` / `top_trader_account_ratio`: `skipped`, rows `0`, dates `nan` to `nan`, missing `100.0%`, used `False`
- `Binance USD-M Futures` / `top_trader_position_ratio`: `skipped`, rows `0`, dates `nan` to `nan`, missing `100.0%`, used `False`
- `Binance USD-M Futures` / `taker_buy_sell_ratio`: `skipped`, rows `0`, dates `nan` to `nan`, missing `100.0%`, used `False`
- `Binance USD-M Futures` / `basis`: `skipped`, rows `0`, dates `nan` to `nan`, missing `100.0%`, used `False`

## Derivatives Impact
- No derivatives impact comparison ran because no valid derivative feature entered the model feature set.

## Feature Group Watchlist
- `dollar_rates_only` / `logistic_linear`: samples `99`, accuracy `68.7%`, Sharpe `1.01`, drawdown `-18.2%`, Brier `0.215`, calibration error `0.097`, reliability `Medium confidence`.
- Remains strongest diagnostic group: `True`
- It is not selected unless it passes the official 30d selection rules in the main leaderboard.

## Feature Group Stability
- `all_features` / `logistic_linear`: `12` stable slices
- `all_features` / `random_forest`: `12` stable slices
- `dollar_rates_only` / `logistic_linear`: `12` stable slices
- `price_momentum_only` / `random_forest`: `12` stable slices
- `all_features` / `hgb`: `11` stable slices

## Conclusion
- Derivatives improved 30d official diagnostics: `False`
- `no_valid_edge` remains correct: `True`
- Recommended next step: Keep no_valid_edge and prioritize validated derivative history plus feature pruning before adding more models.
