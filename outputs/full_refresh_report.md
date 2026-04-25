# Full Refresh Report

## Run Summary
- Run ID: `btc_research_20260425T065919Z`
- Created at: `2026-04-25T06:59:19+00:00`
- Schema version: `1.0`
- Quick mode: `False`
- Data snapshot hash: `d20271283a19eee5`
- Period: `2015-01-01` to `2026-04-25`
- Feature count: `122`
- Models run: `buy_hold_direction, momentum_30d, momentum_90d, random_permutation, logistic_linear, hgb, random_forest`
- Warnings: FRED and Coin Metrics are revised historical data unless point-in-time vintage data is added.

## Data Source Summary
Worked sources:
- `coinbase_btc_usd_candles` (Coinbase Exchange): 3933 rows, 2015-07-20 to 2026-04-25
- `yahoo_btc_usd` (Yahoo chart API): 4239 rows, 2014-09-17 to 2026-04-25
- `yahoo_market_proxies` (Yahoo chart API): 4296 rows, 2014-01-02 to 2026-04-25
- `fred_macro` (FRED CSV downloads): 26231 rows, 1954-07-01 to 2026-04-24
- `coinmetrics_btc_daily` (Coin Metrics Community API): 4497 rows, 2014-01-01 to 2026-04-24
- `alternative_fear_greed` (Alternative.me Fear & Greed): 3002 rows, 2018-02-01 to 2026-04-25
- `defillama_stablecoins` (DefiLlama Stablecoins API): 3070 rows, 2017-11-29 to 2026-04-25

Failed sources:
- `binance_funding_rate`: HTTPSConnectionPool(host='fapi.binance.com', port=443): Max retries exceeded with url: /fapi/v1/fundingRate?symbol=BTCUSDT&limit=1000 (Caused by ConnectTimeoutError(<HTTPSConnection(host='fapi.binance.com', port=443) at 
- `binance_open_interest`: HTTPSConnectionPool(host='fapi.binance.com', port=443): Max retries exceeded with url: /futures/data/openInterestHist?symbol=BTCUSDT&period=1d&limit=500 (Caused by ConnectTimeoutError(<HTTPSConnection(host='fapi.binance.
- `binance_long_short_ratio`: HTTPSConnectionPool(host='fapi.binance.com', port=443): Max retries exceeded with url: /futures/data/globalLongShortAccountRatio?symbol=BTCUSDT&period=1d&limit=500 (Caused by ConnectTimeoutError(<HTTPSConnection(host='fa
- `binance_taker_buy_sell_ratio`: HTTPSConnectionPool(host='fapi.binance.com', port=443): Max retries exceeded with url: /futures/data/takerlongshortRatio?symbol=BTCUSDT&period=1d&limit=500 (Caused by ConnectTimeoutError(<HTTPSConnection(host='fapi.binan
- `binance_basis`: HTTPSConnectionPool(host='fapi.binance.com', port=443): Max retries exceeded with url: /futures/data/basis?pair=BTCUSDT&contractType=PERPETUAL&period=1d&limit=500 (Caused by ConnectTimeoutError(<HTTPSConnection(host='fap

Skipped sources:
- `coingecko_btc_usd`: Skipped because Coinbase/Yahoo price data was available.
- `manual_csv`: No manual CSV inputs were configured for this run.

Unavailable sources:
- `spot_btc_etf_flows`: No free source configured; ETF flows were not fabricated.

Key source checks:
- Binance derivatives worked: `False`
- DefiLlama stablecoins worked: `True`
- Coinbase/Yahoo/FRED/Coin Metrics worked: `True`

## Model Quality Summary
30d official leaderboard:
- `momentum_90d` rank 1: samples 99, directional accuracy 55.6%, Brier 0.249, after-cost return 840.4%, reliability `Low confidence`, selection eligible `False`
- `random_forest` rank 2: samples 99, directional accuracy 51.5%, Brier 0.243, after-cost return 711.2%, reliability `Low confidence`, selection eligible `False`
- `momentum_30d` rank 3: samples 99, directional accuracy 51.5%, Brier 0.257, after-cost return 442.6%, reliability `Low confidence`, selection eligible `False`
- `hgb` rank 4: samples 99, directional accuracy 50.5%, Brier 0.258, after-cost return 863.5%, reliability `Low confidence`, selection eligible `False`
- `buy_hold_direction` rank 5: samples 99, directional accuracy 50.5%, Brier 0.485, after-cost return 740.6%, reliability `Low confidence`, selection eligible `False`
- `logistic_linear` rank 6: samples 99, directional accuracy 45.5%, Brier 0.278, after-cost return 215.5%, reliability `Low confidence`, selection eligible `False`
- `random_permutation` rank 7: samples 99, directional accuracy 44.4%, Brier 0.272, after-cost return 53.8%, reliability `Low confidence`, selection eligible `False`

Selected model result:
- Selected model: `no_valid_edge`
- Signal: `neutral`
- Probability up: `50.0%`
- Model-implied forecast price: `$77,556`
- Reliability label: `Low confidence`
- no_valid_edge triggered: `True`

Baseline and usefulness checks for the top selected row shown in the leaderboard context:
- Beats buy-and-hold direction: `yes`
- Beats 30d momentum: `yes`
- Beats 90d momentum: `no`
- Beats random/permutation baseline: `yes`
- Useful model: `no`
- Sample count: `99`
- Reliability label: `Low confidence`

## Quick vs Full Refresh Comparison
- Quick run ID: `btc_research_20260425T064120Z`
- Full run ID: `btc_research_20260425T065919Z`
- Period changed: `2018-01-01` to `2026-04-25` -> `2015-01-01` to `2026-04-25`
- Feature count changed: `122` -> `122`
- Selected model changed: `hgb` -> `no_valid_edge`
- Confidence changed: `Medium confidence` -> `Low confidence`
- Signal changed: `neutral` -> `neutral`
- 30d model-implied forecast price changed: `$70,634` -> `$77,556`

Data source status changes:
- `binance_basis`: `skipped` -> `failed`
- `binance_funding_rate`: `skipped` -> `failed`
- `binance_long_short_ratio`: `skipped` -> `failed`
- `binance_open_interest`: `skipped` -> `failed`
- `binance_taker_buy_sell_ratio`: `skipped` -> `failed`

## Edge And Deployment Recommendation
- Edge still holds: `False`
- Final confidence level: `Low confidence`
- Deployment recommendation: `deploy`
- Reason: all validation gates passed; deploying reports the stronger full-refresh result, including no_valid_edge if no model qualifies.

Validation gates completed:
- Full refresh completed: `true`
- pytest passed: `true`
- Schemas valid: `true`
- quick_mode false: `True`
- schema_version exists: `True`
- data_snapshot_hash exists: `True`
- Streamlit precomputed-file boot smoke test: `true`
- Required outputs present: `true`
- Leakage/schema validation errors: `none observed`
