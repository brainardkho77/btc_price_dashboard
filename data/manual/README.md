# Manual Binance CSV Fallbacks

Use these templates only when the official Binance USD-M Futures endpoints are unavailable from the runtime network.

Rules:
- Keep one row per UTC date.
- Use ISO dates: `YYYY-MM-DD`.
- Leave files with only headers if no verified data is available.
- Do not paste estimates or fabricated values.

Templates:
- `btc_funding.csv`: `date,funding_rate`
- `btc_open_interest.csv`: `date,sum_open_interest,sum_open_interest_value`
- `btc_long_short_ratio.csv`: `date,long_short_ratio,long_account,short_account`
- `btc_taker_buy_sell_ratio.csv`: `date,taker_buy_sell_ratio,taker_buy_volume,taker_sell_volume`
- `btc_basis.csv`: `date,basis,basis_rate,annualized_basis_rate`
