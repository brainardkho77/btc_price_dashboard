# Feature Pruning Summary

## Run Summary
- Asset: `S&P 500 / SPX`
- Run ID: `spx_research_20260507T130913Z`
- Selected model: `no_valid_edge`
- Signal: `neutral`
- Reliability: `Low confidence`
- Current all-feature best ML: `logistic_linear` at `64.6%` accuracy, Brier `0.230`, net return `198.2%`.
- Best baseline: `buy_hold_direction` at `67.7%` accuracy.

## Best Pruned Candidate
- `all_features_official` / `logistic_linear`: samples `99`, accuracy `64.6%`, Brier `0.230`, calibration `0.127`, Sharpe `0.90`, drawdown `-22.9%`, net return `198.2%`.
- Stability evidence: bootstrap lower bound `55.6%`, bootstrap upper bound `73.7%`, permutation p-value `0.745`.
- Promotion eligible: `False`. Reason: `current_all_features_reference`

## Promotion Decision
- No pruned feature set passed the strict promotion rules. The official conclusion is unchanged.

## Promising But Rejected Candidates
- `all_features_official` / `logistic_linear`: label `promising_but_rejected`, accuracy `64.6%`, net return `198.2%`, CI low `n/a`, p-value `n/a`, reason `current_all_features_reference`.
- `all_features_official` / `random_forest`: label `promising_but_rejected`, accuracy `62.6%`, net return `158.8%`, CI low `n/a`, p-value `n/a`, reason `current_all_features_reference`.
- `all_features_official` / `hgb`: label `promising_but_rejected`, accuracy `59.6%`, net return `143.3%`, CI low `n/a`, p-value `n/a`, reason `current_all_features_reference`.

## Signal Quality
- Signal quality rows were unavailable.

## Strongest 30d Factors
- `macro_vix_ret_1d` (`risk_assets_only`): IC `-0.264`, p-value `0.007`, coverage `100.0%`, keep `True`.
- `cross_asset_btc_ret_30d` (`risk_assets_only`): IC `0.249`, p-value `0.011`, coverage `100.0%`, keep `True`.
- `macro_us_10y_2y_spread_z_365d` (`dollar_rates_only`): IC `0.246`, p-value `0.012`, coverage `100.0%`, keep `True`.
- `macro_financial_stress_index` (`risk_assets_only`): IC `0.235`, p-value `0.018`, coverage `100.0%`, keep `True`.
- `momentum_ret_180d` (`price_momentum_only`): IC `-0.217`, p-value `0.029`, coverage `100.0%`, keep `True`.
- `macro_financial_stress_index_chg_90d` (`risk_assets_only`): IC `0.212`, p-value `0.033`, coverage `100.0%`, keep `True`.
- `momentum_rsi_14d` (`price_momentum_only`): IC `-0.203`, p-value `0.041`, coverage `100.0%`, keep `True`.
- `volatility_realized_90d` (`price_momentum_only`): IC `0.199`, p-value `0.046`, coverage `100.0%`, keep `True`.
- `volatility_drawdown_90d` (`price_momentum_only`): IC `-0.197`, p-value `0.047`, coverage `100.0%`, keep `True`.
- `trend_dist_sma_200d` (`price_momentum_only`): IC `-0.195`, p-value `0.050`, coverage `100.0%`, keep `True`.

## Noisy Or Redundant Factor Reasons
- `weak_ic`: `47` features

## Recommendation
- Neutral signal justified: `True`
- Next step: only consider more model complexity after a promoted feature set survives these pruning and signal-quality checks.
