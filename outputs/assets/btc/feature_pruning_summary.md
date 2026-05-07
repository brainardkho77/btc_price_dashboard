# Feature Pruning Summary

## Run Summary
- Asset: `BTC`
- Run ID: `btc_research_20260507T183236Z`
- Selected model: `no_valid_edge`
- Signal: `neutral`
- Reliability: `Low confidence`
- Current all-feature best ML: `logistic_linear` at `50.0%` accuracy, Brier `0.266`, net return `450.9%`.
- Best baseline: `momentum_90d` at `55.0%` accuracy.

## Best Pruned Candidate
- `all_features_official` / `logistic_linear`: samples `100`, accuracy `50.0%`, Brier `0.266`, calibration `0.113`, Sharpe `0.62`, drawdown `-66.6%`, net return `450.9%`.
- Stability evidence: bootstrap lower bound `41.0%`, bootstrap upper bound `60.0%`, permutation p-value `0.605`.
- Promotion eligible: `False`. Reason: `current_all_features_reference`

## Promotion Decision
- No pruned feature set passed the strict promotion rules. The official conclusion is unchanged.

## Promising But Rejected Candidates
- `all_features_official` / `logistic_linear`: label `promising_but_rejected`, accuracy `50.0%`, net return `450.9%`, CI low `n/a`, p-value `n/a`, reason `current_all_features_reference`.
- `all_features_official` / `hgb`: label `promising_but_rejected`, accuracy `47.0%`, net return `389.5%`, CI low `n/a`, p-value `n/a`, reason `current_all_features_reference`.
- `all_features_official` / `random_forest`: label `promising_but_rejected`, accuracy `44.0%`, net return `246.3%`, CI low `n/a`, p-value `n/a`, reason `current_all_features_reference`.

## BTC No-Edge Drilldown
- `btc_dollar_rates_cycle` / `logistic_linear` was unavailable in the pruning report.

## Signal Quality
- Signal quality rows were unavailable.

## Strongest 30d Factors
- `derivatives_sum_open_interest_value` (`derivatives_only`): IC `-0.355`, p-value `0.002`, coverage `67.0%`, keep `True`.
- `macro_us_10y_2y_spread_z_365d` (`dollar_rates_only`): IC `0.296`, p-value `0.002`, coverage `100.0%`, keep `True`.
- `cycle_cos_4y` (`price_momentum_only`): IC `0.288`, p-value `0.003`, coverage `100.0%`, keep `True`.
- `derivatives_sum_open_interest_z_365d` (`derivatives_only`): IC `-0.269`, p-value `0.028`, coverage `64.0%`, keep `True`.
- `onchain_supply_z_365d` (`onchain_only`): IC `-0.257`, p-value `0.009`, coverage `100.0%`, keep `True`.
- `macro_financial_stress_index_chg_7d` (`risk_assets_only`): IC `0.250`, p-value `0.011`, coverage `100.0%`, keep `True`.
- `derivatives_taker_buy_sell_ratio_z_90d` (`derivatives_only`): IC `-0.245`, p-value `0.044`, coverage `65.0%`, keep `True`.
- `macro_real_5y_yield_chg_90d` (`dollar_rates_only`): IC `-0.240`, p-value `0.014`, coverage `100.0%`, keep `True`.
- `macro_us_10y_2y_spread_chg_90d` (`dollar_rates_only`): IC `0.240`, p-value `0.014`, coverage `100.0%`, keep `True`.
- `macro_vix_ret_30d` (`risk_assets_only`): IC `0.239`, p-value `0.015`, coverage `100.0%`, keep `True`.

## Noisy Or Redundant Factor Reasons
- `weak_ic`: `67` features

## Recommendation
- Neutral signal justified: `True`
- Next step: only consider more model complexity after a promoted feature set survives these pruning and signal-quality checks.
