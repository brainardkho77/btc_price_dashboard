# Feature Pruning Summary

## Run Summary
- Asset: `BTC`
- Run ID: `btc_research_20260506T060635Z`
- Selected model: `no_valid_edge`
- Signal: `neutral`
- Reliability: `Low confidence`
- Current all-feature best ML: `hgb` at `49.0%` accuracy, Brier `0.270`, net return `724.8%`.
- Best baseline: `momentum_90d` at `55.0%` accuracy.

## Best Pruned Candidate
- `all_features_official` / `hgb`: samples `100`, accuracy `49.0%`, Brier `0.270`, calibration `0.118`, Sharpe `0.71`, drawdown `-42.9%`, net return `724.8%`.
- Stability evidence: bootstrap lower bound `39.0%`, bootstrap upper bound `59.0%`, permutation p-value `0.681`.
- Promotion eligible: `False`. Reason: `current_all_features_reference`

## Promotion Decision
- No pruned feature set passed the strict promotion rules. The official conclusion is unchanged.

## Promising But Rejected Candidates
- `all_features_official` / `hgb`: label `promising_but_rejected`, accuracy `49.0%`, net return `724.8%`, CI low `n/a`, p-value `n/a`, reason `current_all_features_reference`.
- `all_features_official` / `logistic_linear`: label `promising_but_rejected`, accuracy `49.0%`, net return `353.4%`, CI low `n/a`, p-value `n/a`, reason `current_all_features_reference`.
- `all_features_official` / `random_forest`: label `promising_but_rejected`, accuracy `44.0%`, net return `239.5%`, CI low `n/a`, p-value `n/a`, reason `current_all_features_reference`.

## BTC No-Edge Drilldown
- `btc_dollar_rates_cycle` / `logistic_linear` was unavailable in the pruning report.

## Signal Quality
- Signal quality rows were unavailable.

## Strongest 30d Factors
- `derivatives_sum_open_interest_value` (`derivatives_only`): IC `-0.354`, p-value `0.002`, coverage `67.0%`, keep `True`.
- `macro_us_10y_2y_spread_z_365d` (`dollar_rates_only`): IC `0.296`, p-value `0.002`, coverage `100.0%`, keep `True`.
- `cycle_cos_4y` (`price_momentum_only`): IC `0.288`, p-value `0.003`, coverage `100.0%`, keep `True`.
- `derivatives_sum_open_interest_z_365d` (`derivatives_only`): IC `-0.277`, p-value `0.023`, coverage `64.0%`, keep `True`.
- `onchain_supply_z_365d` (`onchain_only`): IC `-0.257`, p-value `0.009`, coverage `100.0%`, keep `True`.
- `macro_financial_stress_index_chg_7d` (`risk_assets_only`): IC `0.248`, p-value `0.011`, coverage `100.0%`, keep `True`.
- `macro_real_5y_yield_chg_90d` (`dollar_rates_only`): IC `-0.240`, p-value `0.014`, coverage `100.0%`, keep `True`.
- `macro_us_10y_2y_spread_chg_90d` (`dollar_rates_only`): IC `0.240`, p-value `0.014`, coverage `100.0%`, keep `True`.
- `macro_vix_ret_30d` (`risk_assets_only`): IC `0.239`, p-value `0.015`, coverage `100.0%`, keep `True`.
- `derivatives_taker_buy_sell_ratio_z_90d` (`derivatives_only`): IC `-0.229`, p-value `0.062`, coverage `65.0%`, keep `True`.

## Noisy Or Redundant Factor Reasons
- `weak_ic`: `68` features

## Recommendation
- Neutral signal justified: `True`
- Next step: only consider more model complexity after a promoted feature set survives these pruning and signal-quality checks.
