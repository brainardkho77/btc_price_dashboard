# Feature Pruning Summary

## Run Summary
- Asset: `SOL`
- Run ID: `sol_research_20260529T101537Z`
- Selected model: `random_forest`
- Signal: `neutral`
- Reliability: `Medium confidence`
- Current all-feature best ML: `random_forest` at `69.4%` accuracy, Brier `0.236`, net return `1034.3%`.
- Best baseline: `random_permutation` at `55.6%` accuracy.

## Best Pruned Candidate
- `all_features_official` / `random_forest`: samples `36`, accuracy `69.4%`, Brier `0.236`, calibration `0.075`, Sharpe `1.34`, drawdown `-37.4%`, net return `1034.3%`.
- Stability evidence: bootstrap lower bound `55.6%`, bootstrap upper bound `83.3%`, permutation p-value `0.006`.
- Promotion eligible: `False`. Reason: `current_all_features_reference`

## Promotion Decision
- No pruned feature set passed the strict promotion rules. The official conclusion is unchanged.

## Promising But Rejected Candidates
- `all_features_official` / `logistic_linear`: label `promising_but_rejected`, accuracy `69.4%`, net return `1240.2%`, CI low `n/a`, p-value `n/a`, reason `current_all_features_reference`.
- `all_features_official` / `random_forest`: label `promising_but_rejected`, accuracy `69.4%`, net return `1034.3%`, CI low `n/a`, p-value `n/a`, reason `current_all_features_reference`.
- `all_features_official` / `hgb`: label `promising_but_rejected`, accuracy `63.9%`, net return `663.5%`, CI low `n/a`, p-value `n/a`, reason `current_all_features_reference`.

## Signal Quality
- Signal quality rows were unavailable.

## Strongest 30d Factors
- `macro_us_3m_bill_rate` (`dollar_rates_only`): IC `0.426`, p-value `0.006`, coverage `100.0%`, keep `True`.
- `btc_dominance_regime_btc_safe_haven` (`btc_dominance_regime`): IC `0.421`, p-value `0.007`, coverage `100.0%`, keep `True`.
- `macro_real_5y_yield` (`dollar_rates_only`): IC `0.400`, p-value `0.011`, coverage `100.0%`, keep `True`.
- `cross_asset_tlt_z_365d` (`risk_assets_only`): IC `-0.394`, p-value `0.012`, coverage `100.0%`, keep `True`.
- `solana_ecosystem_tvl_log` (`sol_ecosystem_only`): IC `-0.389`, p-value `0.014`, coverage `100.0%`, keep `True`.
- `cross_asset_tlt_ret_30d` (`risk_assets_only`): IC `-0.388`, p-value `0.014`, coverage `100.0%`, keep `True`.
- `trend_log_price` (`price_momentum_only`): IC `-0.385`, p-value `0.015`, coverage `100.0%`, keep `True`.
- `macro_fed_balance_sheet_z_365d` (`macro_liquidity_only`): IC `-0.385`, p-value `0.015`, coverage `100.0%`, keep `True`.
- `trend_dist_ath` (`price_momentum_only`): IC `-0.383`, p-value `0.016`, coverage `100.0%`, keep `True`.
- `solana_ecosystem_stablecoin_supply_log` (`sol_ecosystem_only`): IC `-0.377`, p-value `0.018`, coverage `100.0%`, keep `True`.

## Noisy Or Redundant Factor Reasons
- `weak_ic`: `30` features

## Recommendation
- Neutral signal justified: `True`
- Next step: only consider more model complexity after a promoted feature set survives these pruning and signal-quality checks.
