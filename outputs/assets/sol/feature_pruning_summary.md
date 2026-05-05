# Feature Pruning Summary

## Run Summary
- Asset: `SOL`
- Run ID: `sol_research_20260505T162108Z`
- Selected model: `random_forest`
- Signal: `neutral`
- Reliability: `Medium confidence`
- Current all-feature best ML: `random_forest` at `69.4%` accuracy, Brier `0.215`, net return `1114.7%`.
- Best baseline: `random_permutation` at `55.6%` accuracy.

## Best Pruned Candidate
- `price_plus_risk_assets` / `logistic_linear`: samples `36`, accuracy `69.4%`, Brier `0.218`, calibration `0.080`, Sharpe `1.15`, drawdown `-37.4%`, net return `576.4%`.
- Stability evidence: bootstrap lower bound `55.6%`, bootstrap upper bound `83.3%`, permutation p-value `0.032`.
- Promotion eligible: `False`. Reason: `did_not_improve_current_all_features_without_material_metric_worsening`

## Promotion Decision
- No pruned feature set passed the strict promotion rules. The official conclusion is unchanged.

## Promising But Rejected Candidates
- `all_features` / `logistic_linear`: label `promising_but_unstable`, accuracy `66.7%`, net return `942.1%`, CI low `50.0%`, p-value `0.048`, reason `did_not_improve_current_all_features_without_material_metric_worsening; failed_bootstrap_or_permutation_stability_check`.
- `no_onchain` / `logistic_linear`: label `promising_but_unstable`, accuracy `66.7%`, net return `942.1%`, CI low `50.0%`, p-value `0.048`, reason `did_not_improve_current_all_features_without_material_metric_worsening; failed_bootstrap_or_permutation_stability_check`.
- `no_derivatives` / `logistic_linear`: label `promising_but_unstable`, accuracy `66.7%`, net return `942.1%`, CI low `50.0%`, p-value `0.048`, reason `did_not_improve_current_all_features_without_material_metric_worsening; failed_bootstrap_or_permutation_stability_check`.
- `no_polymarket` / `logistic_linear`: label `promising_but_unstable`, accuracy `66.7%`, net return `942.1%`, CI low `50.0%`, p-value `0.048`, reason `did_not_improve_current_all_features_without_material_metric_worsening; failed_bootstrap_or_permutation_stability_check`.
- `sol_dollar_rates_only` / `random_forest`: label `promising_but_unstable`, accuracy `66.7%`, net return `767.3%`, CI low `50.0%`, p-value `0.024`, reason `did_not_improve_current_all_features_without_material_metric_worsening; failed_bootstrap_or_permutation_stability_check`.
- `price_plus_dollar_rates` / `random_forest`: label `promising_but_unstable`, accuracy `63.9%`, net return `587.7%`, CI low `47.2%`, p-value `0.092`, reason `did_not_improve_current_all_features_without_material_metric_worsening; failed_bootstrap_or_permutation_stability_check`.
- `sol_rates_risk_price` / `hgb`: label `promising_but_unstable`, accuracy `63.9%`, net return `581.3%`, CI low `47.2%`, p-value `0.088`, reason `did_not_improve_current_all_features_without_material_metric_worsening; failed_bootstrap_or_permutation_stability_check`.
- `sol_risk_assets_only` / `logistic_linear`: label `promising_but_unstable`, accuracy `63.9%`, net return `532.4%`, CI low `47.2%`, p-value `0.112`, reason `did_not_improve_current_all_features_without_material_metric_worsening; failed_bootstrap_or_permutation_stability_check`.

## Signal Quality
- `price_plus_macro` / `logistic_linear`: long signals `21`, abstention `41.7%`, long hit rate `66.7%`, long average return `16.1%`, net return `987.2%`.
- `price_plus_risk_assets` / `logistic_linear`: long signals `21`, abstention `41.7%`, long hit rate `66.7%`, long average return `13.3%`, net return `576.4%`.
- `price_momentum_only` / `logistic_linear`: long signals `14`, abstention `61.1%`, long hit rate `64.3%`, long average return `17.1%`, net return `394.5%`.
- `core_price_macro_risk` / `logistic_linear`: long signals `22`, abstention `38.9%`, long hit rate `63.6%`, long average return `14.8%`, net return `863.2%`.
- `all_features` / `logistic_linear`: long signals `24`, abstention `33.3%`, long hit rate `62.5%`, long average return `13.9%`, net return `942.1%`.

## Strongest 30d Factors
- `cross_asset_tlt_z_365d` (`risk_assets_only`): IC `-0.394`, p-value `0.012`, coverage `100.0%`, keep `True`.
- `cross_asset_tlt_ret_30d` (`risk_assets_only`): IC `-0.388`, p-value `0.014`, coverage `100.0%`, keep `True`.
- `macro_fed_balance_sheet_z_365d` (`macro_liquidity_only`): IC `-0.385`, p-value `0.015`, coverage `100.0%`, keep `True`.
- `trend_log_price` (`price_momentum_only`): IC `-0.385`, p-value `0.015`, coverage `100.0%`, keep `True`.
- `trend_dist_ath` (`price_momentum_only`): IC `-0.383`, p-value `0.016`, coverage `100.0%`, keep `True`.
- `stablecoins_supply_log` (`stablecoins_only`): IC `-0.367`, p-value `0.022`, coverage `100.0%`, keep `True`.
- `macro_reverse_repo` (`macro_liquidity_only`): IC `0.362`, p-value `0.023`, coverage `100.0%`, keep `True`.
- `trend_dist_sma_20d` (`price_momentum_only`): IC `0.358`, p-value `0.025`, coverage `100.0%`, keep `True`.
- `trend_dist_sma_10d` (`price_momentum_only`): IC `0.347`, p-value `0.031`, coverage `100.0%`, keep `True`.
- `macro_fed_funds_rate` (`dollar_rates_only`): IC `0.345`, p-value `0.032`, coverage `100.0%`, keep `True`.

## Noisy Or Redundant Factor Reasons
- `weak_ic`: `16` features

## Recommendation
- Neutral signal justified: `True`
- Next step: only consider more model complexity after a promoted feature set survives these pruning and signal-quality checks.
