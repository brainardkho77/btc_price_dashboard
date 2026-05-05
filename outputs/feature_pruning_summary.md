# Feature Pruning Summary

## Run Summary
- Asset: `BTC`
- Run ID: `btc_research_20260505T051934Z`
- Selected model: `no_valid_edge`
- Signal: `neutral`
- Reliability: `Low confidence`
- Current all-feature best ML: `random_forest` at `54.0%` accuracy, Brier `0.247`, net return `1086.8%`.
- Best baseline: `momentum_90d` at `55.0%` accuracy.

## Best Pruned Candidate
- `btc_dollar_rates_cycle` / `logistic_linear`: samples `100`, accuracy `58.0%`, Brier `0.226`, calibration `0.070`, Sharpe `0.87`, drawdown `-43.8%`, net return `1597.1%`.
- Stability evidence: bootstrap lower bound `45.5%`, bootstrap upper bound `67.8%`, permutation p-value `0.088`.
- Promotion eligible: `False`. Reason: `failed_bootstrap_or_permutation_stability_check`

## Promotion Decision
- No pruned feature set passed the strict promotion rules. The official conclusion is unchanged.

## Promising But Rejected Candidates
- `btc_dollar_rates_cycle` / `logistic_linear`: label `promising_but_unstable`, accuracy `58.0%`, net return `1597.1%`, CI low `45.5%`, p-value `0.088`, reason `failed_bootstrap_or_permutation_stability_check`.
- `core_price_macro_risk` / `random_forest`: label `promising_but_unstable`, accuracy `58.0%`, net return `1358.3%`, CI low `49.0%`, p-value `0.064`, reason `did_not_improve_current_all_features_without_material_metric_worsening; failed_bootstrap_or_permutation_stability_check`.
- `btc_dollar_rates_cycle` / `random_forest`: label `promising_but_unstable`, accuracy `57.0%`, net return `1503.4%`, CI low `47.0%`, p-value `0.120`, reason `did_not_improve_current_all_features_without_material_metric_worsening; failed_bootstrap_or_permutation_stability_check`.
- `price_plus_stablecoins` / `logistic_linear`: label `promising_but_unstable`, accuracy `57.0%`, net return `1321.8%`, CI low `47.2%`, p-value `0.104`, reason `failed_bootstrap_or_permutation_stability_check`.
- `price_plus_dollar_rates` / `random_forest`: label `promising_but_unstable`, accuracy `57.0%`, net return `1132.3%`, CI low `47.2%`, p-value `0.104`, reason `failed_bootstrap_or_permutation_stability_check`.
- `btc_core_dollar_derivatives_cycle` / `random_forest`: label `promising_but_unstable`, accuracy `57.0%`, net return `1132.3%`, CI low `47.2%`, p-value `0.104`, reason `failed_bootstrap_or_permutation_stability_check`.
- `price_momentum_only` / `logistic_linear`: label `promising_but_unstable`, accuracy `56.0%`, net return `1211.0%`, CI low `46.2%`, p-value `0.159`, reason `did_not_improve_current_all_features_without_material_metric_worsening; failed_bootstrap_or_permutation_stability_check`.
- `price_plus_risk_assets` / `logistic_linear`: label `promising_but_unstable`, accuracy `56.0%`, net return `768.9%`, CI low `47.0%`, p-value `0.191`, reason `failed_official_selection_gates; failed_bootstrap_or_permutation_stability_check; low_reliability`.

## BTC No-Edge Drilldown
- `btc_dollar_rates_cycle` / `logistic_linear` is classified as `promising but unstable`: accuracy `58.0%`, net return `1597.1%`, bootstrap lower bound `45.5%`, permutation p-value `0.088`.
- Rejection reason: `failed_bootstrap_or_permutation_stability_check`

## Signal Quality
- `core_price_macro_risk` / `random_forest`: long signals `63`, abstention `37.0%`, long hit rate `57.1%`, long average return `6.4%`, net return `1358.3%`.
- `btc_dollar_rates_cycle` / `random_forest`: long signals `60`, abstention `40.0%`, long hit rate `56.7%`, long average return `6.8%`, net return `1503.4%`.
- `btc_dollar_rates_cycle` / `logistic_linear`: long signals `71`, abstention `29.0%`, long hit rate `56.3%`, long average return `5.8%`, net return `1597.1%`.
- `price_plus_stablecoins` / `logistic_linear`: long signals `64`, abstention `36.0%`, long hit rate `56.2%`, long average return `6.2%`, net return `1321.8%`.
- `price_plus_dollar_rates` / `random_forest`: long signals `64`, abstention `36.0%`, long hit rate `56.2%`, long average return `6.1%`, net return `1132.3%`.

## Strongest 30d Factors
- `derivatives_sum_open_interest_value` (`derivatives_only`): IC `-0.356`, p-value `0.002`, coverage `67.0%`, keep `True`.
- `derivatives_taker_buy_sell_ratio_z_90d` (`derivatives_only`): IC `-0.298`, p-value `0.013`, coverage `65.0%`, keep `True`.
- `cycle_cos_4y` (`price_momentum_only`): IC `0.288`, p-value `0.003`, coverage `100.0%`, keep `True`.
- `onchain_supply_z_365d` (`onchain_only`): IC `-0.257`, p-value `0.009`, coverage `100.0%`, keep `True`.
- `macro_vix_ret_30d` (`risk_assets_only`): IC `0.239`, p-value `0.015`, coverage `100.0%`, keep `True`.
- `cross_asset_spx_ret_30d` (`risk_assets_only`): IC `-0.210`, p-value `0.034`, coverage `100.0%`, keep `True`.
- `derivatives_sum_open_interest_z_90d` (`derivatives_only`): IC `-0.207`, p-value `0.090`, coverage `66.0%`, keep `True`.
- `macro_reverse_repo_chg_7d` (`macro_liquidity_only`): IC `0.198`, p-value `0.046`, coverage `100.0%`, keep `True`.
- `derivatives_sum_open_interest` (`derivatives_only`): IC `-0.198`, p-value `0.104`, coverage `67.0%`, keep `True`.
- `cross_asset_gold_ret_30d` (`risk_assets_only`): IC `-0.185`, p-value `0.063`, coverage `100.0%`, keep `True`.

## Noisy Or Redundant Factor Reasons
- `weak_ic`: `49` features

## Recommendation
- Neutral signal justified: `True`
- Next step: only consider more model complexity after a promoted feature set survives these pruning and signal-quality checks.
