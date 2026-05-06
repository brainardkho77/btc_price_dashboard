# SOL Edge Validation + BTC No-Edge Drilldown

## Run Summary
- Run ID: `btc_research_20260506T165426Z`
- Asset: `BTC`
- Selected model: `no_valid_edge`
- Signal: `neutral`
- Reliability: `Low confidence`

## BTC No-Edge Drilldown
- `price_plus_stablecoins` / `logistic_linear`: category `promising_but_statistically_unstable`, accuracy `57.0%`, CI low `47.2%`, p-value `0.104`.
  Rejection: `failed_bootstrap_or_permutation_stability_check; failed_regime_stability_check`.
- `btc_dollar_rates_cycle` / `logistic_linear`: category `promising_but_statistically_unstable`, accuracy `57.0%`, CI low `46.2%`, p-value `0.100`.
  Rejection: `failed_bootstrap_or_permutation_stability_check`.
- `price_plus_dollar_rates` / `logistic_linear`: category `genuinely_noisy`, accuracy `53.0%`, CI low `43.0%`, p-value `0.355`.
  Rejection: `failed_official_model_selection_gates; failed_bootstrap_or_permutation_stability_check`.

## FRED Follow-Up
- `T10YIE` worked on this refresh.

## Conclusion
- No new model family or paid data source was introduced.
- Polymarket remains diagnostic-only.
- BTC remains neutral unless a candidate passes all official gates naturally.
