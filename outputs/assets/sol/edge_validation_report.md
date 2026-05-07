# SOL Edge Validation + BTC No-Edge Drilldown

## Run Summary
- Run ID: `sol_research_20260507T125935Z`
- Asset: `SOL`
- Selected model: `logistic_linear`
- Signal: `neutral`
- Reliability: `Medium confidence`

## SOL Selection Audit
- `all_features` / `logistic_linear`: accuracy `69.4%`, after-cost return `982.7%`, conclusion `current_official_reference`.
- `price_plus_risk_assets` / `logistic_linear`: accuracy `69.4%`, after-cost return `788.0%`, conclusion `accuracy_not_enough_without_reference_improvement`.
- `sol_dollar_rates_only` / `random_forest`: accuracy `69.4%`, after-cost return `984.0%`, conclusion `accuracy_not_enough_without_reference_improvement`.
- `price_plus_macro` / `logistic_linear`: accuracy `69.4%`, after-cost return `936.4%`, conclusion `accuracy_not_enough_without_reference_improvement`.
- `sol_price_plus_ecosystem` / `logistic_linear`: accuracy `63.9%`, after-cost return `458.4%`, conclusion `rejected_quality_tradeoff`.

## SOL Signal Policy Deployment Check
- `sol_price_plus_ecosystem` / `logistic_linear` has `12` active signals, hit rate `75.0%`, after-cost return `255.0%`.
- Deployment role: `diagnostic_interpretation_layer`. Conclusion: `useful_interpretation_limited_active_support`.
- The signal policy remains an interpretation layer; it cannot change `selected_model`.

## FRED Follow-Up
- `T10YIE` worked on this refresh.

## Conclusion
- No new model family or paid data source was introduced.
- Polymarket remains diagnostic-only.
- BTC remains neutral unless a candidate passes all official gates naturally.
