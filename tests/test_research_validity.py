import numpy as np
import pandas as pd
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

sys.path.append(str(Path(__file__).resolve().parents[1]))

import research_pipeline as rp
from research_config import ResearchConfig, build_source_specs, get_asset_config
import data_sources
from data_sources import FRED_SERIES, fetch_fred_api_series, fred_api_key, fred_api_key_configured, sanitize_fred_error
from research_pipeline import (
    MANUAL_DERIVATIVE_SPECS,
    _is_monthly_polymarket_event,
    _parse_polymarket_threshold,
    _parse_binance_archive_zip,
    _defillama_chart_frame,
    apply_release_lags,
    build_features,
    build_target_frame,
    build_walk_forward_windows,
    compute_metrics,
    decorate_summary,
    initial_data_availability,
    raw_column_source_group,
    select_feature_columns,
    select_primary_model,
    validate_manual_derivative_csv,
    window_fingerprint,
)
from diagnostic_outputs import (
    _btc_trend_masks,
    _material_worsening,
    active_signal_floor,
    asset_deployability_report,
    asset_feature_set_and_policy_diagnostics,
    btc_no_edge_drilldown,
    candidate_feature_sets,
    derivatives_coverage,
    feature_pruning_report,
    feature_group_for_feature,
    fred_macro_impact_report,
    fred_vs_fallback_summary,
    high_confidence_signal_floor,
    is_official_long_policy_allowed,
    latest_signal_interpretation_frame,
    macro_candidate_impact_report,
    select_nested_pruned_features,
    sol_selection_audit,
    sol_deployability_audit,
    sol_signal_policy_deployment_check,
    sol_stability_candidate_pairs,
)
from schemas import DIAGNOSTIC_OUTPUT_SCHEMAS, OUTPUT_SCHEMAS, empty_diagnostic_frame, empty_output_frame, validate_frame


def synthetic_research_raw(rows=1500):
    dates = pd.date_range("2018-01-01", periods=rows, freq="D")
    base = 10_000 * np.exp(np.linspace(0, 1.1, rows) + np.sin(np.arange(rows) / 35) * 0.08)
    price = pd.Series(base, index=dates)
    raw = pd.DataFrame(
        {
            "btc_open": price * 0.99,
            "btc_high": price * 1.02,
            "btc_low": price * 0.98,
            "btc_close": price,
            "btc_volume": 1_000_000 + np.arange(rows) * 100,
            "eth_close": price / 10,
            "spx_close": 3000 + np.arange(rows),
            "nasdaq_close": 8000 + np.arange(rows),
            "vix_close": 20 + np.sin(np.arange(rows) / 20),
            "dxy_close": 100 + np.cos(np.arange(rows) / 40),
            "gold_close": 1500 + np.arange(rows) * 0.2,
            "tlt_close": 100 + np.sin(np.arange(rows) / 50),
            "cm_active_addresses": 700_000 + np.arange(rows),
            "cm_tx_count": 300_000 + np.arange(rows),
            "cm_hash_rate": 100_000_000 + np.arange(rows) * 1000,
            "cm_mvrv": 2.0 + np.sin(np.arange(rows) / 90) * 0.2,
            "cm_exchange_inflow_usd": 1_000_000 + np.arange(rows) * 10,
            "cm_exchange_outflow_usd": 1_100_000 + np.arange(rows) * 11,
            "cm_market_cap_usd": price * 19_000_000,
            "cm_exchange_supply_usd": price * 1_000_000,
            "cm_spot_volume_usd": 5_000_000 + np.arange(rows) * 100,
            "fear_greed_value": 50 + np.sin(np.arange(rows) / 30) * 20,
            "us_10y_yield": 4.0 + np.sin(np.arange(rows) / 100) * 0.5,
            "trade_weighted_usd": 120 + np.cos(np.arange(rows) / 70),
            "m2_money_supply": 20_000 + np.arange(rows),
            "stablecoin_total_circulating_usd": 100_000_000_000 + np.arange(rows) * 10_000_000,
            "binance_funding_rate": 0.0001 + np.sin(np.arange(rows) / 10) * 0.00005,
        },
        index=dates,
    )
    return raw


def synthetic_sol_raw(rows=900):
    dates = pd.date_range("2021-01-01", periods=rows, freq="D")
    price = pd.Series(25 * np.exp(np.linspace(0, 1.0, rows) + np.sin(np.arange(rows) / 25) * 0.12), index=dates)
    return pd.DataFrame(
        {
            "asset_open": price * 0.99,
            "asset_high": price * 1.03,
            "asset_low": price * 0.97,
            "asset_close": price,
            "asset_volume": 5_000_000 + np.arange(rows) * 1000,
            "btc_proxy_close": 30_000 * np.exp(np.linspace(0, 0.5, rows)),
            "eth_close": 1500 * np.exp(np.linspace(0, 0.45, rows)),
            "spx_close": 3800 + np.arange(rows),
            "vix_close": 22 + np.sin(np.arange(rows) / 30),
            "us_10y_yield": 3.5 + np.sin(np.arange(rows) / 100) * 0.4,
            "trade_weighted_usd": 120 + np.cos(np.arange(rows) / 70),
            "m2_money_supply": 20_000 + np.arange(rows),
            "stablecoin_total_circulating_usd": 120_000_000_000 + np.arange(rows) * 5_000_000,
            "solana_stablecoin_circulating_usd": 5_000_000_000 + np.arange(rows) * 3_000_000,
            "solana_tvl_usd": 3_000_000_000 + np.arange(rows) * 2_000_000,
            "solana_dex_volume_usd": 500_000_000 + np.arange(rows) * 1_000_000,
            "solana_fees_revenue_usd": 1_000_000 + np.arange(rows) * 2_000,
            "fear_greed_value": 50 + np.sin(np.arange(rows) / 30) * 20,
        },
        index=dates,
    )


def test_asset_config_resolves_btc_sol_and_spx_symbols():
    btc = get_asset_config("btc")
    sol = get_asset_config("sol")
    spx = get_asset_config("spx")
    assert btc.coinbase_product == "BTC-USD"
    assert btc.enable_derivatives is True
    assert btc.enable_onchain is True
    assert sol.coinbase_product == "SOL-USD"
    assert sol.yahoo_symbol == "SOL-USD"
    assert sol.coingecko_id == "solana"
    assert sol.enable_derivatives is False
    assert sol.enable_onchain is False
    assert spx.display_name == "S&P 500 / SPX"
    assert spx.yahoo_symbol == "SPY"
    assert spx.coinbase_product is None
    assert spx.coingecko_id is None
    assert spx.enable_coinbase is False
    assert spx.enable_coingecko is False
    assert spx.enable_derivatives is False
    assert spx.enable_onchain is False
    assert spx.enable_polymarket is False
    assert spx.enable_crypto_sentiment is False
    assert spx.enable_stablecoins is False


def test_sol_source_specs_record_skipped_sources_honestly():
    sol = get_asset_config("sol")
    availability = initial_data_availability("run", "2026-01-01T00:00:00+00:00", build_source_specs(sol))
    assert "coinbase_sol_usd_candles" in set(availability["dataset"])
    assert "yahoo_sol_usd" in set(availability["dataset"])
    assert "coingecko_sol_usd" in set(availability["dataset"])
    assert "spot_sol_etf_flows" in set(availability["dataset"])
    assert "defillama_solana_tvl" in set(availability["dataset"])
    assert "defillama_solana_dex_volume" in set(availability["dataset"])
    etf = availability[availability["dataset"] == "spot_sol_etf_flows"].iloc[0]
    assert etf["status"] == "unavailable"
    assert bool(etf["is_used_in_model"]) is False


def test_spx_source_specs_mark_crypto_native_sources_not_used():
    spx = get_asset_config("spx")
    specs = {spec.dataset: spec for spec in build_source_specs(spx)}
    assert specs["coinbase_spx_usd_candles"].endpoint == "not configured for this asset"
    assert specs["coinbase_spx_usd_candles"].is_used_in_model is False
    assert specs["yahoo_spx_usd"].endpoint.endswith("/SPY")
    assert specs["coingecko_spx_usd"].endpoint == "not configured for this asset"
    assert specs["coingecko_spx_usd"].is_used_in_model is False
    assert specs["coinmetrics_spx_daily"].is_used_in_model is False
    assert specs["alternative_fear_greed"].is_used_in_model is False
    assert specs["defillama_stablecoins"].is_used_in_model is False
    assert specs["polymarket_spx_monthly_ladders"].endpoint == "not configured for this asset"
    assert specs["polymarket_spx_monthly_ladders"].is_used_in_model is False


def test_fred_api_key_is_environment_only_and_source_specs_include_fallback(monkeypatch, tmp_path):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.setattr(data_sources, "ROOT", tmp_path)
    assert fred_api_key_configured() is False
    monkeypatch.setenv("FRED_API_KEY", "test_key_from_environment")
    assert fred_api_key_configured() is True
    specs = {spec.dataset: spec for spec in build_source_specs(get_asset_config("btc"))}
    assert "fred_macro_api" in specs
    assert "fred_macro" in specs
    assert specs["fred_macro_api"].source == "FRED API"
    assert specs["fred_macro_api"].endpoint == "https://api.stlouisfed.org/fred/series/observations"
    assert specs["fred_macro"].source == "FRED CSV downloads"
    env_example = Path(__file__).resolve().parents[1] / ".env.example"
    assert env_example.read_text(encoding="utf-8").strip() == "FRED_API_KEY="
    assert "DFII10" in FRED_SERIES
    assert "WTREGEN" in FRED_SERIES


def test_fred_api_key_can_load_from_ignored_dotenv(monkeypatch, tmp_path):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.setattr(data_sources, "ROOT", tmp_path)
    key_name = "FRED" + "_API_KEY"
    (tmp_path / ".env").write_text(f"{key_name}=dotenv_secret_value\n", encoding="utf-8")
    assert fred_api_key() == "dotenv_secret_value"
    assert fred_api_key_configured() is True


def test_fred_api_request_uses_live_endpoint_and_sanitizes_failures(monkeypatch):
    class FakeResponse:
        status_code = 200

        def json(self):
            return {"observations": [{"date": "2024-01-01", "value": "4.25"}]}

    captured = {}

    def fake_get(url, params, headers, timeout):
        captured["url"] = url
        captured["params"] = params
        captured["headers"] = headers
        return FakeResponse()

    monkeypatch.setenv("FRED_API_KEY", "secret_value_for_test")
    monkeypatch.setattr(data_sources.requests, "get", fake_get)
    series = fetch_fred_api_series("DGS10")
    assert captured["url"] == "https://api.stlouisfed.org/fred/series/observations"
    assert captured["params"]["api_key"] == "secret_value_for_test"
    assert float(series.iloc[0]) == 4.25
    assert "secret_value_for_test" not in sanitize_fred_error("url?api_key=secret_value_for_test")


def test_sol_target_generation_uses_future_sol_prices_only():
    config = ResearchConfig()
    raw = synthetic_sol_raw()
    feature_result = build_features(raw, config)
    frame = build_target_frame(raw, feature_result.features, feature_result.feature_cols, 30)
    row_date = raw.index[100]
    expected = np.log(raw.loc[raw.index[130], "asset_close"] / raw.loc[row_date, "asset_close"])
    assert np.isclose(frame.loc[row_date, "target_log_return"], expected)
    assert pd.isna(frame["target_log_return"].iloc[-1])
    assert "cross_asset_btc_ret_30d" in feature_result.features.columns


def test_spx_yahoo_target_uses_adjusted_close_when_available(monkeypatch, tmp_path):
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    chart = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [100.0, 101.0, 102.0, 103.0],
            "adjclose": [90.0, 91.0, 92.0, 93.0],
            "volume": [1_000, 1_100, 1_200, 1_300],
        },
        index=dates,
    )
    monkeypatch.setattr(rp, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(rp, "fetch_yahoo_chart", lambda symbol: chart)
    frame = rp.fetch_yahoo_asset_price(get_asset_config("spx"), force=True)
    assert frame["asset_close"].tolist() == [90.0, 91.0, 92.0, 93.0]
    assert frame.attrs["adjusted_close_used"] is True


def test_spx_feature_generation_has_no_same_target_spx_proxy_when_raw_is_filtered():
    raw = synthetic_research_raw(rows=900).rename(
        columns={
            "btc_open": "asset_open",
            "btc_high": "asset_high",
            "btc_low": "asset_low",
            "btc_close": "asset_close",
            "btc_volume": "asset_volume",
        }
    )
    raw = raw.drop(columns=["spx_close"], errors="ignore")
    feature_result = build_features(raw, ResearchConfig())
    assert "cross_asset_spx_ret_30d" not in feature_result.features.columns
    assert "cross_asset_spx_ret_30d" not in feature_result.feature_cols


def test_solana_ecosystem_features_are_audited_and_candidate_packed():
    config = ResearchConfig()
    raw = synthetic_sol_raw(rows=900)
    feature_result = build_features(raw, config)
    solana_features = [col for col in feature_result.features.columns if col.startswith("solana_ecosystem_")]
    assert solana_features
    audit = feature_result.feature_audit[feature_result.feature_audit["source"] == "solana_ecosystem"]
    assert not audit.empty
    assert audit["release_delay_days"].eq(1).all()
    sets = candidate_feature_sets(feature_result.feature_cols, "sol")
    assert "sol_ecosystem_pack" in sets
    assert any(col.startswith("solana_ecosystem_") for col in sets["sol_ecosystem_pack"])


def test_target_generation_uses_future_prices_only():
    config = ResearchConfig()
    raw = synthetic_research_raw()
    feature_result = build_features(raw, config)
    frame = build_target_frame(raw, feature_result.features, feature_result.feature_cols, 30)
    row_date = raw.index[100]
    expected = np.log(raw.loc[raw.index[130], "btc_close"] / raw.loc[row_date, "btc_close"])
    assert np.isclose(frame.loc[row_date, "target_log_return"], expected)
    assert pd.isna(frame["target_log_return"].iloc[-1])


def test_future_and_target_columns_do_not_enter_feature_cols():
    config = ResearchConfig()
    dates = pd.date_range("2020-01-01", periods=500, freq="D")
    features = pd.DataFrame(
        {
            "momentum_ret_30d": np.random.default_rng(1).normal(size=500),
            "future_return_30d": np.random.default_rng(2).normal(size=500),
            "target_up": np.random.default_rng(3).normal(size=500),
            "shifted_target_30d": np.random.default_rng(4).normal(size=500),
            "forward_log_return": np.random.default_rng(5).normal(size=500),
        },
        index=dates,
    )
    cols = select_feature_columns(features, config)
    assert "momentum_ret_30d" in cols
    assert "future_return_30d" not in cols
    assert "target_up" not in cols
    assert "shifted_target_30d" not in cols
    assert "forward_log_return" not in cols


def test_non_price_external_features_are_lagged_conservatively():
    config = ResearchConfig()
    raw = synthetic_research_raw(rows=80)
    raw["polymarket_upside_probability"] = np.linspace(0.2, 0.8, len(raw))
    raw["real_10y_yield"] = np.linspace(1.0, 2.0, len(raw))
    raw["us_3m_bill_rate"] = np.linspace(4.0, 5.0, len(raw))
    raw["treasury_general_account"] = np.linspace(700_000, 800_000, len(raw))
    raw["solana_tvl_usd"] = np.linspace(1_000_000_000, 2_000_000_000, len(raw))
    released = apply_release_lags(raw, config)
    assert released["btc_close"].iloc[10] == raw["btc_close"].iloc[10]
    assert released["binance_funding_rate"].iloc[10] == raw["binance_funding_rate"].iloc[9]
    assert released["stablecoin_total_circulating_usd"].iloc[10] == raw["stablecoin_total_circulating_usd"].iloc[9]
    assert released["us_10y_yield"].iloc[10] == raw["us_10y_yield"].iloc[8]
    assert released["m2_money_supply"].iloc[40] == raw["m2_money_supply"].iloc[10]
    assert released["cm_active_addresses"].iloc[10] == raw["cm_active_addresses"].iloc[8]
    assert released["polymarket_upside_probability"].iloc[10] == raw["polymarket_upside_probability"].iloc[9]
    assert released["real_10y_yield"].iloc[10] == raw["real_10y_yield"].iloc[8]
    assert released["us_3m_bill_rate"].iloc[40] == raw["us_3m_bill_rate"].iloc[10]
    assert released["treasury_general_account"].iloc[40] == raw["treasury_general_account"].iloc[10]
    assert released["solana_tvl_usd"].iloc[10] == raw["solana_tvl_usd"].iloc[9]
    assert raw_column_source_group("solana_tvl_usd") == "solana_ecosystem"


def test_monthly_date_alignment_for_30d_official_windows():
    config = ResearchConfig(quick_initial_train_days=365)
    raw = synthetic_research_raw(rows=900)
    feature_result = build_features(raw, config)
    frame = build_target_frame(raw, feature_result.features, feature_result.feature_cols, 30)
    windows = build_walk_forward_windows(frame, 30, "official_monthly", config, quick=True)
    months = [w.test_date.to_period("M") for w in windows]
    assert len(months) == len(set(months))
    assert all((b.test_date - a.test_date).days >= 25 for a, b in zip(windows, windows[1:]))


def test_identical_model_windows_are_enforced_by_shared_fingerprint():
    config = ResearchConfig(quick_initial_train_days=365)
    raw = synthetic_research_raw(rows=900)
    feature_result = build_features(raw, config)
    frame = build_target_frame(raw, feature_result.features, feature_result.feature_cols, 90)
    windows_a = build_walk_forward_windows(frame, 90, "official_quarterly", config, quick=True)
    windows_b = build_walk_forward_windows(frame, 90, "official_quarterly", config, quick=True)
    assert window_fingerprint(windows_a) == window_fingerprint(windows_b)


def test_transaction_costs_reduce_returns_when_turnover_exists():
    dates = pd.date_range("2021-01-01", periods=6, freq="ME")
    preds = pd.DataFrame(
        {
            "probability_up": [0.9, 0.1, 0.9, 0.1, 0.9, 0.1],
            "expected_log_return": [0.05, -0.03, 0.04, -0.02, 0.03, -0.01],
            "actual_log_return": [0.04, -0.02, 0.03, -0.01, 0.02, -0.01],
            "actual_return": np.exp([0.04, -0.02, 0.03, -0.01, 0.02, -0.01]) - 1,
            "actual_up": [1, 0, 1, 0, 1, 0],
            "predicted_up": [1, 0, 1, 0, 1, 0],
            "position": [1, 0, 1, 0, 1, 0],
        },
        index=dates,
    )
    metrics, returns_frame = compute_metrics(preds, 30, "official_monthly", transaction_cost_bps=25)
    assert returns_frame["tc_adjusted_strategy_return"].sum() < returns_frame["gross_strategy_return"].sum()
    assert metrics["tc_adjusted_return"] < metrics["gross_return"]


def test_diagnostic_schemas_include_required_outputs():
    for filename in [
        "model_rejection_reasons.csv",
        "feature_signal_diagnostics.csv",
        "feature_group_ablation.csv",
        "model_regime_breakdown.csv",
        "derivatives_coverage.csv",
        "derivatives_impact.csv",
        "feature_group_stability.csv",
        "polymarket_coverage.csv",
        "polymarket_feature_diagnostics.csv",
        "polymarket_impact.csv",
        "factor_quality_scorecard.csv",
        "signal_quality_report.csv",
        "pruned_feature_leaderboard.csv",
        "feature_pruning_report.csv",
        "sol_stability_report.csv",
        "signal_policy_report.csv",
        "asset_feature_set_leaderboard.csv",
        "fred_macro_impact_report.csv",
        "macro_candidate_impact_report.csv",
        "fred_vs_fallback_summary.csv",
        "sol_selection_audit.csv",
        "sol_signal_policy_deployment.csv",
        "sol_deployability_audit.csv",
        "btc_no_edge_drilldown.csv",
        "asset_deployability_report.csv",
        "sol_deployability_audit.csv",
    ]:
        assert filename in DIAGNOSTIC_OUTPUT_SCHEMAS
    assert "latest_signal_interpretation.csv" in OUTPUT_SCHEMAS
    assert "balanced_accuracy" in DIAGNOSTIC_OUTPUT_SCHEMAS["derivatives_impact.csv"]
    assert "calibration_error" in DIAGNOSTIC_OUTPUT_SCHEMAS["derivatives_impact.csv"]
    assert "beats_momentum_90d" in DIAGNOSTIC_OUTPUT_SCHEMAS["derivatives_impact.csv"]
    assert "passes_official_gates" in DIAGNOSTIC_OUTPUT_SCHEMAS["polymarket_impact.csv"]
    assert "promotion_eligible" in DIAGNOSTIC_OUTPUT_SCHEMAS["pruned_feature_leaderboard.csv"]
    assert "long_hit_rate" in DIAGNOSTIC_OUTPUT_SCHEMAS["signal_quality_report.csv"]
    assert "correlation_cluster" in DIAGNOSTIC_OUTPUT_SCHEMAS["factor_quality_scorecard.csv"]
    assert "improvement_vs_all_features_accuracy" in DIAGNOSTIC_OUTPUT_SCHEMAS["feature_pruning_report.csv"]
    assert "passes_stability_check" in DIAGNOSTIC_OUTPUT_SCHEMAS["sol_stability_report.csv"]
    assert "active_signal_count" in DIAGNOSTIC_OUTPUT_SCHEMAS["signal_policy_report.csv"]
    assert "promotion_eligible" in DIAGNOSTIC_OUTPUT_SCHEMAS["asset_feature_set_leaderboard.csv"]
    assert "series_id" in DIAGNOSTIC_OUTPUT_SCHEMAS["fred_macro_impact_report.csv"]
    assert "tc_adjusted_return" in DIAGNOSTIC_OUTPUT_SCHEMAS["macro_candidate_impact_report.csv"]
    assert "accuracy_delta" in DIAGNOSTIC_OUTPUT_SCHEMAS["fred_vs_fallback_summary.csv"]
    assert "audit_conclusion" in DIAGNOSTIC_OUTPUT_SCHEMAS["sol_selection_audit.csv"]
    assert "can_change_selected_model" in DIAGNOSTIC_OUTPUT_SCHEMAS["sol_signal_policy_deployment.csv"]
    assert "rejection_category" in DIAGNOSTIC_OUTPUT_SCHEMAS["btc_no_edge_drilldown.csv"]
    assert "equity_chart_role" in DIAGNOSTIC_OUTPUT_SCHEMAS["asset_deployability_report.csv"]
    assert "unavailable_gates" in DIAGNOSTIC_OUTPUT_SCHEMAS["sol_deployability_audit.csv"]


def test_btc_up_down_slice_creation_uses_known_btc_history():
    dates = pd.date_range("2023-01-01", periods=240, freq="D")
    btc = pd.Series(np.r_[np.linspace(100, 200, 140), np.linspace(200, 120, 100)], index=dates)
    raw = pd.DataFrame({"btc_proxy_close": btc}, index=dates)
    masks = _btc_trend_masks(raw, dates)
    assert masks["BTC-up"].any()
    assert masks["BTC-down"].any()
    assert not (masks["BTC-up"] & masks["BTC-down"]).any()


def test_fred_macro_impact_report_schema_and_series_status():
    config = ResearchConfig()
    raw = synthetic_research_raw(rows=900)
    feature_result = build_features(raw, config)
    fred_audit = feature_result.feature_audit[feature_result.feature_audit["source"].str.startswith("fred_macro")]
    feature_name = str(fred_audit.iloc[0]["feature_name"])
    availability = pd.DataFrame(
        [
            {
                "run_id": "btc_research_test",
                "generated_at": "2026-01-01T00:00:00+00:00",
                "source": "FRED API",
                "endpoint": "https://api.stlouisfed.org/fred/series/observations",
                "dataset": "fred_macro_api:DGS10",
                "status": "worked",
                "requested_fields": "us_10y_yield",
                "available_fields": "us_10y_yield",
                "rows": 900,
                "first_date": "2018-01-01",
                "last_date": "2020-06-18",
                "failure_reason": "",
                "is_used_in_model": True,
                "revision_warning": "FRED API data is revised historical data, not point-in-time vintages.",
            }
        ]
    )
    feature_signal = pd.DataFrame(
        [
            {
                "feature_name": feature_name,
                "horizon": 30,
                "window_type": "official_monthly",
                "information_coefficient": 0.12,
                "ic_p_value": 0.05,
            }
        ]
    )
    factor_quality = pd.DataFrame(
        [
            {
                "feature_name": feature_name,
                "horizon": 30,
                "window_type": "official_monthly",
                "keep_candidate": True,
            }
        ]
    )
    report = fred_macro_impact_report("btc_research_test", raw, feature_result, config, availability, feature_signal, factor_quality)
    validate_frame("fred_macro_impact_report.csv", report)
    assert not report.empty
    assert "DGS10" in set(report["series_id"])
    assert report["release_lag_days"].isin([2, 30]).all()


def test_macro_candidate_and_fred_summary_reports_validate_without_baseline(tmp_path):
    asset_rows = pd.DataFrame(
        [
            {
                "candidate_feature_set": "btc_dollar_rates_cycle",
                "model_name": "logistic_linear",
                "horizon": 30,
                "window_type": "official_monthly",
                "n_features": 8,
                "n_samples": 36,
                "directional_accuracy": 0.57,
                "balanced_accuracy": 0.56,
                "brier_score": 0.24,
                "calibration_error": 0.08,
                "net_return": 1.2,
                "sharpe": 0.8,
                "max_drawdown": -0.2,
                "bootstrap_ci_low": 0.48,
                "bootstrap_ci_high": 0.66,
                "permutation_p_value": 0.18,
                "beats_buy_hold": True,
                "beats_momentum_30d": True,
                "beats_momentum_90d": False,
                "beats_random_baseline": True,
                "promotion_eligible": False,
                "rejection_reason": "failed_bootstrap_or_permutation_stability_check",
            }
        ]
    )
    macro_report = macro_candidate_impact_report("btc_research_test", asset_rows)
    validate_frame("macro_candidate_impact_report.csv", macro_report)
    latest = pd.DataFrame([{"is_primary_objective": True, "selected_model": "no_valid_edge"}])
    leaderboard = pd.DataFrame(
        [
            {
                "horizon": 30,
                "window_type": "official_monthly",
                "model": "momentum_90d",
                "rank": 1,
                "directional_accuracy": 0.55,
            }
        ]
    )
    availability = pd.DataFrame(
        [
            {"dataset": "fred_macro_api", "status": "worked"},
            {"dataset": "fred_macro", "status": "skipped"},
        ]
    )
    summary = fred_vs_fallback_summary("unit_research_test", tmp_path, None, latest, leaderboard, availability, macro_report)
    validate_frame("fred_vs_fallback_summary.csv", summary)
    assert summary.iloc[0]["notes"] == "baseline_unavailable"


def test_sol_selection_audit_explains_higher_accuracy_rejection():
    rows = pd.DataFrame(
        [
            {
                "run_id": "sol_research_test",
                "asset_id": "sol",
                "horizon": 30,
                "window_type": "official_monthly",
                "candidate_feature_set": "all_features",
                "model_name": "logistic_linear",
                "feature_selection_method": "all_features_reference",
                "n_features": 100,
                "n_samples": 36,
                "directional_accuracy": 0.694,
                "balanced_accuracy": 0.694,
                "brier_score": 0.25,
                "calibration_error": 0.11,
                "sharpe": 1.1,
                "max_drawdown": -0.37,
                "net_return": 9.80,
                "beats_buy_hold": True,
                "beats_momentum_30d": True,
                "beats_momentum_90d": True,
                "beats_random_baseline": True,
                "beats_current_reference": False,
                "material_worsening": False,
                "regime_stability_pass": False,
                "bootstrap_ci_low": np.nan,
                "bootstrap_ci_high": np.nan,
                "permutation_p_value": np.nan,
                "promotion_eligible": False,
                "reliability_label": "Medium confidence",
                "window_fingerprint": "",
                "rejection_reason": "current_all_features_reference",
            },
            {
                "run_id": "sol_research_test",
                "asset_id": "sol",
                "horizon": 30,
                "window_type": "official_monthly",
                "candidate_feature_set": "price_plus_risk_assets",
                "model_name": "logistic_linear",
                "feature_selection_method": "nested_train_only_pruned",
                "n_features": 20,
                "n_samples": 36,
                "directional_accuracy": 0.722,
                "balanced_accuracy": 0.722,
                "brier_score": 0.23,
                "calibration_error": 0.08,
                "sharpe": 1.2,
                "max_drawdown": -0.23,
                "net_return": 9.50,
                "beats_buy_hold": True,
                "beats_momentum_30d": True,
                "beats_momentum_90d": True,
                "beats_random_baseline": True,
                "beats_current_reference": False,
                "material_worsening": False,
                "regime_stability_pass": True,
                "bootstrap_ci_low": 0.58,
                "bootstrap_ci_high": 0.86,
                "permutation_p_value": 0.02,
                "promotion_eligible": False,
                "reliability_label": "Medium confidence",
                "window_fingerprint": "abc",
                "rejection_reason": "did_not_improve_current_all_features_reference",
            },
        ]
    )
    audit = sol_selection_audit("sol_research_test", rows)
    validate_frame("sol_selection_audit.csv", audit)
    challenger = audit[audit["candidate_feature_set"] == "price_plus_risk_assets"].iloc[0]
    assert challenger["audit_conclusion"] == "higher_accuracy_but_lower_after_cost_return"
    assert challenger["accuracy_delta_vs_reference"] > 0
    assert challenger["return_delta_vs_reference"] < 0


def test_sol_signal_policy_deployment_cannot_change_model_and_flags_limited_support():
    policy = pd.DataFrame(
        [
            {
                "run_id": "sol_research_test",
                "asset_id": "sol",
                "horizon": 30,
                "window_type": "official_monthly",
                "candidate_feature_set": "sol_price_plus_ecosystem",
                "model_name": "logistic_linear",
                "n_samples": 36,
                "policy_source": "train_calibration_only",
                "long_threshold_median": 0.55,
                "expected_return_min_median": 0.0,
                "threshold_050_used": False,
                "threshold_050_diagnostic_only": True,
                "active_signal_count": 12,
                "active_coverage": 0.333,
                "abstention_rate": 0.667,
                "active_hit_rate": 0.75,
                "missed_up_month_rate": 0.5,
                "after_cost_return": 2.55,
                "max_drawdown": -0.20,
                "bootstrap_ci_low": 0.52,
                "bootstrap_ci_high": 0.92,
                "permutation_p_value": 0.04,
                "risk_off_count": 8,
                "risk_off_rate": 0.22,
                "risk_off_hit_rate": 0.60,
                "risk_off_avg_return": -0.05,
                "risk_off_probability_threshold_median": 0.45,
                "valid_signal_policy": True,
                "high_confidence_policy_eligible": False,
                "promoted_signal_policy": True,
                "rejection_reason": "promotion_eligible",
            }
        ]
    )
    latest_interp = pd.DataFrame(
        [
            {
                "selected_model": "logistic_linear",
                "strategy_action": "cash",
                "risk_label": "Neutral / no edge",
            }
        ]
    )
    report = sol_signal_policy_deployment_check("sol_research_test", policy, latest_interp)
    validate_frame("sol_signal_policy_deployment.csv", report)
    row = report.iloc[0]
    assert bool(row["can_change_selected_model"]) is False
    assert bool(row["enough_active_signals"]) is True
    assert bool(row["high_confidence_active_floor_met"]) is False
    assert row["audit_conclusion"] == "useful_interpretation_limited_active_support"


def _sol_feature_candidate(feature_set="all_features", model_name="logistic_linear", **overrides):
    row = {
        "run_id": "sol_research_test",
        "asset_id": "sol",
        "horizon": 30,
        "window_type": "official_monthly",
        "candidate_feature_set": feature_set,
        "model_name": model_name,
        "feature_selection_method": "all_features_reference",
        "n_features": 100,
        "n_samples": 36,
        "directional_accuracy": 0.694,
        "balanced_accuracy": 0.694,
        "brier_score": 0.25,
        "calibration_error": 0.11,
        "sharpe": 1.1,
        "max_drawdown": -0.37,
        "net_return": 9.80,
        "beats_buy_hold": True,
        "beats_momentum_30d": True,
        "beats_momentum_90d": True,
        "beats_random_baseline": True,
        "beats_current_reference": False,
        "material_worsening": False,
        "regime_stability_pass": False,
        "bootstrap_ci_low": np.nan,
        "bootstrap_ci_high": np.nan,
        "permutation_p_value": np.nan,
        "promotion_eligible": False,
        "reliability_label": "Medium confidence",
        "window_fingerprint": "",
        "rejection_reason": "current_all_features_reference",
    }
    row.update(overrides)
    return row


def test_sol_deployability_audit_blocks_missing_gate_sources():
    latest = pd.DataFrame(
        [
            {
                "run_id": "sol_research_test",
                "generated_at": "2026-01-01T00:00:00+00:00",
                "as_of_date": "2026-01-01",
                "horizon": 30,
                "selected_model": "logistic_linear",
                "signal": "neutral",
                "predicted_probability_up": 0.44,
                "expected_return": -0.04,
                "reliability_label": "Medium confidence",
                "is_primary_objective": True,
            }
        ]
    )
    interp = pd.DataFrame(
        [
            {
                "selected_model": "logistic_linear",
                "selected_feature_set": "all_features",
                "strategy_action": "cash",
                "risk_label": "Neutral / no edge",
            }
        ]
    )
    feature_sets = pd.DataFrame([_sol_feature_candidate()])
    out = sol_deployability_audit(
        "sol_research_test",
        "2026-01-01T00:00:00+00:00",
        latest,
        interp,
        feature_sets,
        empty_diagnostic_frame("signal_policy_report.csv"),
        empty_diagnostic_frame("sol_stability_report.csv"),
    )
    validate_frame("sol_deployability_audit.csv", out)
    row = out.iloc[0]
    assert bool(row["is_selected_model"]) is True
    assert row["deployability_decision"] == "diagnostic_only"
    assert "bootstrap_ci_low_above_50" in row["unavailable_gates"]
    assert "signal_policy_available" in row["failed_gates"]


def test_sol_deployability_audit_only_includes_available_challengers():
    latest = pd.DataFrame(
        [
            {
                "horizon": 30,
                "selected_model": "logistic_linear",
                "signal": "neutral",
                "predicted_probability_up": 0.44,
                "expected_return": -0.04,
                "is_primary_objective": True,
            }
        ]
    )
    interp = pd.DataFrame([{"selected_model": "logistic_linear", "selected_feature_set": "all_features", "strategy_action": "cash", "risk_label": "Neutral / no edge"}])
    feature_sets = pd.DataFrame(
        [
            _sol_feature_candidate(),
            _sol_feature_candidate("price_plus_macro", "logistic_linear", bootstrap_ci_low=0.56, permutation_p_value=0.03),
        ]
    )
    out = sol_deployability_audit(
        "sol_research_test",
        "2026-01-01T00:00:00+00:00",
        latest,
        interp,
        feature_sets,
        empty_diagnostic_frame("signal_policy_report.csv"),
        empty_diagnostic_frame("sol_stability_report.csv"),
    )
    assert set(out["feature_set"]) == {"all_features", "price_plus_macro"}


def test_sol_deployability_requires_latest_long_action_and_high_active_floor():
    latest = pd.DataFrame(
        [
            {
                "horizon": 30,
                "selected_model": "logistic_linear",
                "signal": "neutral",
                "predicted_probability_up": 0.44,
                "expected_return": -0.04,
                "is_primary_objective": True,
            }
        ]
    )
    interp = pd.DataFrame([{"selected_model": "logistic_linear", "selected_feature_set": "all_features", "strategy_action": "cash", "risk_label": "Neutral / no edge"}])
    feature_sets = pd.DataFrame(
        [
            _sol_feature_candidate(
                bootstrap_ci_low=0.56,
                permutation_p_value=0.03,
                regime_stability_pass=True,
            )
        ]
    )
    policy = pd.DataFrame(
        [
            {
                "horizon": 30,
                "window_type": "official_monthly",
                "candidate_feature_set": "all_features",
                "model_name": "logistic_linear",
                "n_samples": 36,
                "policy_source": "train_calibration_only",
                "active_signal_count": 12,
                "active_coverage": 0.33,
                "active_hit_rate": 0.75,
                "missed_up_month_rate": 0.5,
                "valid_signal_policy": True,
            }
        ]
    )
    stability = pd.DataFrame(
        [
            {
                "candidate_feature_set": "all_features",
                "model_name": "logistic_linear",
                "period_slice": slice_name,
                "directional_accuracy": 0.65,
                "passes_stability_check": True,
            }
            for slice_name in ["BTC-up", "BTC-down", "2023-2024", "2024-present"]
        ]
    )
    out = sol_deployability_audit("sol_research_test", "2026-01-01T00:00:00+00:00", latest, interp, feature_sets, policy, stability)
    row = out.iloc[0]
    assert row["deployability_decision"] == "diagnostic_only"
    assert "high_confidence_active_signal_floor" in row["failed_gates"]
    assert "latest_deployable_action" in row["failed_gates"]


def test_btc_no_edge_drilldown_classifies_target_candidates():
    rows = pd.DataFrame(
        [
            {
                "run_id": "btc_research_test",
                "asset_id": "btc",
                "horizon": 30,
                "window_type": "official_monthly",
                "candidate_feature_set": "all_features",
                "model_name": "hgb",
                "n_samples": 100,
                "directional_accuracy": 0.55,
                "brier_score": 0.25,
                "calibration_error": 0.09,
                "net_return": 11.0,
                "max_drawdown": -0.42,
                "beats_buy_hold": True,
                "beats_momentum_30d": True,
                "beats_momentum_90d": False,
                "beats_random_baseline": True,
                "beats_current_reference": False,
                "material_worsening": False,
                "regime_stability_pass": False,
                "bootstrap_ci_low": np.nan,
                "bootstrap_ci_high": np.nan,
                "permutation_p_value": np.nan,
                "promotion_eligible": False,
                "reliability_label": "Low confidence",
                "rejection_reason": "current_all_features_reference",
            },
            {
                "run_id": "btc_research_test",
                "asset_id": "btc",
                "horizon": 30,
                "window_type": "official_monthly",
                "candidate_feature_set": "price_plus_stablecoins",
                "model_name": "logistic_linear",
                "n_samples": 100,
                "directional_accuracy": 0.57,
                "brier_score": 0.24,
                "calibration_error": 0.08,
                "net_return": 13.0,
                "max_drawdown": -0.54,
                "beats_buy_hold": True,
                "beats_momentum_30d": True,
                "beats_momentum_90d": True,
                "beats_random_baseline": True,
                "beats_current_reference": True,
                "material_worsening": True,
                "regime_stability_pass": False,
                "bootstrap_ci_low": 0.47,
                "bootstrap_ci_high": 0.66,
                "permutation_p_value": 0.11,
                "promotion_eligible": False,
                "reliability_label": "Medium confidence",
                "rejection_reason": "materially_worsened_brier_calibration_or_drawdown; failed_regime_stability_check",
            },
            {
                "run_id": "btc_research_test",
                "asset_id": "btc",
                "horizon": 30,
                "window_type": "official_monthly",
                "candidate_feature_set": "btc_dollar_rates_cycle",
                "model_name": "logistic_linear",
                "n_samples": 100,
                "directional_accuracy": 0.56,
                "brier_score": 0.24,
                "calibration_error": 0.13,
                "net_return": 10.8,
                "max_drawdown": -0.42,
                "beats_buy_hold": True,
                "beats_momentum_30d": True,
                "beats_momentum_90d": True,
                "beats_random_baseline": True,
                "beats_current_reference": False,
                "material_worsening": True,
                "regime_stability_pass": True,
                "bootstrap_ci_low": 0.46,
                "bootstrap_ci_high": 0.67,
                "permutation_p_value": 0.15,
                "promotion_eligible": False,
                "reliability_label": "Medium confidence",
                "rejection_reason": "failed_bootstrap_or_permutation_stability_check",
            },
        ]
    )
    report = btc_no_edge_drilldown("btc_research_test", rows)
    validate_frame("btc_no_edge_drilldown.csv", report)
    categories = dict(zip(report["candidate_feature_set"], report["rejection_category"]))
    assert categories["price_plus_stablecoins"] == "high_return_but_poor_calibration_or_drawdown"
    assert categories["btc_dollar_rates_cycle"] == "promising_but_statistically_unstable"


def test_sol_stability_report_candidate_pairs_include_selected_model():
    pairs = sol_stability_candidate_pairs("sol")
    assert ("all_features", "random_forest") in pairs
    assert ("price_plus_risk_assets", "logistic_linear") in pairs
    assert sol_stability_candidate_pairs("btc") == []


def test_feature_pruning_report_marks_unstable_candidates_not_promoted():
    run_id = "btc_research_test"
    base = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "horizon": 30,
                "window_type": "official_monthly",
                "model": "logistic_linear",
                "sample_count": 36,
                "directional_accuracy": 0.52,
                "brier_score": 0.24,
                "calibration_error": 0.10,
                "tc_adjusted_return": 0.20,
                "sharpe": 0.40,
                "max_drawdown": -0.20,
                "beats_buy_hold_direction": False,
                "beats_momentum_30d": False,
                "beats_momentum_90d": False,
                "beats_random_baseline": True,
                "reliability_label": "Low confidence",
                "notes": "current all features",
            }
        ]
    )
    pruned = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "asset_id": "btc",
                "horizon": 30,
                "window_type": "official_monthly",
                "candidate_feature_set": "btc_dollar_rates_cycle",
                "model_name": "logistic_linear",
                "n_features": 10,
                "median_selected_features": 7,
                "n_samples": 36,
                "directional_accuracy": 0.58,
                "balanced_accuracy": 0.58,
                "brier_score": 0.23,
                "calibration_error": 0.09,
                "sharpe": 0.70,
                "max_drawdown": -0.15,
                "net_return": 0.40,
                "beats_buy_hold": True,
                "beats_momentum_30d": True,
                "beats_momentum_90d": True,
                "beats_random_baseline": True,
                "selection_eligible": True,
                "promotion_eligible": False,
                "reliability_label": "Medium confidence",
                "window_fingerprint": "abc",
                "rejection_reason": "failed_bootstrap_or_permutation_stability_check",
            }
        ]
    )
    signal = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "asset_id": "btc",
                "horizon": 30,
                "window_type": "official_monthly",
                "candidate_feature_set": "btc_dollar_rates_cycle",
                "model_name": "logistic_linear",
                "bootstrap_ci_low": 0.49,
                "bootstrap_ci_high": 0.68,
                "permutation_p_value": 0.20,
            }
        ]
    )
    report = feature_pruning_report(run_id, pruned, signal, base)
    row = report[report["candidate_feature_set"] == "btc_dollar_rates_cycle"].iloc[0]
    assert row["promotion_decision"] == "reject"
    assert row["report_label"] == "promising_but_unstable"
    assert row["bootstrap_ci_low"] < 0.50


def test_streamlit_optional_new_diagnostics_can_be_empty_with_schema():
    for filename in [
        "feature_pruning_report.csv",
        "sol_stability_report.csv",
        "signal_policy_report.csv",
        "asset_feature_set_leaderboard.csv",
        "fred_macro_impact_report.csv",
        "macro_candidate_impact_report.csv",
        "fred_vs_fallback_summary.csv",
        "sol_selection_audit.csv",
        "sol_signal_policy_deployment.csv",
        "btc_no_edge_drilldown.csv",
    ]:
        frame = empty_diagnostic_frame(filename)
        validate_frame(filename, frame)
    validate_frame("latest_signal_interpretation.csv", empty_output_frame("latest_signal_interpretation.csv"))


def test_signal_policy_threshold_and_active_floors_are_enforced():
    assert is_official_long_policy_allowed(0.50, 0.05) is False
    assert is_official_long_policy_allowed(0.55, 0.0) is True
    assert active_signal_floor(36) == 12
    assert high_confidence_signal_floor(36) == 18
    assert active_signal_floor(100) == 20
    assert high_confidence_signal_floor(100) == 30


def test_no_valid_edge_keeps_neutral_no_edge_interpretation():
    latest = pd.DataFrame(
        [
            {
                "run_id": "btc_research_test",
                "generated_at": "2026-01-01T00:00:00+00:00",
                "as_of_date": "2026-01-01",
                "horizon": 30,
                "selected_model": "no_valid_edge",
                "predicted_probability_up": 0.50,
                "expected_return": 0.0,
                "reliability_label": "Low confidence",
                "is_primary_objective": True,
            }
        ]
    )
    out = latest_signal_interpretation_frame(
        "btc_research_test",
        "2026-01-01T00:00:00+00:00",
        latest,
        empty_diagnostic_frame("signal_policy_report.csv"),
        empty_diagnostic_frame("asset_feature_set_leaderboard.csv"),
    )
    row = out.iloc[0]
    assert row["selected_model"] == "no_valid_edge"
    assert row["strategy_action"] == "cash"
    assert row["risk_label"] == "Neutral / no edge"
    assert bool(row["signal_policy_promoted"]) is False


def test_asset_deployability_no_valid_edge_uses_benchmark_context():
    latest = pd.DataFrame(
        [
            {
                "run_id": "btc_research_test",
                "generated_at": "2026-01-01T00:00:00+00:00",
                "as_of_date": "2026-01-01",
                "horizon": 30,
                "selected_model": "no_valid_edge",
                "selection_reason": "No 30d official model beat baselines.",
                "signal": "neutral",
                "reliability_label": "Low confidence",
                "is_primary_objective": True,
            }
        ]
    )
    interp = latest_signal_interpretation_frame(
        "btc_research_test",
        "2026-01-01T00:00:00+00:00",
        latest.assign(predicted_probability_up=0.5, expected_return=0.0),
        empty_diagnostic_frame("signal_policy_report.csv"),
        empty_diagnostic_frame("asset_feature_set_leaderboard.csv"),
    )
    leaderboard = pd.DataFrame(
        [
            {
                "horizon": 30,
                "window_type": "official_monthly",
                "model": "momentum_90d",
                "directional_accuracy": 0.56,
                "tc_adjusted_return": 0.2,
                "brier_score": 0.24,
                "max_drawdown": -0.2,
            },
            {
                "horizon": 30,
                "window_type": "official_monthly",
                "model": "logistic_linear",
                "directional_accuracy": 0.52,
                "tc_adjusted_return": 0.1,
                "brier_score": 0.25,
                "max_drawdown": -0.3,
            },
        ]
    )
    out = asset_deployability_report("btc_research_test", latest, interp, leaderboard, {"asset_name": "Bitcoin"})
    validate_frame("asset_deployability_report.csv", out)
    row = out.iloc[0]
    assert row["deployability_decision"] == "no_valid_edge"
    assert row["equity_chart_role"] == "Benchmark context only"
    assert row["equity_chart_model"] == "momentum_90d"


def test_signal_policy_cannot_change_selected_model_and_risk_off_is_not_short():
    latest = pd.DataFrame(
        [
            {
                "run_id": "sol_research_test",
                "generated_at": "2026-01-01T00:00:00+00:00",
                "as_of_date": "2026-01-01",
                "horizon": 30,
                "selected_model": "random_forest",
                "predicted_probability_up": 0.34,
                "expected_return": -0.12,
                "reliability_label": "Medium confidence",
                "is_primary_objective": True,
            }
        ]
    )
    policy = pd.DataFrame(
        [
            {
                "run_id": "sol_research_test",
                "asset_id": "sol",
                "horizon": 30,
                "window_type": "official_monthly",
                "candidate_feature_set": "all_features",
                "model_name": "random_forest",
                "n_samples": 36,
                "policy_source": "train_calibration_only",
                "long_threshold_median": 0.55,
                "expected_return_min_median": 0.0,
                "threshold_050_used": False,
                "threshold_050_diagnostic_only": True,
                "active_signal_count": 12,
                "active_coverage": 0.33,
                "abstention_rate": 0.67,
                "active_hit_rate": 0.70,
                "missed_up_month_rate": 0.20,
                "after_cost_return": 0.20,
                "max_drawdown": -0.10,
                "bootstrap_ci_low": 0.52,
                "bootstrap_ci_high": 0.84,
                "permutation_p_value": 0.05,
                "risk_off_count": 10,
                "risk_off_rate": 0.28,
                "risk_off_hit_rate": 0.70,
                "risk_off_avg_return": -0.08,
                "risk_off_probability_threshold_median": 0.40,
                "valid_signal_policy": True,
                "high_confidence_policy_eligible": False,
                "promoted_signal_policy": True,
                "rejection_reason": "promotion_eligible",
            }
        ]
    )
    features = empty_diagnostic_frame("asset_feature_set_leaderboard.csv")
    out = latest_signal_interpretation_frame("sol_research_test", "2026-01-01T00:00:00+00:00", latest, policy, features)
    row = out.iloc[0]
    assert row["selected_model"] == "random_forest"
    assert row["strategy_action"] == "cash"
    assert row["risk_label"] == "High downside risk"


def test_material_worsening_rules_reject_candidates():
    reference = {"brier_score": 0.20, "calibration_error": 0.08, "max_drawdown": -0.15}
    assert _material_worsening({"brier_score": 0.235, "calibration_error": 0.08, "max_drawdown": -0.15}, reference) is True
    assert _material_worsening({"brier_score": 0.20, "calibration_error": 0.115, "max_drawdown": -0.15}, reference) is True
    assert _material_worsening({"brier_score": 0.20, "calibration_error": 0.08, "max_drawdown": -0.27}, reference) is True
    assert _material_worsening({"brier_score": 0.21, "calibration_error": 0.09, "max_drawdown": -0.20}, reference) is False


def test_feature_group_mapping_is_stable():
    assert feature_group_for_feature("momentum_ret_30d") == "price_momentum_only"
    assert feature_group_for_feature("macro_m2_money_supply_ret_30d") == "macro_liquidity_only"
    assert feature_group_for_feature("macro_us_10y_yield") == "dollar_rates_only"
    assert feature_group_for_feature("cross_asset_spx_ret_30d") == "risk_assets_only"
    assert feature_group_for_feature("stablecoins_supply_chg_30d") == "stablecoins_only"
    assert feature_group_for_feature("onchain_mvrv") == "onchain_only"
    assert feature_group_for_feature("derivatives_funding_rate") == "derivatives_only"
    assert feature_group_for_feature("polymarket_ladder_skew") == "prediction_markets_only"
    assert feature_group_for_feature("solana_ecosystem_tvl_chg_30d") == "sol_ecosystem_only"


def test_polymarket_threshold_parsing_and_monthly_event_filter():
    assert _parse_polymarket_threshold("Will Bitcoin reach $115,000 in May?") == 115000
    assert _parse_polymarket_threshold("Will Solana dip to $75 in May?") == 75
    assert _parse_polymarket_threshold("Will Bitcoin reach $150k in May?") == 150000
    monthly = {
        "title": "What price will Bitcoin hit in May?",
        "startDate": "2026-05-01T00:00:00Z",
        "endDate": "2026-06-01T04:00:00Z",
    }
    annual = {
        "title": "What price will Bitcoin hit in 2026?",
        "startDate": "2025-11-01T00:00:00Z",
        "endDate": "2027-01-01T00:00:00Z",
    }
    weekly = {
        "title": "Bitcoin above ___ on May 4?",
        "startDate": "2026-05-01T00:00:00Z",
        "endDate": "2026-05-04T00:00:00Z",
    }
    assert _is_monthly_polymarket_event(monthly, "Bitcoin") is True
    assert _is_monthly_polymarket_event(annual, "Bitcoin") is False
    assert _is_monthly_polymarket_event(weekly, "Bitcoin") is False


def test_polymarket_features_are_quarantined_from_official_feature_cols():
    config = ResearchConfig(min_feature_valid_ratio=0.40)
    raw = synthetic_research_raw(rows=950)
    raw["polymarket_implied_median_price"] = raw["btc_close"] * 1.05
    raw["polymarket_upside_probability"] = 0.55
    raw["polymarket_downside_probability"] = 0.35
    raw["polymarket_ladder_skew"] = 0.20
    raw["polymarket_ladder_width"] = 0.25
    raw["polymarket_market_count"] = 12
    feature_result = build_features(raw, config)
    assert any(col.startswith("polymarket_") for col in feature_result.features.columns)
    assert not any(col.startswith("polymarket_") for col in feature_result.feature_cols)
    audit = feature_result.feature_audit[feature_result.feature_audit["feature_name"].str.startswith("polymarket_")]
    assert not audit.empty
    assert audit["used_in_model"].eq(False).all()


def test_candidate_feature_sets_keep_polymarket_diagnostic_only():
    cols = [
        "momentum_ret_30d",
        "macro_us_10y_yield",
        "macro_real_10y_yield",
        "macro_treasury_general_account_ret_30d",
        "cross_asset_spx_ret_30d",
        "stablecoins_supply_chg_30d",
        "derivatives_funding_rate",
        "derivatives_top_trader_account_long_short_ratio_z_90d",
        "solana_ecosystem_tvl_chg_30d",
        "polymarket_ladder_skew",
    ]
    sets = candidate_feature_sets(cols, "btc")
    assert "polymarket_ladder_skew" not in sets["no_polymarket"]
    assert "polymarket_ladder_skew" not in sets["all_features"]
    assert "btc_derivatives_v2_pack" in sets
    assert "derivatives_top_trader_account_long_short_ratio_z_90d" in sets["btc_derivatives_v2_pack"]
    assert "btc_macro_liquidity_v2" in sets
    sol_sets = candidate_feature_sets(cols, "sol")
    assert "sol_ecosystem_pack" in sol_sets
    assert "solana_ecosystem_tvl_chg_30d" in sol_sets["sol_ecosystem_pack"]
    assert "sol_price_plus_ecosystem" in sol_sets
    spx_sets = candidate_feature_sets(cols, "spx")
    assert "spx_macro_risk" in spx_sets
    assert "spx_rates_liquidity" in spx_sets
    assert "spx_cross_asset_risk" in spx_sets
    assert "spx_price_macro_risk" in spx_sets
    assert "polymarket_ladder_skew" not in spx_sets["all_features"]


def test_nested_feature_pruning_uses_train_columns_only_and_drops_correlated_features():
    config = ResearchConfig(min_feature_valid_ratio=0.40)
    rng = np.random.default_rng(123)
    dates = pd.date_range("2020-01-01", periods=260, freq="D")
    target = rng.normal(size=len(dates))
    frame = pd.DataFrame(
        {
            "target_log_return": target,
            "target_up": (target > 0).astype(int),
            "asset_close": 100 + np.arange(len(dates)),
            "signal_a": target + rng.normal(scale=0.01, size=len(dates)),
            "signal_b": target + rng.normal(scale=0.01, size=len(dates)),
            "test_only_signal": np.nan,
        },
        index=dates,
    )
    frame.loc[dates[-20:], "test_only_signal"] = target[-20:]
    train = frame.iloc[:-20]
    selected = select_nested_pruned_features(
        train,
        ["signal_a", "signal_b", "test_only_signal"],
        config,
        max_features=3,
        corr_threshold=0.85,
    )
    assert "test_only_signal" not in selected
    assert len({"signal_a", "signal_b"} & set(selected)) == 1


def test_manual_derivative_csv_validation_statuses(tmp_path):
    spec = MANUAL_DERIVATIVE_SPECS["binance_funding_rate"]
    valid = tmp_path / "valid.csv"
    valid.write_text("date,funding_rate\n2024-01-01,0.0001\n2024-01-02,-0.0002\n", encoding="utf-8")
    result = validate_manual_derivative_csv(valid, spec)
    assert result.status == "worked"
    assert result.frame is not None
    assert list(result.frame.columns) == ["binance_funding_rate"]

    empty = tmp_path / "empty.csv"
    empty.write_text("date,funding_rate\n", encoding="utf-8")
    assert validate_manual_derivative_csv(empty, spec).status == "skipped"

    missing = tmp_path / "missing.csv"
    missing.write_text("date\n2024-01-01\n", encoding="utf-8")
    assert validate_manual_derivative_csv(missing, spec).status == "failed"

    duplicate = tmp_path / "duplicate.csv"
    duplicate.write_text("date,funding_rate\n2024-01-01,0.1\n2024-01-01,0.2\n", encoding="utf-8")
    assert validate_manual_derivative_csv(duplicate, spec).status == "failed"

    non_numeric = tmp_path / "non_numeric.csv"
    non_numeric.write_text("date,funding_rate\n2024-01-01,abc\n", encoding="utf-8")
    assert validate_manual_derivative_csv(non_numeric, spec).status == "failed"

    impossible = tmp_path / "impossible.csv"
    impossible.write_text("date,funding_rate\n2024-01-01,2.0\n", encoding="utf-8")
    assert validate_manual_derivative_csv(impossible, spec).status == "failed"


def test_archive_ratio_manual_csvs_validate_with_real_available_columns(tmp_path):
    long_short = tmp_path / "long_short.csv"
    long_short.write_text("date,long_short_ratio\n2024-01-01,1.25\n2024-01-02,0.95\n", encoding="utf-8")
    long_result = validate_manual_derivative_csv(long_short, MANUAL_DERIVATIVE_SPECS["binance_long_short_ratio"])
    assert long_result.status == "worked"
    assert list(long_result.frame.columns) == ["binance_long_short_ratio"]

    taker = tmp_path / "taker.csv"
    taker.write_text("date,taker_buy_sell_ratio\n2024-01-01,1.10\n2024-01-02,0.80\n", encoding="utf-8")
    taker_result = validate_manual_derivative_csv(taker, MANUAL_DERIVATIVE_SPECS["binance_taker_buy_sell_ratio"])
    assert taker_result.status == "worked"
    assert list(taker_result.frame.columns) == ["binance_taker_buy_sell_ratio"]

    top_account = tmp_path / "top_account.csv"
    top_account.write_text("date,long_short_ratio\n2024-01-01,1.05\n2024-01-02,0.90\n", encoding="utf-8")
    top_result = validate_manual_derivative_csv(top_account, MANUAL_DERIVATIVE_SPECS["binance_top_trader_account_ratio"])
    assert top_result.status == "worked"
    assert list(top_result.frame.columns) == ["binance_top_trader_account_long_short_ratio"]


def test_defillama_chart_parser_handles_list_payloads(monkeypatch, tmp_path):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"totalDataChart": [[1_700_000_000, 123.4], [1_700_086_400, 234.5]]}

    monkeypatch.setattr(rp, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(rp.requests, "get", lambda *args, **kwargs: FakeResponse())
    frame = _defillama_chart_frame(
        dataset="defillama_test_chart",
        endpoint="https://api.llama.fi/overview/dexs/Solana",
        output_column="solana_dex_volume_usd",
    )
    assert list(frame.columns) == ["solana_dex_volume_usd"]
    assert len(frame) == 2
    assert frame["solana_dex_volume_usd"].iloc[-1] == 234.5


def test_binance_recent_window_endpoints_are_not_treated_as_full_history(monkeypatch):
    calls = []

    def fake_chunked(endpoint, params, **kwargs):
        calls.append({"endpoint": endpoint, **kwargs})
        return [
            {
                "timestamp": 1_700_000_000_000,
                "sumOpenInterest": "1",
                "sumOpenInterestValue": "2",
                "longShortRatio": "1.1",
                "longAccount": "0.52",
                "shortAccount": "0.48",
            }
        ]

    monkeypatch.setattr(rp, "_fetch_binance_chunked", fake_chunked)
    monkeypatch.setattr(rp, "_write_cached_dataset", lambda dataset, frame: frame)
    rp.fetch_binance_open_interest()
    rp.fetch_binance_top_trader_account_ratio()
    assert calls[0]["recent_days"] == 29
    assert calls[1]["recent_days"] == 29


def test_binance_archive_zip_parser_aggregates_daily_metrics(tmp_path):
    archive_path = tmp_path / "BTCUSDT-metrics-2024-01-01.zip"
    csv = "\n".join(
        [
            "create_time,symbol,sum_open_interest,sum_open_interest_value,count_long_short_ratio,sum_taker_long_short_vol_ratio",
            "2024-01-01 00:00:00,BTCUSDT,10,1000,1.20,0.80",
            "2024-01-01 00:05:00,BTCUSDT,12,1200,1.40,1.00",
            "",
        ]
    )
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("BTCUSDT-metrics-2024-01-01.csv", csv)
    parsed = _parse_binance_archive_zip(archive_path)
    assert len(parsed) == 1
    assert parsed.iloc[0]["sum_open_interest"] == 11
    assert parsed.iloc[0]["sum_open_interest_value"] == 1100
    assert np.isclose(parsed.iloc[0]["count_long_short_ratio"], 1.3)
    assert np.isclose(parsed.iloc[0]["sum_taker_long_short_vol_ratio"], 0.9)


def test_derivatives_impact_uses_identical_windows():
    config = ResearchConfig(quick_initial_train_days=365, min_feature_valid_ratio=0.40)
    raw = synthetic_research_raw(rows=950)
    feature_result = build_features(raw, config)
    derivative_cols = [col for col in feature_result.feature_cols if col.startswith("derivatives_")]
    assert derivative_cols
    without_derivatives = [col for col in feature_result.feature_cols if col not in derivative_cols]

    with_frame = build_target_frame(raw, feature_result.features, feature_result.feature_cols, 30)
    without_frame = build_target_frame(raw, feature_result.features, without_derivatives, 30)
    with_windows = build_walk_forward_windows(with_frame, 30, "official_monthly", config, quick=True)
    without_windows = build_walk_forward_windows(without_frame, 30, "official_monthly", config, quick=True)
    assert window_fingerprint(with_windows) == window_fingerprint(without_windows)


def test_no_derivative_features_when_raw_derivatives_missing():
    config = ResearchConfig()
    raw = synthetic_research_raw().drop(columns=["binance_funding_rate"])
    feature_result = build_features(raw, config)
    assert not any(col.startswith("derivatives_") for col in feature_result.feature_cols)


def test_derivative_feature_threshold_allows_valid_partial_history():
    config = ResearchConfig(min_feature_valid_ratio=0.55, min_derivative_feature_valid_ratio=0.45)
    features = pd.DataFrame(
        {
            "derivatives_open_interest": [np.nan] * 55 + list(range(45)),
            "macro_sparse": [np.nan] * 55 + list(range(45)),
        }
    )
    selected = select_feature_columns(features, config)
    assert "derivatives_open_interest" in selected
    assert "macro_sparse" not in selected


def test_no_valid_edge_selection_fallback_for_empty_leaderboard():
    selected, reason, reliability = select_primary_model(pd.DataFrame())
    assert selected == "no_valid_edge"
    assert "No 30d official model" in reason
    assert reliability == "Low confidence"


def test_useful_model_must_beat_both_momentum_baselines():
    summary = {
        "sample_count": 99,
        "directional_accuracy": 0.54,
        "tc_adjusted_return": 2.0,
        "calibration_error": 0.10,
    }
    baselines = {
        "buy_hold_direction": {"directional_accuracy": 0.50, "tc_adjusted_return": 1.0},
        "momentum_30d": {"directional_accuracy": 0.51, "tc_adjusted_return": 1.0},
        "momentum_90d": {"directional_accuracy": 0.56, "tc_adjusted_return": 1.0},
        "random_permutation": {"directional_accuracy": 0.44, "tc_adjusted_return": 0.0},
    }
    decorated = decorate_summary(summary, baselines, 30, True, "random_forest")
    assert decorated["beats_momentum_30d"] is True
    assert decorated["beats_momentum_90d"] is False
    assert decorated["useful_model"] is False
    assert decorated["selection_eligible"] is False


def test_derivatives_coverage_does_not_mark_failed_source_used():
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    raw = pd.DataFrame({"binance_sum_open_interest": [1.0, 2.0, 3.0]}, index=dates)
    feature_result = SimpleNamespace(
        feature_audit=pd.DataFrame(
            {
                "raw_metric": ["binance_sum_open_interest"],
                "used_in_model": [True],
            }
        )
    )
    availability = pd.DataFrame(
        [
            {
                "source": "Binance USD-M Futures",
                "dataset": "binance_open_interest",
                "status": "failed",
                "rows": 0,
                "first_date": "",
                "last_date": "",
                "is_used_in_model": False,
                "failure_reason": "timeout",
            },
            {
                "source": "manual_csv",
                "dataset": "binance_open_interest",
                "status": "worked",
                "rows": 3,
                "first_date": "2024-01-01",
                "last_date": "2024-01-03",
                "is_used_in_model": True,
                "failure_reason": "",
            },
        ]
    )
    coverage = derivatives_coverage("run", raw, feature_result, availability)
    failed = coverage[coverage["source"] == "Binance USD-M Futures"].iloc[0]
    manual = coverage[coverage["source"] == "manual_csv"].iloc[0]
    assert bool(failed["used_in_model"]) is False
    assert bool(manual["used_in_model"]) is True
