from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from research_config import ResearchConfig
from schemas import empty_diagnostic_frame, write_diagnostic_csv


BASELINE_DIR = Path("/tmp/btc_refresh_baseline")


def preserve_quick_baseline(output_dir: Path, refresh: bool) -> Optional[Path]:
    baseline_dir = Path(f"/tmp/btc_refresh_baseline_{output_dir.name}")
    if not refresh:
        return None
    manifest_path = output_dir / "run_manifest.json"
    if not manifest_path.exists():
        return baseline_dir if baseline_dir.exists() else BASELINE_DIR if BASELINE_DIR.exists() else None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return baseline_dir if baseline_dir.exists() else BASELINE_DIR if BASELINE_DIR.exists() else None
    if not manifest.get("quick_mode"):
        return baseline_dir if baseline_dir.exists() else BASELINE_DIR if BASELINE_DIR.exists() else None

    baseline_dir.mkdir(parents=True, exist_ok=True)
    for filename in [
        "run_manifest.json",
        "data_availability.csv",
        "model_leaderboard.csv",
        "latest_forecast.csv",
        "feature_audit.csv",
    ]:
        src = output_dir / filename
        if src.exists():
            shutil.copy2(src, baseline_dir / filename)
    return baseline_dir


def feature_group_for_feature(feature: str) -> str:
    if feature.startswith("polymarket_"):
        return "prediction_markets_only"
    if feature.startswith("derivatives_"):
        return "derivatives_only"
    if feature.startswith("stablecoins_"):
        return "stablecoins_only"
    if feature.startswith("onchain_"):
        return "onchain_only"
    if feature.startswith("cross_asset_") or feature.startswith("macro_vix"):
        return "risk_assets_only"
    if feature.startswith("macro_dxy") or "trade_weighted_usd" in feature or "yield" in feature or "fed_funds" in feature or "spread" in feature or "breakeven" in feature:
        return "dollar_rates_only"
    if "fed_balance_sheet" in feature or "m2_money_supply" in feature or "reverse_repo" in feature:
        return "macro_liquidity_only"
    return "price_momentum_only"


def feature_group_columns(feature_cols: Sequence[str]) -> Dict[str, List[str]]:
    groups = {
        "price_momentum_only": [],
        "macro_liquidity_only": [],
        "dollar_rates_only": [],
        "risk_assets_only": [],
        "stablecoins_only": [],
        "onchain_only": [],
        "derivatives_only": [],
        "prediction_markets_only": [],
        "all_features": list(feature_cols),
    }
    for col in feature_cols:
        groups.setdefault(feature_group_for_feature(col), []).append(col)
    return groups


def min_samples_for(horizon: int, window_type: str) -> int:
    if horizon == 30 and window_type == "official_monthly":
        return 24
    if horizon == 90 and window_type == "official_quarterly":
        return 12
    if horizon == 180:
        return 8
    return 12


def model_rejection_reasons(summary: pd.DataFrame, selected_model: str) -> pd.DataFrame:
    if summary.empty:
        return empty_diagnostic_frame("model_rejection_reasons.csv")
    rows = []
    for _, row in summary.iterrows():
        model = str(row["model"])
        is_baseline = model in {"buy_hold_direction", "momentum_30d", "momentum_90d", "random_permutation"}
        best_baseline_return = max(
            [
                row.get("buy_hold_tc_adjusted_return", -np.inf),
                row.get("momentum_30d_tc_adjusted_return", -np.inf),
                row.get("momentum_90d_tc_adjusted_return", -np.inf),
                row.get("random_tc_adjusted_return", -np.inf),
            ]
        )
        passes_transaction_cost = bool(row.get("tc_adjusted_return", -np.inf) > best_baseline_return and row.get("tc_adjusted_return", 0) > 0)
        passes_calibration = bool(row.get("calibration_error", 1) <= 0.15)
        passes_sample = bool(row.get("sample_count", 0) >= min_samples_for(int(row["horizon"]), str(row["window_type"])))
        passes_reliability = bool(str(row.get("reliability_label", "")) != "Low confidence")
        reasons = []
        if model == selected_model and selected_model != "no_valid_edge":
            reasons.append("selected_model")
        else:
            if is_baseline:
                reasons.append("baseline_not_selectable")
            if int(row["horizon"]) == 180:
                reasons.append("diagnostic_horizon_not_selection_eligible")
            if not bool(row.get("is_official", False)):
                reasons.append("sensitivity_window_not_selection_eligible")
            if not passes_sample:
                reasons.append("below_sample_threshold")
            if not bool(row.get("beats_buy_hold_direction", False)):
                reasons.append("did_not_beat_buy_hold")
            if not bool(row.get("beats_momentum_30d", False)):
                reasons.append("did_not_beat_30d_momentum")
            if not bool(row.get("beats_momentum_90d", False)):
                reasons.append("did_not_beat_90d_momentum")
            if not bool(row.get("beats_random_baseline", False)):
                reasons.append("did_not_beat_random_baseline")
            if not passes_transaction_cost:
                reasons.append("failed_transaction_cost_check")
            if not passes_calibration:
                reasons.append("failed_calibration_check")
            if not passes_reliability:
                reasons.append("low_reliability")
            if not reasons:
                reasons.append("not_selected_by_primary_30d_rules")
        rows.append(
            {
                "run_id": row["run_id"],
                "horizon": row["horizon"],
                "window_type": row["window_type"],
                "model_name": model,
                "is_baseline": is_baseline,
                "n_samples": int(row.get("sample_count", 0)),
                "directional_accuracy": row.get("directional_accuracy", np.nan),
                "beats_buy_hold": bool(row.get("beats_buy_hold_direction", False)),
                "beats_momentum_30d": bool(row.get("beats_momentum_30d", False)),
                "beats_momentum_90d": bool(row.get("beats_momentum_90d", False)),
                "beats_random_baseline": bool(row.get("beats_random_baseline", False)),
                "passes_transaction_cost_check": passes_transaction_cost,
                "passes_calibration_check": passes_calibration,
                "passes_sample_threshold": passes_sample,
                "passes_reliability_check": passes_reliability,
                "final_rejection_reason": "; ".join(reasons),
            }
        )
    return pd.DataFrame(rows)


def _corr_stats(x: pd.Series, y: pd.Series) -> Tuple[float, float, float, float]:
    pair = pd.concat([x, y], axis=1).dropna()
    if len(pair) < 4:
        return np.nan, np.nan, np.nan, np.nan
    pearson = float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))
    ranked = pair.rank(method="average")
    spearman = float(ranked.iloc[:, 0].corr(ranked.iloc[:, 1]))
    r = max(min(spearman, 0.999999), -0.999999) if not pd.isna(spearman) else np.nan
    if pd.isna(r):
        return pearson, spearman, np.nan, np.nan
    t_stat = float(r * math.sqrt((len(pair) - 2) / max(1e-12, 1 - r * r)))
    p_value = float(math.erfc(abs(t_stat) / math.sqrt(2)))
    return pearson, spearman, t_stat, p_value


def feature_signal_diagnostics(
    run_id: str,
    raw: pd.DataFrame,
    feature_result,
    config: ResearchConfig,
) -> pd.DataFrame:
    from research_pipeline import build_target_frame, build_walk_forward_windows

    audit = feature_result.feature_audit.set_index("feature_name")
    rows = []
    for horizon, window_type in [(30, "official_monthly"), (90, "official_quarterly")]:
        target_frame = build_target_frame(raw, feature_result.features, feature_result.feature_cols, horizon)
        windows = build_walk_forward_windows(target_frame, horizon, window_type, config, quick=False)
        dates = pd.DatetimeIndex([w.test_date for w in windows])
        sample = target_frame.loc[target_frame.index.intersection(dates)]
        target = sample["target_log_return"]
        for feature in feature_result.feature_cols:
            series = sample[feature]
            valid = pd.concat([series, target], axis=1).dropna()
            pearson, spearman, t_stat, p_value = _corr_stats(series, target)
            valid_feature = feature_result.features[feature].dropna()
            rows.append(
                {
                    "run_id": run_id,
                    "feature_name": feature,
                    "source": audit.at[feature, "source"] if feature in audit.index else "",
                    "feature_group": feature_group_for_feature(feature),
                    "horizon": horizon,
                    "window_type": window_type,
                    "n_samples": int(len(valid)),
                    "pearson_corr": pearson,
                    "spearman_corr": spearman,
                    "information_coefficient": spearman,
                    "ic_t_stat": t_stat,
                    "ic_p_value": p_value,
                    "missing_pct": float(1 - series.notna().mean()) if len(series) else 1.0,
                    "first_date": valid_feature.index.min().date().isoformat() if len(valid_feature) else "",
                    "last_date": valid_feature.index.max().date().isoformat() if len(valid_feature) else "",
                    "direction": "positive" if pd.notna(spearman) and spearman > 0 else "negative" if pd.notna(spearman) and spearman < 0 else "flat",
                    "notes": "research_only_not_used_for_model_selection",
                }
            )
    return pd.DataFrame(rows).sort_values(["horizon", "information_coefficient"], key=lambda s: s.abs() if pd.api.types.is_numeric_dtype(s) else s, ascending=False)


def _predict_feature_set(
    raw: pd.DataFrame,
    feature_result,
    feature_cols: Sequence[str],
    horizon: int,
    window_type: str,
    config: ResearchConfig,
    *,
    quick: bool,
    candidate_dates: Optional[pd.DatetimeIndex] = None,
) -> Dict[str, pd.DataFrame]:
    from research_pipeline import (
        build_target_frame,
        build_walk_forward_windows,
        predict_baseline_windows,
        predict_model_windows,
    )

    target_frame = build_target_frame(raw, feature_result.features, feature_cols, horizon)
    windows = build_walk_forward_windows(target_frame, horizon, window_type, config, quick=quick)
    if candidate_dates is not None:
        allowed = set(pd.DatetimeIndex(candidate_dates))
        windows = [window for window in windows if window.test_date in allowed]
    predictions_by_model: Dict[str, pd.DataFrame] = {}
    for baseline in config.baseline_models:
        preds = predict_baseline_windows(baseline, target_frame, windows, config)
        if not preds.empty:
            predictions_by_model[baseline] = preds
    if len(feature_cols) >= 5:
        for model_name in config.first_model_set:
            preds = predict_model_windows(model_name, target_frame, feature_cols, windows, config, quick=quick)
            if not preds.empty:
                predictions_by_model[model_name] = preds
    if not predictions_by_model:
        return {}
    common_dates = None
    for preds in predictions_by_model.values():
        common_dates = preds.index if common_dates is None else common_dates.intersection(preds.index)
    if common_dates is None or len(common_dates) == 0:
        return {}
    return {model: preds.loc[common_dates].sort_index() for model, preds in predictions_by_model.items()}


def _score_feature_set(
    raw: pd.DataFrame,
    feature_result,
    feature_cols: Sequence[str],
    horizon: int,
    window_type: str,
    config: ResearchConfig,
    *,
    quick: bool,
    candidate_dates: Optional[pd.DatetimeIndex] = None,
) -> Dict[str, dict]:
    from research_pipeline import BASELINE_MODELS, compute_metrics, decorate_summary

    predictions_by_model = _predict_feature_set(
        raw,
        feature_result,
        feature_cols,
        horizon,
        window_type,
        config,
        quick=quick,
        candidate_dates=candidate_dates,
    )
    if not predictions_by_model:
        return {}

    summaries: Dict[str, dict] = {}
    for model, preds in predictions_by_model.items():
        metrics, _ = compute_metrics(preds, horizon, window_type, config.transaction_cost_bps)
        metrics.update(
            {
                "run_id": "",
                "horizon": horizon,
                "window_type": window_type,
                "is_official": True,
                "model": model,
                "threshold": float(preds["threshold"].mean()),
                "threshold_source": "baseline_rule" if model in BASELINE_MODELS else "nested_calibration",
            }
        )
        summaries[model] = metrics
    baseline_summaries = {name: summaries[name] for name in BASELINE_MODELS if name in summaries}
    return {model: decorate_summary(summary, baseline_summaries, horizon, True, model) for model, summary in summaries.items()}


def feature_group_ablation(
    run_id: str,
    raw: pd.DataFrame,
    feature_result,
    config: ResearchConfig,
    *,
    quick: bool,
) -> pd.DataFrame:
    rows = []
    groups = feature_group_columns(feature_result.feature_cols)
    for horizon, window_type in [(30, "official_monthly"), (90, "official_quarterly")]:
        for group_name, group_cols in groups.items():
            summaries = _score_feature_set(raw, feature_result, group_cols, horizon, window_type, config, quick=quick)
            for model_name in config.baseline_models + config.first_model_set:
                summary = summaries.get(model_name)
                if summary is None:
                    rows.append(
                        {
                            "run_id": run_id,
                            "horizon": horizon,
                            "window_type": window_type,
                            "feature_group": group_name,
                            "model_name": model_name,
                            "n_features": len(group_cols) if model_name not in config.baseline_models else 0,
                            "n_samples": 0,
                            "directional_accuracy": np.nan,
                            "balanced_accuracy": np.nan,
                            "brier_score": np.nan,
                            "calibration_error": np.nan,
                            "sharpe": np.nan,
                            "max_drawdown": np.nan,
                            "net_return": np.nan,
                            "beats_buy_hold": False,
                            "beats_momentum": False,
                            "reliability_label": "Low confidence",
                        }
                    )
                    continue
                beats_momentum = summary.get("beats_momentum_30d", False) if horizon == 30 else summary.get("beats_momentum_90d", False)
                rows.append(
                    {
                        "run_id": run_id,
                        "horizon": horizon,
                        "window_type": window_type,
                        "feature_group": group_name,
                        "model_name": model_name,
                        "n_features": len(group_cols) if model_name not in config.baseline_models else 0,
                        "n_samples": int(summary.get("sample_count", 0)),
                        "directional_accuracy": summary.get("directional_accuracy", np.nan),
                        "balanced_accuracy": summary.get("balanced_accuracy", np.nan),
                        "brier_score": summary.get("brier_score", np.nan),
                        "calibration_error": summary.get("calibration_error", np.nan),
                        "sharpe": summary.get("sharpe", np.nan),
                        "max_drawdown": summary.get("max_drawdown", np.nan),
                        "net_return": summary.get("tc_adjusted_return", np.nan),
                        "beats_buy_hold": bool(summary.get("beats_buy_hold_direction", False)),
                        "beats_momentum": bool(beats_momentum),
                        "reliability_label": summary.get("reliability_label", "Low confidence"),
                    }
                )
    return pd.DataFrame(rows)


def _max_drawdown_from_returns(returns: pd.Series) -> float:
    if returns.empty:
        return np.nan
    equity = (1 + returns).cumprod()
    return float((equity / equity.cummax() - 1).min())


def _regime_masks(raw: pd.DataFrame, dates: pd.DatetimeIndex) -> Dict[str, pd.Series]:
    price_col = "asset_close" if "asset_close" in raw.columns else "btc_close"
    price = raw[price_col].astype(float)
    log_ret = np.log(price).diff()
    sma_200 = price.rolling(200, min_periods=100).mean()
    vol_90 = log_ret.rolling(90, min_periods=30).std()
    vol_median = vol_90.expanding(min_periods=180).median()
    rates = raw["us_10y_yield"] if "us_10y_yield" in raw else pd.Series(index=raw.index, dtype=float)
    rates_median = rates.expanding(min_periods=180).median()
    halvings = [pd.Timestamp("2016-07-09"), pd.Timestamp("2020-05-11"), pd.Timestamp("2024-04-20")]
    etf_date = pd.Timestamp("2024-01-11")

    aligned = pd.DataFrame(index=dates)
    aligned["bull_market"] = price.reindex(dates) >= sma_200.reindex(dates)
    aligned["bear_market"] = price.reindex(dates) < sma_200.reindex(dates)
    aligned["high_volatility"] = vol_90.reindex(dates) >= vol_median.reindex(dates)
    aligned["low_volatility"] = vol_90.reindex(dates) < vol_median.reindex(dates)
    aligned["high_rate"] = rates.reindex(dates) >= rates_median.reindex(dates)
    aligned["low_rate"] = rates.reindex(dates) < rates_median.reindex(dates)
    aligned["pre_halving"] = False
    aligned["post_halving"] = False
    for halving in halvings:
        aligned["pre_halving"] = aligned["pre_halving"] | ((dates >= halving - pd.Timedelta(days=365)) & (dates < halving))
        aligned["post_halving"] = aligned["post_halving"] | ((dates >= halving) & (dates <= halving + pd.Timedelta(days=365)))
    aligned["pre_etf"] = dates < etf_date
    aligned["post_etf"] = dates >= etf_date
    return {col: aligned[col].fillna(False) for col in aligned.columns}


def model_regime_breakdown(run_id: str, raw: pd.DataFrame, equity: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    official = equity[equity["window_type"].isin(["official_monthly", "official_quarterly"])].copy()
    if official.empty:
        return empty_diagnostic_frame("model_regime_breakdown.csv")
    official["date"] = pd.to_datetime(official["date"])
    reliability = summary.set_index(["horizon", "window_type", "model"])["reliability_label"].to_dict() if not summary.empty else {}
    for (horizon, window_type, model), group in official.groupby(["horizon", "window_type", "model"]):
        group = group.sort_values("date").copy()
        dates = pd.DatetimeIndex(group["date"])
        masks = _regime_masks(raw, dates)
        for regime, mask in masks.items():
            part = group.loc[mask.to_numpy()]
            if part.empty:
                continue
            pred_up = (part["signal"] == "long").astype(int)
            return_col = "asset_return" if "asset_return" in part.columns else "btc_return"
            actual_up = (part[return_col] > 0).astype(int)
            returns = part["tc_adjusted_strategy_return"].astype(float)
            periods = 12.0 if int(horizon) == 30 else 4.0 if int(horizon) == 90 else 2.0
            std = float(returns.std(ddof=1))
            rows.append(
                {
                    "run_id": run_id,
                    "horizon": horizon,
                    "window_type": window_type,
                    "model_name": model,
                    "regime": regime,
                    "n_samples": int(len(part)),
                    "directional_accuracy": float((pred_up == actual_up).mean()),
                    "net_return": float((1 + returns).prod() - 1),
                    "sharpe": float((returns.mean() / std) * math.sqrt(periods)) if std > 0 else np.nan,
                    "max_drawdown": _max_drawdown_from_returns(returns),
                    "reliability_label": reliability.get((horizon, window_type, model), "Low confidence"),
                }
            )
    return pd.DataFrame(rows)


def derivatives_impact(
    run_id: str,
    raw: pd.DataFrame,
    feature_result,
    config: ResearchConfig,
    *,
    quick: bool,
) -> pd.DataFrame:
    derivative_cols = [col for col in feature_result.feature_cols if feature_group_for_feature(col) == "derivatives_only"]
    if not derivative_cols:
        return empty_diagnostic_frame("derivatives_impact.csv")
    rows = []
    without_derivatives = [col for col in feature_result.feature_cols if col not in derivative_cols]
    for horizon, window_type in [(30, "official_monthly"), (90, "official_quarterly")]:
        for included, cols in [(False, without_derivatives), (True, feature_result.feature_cols)]:
            summaries = _score_feature_set(raw, feature_result, cols, horizon, window_type, config, quick=quick)
            for model_name in config.first_model_set:
                summary = summaries.get(model_name)
                if summary is None:
                    continue
                rows.append(
                    {
                        "run_id": run_id,
                        "horizon": horizon,
                        "window_type": window_type,
                        "model_name": model_name,
                        "derivatives_included": included,
                        "n_features": len(cols),
                        "n_samples": int(summary.get("sample_count", 0)),
                        "directional_accuracy": summary.get("directional_accuracy", np.nan),
                        "balanced_accuracy": summary.get("balanced_accuracy", np.nan),
                        "brier_score": summary.get("brier_score", np.nan),
                        "calibration_error": summary.get("calibration_error", np.nan),
                        "sharpe": summary.get("sharpe", np.nan),
                        "max_drawdown": summary.get("max_drawdown", np.nan),
                        "net_return": summary.get("tc_adjusted_return", np.nan),
                        "beats_buy_hold": bool(summary.get("beats_buy_hold_direction", False)),
                        "beats_momentum_30d": bool(summary.get("beats_momentum_30d", False)),
                        "beats_momentum_90d": bool(summary.get("beats_momentum_90d", False)),
                        "reliability_label": summary.get("reliability_label", "Low confidence"),
                    }
                )
    return pd.DataFrame(rows)


def polymarket_feature_columns(feature_result) -> List[str]:
    cols = []
    for col in feature_result.features.columns:
        if col.startswith("polymarket_") and pd.api.types.is_numeric_dtype(feature_result.features[col]):
            cols.append(col)
    return cols


def polymarket_coverage(run_id: str, raw: pd.DataFrame, feature_result, availability: pd.DataFrame) -> pd.DataFrame:
    records = availability[availability["dataset"].astype(str).str.startswith("polymarket_")]
    if records.empty:
        return empty_diagnostic_frame("polymarket_coverage.csv")
    audit = feature_result.feature_audit
    poly_cols = [col for col in raw.columns if col.startswith("polymarket_")]
    rows = []
    for _, record in records.iterrows():
        status = str(record.get("status", "skipped"))
        raw_missing = float(raw[poly_cols].isna().mean().mean()) if poly_cols and status == "worked" else 1.0
        used = False
        if not audit.empty:
            used = bool(audit[(audit["feature_name"].astype(str).str.startswith("polymarket_")) & (audit["used_in_model"] == True)].shape[0])
        rows.append(
            {
                "run_id": run_id,
                "asset_id": str(record["dataset"]).replace("polymarket_", "").replace("_monthly_ladders", ""),
                "source": record.get("source", "Polymarket Gamma/CLOB APIs"),
                "status": status,
                "rows": int(record.get("rows", 0)),
                "events": int(raw["polymarket_event_count"].max()) if "polymarket_event_count" in raw and raw["polymarket_event_count"].notna().any() else 0,
                "markets": int(raw["polymarket_discovered_markets"].max()) if "polymarket_discovered_markets" in raw and raw["polymarket_discovered_markets"].notna().any() else int(raw["polymarket_market_count"].max()) if "polymarket_market_count" in raw and raw["polymarket_market_count"].notna().any() else 0,
                "first_date": record.get("first_date", ""),
                "last_date": record.get("last_date", ""),
                "missing_pct": raw_missing,
                "used_in_model": used,
                "failure_reason": record.get("failure_reason", ""),
            }
        )
    return pd.DataFrame(rows)


def polymarket_feature_diagnostics(
    run_id: str,
    raw: pd.DataFrame,
    feature_result,
    config: ResearchConfig,
) -> pd.DataFrame:
    from research_pipeline import build_target_frame, build_walk_forward_windows

    poly_cols = polymarket_feature_columns(feature_result)
    if not poly_cols:
        return empty_diagnostic_frame("polymarket_feature_diagnostics.csv")
    asset_id = run_id.split("_research", 1)[0] if "_research" in run_id else ""
    rows = []
    all_cols = list(dict.fromkeys(list(feature_result.feature_cols) + poly_cols))
    for horizon, window_type in [(30, "official_monthly"), (90, "official_quarterly")]:
        target_frame = build_target_frame(raw, feature_result.features, all_cols, horizon)
        windows = build_walk_forward_windows(target_frame, horizon, window_type, config, quick=False)
        dates = pd.DatetimeIndex([w.test_date for w in windows])
        sample = target_frame.loc[target_frame.index.intersection(dates)]
        target = sample["target_log_return"] if "target_log_return" in sample else pd.Series(dtype=float)
        for feature in poly_cols:
            series = sample[feature] if feature in sample else pd.Series(index=sample.index, dtype=float)
            valid = pd.concat([series, target], axis=1).dropna()
            pearson, spearman, t_stat, p_value = _corr_stats(series, target)
            valid_feature = feature_result.features[feature].dropna()
            rows.append(
                {
                    "run_id": run_id,
                    "asset_id": asset_id,
                    "feature_name": feature,
                    "horizon": horizon,
                    "window_type": window_type,
                    "n_samples": int(len(valid)),
                    "coverage_pct": float(series.notna().mean()) if len(series) else 0.0,
                    "pearson_corr": pearson,
                    "spearman_corr": spearman,
                    "information_coefficient": spearman,
                    "ic_t_stat": t_stat,
                    "ic_p_value": p_value,
                    "first_date": valid_feature.index.min().date().isoformat() if len(valid_feature) else "",
                    "last_date": valid_feature.index.max().date().isoformat() if len(valid_feature) else "",
                    "used_in_official_model": feature in set(feature_result.feature_cols),
                    "notes": "diagnostic_only_prediction_market_factor",
                }
            )
    return pd.DataFrame(rows)


def polymarket_impact(
    run_id: str,
    raw: pd.DataFrame,
    feature_result,
    config: ResearchConfig,
    *,
    quick: bool,
) -> pd.DataFrame:
    from research_pipeline import build_target_frame, build_walk_forward_windows

    poly_cols = polymarket_feature_columns(feature_result)
    if not poly_cols:
        return empty_diagnostic_frame("polymarket_impact.csv")
    asset_id = run_id.split("_research", 1)[0] if "_research" in run_id else ""
    valid_poly_dates = feature_result.features[poly_cols].dropna(how="all").index
    if len(valid_poly_dates) == 0:
        return empty_diagnostic_frame("polymarket_impact.csv")
    official_cols = list(feature_result.feature_cols)
    with_polymarket = list(dict.fromkeys(official_cols + poly_cols))
    rows = []
    for horizon, window_type in [(30, "official_monthly"), (90, "official_quarterly")]:
        target_frame = build_target_frame(raw, feature_result.features, with_polymarket, horizon)
        all_windows = build_walk_forward_windows(target_frame, horizon, window_type, config, quick=quick)
        covered_window_dates = pd.DatetimeIndex([window.test_date for window in all_windows if window.test_date in set(pd.DatetimeIndex(valid_poly_dates))])
        coverage_ratio = (len(covered_window_dates) / len(all_windows)) if all_windows else 0.0
        coverage_ok = bool(len(covered_window_dates) >= 24 and coverage_ratio >= config.min_feature_valid_ratio)
        summaries_by_included = {}
        for included, cols in [(False, official_cols), (True, with_polymarket)]:
            summaries_by_included[included] = _score_feature_set(
                raw,
                feature_result,
                cols,
                horizon,
                window_type,
                config,
                quick=quick,
                candidate_dates=pd.DatetimeIndex(valid_poly_dates),
            )
        for included, cols in [(False, official_cols), (True, with_polymarket)]:
            summaries = summaries_by_included[included]
            for model_name in config.first_model_set:
                summary = summaries.get(model_name)
                if summary is None:
                    rows.append(
                        {
                            "run_id": run_id,
                            "asset_id": asset_id,
                            "horizon": horizon,
                            "window_type": window_type,
                            "model_name": model_name,
                            "polymarket_included": included,
                            "n_features": len(cols),
                            "n_polymarket_features": len(poly_cols) if included else 0,
                            "n_samples": 0,
                            "directional_accuracy": np.nan,
                            "balanced_accuracy": np.nan,
                            "brier_score": np.nan,
                            "calibration_error": np.nan,
                            "sharpe": np.nan,
                            "max_drawdown": np.nan,
                            "net_return": np.nan,
                            "beats_buy_hold": False,
                            "beats_momentum_30d": False,
                            "beats_momentum_90d": False,
                            "beats_random_baseline": False,
                            "reliability_label": "Low confidence",
                            "passes_official_gates": False,
                        }
                    )
                    continue
                baseline_summary = summaries_by_included.get(False, {}).get(model_name, {})
                improves_without_polymarket = bool(
                    included
                    and summary.get("directional_accuracy", -np.inf) > baseline_summary.get("directional_accuracy", np.inf)
                    and summary.get("tc_adjusted_return", -np.inf) >= baseline_summary.get("tc_adjusted_return", np.inf)
                )
                passes = bool(
                    included
                    and horizon == 30
                    and window_type == "official_monthly"
                    and coverage_ok
                    and improves_without_polymarket
                    and summary.get("selection_eligible", False)
                    and summary.get("sample_count", 0) >= 24
                )
                rows.append(
                    {
                        "run_id": run_id,
                        "asset_id": asset_id,
                        "horizon": horizon,
                        "window_type": window_type,
                        "model_name": model_name,
                        "polymarket_included": included,
                        "n_features": len(cols),
                        "n_polymarket_features": len(poly_cols) if included else 0,
                        "n_samples": int(summary.get("sample_count", 0)),
                        "directional_accuracy": summary.get("directional_accuracy", np.nan),
                        "balanced_accuracy": summary.get("balanced_accuracy", np.nan),
                        "brier_score": summary.get("brier_score", np.nan),
                        "calibration_error": summary.get("calibration_error", np.nan),
                        "sharpe": summary.get("sharpe", np.nan),
                        "max_drawdown": summary.get("max_drawdown", np.nan),
                        "net_return": summary.get("tc_adjusted_return", np.nan),
                        "beats_buy_hold": bool(summary.get("beats_buy_hold_direction", False)),
                        "beats_momentum_30d": bool(summary.get("beats_momentum_30d", False)),
                        "beats_momentum_90d": bool(summary.get("beats_momentum_90d", False)),
                        "beats_random_baseline": bool(summary.get("beats_random_baseline", False)),
                        "reliability_label": summary.get("reliability_label", "Low confidence"),
                        "passes_official_gates": passes,
                    }
                )
    return pd.DataFrame(rows)


DERIVATIVE_DATASETS = {
    "binance_funding_rate": {
        "metric": "funding_rate",
        "columns": ["binance_funding_rate"],
    },
    "binance_open_interest": {
        "metric": "open_interest",
        "columns": ["binance_sum_open_interest", "binance_sum_open_interest_value"],
    },
    "binance_long_short_ratio": {
        "metric": "long_short_ratio",
        "columns": ["binance_long_short_ratio", "binance_long_account", "binance_short_account"],
    },
    "binance_taker_buy_sell_ratio": {
        "metric": "taker_buy_sell_ratio",
        "columns": ["binance_taker_buy_sell_ratio", "binance_taker_buy_volume", "binance_taker_sell_volume"],
    },
    "binance_basis": {
        "metric": "basis",
        "columns": ["binance_basis", "binance_basis_rate", "binance_annualized_basis_rate"],
    },
}


def derivatives_coverage(run_id: str, raw: pd.DataFrame, feature_result, availability: pd.DataFrame) -> pd.DataFrame:
    audit = feature_result.feature_audit
    rows = []
    derivative_records = availability[availability["dataset"].isin(DERIVATIVE_DATASETS)].copy()
    if derivative_records.empty:
        return empty_diagnostic_frame("derivatives_coverage.csv")
    for _, record in derivative_records.iterrows():
        dataset = str(record["dataset"])
        spec = DERIVATIVE_DATASETS[dataset]
        cols = [col for col in spec["columns"] if col in raw.columns]
        raw_missing = float(raw[cols].isna().mean().mean()) if cols else 1.0
        used = False
        record_worked = str(record["status"]) == "worked"
        record_used = bool(record.get("is_used_in_model", False))
        if record_worked and record_used and not audit.empty and cols:
            used = bool(audit[(audit["raw_metric"].isin(cols)) & (audit["used_in_model"] == True)].shape[0])
        rows.append(
            {
                "run_id": run_id,
                "source": record["source"],
                "metric": spec["metric"],
                "status": record["status"],
                "rows": int(record.get("rows", 0)),
                "first_date": record.get("first_date", ""),
                "last_date": record.get("last_date", ""),
                "missing_pct": raw_missing if record_worked else 1.0,
                "used_in_model": used,
                "failure_reason": record.get("failure_reason", ""),
            }
        )
    return pd.DataFrame(rows)


def _period_slice_masks(raw: pd.DataFrame, dates: pd.DatetimeIndex) -> Dict[str, pd.Series]:
    aligned = pd.DataFrame(index=dates)
    aligned["2015-2017"] = (dates >= pd.Timestamp("2015-01-01")) & (dates <= pd.Timestamp("2017-12-31"))
    aligned["2018-2020"] = (dates >= pd.Timestamp("2018-01-01")) & (dates <= pd.Timestamp("2020-12-31"))
    aligned["2021-2023"] = (dates >= pd.Timestamp("2021-01-01")) & (dates <= pd.Timestamp("2023-12-31"))
    aligned["2024-present"] = dates >= pd.Timestamp("2024-01-01")
    aligned["pre-ETF"] = dates < pd.Timestamp("2024-01-11")
    aligned["post-ETF"] = dates >= pd.Timestamp("2024-01-11")
    regimes = _regime_masks(raw, dates)
    aligned["high-rate"] = regimes["high_rate"].to_numpy()
    aligned["low-rate"] = regimes["low_rate"].to_numpy()
    aligned["high-vol"] = regimes["high_volatility"].to_numpy()
    aligned["low-vol"] = regimes["low_volatility"].to_numpy()
    return {col: aligned[col].fillna(False) for col in aligned.columns}


def feature_group_stability(
    run_id: str,
    raw: pd.DataFrame,
    feature_result,
    config: ResearchConfig,
    *,
    quick: bool,
) -> pd.DataFrame:
    from research_pipeline import compute_metrics

    rows = []
    groups = feature_group_columns(feature_result.feature_cols)
    scoped_models = {
        "dollar_rates_only": ["logistic_linear"],
        "all_features": ["logistic_linear", "hgb", "random_forest"],
        "price_momentum_only": ["momentum_30d", "momentum_90d", "random_forest"],
    }
    horizon = 30
    window_type = "official_monthly"
    for group_name, model_names in scoped_models.items():
        group_cols = groups.get(group_name, [])
        predictions_by_model = _predict_feature_set(raw, feature_result, group_cols, horizon, window_type, config, quick=quick)
        for model_name in model_names:
            preds = predictions_by_model.get(model_name)
            masks = _period_slice_masks(raw, pd.DatetimeIndex(preds.index if preds is not None else []))
            for period_slice, mask in masks.items():
                if preds is None or preds.empty:
                    part = pd.DataFrame()
                else:
                    part = preds.loc[mask.to_numpy()]
                if part.empty:
                    rows.append(
                        {
                            "run_id": run_id,
                            "feature_group": group_name,
                            "model_name": model_name,
                            "horizon": horizon,
                            "window_type": window_type,
                            "period_slice": period_slice,
                            "n_samples": 0,
                            "directional_accuracy": np.nan,
                            "net_return": np.nan,
                            "sharpe": np.nan,
                            "max_drawdown": np.nan,
                            "reliability_label": "Low confidence",
                        }
                    )
                    continue
                metrics, _ = compute_metrics(part, horizon, window_type, config.transaction_cost_bps)
                rows.append(
                    {
                        "run_id": run_id,
                        "feature_group": group_name,
                        "model_name": model_name,
                        "horizon": horizon,
                        "window_type": window_type,
                        "period_slice": period_slice,
                        "n_samples": int(metrics.get("sample_count", 0)),
                        "directional_accuracy": metrics.get("directional_accuracy", np.nan),
                        "net_return": metrics.get("tc_adjusted_return", np.nan),
                        "sharpe": metrics.get("sharpe", np.nan),
                        "max_drawdown": metrics.get("max_drawdown", np.nan),
                        "reliability_label": "Low confidence",
                    }
                )
    return pd.DataFrame(rows)


def _fmt_pct(value: object) -> str:
    try:
        if pd.isna(value):
            return "n/a"
        return f"{float(value):.1%}"
    except Exception:
        return "n/a"


def _fmt_usd(value: object) -> str:
    try:
        if pd.isna(value):
            return "n/a"
        return f"${float(value):,.0f}"
    except Exception:
        return "n/a"


def _fmt_num(value: object, digits: int = 2) -> str:
    try:
        if pd.isna(value):
            return "n/a"
        return f"{float(value):.{digits}f}"
    except Exception:
        return "n/a"


def write_full_refresh_diagnostics_report(output_dir: Path, baseline_dir: Optional[Path]) -> Path:
    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    asset_name = str(manifest.get("asset_name") or "BTC")
    latest = pd.read_csv(output_dir / "latest_forecast.csv")
    leaderboard = pd.read_csv(output_dir / "model_leaderboard.csv")
    rejection = pd.read_csv(output_dir / "csv" / "model_rejection_reasons.csv")
    ablation = pd.read_csv(output_dir / "csv" / "feature_group_ablation.csv")
    regimes = pd.read_csv(output_dir / "csv" / "model_regime_breakdown.csv")
    availability = pd.read_csv(output_dir / "data_availability.csv")
    coverage_path = output_dir / "csv" / "derivatives_coverage.csv"
    coverage = pd.read_csv(coverage_path) if coverage_path.exists() else pd.DataFrame()
    stability_path = output_dir / "csv" / "feature_group_stability.csv"
    stability = pd.read_csv(stability_path) if stability_path.exists() else pd.DataFrame()
    primary = latest[latest["is_primary_objective"] == True].iloc[0]
    full_30 = leaderboard[(leaderboard["horizon"] == 30) & (leaderboard["window_type"] == "official_monthly")].copy()

    quick_text = "Quick baseline was unavailable, so quick-vs-full comparisons are limited."
    if baseline_dir and (baseline_dir / "run_manifest.json").exists():
        quick_manifest = json.loads((baseline_dir / "run_manifest.json").read_text(encoding="utf-8"))
        quick_latest = pd.read_csv(baseline_dir / "latest_forecast.csv")
        quick_lb = pd.read_csv(baseline_dir / "model_leaderboard.csv")
        quick_primary = quick_latest[quick_latest["is_primary_objective"] == True].iloc[0]
        quick_30 = quick_lb[(quick_lb["horizon"] == 30) & (quick_lb["window_type"] == "official_monthly")]
        quick_text = (
            f"Quick mode used `{quick_manifest.get('start_date')}` to `{quick_manifest.get('end_date')}` with "
            f"`{int(quick_30['sample_count'].iloc[0]) if not quick_30.empty else 0}` 30d official samples and selected "
            f"`{quick_primary['selected_model']}`. Full refresh used `{manifest.get('start_date')}` to `{manifest.get('end_date')}` "
            f"with `{int(full_30['sample_count'].iloc[0]) if not full_30.empty else 0}` 30d official samples and selected "
            f"`{primary['selected_model']}`. The quick edge disappeared because the broader 2015+ sample added regimes where the ML models "
            "did not beat the 90d momentum baseline or the full baseline set after costs."
        )

    ml_30 = full_30[~full_30["model"].isin(["buy_hold_direction", "momentum_30d", "momentum_90d", "random_permutation"])]
    baseline_30 = full_30[full_30["model"].isin(["buy_hold_direction", "momentum_30d", "momentum_90d", "random_permutation"])]
    best_ml = ml_30.sort_values("directional_accuracy", ascending=False).head(1)
    best_baseline = baseline_30.sort_values("directional_accuracy", ascending=False).head(1)
    baseline_winners = baseline_30[baseline_30["directional_accuracy"] > (best_ml["directional_accuracy"].iloc[0] if not best_ml.empty else -1)]

    rejection_30_ml = rejection[
        (rejection["horizon"] == 30)
        & (rejection["window_type"] == "official_monthly")
        & (~rejection["is_baseline"])
    ][["model_name", "final_rejection_reason"]]

    ablation_30 = ablation[(ablation["horizon"] == 30) & (ablation["window_type"] == "official_monthly")]
    ablation_best = ablation_30.sort_values("directional_accuracy", ascending=False).head(10)
    watchlist = ablation_30[
        (ablation_30["feature_group"] == "dollar_rates_only")
        & (ablation_30["model_name"] == "logistic_linear")
    ].head(1)
    regime_30 = regimes[(regimes["horizon"] == 30) & (regimes["window_type"] == "official_monthly")]
    regime_best = regime_30.sort_values("directional_accuracy", ascending=False).head(10)

    derivatives_worked = availability[availability["dataset"].str.startswith("binance_")]["status"].eq("worked").any()
    manual_used = availability[
        (availability["source"] == "manual_csv")
        & (availability["dataset"].str.startswith("binance_"))
        & (availability["is_used_in_model"] == True)
    ]["status"].eq("worked").any()
    derivatives_impact_path = output_dir / "csv" / "derivatives_impact.csv"
    derivatives_impact_frame = pd.read_csv(derivatives_impact_path) if derivatives_impact_path.exists() else pd.DataFrame()

    report = [
        "# Full Refresh Diagnostics",
        "",
        "## Summary",
        f"- Full refresh run ID: `{manifest.get('run_id')}`",
        f"- Selected model: `{primary['selected_model']}`",
        f"- Signal: `{primary['signal']}`",
        f"- Reliability: `{primary['reliability_label']}`",
        f"- no_valid_edge triggered correctly: `{primary['selected_model'] == 'no_valid_edge'}`",
        "",
        "## Why Quick Mode Looked Better",
        quick_text,
        "",
        "## 30d Official Model Comparison",
        f"- Best ML model: `{best_ml.iloc[0]['model'] if not best_ml.empty else 'none'}` with directional accuracy `{_fmt_pct(best_ml.iloc[0]['directional_accuracy']) if not best_ml.empty else 'n/a'}`",
        f"- Best baseline: `{best_baseline.iloc[0]['model'] if not best_baseline.empty else 'none'}` with directional accuracy `{_fmt_pct(best_baseline.iloc[0]['directional_accuracy']) if not best_baseline.empty else 'n/a'}`",
        f"- Baselines that beat the best ML model: `{', '.join(baseline_winners['model'].astype(str).tolist()) if not baseline_winners.empty else 'none'}`",
        "",
        "Validation gates failed by ML model:",
    ]
    for _, row in rejection_30_ml.iterrows():
        report.append(f"- `{row['model_name']}`: {row['final_rejection_reason']}")

    report.extend(
        [
            "",
            "## Feature Group Findings",
        ]
    )
    for _, row in ablation_best.iterrows():
        report.append(
            f"- `{row['feature_group']}` / `{row['model_name']}`: samples `{int(row['n_samples'])}`, "
            f"accuracy `{_fmt_pct(row['directional_accuracy'])}`, net return `{_fmt_pct(row['net_return'])}`, "
            f"reliability `{row['reliability_label']}`"
        )

    report.extend(["", "## Feature Group Watchlist"])
    if watchlist.empty:
        report.append("- `dollar_rates_only` / `logistic_linear`: unavailable in this run.")
    else:
        row = watchlist.iloc[0]
        report.append(
            f"- `dollar_rates_only` / `logistic_linear`: samples `{int(row['n_samples'])}`, "
            f"accuracy `{_fmt_pct(row['directional_accuracy'])}`, Sharpe `{_fmt_num(row['sharpe'])}`, "
            f"max drawdown `{_fmt_pct(row['max_drawdown'])}`, Brier `{_fmt_num(row['brier_score'], 3)}`, "
            f"calibration error `{_fmt_num(row['calibration_error'], 3)}`, "
            f"beats buy-hold `{bool(row['beats_buy_hold'])}`, beats momentum `{bool(row['beats_momentum'])}`."
        )
        report.append("- It remains diagnostic only because feature-group results are not official model-selection inputs and must prove stability across regimes first.")

    report.extend(
        [
            "",
            "## Regime Findings",
        ]
    )
    for _, row in regime_best.iterrows():
        report.append(
            f"- `{row['model_name']}` in `{row['regime']}`: samples `{int(row['n_samples'])}`, "
            f"accuracy `{_fmt_pct(row['directional_accuracy'])}`, net return `{_fmt_pct(row['net_return'])}`, "
            f"reliability `{row['reliability_label']}`"
        )

    report.extend(
        [
            "",
            "## Binance Derivatives",
            f"- Binance derivatives recovered: `{derivatives_worked}`",
            f"- Manual derivatives used: `{manual_used}`",
        ]
    )
    if not coverage.empty:
        coverage_summary = coverage.groupby(["source", "status"]).size().reset_index(name="count")
        for _, row in coverage_summary.iterrows():
            report.append(f"- Coverage `{row['source']}` / `{row['status']}`: `{int(row['count'])}` metrics")
    if derivatives_impact_frame.empty:
        report.append("- Derivatives impact test was not run because no derivative features were available.")
    else:
        best_deriv = derivatives_impact_frame.sort_values("directional_accuracy", ascending=False).head(1).iloc[0]
        report.append(
            f"- Best derivatives impact row: `{best_deriv['model_name']}`, derivatives included `{best_deriv['derivatives_included']}`, "
            f"accuracy `{_fmt_pct(best_deriv['directional_accuracy'])}`."
        )
    if not stability.empty:
        stable_rows = stability[
            (stability["horizon"] == 30)
            & (stability["window_type"] == "official_monthly")
            & (stability["n_samples"] >= 8)
            & (stability["directional_accuracy"] >= 0.55)
            & (stability["net_return"] > 0)
        ]
        stable_groups = sorted(stable_rows["feature_group"].unique().tolist())
        report.append(f"- Stable feature groups with positive 30d slices: `{', '.join(stable_groups) if stable_groups else 'none'}`")

    report.extend(
        [
            "",
            "## Conclusion",
            f"- The full refresh did not validate a 30d {asset_name} directional edge.",
            "- `no_valid_edge` remains the correct conclusion unless future data reliability or feature diagnostics materially improve.",
            "- Recommended next step: improve Binance/manual derivatives coverage, then rerun this diagnostic sprint before adding more models.",
        ]
    )

    path = output_dir / "full_refresh_diagnostics.md"
    path.write_text("\n".join(report) + "\n", encoding="utf-8")
    return path


def write_derivatives_recovery_report(output_dir: Path) -> Path:
    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    latest = pd.read_csv(output_dir / "latest_forecast.csv")
    availability = pd.read_csv(output_dir / "data_availability.csv")
    coverage = pd.read_csv(output_dir / "csv" / "derivatives_coverage.csv")
    impact = pd.read_csv(output_dir / "csv" / "derivatives_impact.csv")
    ablation = pd.read_csv(output_dir / "csv" / "feature_group_ablation.csv")
    stability = pd.read_csv(output_dir / "csv" / "feature_group_stability.csv")
    primary = latest[latest["is_primary_objective"] == True].iloc[0]

    binance_recovered = availability[
        (availability["source"] == "Binance USD-M Futures")
        & (availability["dataset"].str.startswith("binance_"))
    ]["status"].eq("worked").any()
    manual_used = availability[
        (availability["source"] == "manual_csv")
        & (availability["dataset"].str.startswith("binance_"))
        & (availability["is_used_in_model"] == True)
    ]["status"].eq("worked").any()

    impact_text = "No derivatives impact comparison ran because no valid derivative feature entered the model feature set."
    derivatives_improved = False
    if not impact.empty:
        impact_30 = impact[(impact["horizon"] == 30) & (impact["window_type"] == "official_monthly")]
        with_deriv = impact_30[impact_30["derivatives_included"] == True].sort_values("directional_accuracy", ascending=False).head(1)
        without_deriv = impact_30[impact_30["derivatives_included"] == False].sort_values("directional_accuracy", ascending=False).head(1)
        if not with_deriv.empty and not without_deriv.empty:
            derivatives_improved = bool(with_deriv.iloc[0]["directional_accuracy"] > without_deriv.iloc[0]["directional_accuracy"])
            impact_text = (
                f"Best with derivatives: `{with_deriv.iloc[0]['model_name']}` at `{_fmt_pct(with_deriv.iloc[0]['directional_accuracy'])}`. "
                f"Best without derivatives: `{without_deriv.iloc[0]['model_name']}` at `{_fmt_pct(without_deriv.iloc[0]['directional_accuracy'])}`. "
                f"Improved directional accuracy: `{derivatives_improved}`."
            )

    ablation_30 = ablation[(ablation["horizon"] == 30) & (ablation["window_type"] == "official_monthly")]
    top_ablation = ablation_30.sort_values("directional_accuracy", ascending=False).head(1)
    watchlist = ablation_30[
        (ablation_30["feature_group"] == "dollar_rates_only")
        & (ablation_30["model_name"] == "logistic_linear")
    ].head(1)
    dollar_remains_strongest = (
        not top_ablation.empty
        and not watchlist.empty
        and str(top_ablation.iloc[0]["feature_group"]) == "dollar_rates_only"
        and str(top_ablation.iloc[0]["model_name"]) == "logistic_linear"
    )

    stable = stability[
        (stability["horizon"] == 30)
        & (stability["window_type"] == "official_monthly")
        & (stability["n_samples"] >= 8)
        & (stability["directional_accuracy"] >= 0.55)
        & (stability["net_return"] > 0)
    ]
    stable_counts = stable.groupby(["feature_group", "model_name"]).size().reset_index(name="stable_slices") if not stable.empty else pd.DataFrame()
    stable_counts = stable_counts.sort_values("stable_slices", ascending=False).head(5) if not stable_counts.empty else stable_counts

    recommendation = "Keep no_valid_edge and prioritize validated derivative history plus feature pruning before adding more models."
    if str(primary["selected_model"]) != "no_valid_edge":
        recommendation = "A model passed official 30d rules; review the full leaderboard and calibration before deployment."
    elif derivatives_improved:
        recommendation = "Derivatives improved diagnostics but did not pass official selection; extend historical coverage before adding models."
    elif not dollar_remains_strongest:
        recommendation = "Feature-group leadership changed; inspect stability before pruning or expanding features."

    report = [
        "# Derivatives Recovery Report",
        "",
        "## Run Summary",
        f"- Run ID: `{manifest.get('run_id')}`",
        f"- Selected model: `{primary['selected_model']}`",
        f"- Signal: `{primary['signal']}`",
        f"- Reliability: `{primary['reliability_label']}`",
        f"- no_valid_edge remains correct: `{primary['selected_model'] == 'no_valid_edge'}`",
        "",
        "## Derivatives Coverage",
        f"- Binance derivatives recovered: `{binance_recovered}`",
        f"- Manual CSV derivatives used: `{manual_used}`",
    ]
    if coverage.empty:
        report.append("- No derivatives coverage rows were written.")
    else:
        for _, row in coverage.iterrows():
            report.append(
                f"- `{row['source']}` / `{row['metric']}`: `{row['status']}`, rows `{int(row['rows'])}`, "
                f"dates `{row['first_date']}` to `{row['last_date']}`, missing `{_fmt_pct(row['missing_pct'])}`, "
                f"used `{bool(row['used_in_model'])}`"
            )

    report.extend(
        [
            "",
            "## Derivatives Impact",
            f"- {impact_text}",
            "",
            "## Feature Group Watchlist",
        ]
    )
    if watchlist.empty:
        report.append("- `dollar_rates_only` / `logistic_linear`: unavailable.")
    else:
        row = watchlist.iloc[0]
        report.append(
            f"- `dollar_rates_only` / `logistic_linear`: samples `{int(row['n_samples'])}`, "
            f"accuracy `{_fmt_pct(row['directional_accuracy'])}`, Sharpe `{_fmt_num(row['sharpe'])}`, "
            f"drawdown `{_fmt_pct(row['max_drawdown'])}`, Brier `{_fmt_num(row['brier_score'], 3)}`, "
            f"calibration error `{_fmt_num(row['calibration_error'], 3)}`, reliability `{row['reliability_label']}`."
        )
        report.append(f"- Remains strongest diagnostic group: `{dollar_remains_strongest}`")
        report.append("- It is not selected unless it passes the official 30d selection rules in the main leaderboard.")

    report.extend(["", "## Feature Group Stability"])
    if stable_counts.empty:
        report.append("- No feature group showed broad stable 30d slices under the diagnostic threshold.")
    else:
        for _, row in stable_counts.iterrows():
            report.append(f"- `{row['feature_group']}` / `{row['model_name']}`: `{int(row['stable_slices'])}` stable slices")

    report.extend(
        [
            "",
            "## Conclusion",
            f"- Derivatives improved 30d official diagnostics: `{derivatives_improved}`",
            f"- `no_valid_edge` remains correct: `{primary['selected_model'] == 'no_valid_edge'}`",
            f"- Recommended next step: {recommendation}",
        ]
    )

    path = output_dir / "derivatives_recovery_report.md"
    path.write_text("\n".join(report) + "\n", encoding="utf-8")
    return path


def write_diagnostics(
    *,
    output_dir: Path,
    run_id: str,
    raw: pd.DataFrame,
    feature_result,
    config: ResearchConfig,
    base_outputs: Dict[str, pd.DataFrame],
    latest: pd.DataFrame,
    baseline_dir: Optional[Path],
    quick: bool,
) -> Dict[str, pd.DataFrame]:
    selected_model = str(latest.loc[latest["is_primary_objective"] == True, "selected_model"].iloc[0])
    diagnostics = {
        "model_rejection_reasons.csv": model_rejection_reasons(base_outputs["backtest_summary.csv"], selected_model),
        "feature_signal_diagnostics.csv": feature_signal_diagnostics(run_id, raw, feature_result, config),
        "feature_group_ablation.csv": feature_group_ablation(run_id, raw, feature_result, config, quick=quick),
        "model_regime_breakdown.csv": model_regime_breakdown(
            run_id,
            raw,
            base_outputs["equity_curves.csv"],
            base_outputs["backtest_summary.csv"],
        ),
        "derivatives_coverage.csv": derivatives_coverage(run_id, raw, feature_result, base_outputs["data_availability.csv"]),
        "derivatives_impact.csv": derivatives_impact(run_id, raw, feature_result, config, quick=quick),
        "feature_group_stability.csv": feature_group_stability(run_id, raw, feature_result, config, quick=quick),
        "polymarket_coverage.csv": polymarket_coverage(run_id, raw, feature_result, base_outputs["data_availability.csv"]),
        "polymarket_feature_diagnostics.csv": polymarket_feature_diagnostics(run_id, raw, feature_result, config),
        "polymarket_impact.csv": polymarket_impact(run_id, raw, feature_result, config, quick=quick),
    }
    for filename, frame in diagnostics.items():
        write_diagnostic_csv(filename, frame, output_dir)
    write_full_refresh_diagnostics_report(output_dir, baseline_dir)
    write_derivatives_recovery_report(output_dir)
    return diagnostics
