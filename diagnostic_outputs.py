from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from research_config import MIN_SAMPLE_THRESHOLDS, ResearchConfig
from schemas import empty_diagnostic_frame, empty_output_frame, write_diagnostic_csv, write_schema_csv


BASELINE_DIR = Path("/tmp/btc_refresh_baseline")
LONG_POLICY_THRESHOLDS = (0.50, 0.55, 0.60, 0.65)
EXPECTED_RETURN_FILTERS = (0.0, 0.02, 0.05)
RISK_OFF_PROBABILITY_THRESHOLDS = (0.35, 0.40, 0.45)
VALID_ACTIVE_MIN_COUNT = 12
VALID_ACTIVE_MIN_RATIO = 0.20
HIGH_CONFIDENCE_ACTIVE_MIN_COUNT = 18
HIGH_CONFIDENCE_ACTIVE_MIN_RATIO = 0.30
MATERIAL_BRIER_WORSENING = 0.03
MATERIAL_CALIBRATION_WORSENING = 0.03
MATERIAL_DRAWDOWN_WORSENING = 0.10


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


def active_signal_floor(n_samples: int) -> int:
    return int(max(VALID_ACTIVE_MIN_COUNT, math.ceil(float(n_samples) * VALID_ACTIVE_MIN_RATIO)))


def high_confidence_signal_floor(n_samples: int) -> int:
    return int(max(HIGH_CONFIDENCE_ACTIVE_MIN_COUNT, math.ceil(float(n_samples) * HIGH_CONFIDENCE_ACTIVE_MIN_RATIO)))


def is_official_long_policy_allowed(long_threshold: float, expected_return_min: float = 0.0) -> bool:
    return bool(float(long_threshold) >= 0.55)


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
    if pair.iloc[:, 0].nunique(dropna=True) < 2 or pair.iloc[:, 1].nunique(dropna=True) < 2:
        return np.nan, np.nan, np.nan, np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
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


def _policy_frame(
    preds: pd.DataFrame,
    long_threshold: float,
    expected_return_min: float,
    transaction_cost_bps: float,
    *,
    position_col: str = "policy_position",
) -> pd.DataFrame:
    frame = preds.copy()
    expected_return = np.exp(frame["expected_log_return"].astype(float)) - 1
    frame[position_col] = (
        (frame["probability_up"].astype(float) >= float(long_threshold))
        & (expected_return >= float(expected_return_min))
    ).astype(float)
    cost_rate = float(transaction_cost_bps) / 10000.0
    frame["policy_turnover"] = frame[position_col].diff().abs().fillna(frame[position_col].abs())
    frame["policy_return_after_costs"] = frame[position_col] * frame["actual_return"].astype(float) - frame["policy_turnover"] * cost_rate
    return frame


def _policy_score(
    calibration: pd.DataFrame,
    long_threshold: float,
    expected_return_min: float,
    transaction_cost_bps: float,
) -> dict:
    if calibration.empty:
        return {"active_count": 0, "active_coverage": 0.0, "hit_rate": np.nan, "net_return": -np.inf}
    frame = _policy_frame(calibration, long_threshold, expected_return_min, transaction_cost_bps)
    active = frame[frame["policy_position"] > 0]
    return {
        "active_count": int(len(active)),
        "active_coverage": float(len(active) / len(frame)) if len(frame) else 0.0,
        "hit_rate": float(active["actual_up"].astype(int).mean()) if not active.empty else np.nan,
        "net_return": float((1 + frame["policy_return_after_costs"]).prod() - 1) if len(frame) else -np.inf,
    }


def select_signal_policy_from_calibration(calibration: pd.DataFrame, config: ResearchConfig) -> dict:
    if calibration.empty:
        return {
            "long_threshold": 0.55,
            "expected_return_min": 0.0,
            "policy_valid": False,
            "policy_source": "train_calibration_only",
            "policy_rejection_reason": "empty_calibration",
        }
    min_active = active_signal_floor(len(calibration))
    candidates = []
    for long_threshold in LONG_POLICY_THRESHOLDS:
        for expected_min in EXPECTED_RETURN_FILTERS:
            score = _policy_score(calibration, long_threshold, expected_min, config.transaction_cost_bps)
            diagnostic_only = not is_official_long_policy_allowed(long_threshold, expected_min)
            valid = bool(
                not diagnostic_only
                and score["active_count"] >= min_active
                and score["active_coverage"] >= VALID_ACTIVE_MIN_RATIO
                and score["net_return"] > 0
                and pd.notna(score["hit_rate"])
                and score["hit_rate"] >= 0.50
            )
            candidates.append(
                {
                    "long_threshold": float(long_threshold),
                    "expected_return_min": float(expected_min),
                    "diagnostic_only": diagnostic_only,
                    "valid": valid,
                    **score,
                }
            )
    valid_candidates = [row for row in candidates if row["valid"]]
    if not valid_candidates:
        return {
            "long_threshold": 0.55,
            "expected_return_min": 0.0,
            "policy_valid": False,
            "policy_source": "train_calibration_only",
            "policy_rejection_reason": "no_train_calibration_policy_met_active_count_return_hit_rate_gates",
        }
    selected = sorted(
        valid_candidates,
        key=lambda row: (
            row["net_return"],
            row["hit_rate"] if pd.notna(row["hit_rate"]) else -np.inf,
            row["active_count"],
            -row["long_threshold"],
        ),
        reverse=True,
    )[0]
    return {
        "long_threshold": float(selected["long_threshold"]),
        "expected_return_min": float(selected["expected_return_min"]),
        "policy_valid": True,
        "policy_source": "train_calibration_only",
        "policy_rejection_reason": "",
    }


def select_risk_off_threshold_from_calibration(calibration: pd.DataFrame) -> dict:
    if calibration.empty:
        return {
            "risk_off_threshold": np.nan,
            "risk_off_bucket_count": 0,
            "risk_off_bucket_avg_return": np.nan,
            "risk_off_threshold_source": "unavailable",
        }
    min_bucket = int(max(8, math.ceil(len(calibration) * 0.10)))
    candidates = []
    for threshold in RISK_OFF_PROBABILITY_THRESHOLDS:
        bucket = calibration[calibration["probability_up"].astype(float) <= float(threshold)]
        avg_return = float(bucket["actual_return"].mean()) if not bucket.empty else np.nan
        hit_rate = float((bucket["actual_return"] < 0).mean()) if not bucket.empty else np.nan
        valid = bool(len(bucket) >= min_bucket and pd.notna(avg_return) and avg_return < 0 and pd.notna(hit_rate) and hit_rate >= 0.50)
        candidates.append(
            {
                "risk_off_threshold": float(threshold),
                "risk_off_bucket_count": int(len(bucket)),
                "risk_off_bucket_avg_return": avg_return,
                "risk_off_hit_rate": hit_rate,
                "valid": valid,
            }
        )
    valid_candidates = [row for row in candidates if row["valid"]]
    if not valid_candidates:
        return {
            "risk_off_threshold": np.nan,
            "risk_off_bucket_count": 0,
            "risk_off_bucket_avg_return": np.nan,
            "risk_off_hit_rate": np.nan,
            "risk_off_threshold_source": "unavailable",
        }
    selected = sorted(valid_candidates, key=lambda row: (row["risk_off_threshold"], row["risk_off_bucket_count"]), reverse=True)[0]
    selected["risk_off_threshold_source"] = "train_calibration_probability_bucket"
    return selected


def _add_signal_policy_columns(test_preds: pd.DataFrame, calibration_eval: pd.DataFrame, config: ResearchConfig) -> pd.DataFrame:
    policy = select_signal_policy_from_calibration(calibration_eval, config)
    risk = select_risk_off_threshold_from_calibration(calibration_eval)
    out = test_preds.copy()
    expected_return = np.exp(out["expected_log_return"].astype(float)) - 1
    long_threshold = float(policy["long_threshold"])
    expected_min = float(policy["expected_return_min"])
    risk_threshold = risk.get("risk_off_threshold", np.nan)
    out["policy_long_threshold"] = long_threshold
    out["policy_expected_return_min"] = expected_min
    out["policy_source"] = policy["policy_source"]
    out["policy_valid_in_calibration"] = bool(policy["policy_valid"])
    out["policy_rejection_reason"] = policy["policy_rejection_reason"]
    out["policy_position"] = ((out["probability_up"].astype(float) >= long_threshold) & (expected_return >= expected_min)).astype(float)
    out["risk_off_probability_threshold"] = risk_threshold
    out["risk_off_threshold_source"] = risk.get("risk_off_threshold_source", "unavailable")
    out["risk_off_bucket_count_train"] = int(risk.get("risk_off_bucket_count", 0) or 0)
    out["risk_off_bucket_avg_return_train"] = risk.get("risk_off_bucket_avg_return", np.nan)
    out["risk_off_flag"] = (
        out["probability_up"].astype(float) <= float(risk_threshold)
        if pd.notna(risk_threshold)
        else False
    )
    out["high_downside_flag"] = (
        out["risk_off_flag"].astype(bool)
        & (out["probability_up"].astype(float) <= 0.35)
        & (expected_return < -0.10)
        & (out["risk_off_bucket_count_train"].astype(int) >= 8)
    )
    return out


def _policy_returns_for_predictions(preds: pd.DataFrame, transaction_cost_bps: float) -> pd.DataFrame:
    frame = preds.copy()
    if "policy_position" not in frame:
        frame["policy_position"] = 0.0
    cost_rate = float(transaction_cost_bps) / 10000.0
    frame["policy_turnover"] = frame["policy_position"].astype(float).diff().abs().fillna(frame["policy_position"].astype(float).abs())
    frame["policy_return_after_costs"] = frame["policy_position"].astype(float) * frame["actual_return"].astype(float) - frame["policy_turnover"] * cost_rate
    frame["policy_equity"] = (1 + frame["policy_return_after_costs"]).cumprod()
    return frame


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


def _asset_id_from_run(run_id: str) -> str:
    return run_id.split("_research", 1)[0] if "_research" in run_id else ""


def _sample_fingerprint(dates: Sequence[pd.Timestamp]) -> str:
    import hashlib

    payload = "|".join(pd.Timestamp(date).date().isoformat() for date in pd.DatetimeIndex(dates))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def candidate_feature_sets(feature_cols: Sequence[str], asset_id: str) -> Dict[str, List[str]]:
    cols = [col for col in dict.fromkeys(feature_cols) if not col.startswith("polymarket_")]
    groups = feature_group_columns(cols)

    def only(*group_names: str) -> List[str]:
        selected: List[str] = []
        for group_name in group_names:
            selected.extend(groups.get(group_name, []))
        return list(dict.fromkeys(selected))

    def without(*group_names: str) -> List[str]:
        excluded = set(only(*group_names))
        return [col for col in cols if col not in excluded]

    sets = {
        "all_features": cols,
        "price_momentum_only": groups.get("price_momentum_only", []),
        "price_plus_macro": only("price_momentum_only", "macro_liquidity_only", "dollar_rates_only"),
        "price_plus_risk_assets": only("price_momentum_only", "risk_assets_only"),
        "price_plus_dollar_rates": only("price_momentum_only", "dollar_rates_only"),
        "price_plus_stablecoins": only("price_momentum_only", "stablecoins_only"),
        "core_price_macro_risk": only("price_momentum_only", "macro_liquidity_only", "dollar_rates_only", "risk_assets_only"),
        "no_onchain": without("onchain_only"),
        "no_derivatives": without("derivatives_only"),
        "no_polymarket": [col for col in cols if not col.startswith("polymarket_")],
    }
    cycle_cols = [col for col in cols if "cycle_" in col]
    if asset_id == "btc":
        derivatives_pack = groups.get("derivatives_only", [])
        dollar_rates_cycle = list(dict.fromkeys(only("dollar_rates_only") + cycle_cols))
        sets.update(
            {
                "btc_dollar_rates_cycle": dollar_rates_cycle,
                "btc_derivatives_pack": derivatives_pack,
                "btc_dollar_rates_cycle_plus_derivatives_pack": list(dict.fromkeys(dollar_rates_cycle + derivatives_pack)),
                "btc_core_dollar_derivatives_cycle": list(
                    dict.fromkeys(only("price_momentum_only", "dollar_rates_only", "derivatives_only") + cycle_cols)
                ),
                "btc_no_sparse_derivatives": without("derivatives_only") if len(groups.get("derivatives_only", [])) < 6 else cols,
            }
        )
    if asset_id == "sol":
        sets.update(
            {
                "sol_rates_risk_price": only("price_momentum_only", "dollar_rates_only", "risk_assets_only"),
                "sol_dollar_rates_only": groups.get("dollar_rates_only", []),
                "sol_macro_liquidity_price": only("price_momentum_only", "macro_liquidity_only"),
                "sol_risk_assets_only": groups.get("risk_assets_only", []),
            }
        )
    return {name: list(dict.fromkeys(values)) for name, values in sets.items() if values}


def _feature_valid_threshold(feature: str, config: ResearchConfig) -> float:
    if feature_group_for_feature(feature) == "derivatives_only":
        return float(config.min_derivative_feature_valid_ratio)
    return float(config.min_feature_valid_ratio)


def _training_feature_stats(train: pd.DataFrame, feature: str, target_col: str = "target_log_return") -> dict:
    series = train[feature] if feature in train else pd.Series(index=train.index, dtype=float)
    target = train[target_col] if target_col in train else pd.Series(index=train.index, dtype=float)
    valid = pd.concat([series, target], axis=1).dropna()
    pearson, spearman, t_stat, p_value = _corr_stats(series, target)
    return {
        "feature": feature,
        "n_samples": int(len(valid)),
        "coverage_pct": float(series.notna().mean()) if len(series) else 0.0,
        "ic": spearman,
        "abs_ic": abs(float(spearman)) if pd.notna(spearman) else -np.inf,
        "p_value": p_value,
        "t_stat": t_stat,
    }


def select_nested_pruned_features(
    train: pd.DataFrame,
    candidate_cols: Sequence[str],
    config: ResearchConfig,
    *,
    max_features: int = 30,
    corr_threshold: float = 0.85,
) -> List[str]:
    from research_pipeline import is_leaky_feature_name

    stats = []
    for feature in candidate_cols:
        if feature not in train or is_leaky_feature_name(feature) or feature.startswith("polymarket_"):
            continue
        if not pd.api.types.is_numeric_dtype(train[feature]):
            continue
        row = _training_feature_stats(train, feature)
        min_coverage = _feature_valid_threshold(feature, config)
        if row["coverage_pct"] < min_coverage or row["n_samples"] < 120 or not np.isfinite(row["abs_ic"]):
            continue
        row["group"] = feature_group_for_feature(feature)
        stats.append(row)
    if not stats:
        return []

    stats_frame = pd.DataFrame(stats)
    stats_frame["group_rank"] = stats_frame.groupby("group")["abs_ic"].rank(method="first", ascending=False)
    stats_frame = stats_frame[stats_frame["group_rank"] <= 10].copy()
    stats_frame = stats_frame.sort_values(["abs_ic", "coverage_pct"], ascending=[False, False])

    kept: List[str] = []
    numeric_train = train[[col for col in stats_frame["feature"] if col in train]].copy()
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = numeric_train.corr(method="spearman").abs() if not numeric_train.empty else pd.DataFrame()
    for feature in stats_frame["feature"]:
        if len(kept) >= max_features:
            break
        if corr.empty:
            kept.append(feature)
            continue
        too_close = any(
            feature in corr.index
            and other in corr.columns
            and pd.notna(corr.at[feature, other])
            and corr.at[feature, other] >= corr_threshold
            for other in kept
        )
        if not too_close:
            kept.append(feature)
    return kept


def _predict_model_windows_nested_pruned(
    model_name: str,
    frame: pd.DataFrame,
    candidate_cols: Sequence[str],
    windows: Sequence[object],
    config: ResearchConfig,
    *,
    quick: bool,
) -> Tuple[pd.DataFrame, List[int]]:
    from research_pipeline import _calibrate_probabilities, _model_pipeline, _split_fit_calibration, tune_threshold_nested

    rows = []
    selected_counts: List[int] = []
    valid = frame.dropna(subset=["target_log_return", "target_up", "asset_close"]).copy()
    for idx, window in enumerate(windows):
        train = valid.loc[valid.index < window.test_date].copy()
        test = valid.loc[[window.test_date]].copy()
        if train.empty or test.empty:
            continue
        fit_train, cal = _split_fit_calibration(train, config)
        selected_cols = select_nested_pruned_features(fit_train, candidate_cols, config)
        selected_cols = [col for col in selected_cols if fit_train[col].notna().any()]
        if len(selected_cols) < 5 or fit_train["target_up"].nunique() < 2 or cal["target_up"].nunique() < 2:
            continue

        model = _model_pipeline(model_name, config.random_seed + idx, quick)
        model.fit(fit_train[selected_cols], fit_train["target_up"].astype(int))
        raw_cal_prob = model.predict_proba(cal[selected_cols])[:, 1]
        calibrated_cal_prob, _ = _calibrate_probabilities(raw_cal_prob, raw_cal_prob, cal["target_up"].astype(int).to_numpy())
        fit_up = fit_train.loc[fit_train["target_up"] == 1, "target_log_return"]
        fit_down = fit_train.loc[fit_train["target_up"] == 0, "target_log_return"]
        fit_mean_up = float(fit_up.mean()) if len(fit_up) else float(fit_train["target_log_return"].mean())
        fit_mean_down = float(fit_down.mean()) if len(fit_down) else float(fit_train["target_log_return"].mean())
        calibrated_cal_expected = calibrated_cal_prob * fit_mean_up + (1 - calibrated_cal_prob) * fit_mean_down
        cal_eval = pd.DataFrame(
            {
                "probability_up": calibrated_cal_prob,
                "expected_log_return": calibrated_cal_expected,
                "actual_return": np.exp(cal["target_log_return"].astype(float)) - 1,
                "actual_up": cal["target_up"].astype(int),
            },
            index=cal.index,
        )
        threshold = tune_threshold_nested(cal_eval, config.transaction_cost_bps)

        raw_test_prob = model.predict_proba(test[selected_cols])[:, 1]
        prob, prob_conf = _calibrate_probabilities(raw_test_prob, raw_cal_prob, cal["target_up"].astype(int).to_numpy())
        prob_up = float(prob[0])
        cal_up = cal.loc[cal["target_up"] == 1, "target_log_return"]
        cal_down = cal.loc[cal["target_up"] == 0, "target_log_return"]
        mean_up = float(cal_up.mean()) if len(cal_up) else float(cal["target_log_return"].mean())
        mean_down = float(cal_down.mean()) if len(cal_down) else float(cal["target_log_return"].mean())
        expected_log = prob_up * mean_up + (1 - prob_up) * mean_down
        actual_log = float(test["target_log_return"].iloc[0])
        selected_counts.append(len(selected_cols))
        row = {
            "date": window.test_date,
            "model": model_name,
            "horizon": window.horizon,
            "window_type": window.window_type,
            "is_official": window.is_official,
            "probability_up": prob_up,
            "probability_confidence": prob_conf,
            "expected_log_return": expected_log,
            "actual_log_return": actual_log,
            "actual_return": math.exp(actual_log) - 1,
            "actual_up": int(actual_log > 0),
            "threshold": threshold,
            "threshold_source": "nested_feature_selection_and_calibration",
            "predicted_up": int(prob_up >= threshold),
            "position": float(prob_up >= threshold),
            "train_rows": int(len(fit_train)),
            "calibration_rows": int(len(cal)),
            "feature_count": int(len(selected_cols)),
        }
        policy_input = pd.DataFrame([row]).set_index("date")
        policy_row = _add_signal_policy_columns(policy_input, cal_eval, config).iloc[0].to_dict()
        row.update({key: policy_row[key] for key in policy_row if key not in row or key.startswith(("policy_", "risk_", "high_"))})
        rows.append(row)
    if not rows:
        return pd.DataFrame(), selected_counts
    return pd.DataFrame(rows).set_index("date").sort_index(), selected_counts


def _bootstrap_signal_quality(preds: pd.DataFrame, config: ResearchConfig, *, quick: bool) -> Tuple[float, float, float]:
    if preds.empty:
        return np.nan, np.nan, np.nan
    iterations = config.quick_bootstrap_iterations if quick else min(config.bootstrap_iterations, 250)
    perm_iterations = config.quick_permutation_iterations if quick else min(config.permutation_iterations, 250)
    rng = np.random.default_rng(config.random_seed + len(preds))
    actual = preds["actual_up"].astype(int).to_numpy()
    pred = preds["predicted_up"].astype(int).to_numpy()
    observed = float(np.mean(actual == pred))
    boot = []
    n = len(preds)
    for _ in range(iterations):
        idx = rng.integers(0, n, n)
        boot.append(float(np.mean(actual[idx] == pred[idx])))
    perm_hits = 0
    for _ in range(perm_iterations):
        if float(np.mean(rng.permutation(actual) == pred)) >= observed:
            perm_hits += 1
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), float((perm_hits + 1) / (perm_iterations + 1))


def signal_quality_row(
    run_id: str,
    asset_id: str,
    horizon: int,
    window_type: str,
    candidate_name: str,
    model_name: str,
    preds: pd.DataFrame,
    summary: dict,
    config: ResearchConfig,
    *,
    quick: bool,
) -> dict:
    if preds.empty:
        ci_low, ci_high, p_value = np.nan, np.nan, np.nan
        long = pd.DataFrame()
    else:
        ci_low, ci_high, p_value = _bootstrap_signal_quality(preds, config, quick=quick)
        long = preds[preds["position"].astype(float) > 0]
    long_actual_down = int((long["actual_up"].astype(int) == 0).sum()) if not long.empty else 0
    return {
        "run_id": run_id,
        "asset_id": asset_id,
        "horizon": horizon,
        "window_type": window_type,
        "candidate_feature_set": candidate_name,
        "model_name": model_name,
        "n_samples": int(len(preds)),
        "n_long_signals": int(len(long)),
        "n_neutral_signals": int(len(preds) - len(long)),
        "abstention_rate": float(1 - len(long) / len(preds)) if len(preds) else np.nan,
        "long_hit_rate": float(long["actual_up"].mean()) if not long.empty else np.nan,
        "long_avg_return": float(long["actual_return"].mean()) if not long.empty else np.nan,
        "false_positive_rate": float(long_actual_down / len(long)) if not long.empty else np.nan,
        "avg_probability_when_long": float(long["probability_up"].mean()) if not long.empty else np.nan,
        "realized_return_when_long": float((1 + long["actual_return"]).prod() - 1) if not long.empty else np.nan,
        "brier_score": summary.get("brier_score", np.nan),
        "calibration_error": summary.get("calibration_error", np.nan),
        "net_return": summary.get("tc_adjusted_return", np.nan),
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "permutation_p_value": p_value,
        "reliability_label": summary.get("reliability_label", "Low confidence"),
    }


def factor_quality_scorecard(
    run_id: str,
    raw: pd.DataFrame,
    feature_result,
    config: ResearchConfig,
) -> pd.DataFrame:
    from research_pipeline import build_target_frame, build_walk_forward_windows

    asset_id = _asset_id_from_run(run_id)
    audit = feature_result.feature_audit.set_index("feature_name") if not feature_result.feature_audit.empty else pd.DataFrame()
    rows = []
    for horizon, window_type in [(30, "official_monthly"), (90, "official_quarterly")]:
        target_frame = build_target_frame(raw, feature_result.features, feature_result.feature_cols, horizon)
        windows = build_walk_forward_windows(target_frame, horizon, window_type, config, quick=False)
        dates = pd.DatetimeIndex([window.test_date for window in windows])
        sample = target_frame.loc[target_frame.index.intersection(dates)]
        if sample.empty:
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            corr_frame = sample[list(feature_result.feature_cols)].corr(method="spearman").abs()
        cluster_ids: Dict[str, str] = {}
        cluster_count = 0
        for feature in feature_result.feature_cols:
            if feature in cluster_ids:
                continue
            cluster_count += 1
            cluster_name = f"cluster_{cluster_count:03d}"
            cluster_ids[feature] = cluster_name
            if feature in corr_frame.index:
                neighbors = corr_frame.columns[(corr_frame.loc[feature] >= 0.85).fillna(False)].tolist()
                for neighbor in neighbors:
                    cluster_ids.setdefault(neighbor, cluster_name)
        for feature in feature_result.feature_cols:
            stats = _training_feature_stats(sample, feature)
            period_scores = []
            for mask in _period_slice_masks(raw, sample.index).values():
                part = sample.loc[mask.to_numpy()]
                if len(part) < 8:
                    continue
                _, spearman, _, _ = _corr_stats(part[feature], part["target_log_return"])
                if pd.notna(spearman):
                    period_scores.append(np.sign(spearman))
            stability = float(abs(np.nanmean(period_scores))) if period_scores else np.nan
            min_coverage = _feature_valid_threshold(feature, config)
            keep = bool(
                stats["coverage_pct"] >= min_coverage
                and stats["n_samples"] >= min_samples_for(horizon, window_type)
                and pd.notna(stats["ic"])
                and abs(float(stats["ic"])) >= 0.05
            )
            reasons = []
            if stats["coverage_pct"] < min_coverage:
                reasons.append("below_coverage_threshold")
            if stats["n_samples"] < min_samples_for(horizon, window_type):
                reasons.append("below_sample_threshold")
            if pd.isna(stats["ic"]) or abs(float(stats["ic"])) < 0.05:
                reasons.append("weak_ic")
            rows.append(
                {
                    "run_id": run_id,
                    "asset_id": asset_id,
                    "feature_name": feature,
                    "feature_group": feature_group_for_feature(feature),
                    "source": audit.at[feature, "source"] if feature in audit.index else "",
                    "horizon": horizon,
                    "window_type": window_type,
                    "n_samples": stats["n_samples"],
                    "coverage_pct": stats["coverage_pct"],
                    "information_coefficient": stats["ic"],
                    "ic_p_value": stats["p_value"],
                    "ic_sign": "positive" if pd.notna(stats["ic"]) and stats["ic"] > 0 else "negative" if pd.notna(stats["ic"]) and stats["ic"] < 0 else "flat",
                    "ic_stability_score": stability,
                    "missing_pct": 1 - stats["coverage_pct"],
                    "correlation_cluster": cluster_ids.get(feature, ""),
                    "keep_candidate": keep,
                    "drop_reason": "; ".join(reasons),
                }
            )
    return pd.DataFrame(rows)


def _confidence_lookup(confidence: Optional[pd.DataFrame]) -> Dict[str, dict]:
    if confidence is None or confidence.empty:
        return {}
    conf = confidence[
        (confidence["horizon"] == 30)
        & (confidence["window_type"] == "official_monthly")
        & (confidence["metric"] == "directional_accuracy")
    ].copy()
    return {str(row["model"]): row.to_dict() for _, row in conf.iterrows()}


def _beats_all_baselines(row: pd.Series) -> bool:
    return bool(
        row.get("beats_buy_hold", row.get("beats_buy_hold_direction", False))
        and row.get("beats_momentum_30d", False)
        and row.get("beats_momentum_90d", False)
        and row.get("beats_random_baseline", False)
    )


def _pruning_report_label(row: pd.Series, improvement_accuracy: float) -> str:
    if bool(row.get("promotion_eligible", False)):
        return "promoted"
    reason = str(row.get("rejection_reason", ""))
    accuracy = row.get("directional_accuracy", np.nan)
    net_return = row.get("net_return", np.nan)
    promising = bool(
        pd.notna(accuracy)
        and (
            float(accuracy) >= 0.55
            or (pd.notna(improvement_accuracy) and float(improvement_accuracy) > 0)
            or (pd.notna(net_return) and float(net_return) > 0)
        )
    )
    if "failed_bootstrap_or_permutation_stability_check" in reason and promising:
        return "promising_but_unstable"
    if promising:
        return "promising_but_rejected"
    return "bad_or_noisy"


def feature_pruning_report(
    run_id: str,
    pruned: pd.DataFrame,
    signal: pd.DataFrame,
    base_leaderboard: pd.DataFrame,
    base_confidence: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    from research_pipeline import BASELINE_MODELS

    asset_id = _asset_id_from_run(run_id)
    rows = []
    official_30 = base_leaderboard[
        (base_leaderboard["horizon"] == 30)
        & (base_leaderboard["window_type"] == "official_monthly")
    ].copy()
    official_ml = official_30[~official_30["model"].isin(BASELINE_MODELS)].copy()
    confidence_by_model = _confidence_lookup(base_confidence)
    base_by_model = {str(row["model"]): row.to_dict() for _, row in official_ml.iterrows()}
    best_base = (
        official_ml.sort_values(
            ["directional_accuracy", "brier_score", "sharpe", "max_drawdown", "calibration_error"],
            ascending=[False, True, False, False, True],
            na_position="last",
        )
        .head(1)
        .to_dict("records")
    )
    fallback_base = best_base[0] if best_base else {}

    for _, row in official_ml.iterrows():
        conf = confidence_by_model.get(str(row["model"]), {})
        rows.append(
            {
                "run_id": run_id,
                "asset_id": asset_id,
                "horizon": 30,
                "window_type": "official_monthly",
                "candidate_feature_set": "all_features_official",
                "model_name": row["model"],
                "n_samples": int(row.get("sample_count", 0)),
                "official_30d_accuracy": row.get("directional_accuracy", np.nan),
                "brier_score": row.get("brier_score", np.nan),
                "calibration_error": row.get("calibration_error", np.nan),
                "net_return": row.get("tc_adjusted_return", np.nan),
                "max_drawdown": row.get("max_drawdown", np.nan),
                "improvement_vs_all_features_accuracy": 0.0,
                "improvement_vs_all_features_net_return": 0.0,
                "beats_baselines": bool(
                    row.get("beats_buy_hold_direction", False)
                    and row.get("beats_momentum_30d", False)
                    and row.get("beats_momentum_90d", False)
                    and row.get("beats_random_baseline", False)
                ),
                "bootstrap_ci_low": conf.get("ci_low", np.nan),
                "bootstrap_ci_high": conf.get("ci_high", np.nan),
                "permutation_p_value": conf.get("permutation_p_value", np.nan),
                "promotion_decision": "current_official_reference",
                "rejection_reason": row.get("notes", ""),
                "reliability_label": row.get("reliability_label", "Low confidence"),
                "report_label": "current_all_features",
            }
        )

    if pruned.empty:
        return pd.DataFrame(rows)

    pruned_30 = pruned[
        (pruned["horizon"] == 30)
        & (pruned["window_type"] == "official_monthly")
        & (~pruned["model_name"].isin(BASELINE_MODELS))
    ].copy()
    if pruned_30.empty:
        return pd.DataFrame(rows)

    signal_cols = [
        "run_id",
        "asset_id",
        "horizon",
        "window_type",
        "candidate_feature_set",
        "model_name",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "permutation_p_value",
    ]
    signal_small = signal[signal_cols].copy() if not signal.empty else pd.DataFrame(columns=signal_cols)
    merged = pruned_30.merge(
        signal_small,
        on=["run_id", "asset_id", "horizon", "window_type", "candidate_feature_set", "model_name"],
        how="left",
    )
    for _, row in merged.iterrows():
        base = base_by_model.get(str(row["model_name"]), fallback_base)
        base_acc = base.get("directional_accuracy", np.nan)
        base_return = base.get("tc_adjusted_return", np.nan)
        improvement_accuracy = (
            float(row["directional_accuracy"]) - float(base_acc)
            if pd.notna(row.get("directional_accuracy")) and pd.notna(base_acc)
            else np.nan
        )
        improvement_return = (
            float(row["net_return"]) - float(base_return)
            if pd.notna(row.get("net_return")) and pd.notna(base_return)
            else np.nan
        )
        promoted = bool(row.get("promotion_eligible", False))
        label = _pruning_report_label(row, improvement_accuracy)
        rows.append(
            {
                "run_id": run_id,
                "asset_id": asset_id,
                "horizon": 30,
                "window_type": "official_monthly",
                "candidate_feature_set": row["candidate_feature_set"],
                "model_name": row["model_name"],
                "n_samples": int(row.get("n_samples", 0)),
                "official_30d_accuracy": row.get("directional_accuracy", np.nan),
                "brier_score": row.get("brier_score", np.nan),
                "calibration_error": row.get("calibration_error", np.nan),
                "net_return": row.get("net_return", np.nan),
                "max_drawdown": row.get("max_drawdown", np.nan),
                "improvement_vs_all_features_accuracy": improvement_accuracy,
                "improvement_vs_all_features_net_return": improvement_return,
                "beats_baselines": _beats_all_baselines(row),
                "bootstrap_ci_low": row.get("bootstrap_ci_low", np.nan),
                "bootstrap_ci_high": row.get("bootstrap_ci_high", np.nan),
                "permutation_p_value": row.get("permutation_p_value", np.nan),
                "promotion_decision": "promote" if promoted else "reject",
                "rejection_reason": row.get("rejection_reason", ""),
                "reliability_label": row.get("reliability_label", "Low confidence"),
                "report_label": label,
            }
        )
    return pd.DataFrame(rows)


def _pruned_rejection_reason(summary: dict, promotion_eligible: bool, base_comparison_ok: bool, stability_ok: bool) -> str:
    reasons = []
    if promotion_eligible:
        return "promotion_eligible"
    if summary.get("model") in {"buy_hold_direction", "momentum_30d", "momentum_90d", "random_permutation"}:
        reasons.append("baseline_not_promotable")
    if not summary.get("selection_eligible", False):
        reasons.append("failed_official_selection_gates")
    if not base_comparison_ok:
        reasons.append("did_not_improve_current_all_features_without_material_metric_worsening")
    if not stability_ok:
        reasons.append("failed_bootstrap_or_permutation_stability_check")
    if summary.get("reliability_label") == "Low confidence":
        reasons.append("low_reliability")
    return "; ".join(reasons) if reasons else "not_promoted"


def pruned_feature_diagnostics(
    run_id: str,
    raw: pd.DataFrame,
    feature_result,
    config: ResearchConfig,
    base_leaderboard: pd.DataFrame,
    base_confidence: Optional[pd.DataFrame] = None,
    *,
    quick: bool,
) -> Dict[str, pd.DataFrame]:
    from research_pipeline import BASELINE_MODELS, build_target_frame, build_walk_forward_windows, compute_metrics, decorate_summary, predict_baseline_windows

    asset_id = _asset_id_from_run(run_id)
    candidate_sets = candidate_feature_sets(feature_result.feature_cols, asset_id)
    if quick:
        quick_names = {
            "all_features",
            "price_plus_macro",
            "price_plus_dollar_rates",
            "core_price_macro_risk",
            "btc_dollar_rates_cycle",
            "btc_core_dollar_derivatives_cycle",
            "sol_rates_risk_price",
            "sol_dollar_rates_only",
        }
        candidate_sets = {name: cols for name, cols in candidate_sets.items() if name in quick_names}
    full_model_candidates = {
        "core_price_macro_risk",
        "price_plus_dollar_rates",
        "btc_core_dollar_derivatives_cycle",
        "btc_dollar_rates_cycle",
        "sol_rates_risk_price",
        "sol_dollar_rates_only",
    }
    leaderboard_rows = []
    signal_rows = []
    base_30 = base_leaderboard[(base_leaderboard["horizon"] == 30) & (base_leaderboard["window_type"] == "official_monthly")]
    base_ml = base_30[~base_30["model"].isin(BASELINE_MODELS)].sort_values(
        ["directional_accuracy", "brier_score", "sharpe", "max_drawdown", "calibration_error"],
        ascending=[False, True, False, False, True],
        na_position="last",
    )
    base_best = base_ml.iloc[0].to_dict() if not base_ml.empty else {}

    for horizon, window_type in [(30, "official_monthly")]:
        for candidate_name, candidate_cols in candidate_sets.items():
            target_frame = build_target_frame(raw, feature_result.features, candidate_cols, horizon)
            windows = build_walk_forward_windows(target_frame, horizon, window_type, config, quick=quick)
            if quick and len(windows) > 12:
                windows = windows[-12:]
            if not windows:
                continue
            predictions_by_model: Dict[str, pd.DataFrame] = {}
            selected_counts_by_model: Dict[str, List[int]] = {}
            for baseline in config.baseline_models:
                preds = predict_baseline_windows(baseline, target_frame, windows, config)
                if not preds.empty:
                    predictions_by_model[baseline] = preds
                    selected_counts_by_model[baseline] = [0]
            if len(candidate_cols) >= 5:
                model_names = config.first_model_set if (not quick and candidate_name in full_model_candidates) else ["logistic_linear"]
                for model_name in model_names:
                    preds, selected_counts = _predict_model_windows_nested_pruned(
                        model_name,
                        target_frame,
                        candidate_cols,
                        windows,
                        config,
                        quick=quick,
                    )
                    if not preds.empty:
                        predictions_by_model[model_name] = preds
                        selected_counts_by_model[model_name] = selected_counts
            if not predictions_by_model:
                continue
            common_dates = None
            for preds in predictions_by_model.values():
                common_dates = preds.index if common_dates is None else common_dates.intersection(preds.index)
            if common_dates is None or len(common_dates) == 0:
                continue
            predictions_by_model = {model: preds.loc[common_dates].sort_index() for model, preds in predictions_by_model.items()}

            summaries: Dict[str, dict] = {}
            for model, preds in predictions_by_model.items():
                metrics, _ = compute_metrics(preds, horizon, window_type, config.transaction_cost_bps)
                metrics.update(
                    {
                        "run_id": run_id,
                        "horizon": horizon,
                        "window_type": window_type,
                        "is_official": True,
                        "model": model,
                        "threshold": float(preds["threshold"].mean()),
                        "threshold_source": "baseline_rule" if model in BASELINE_MODELS else "nested_feature_selection_and_calibration",
                    }
                )
                summaries[model] = metrics
            baseline_summaries = {name: summaries[name] for name in BASELINE_MODELS if name in summaries}
            fingerprint = _sample_fingerprint(common_dates)
            candidate_model_names = config.first_model_set if (not quick and candidate_name in full_model_candidates) else ["logistic_linear"]
            for model_name in config.baseline_models + candidate_model_names:
                summary = summaries.get(model_name)
                if summary is None:
                    continue
                decorated = decorate_summary(summary, baseline_summaries, horizon, True, model_name)
                base_ok = True
                if model_name not in BASELINE_MODELS and horizon == 30 and base_best:
                    base_ok = bool(
                        decorated.get("directional_accuracy", -np.inf) > base_best.get("directional_accuracy", np.inf)
                        and decorated.get("brier_score", np.inf) <= base_best.get("brier_score", np.inf) + 0.02
                        and decorated.get("calibration_error", np.inf) <= base_best.get("calibration_error", np.inf) + 0.03
                        and decorated.get("sharpe", -np.inf) >= base_best.get("sharpe", -np.inf) - 0.25
                        and decorated.get("max_drawdown", -np.inf) >= base_best.get("max_drawdown", -np.inf) - 0.10
                    )
                ci_low, _, p_value = _bootstrap_signal_quality(predictions_by_model[model_name], config, quick=quick)
                stability_ok = bool(
                    model_name not in BASELINE_MODELS
                    and pd.notna(ci_low)
                    and ci_low > 0.50
                    and pd.notna(p_value)
                    and p_value <= 0.10
                )
                promotion_eligible = bool(
                    horizon == 30
                    and candidate_name != "all_features"
                    and model_name not in BASELINE_MODELS
                    and decorated.get("selection_eligible", False)
                    and base_ok
                    and stability_ok
                    and decorated.get("sample_count", 0) >= MIN_SAMPLE_THRESHOLDS["30d_official"]
                )
                selected_counts = selected_counts_by_model.get(model_name, [0])
                leaderboard_rows.append(
                    {
                        "run_id": run_id,
                        "asset_id": asset_id,
                        "horizon": horizon,
                        "window_type": window_type,
                        "candidate_feature_set": candidate_name,
                        "model_name": model_name,
                        "n_features": len(candidate_cols) if model_name not in BASELINE_MODELS else 0,
                        "median_selected_features": float(np.nanmedian(selected_counts)) if selected_counts else 0,
                        "n_samples": int(decorated.get("sample_count", 0)),
                        "directional_accuracy": decorated.get("directional_accuracy", np.nan),
                        "balanced_accuracy": decorated.get("balanced_accuracy", np.nan),
                        "brier_score": decorated.get("brier_score", np.nan),
                        "calibration_error": decorated.get("calibration_error", np.nan),
                        "sharpe": decorated.get("sharpe", np.nan),
                        "max_drawdown": decorated.get("max_drawdown", np.nan),
                        "net_return": decorated.get("tc_adjusted_return", np.nan),
                        "beats_buy_hold": bool(decorated.get("beats_buy_hold_direction", False)),
                        "beats_momentum_30d": bool(decorated.get("beats_momentum_30d", False)),
                        "beats_momentum_90d": bool(decorated.get("beats_momentum_90d", False)),
                        "beats_random_baseline": bool(decorated.get("beats_random_baseline", False)),
                        "selection_eligible": bool(decorated.get("selection_eligible", False)),
                        "promotion_eligible": promotion_eligible,
                        "reliability_label": decorated.get("reliability_label", "Low confidence"),
                        "window_fingerprint": fingerprint,
                        "rejection_reason": _pruned_rejection_reason(decorated, promotion_eligible, base_ok, stability_ok),
                    }
                )
                signal_rows.append(
                    signal_quality_row(
                        run_id,
                        asset_id,
                        horizon,
                        window_type,
                        candidate_name,
                        model_name,
                        predictions_by_model[model_name],
                        decorated,
                        config,
                        quick=quick,
                    )
                )
    factor_frame = factor_quality_scorecard(run_id, raw, feature_result, config)
    signal_frame = pd.DataFrame(signal_rows)
    pruned_frame = pd.DataFrame(leaderboard_rows)
    diagnostics = {
        "factor_quality_scorecard.csv": factor_frame,
        "signal_quality_report.csv": signal_frame,
        "pruned_feature_leaderboard.csv": pruned_frame,
        "feature_pruning_report.csv": feature_pruning_report(
            run_id,
            pruned_frame,
            signal_frame,
            base_leaderboard,
            base_confidence,
        ),
        "sol_stability_report.csv": sol_stability_report(run_id, raw, feature_result, config, quick=quick),
    }
    return diagnostics


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


def _btc_trend_masks(raw: pd.DataFrame, dates: pd.DatetimeIndex) -> Dict[str, pd.Series]:
    btc_col = "btc_proxy_close" if "btc_proxy_close" in raw.columns else "btc_close" if "btc_close" in raw.columns else ""
    if not btc_col:
        empty = pd.Series(False, index=dates)
        return {"BTC-up": empty, "BTC-down": empty.copy()}
    btc = raw[btc_col].astype(float)
    btc_ret_90 = np.log(btc / btc.shift(90))
    aligned = btc_ret_90.reindex(dates)
    return {
        "BTC-up": (aligned > 0).fillna(False),
        "BTC-down": (aligned <= 0).fillna(False),
    }


def _period_slice_masks(raw: pd.DataFrame, dates: pd.DatetimeIndex) -> Dict[str, pd.Series]:
    aligned = pd.DataFrame(index=dates)
    aligned["2015-2017"] = (dates >= pd.Timestamp("2015-01-01")) & (dates <= pd.Timestamp("2017-12-31"))
    aligned["2018-2020"] = (dates >= pd.Timestamp("2018-01-01")) & (dates <= pd.Timestamp("2020-12-31"))
    aligned["2021-2023"] = (dates >= pd.Timestamp("2021-01-01")) & (dates <= pd.Timestamp("2023-12-31"))
    aligned["2023-2024"] = (dates >= pd.Timestamp("2023-01-01")) & (dates <= pd.Timestamp("2024-12-31"))
    aligned["2024-present"] = dates >= pd.Timestamp("2024-01-01")
    aligned["pre-ETF"] = dates < pd.Timestamp("2024-01-11")
    aligned["post-ETF"] = dates >= pd.Timestamp("2024-01-11")
    regimes = _regime_masks(raw, dates)
    aligned["high-rate"] = regimes["high_rate"].to_numpy()
    aligned["low-rate"] = regimes["low_rate"].to_numpy()
    aligned["high-vol"] = regimes["high_volatility"].to_numpy()
    aligned["low-vol"] = regimes["low_volatility"].to_numpy()
    btc_masks = _btc_trend_masks(raw, dates)
    aligned["BTC-up"] = btc_masks["BTC-up"].to_numpy()
    aligned["BTC-down"] = btc_masks["BTC-down"].to_numpy()
    return {col: aligned[col].fillna(False) for col in aligned.columns}


def sol_stability_candidate_pairs(asset_id: str) -> List[Tuple[str, str]]:
    if asset_id != "sol":
        return []
    return [
        ("all_features", "random_forest"),
        ("price_plus_risk_assets", "logistic_linear"),
        ("price_plus_macro", "logistic_linear"),
        ("sol_dollar_rates_only", "random_forest"),
    ]


def _predict_candidate_model(
    raw: pd.DataFrame,
    feature_result,
    candidate_cols: Sequence[str],
    model_name: str,
    config: ResearchConfig,
    *,
    quick: bool,
) -> pd.DataFrame:
    from research_pipeline import build_target_frame, build_walk_forward_windows, predict_model_windows

    if len(candidate_cols) < 5:
        return pd.DataFrame()
    horizon = 30
    window_type = "official_monthly"
    target_frame = build_target_frame(raw, feature_result.features, candidate_cols, horizon)
    windows = build_walk_forward_windows(target_frame, horizon, window_type, config, quick=quick)
    if quick and len(windows) > 12:
        windows = windows[-12:]
    if not windows:
        return pd.DataFrame()
    return predict_model_windows(model_name, target_frame, candidate_cols, windows, config, quick=quick)


def _predict_direct_model_windows_with_policy(
    model_name: str,
    frame: pd.DataFrame,
    feature_cols: Sequence[str],
    windows: Sequence[object],
    config: ResearchConfig,
    *,
    quick: bool,
) -> pd.DataFrame:
    from research_pipeline import _calibrate_probabilities, _model_pipeline, _split_fit_calibration, tune_threshold_nested

    rows = []
    valid = frame.dropna(subset=["target_log_return", "target_up", "asset_close"]).copy()
    for idx, window in enumerate(windows):
        train = valid.loc[valid.index < window.test_date].copy()
        test = valid.loc[[window.test_date]].copy()
        if train.empty or test.empty:
            continue
        fit_train, cal = _split_fit_calibration(train, config)
        usable_cols = [c for c in feature_cols if c in fit_train and fit_train[c].notna().any()]
        if len(usable_cols) < 5 or fit_train["target_up"].nunique() < 2 or cal["target_up"].nunique() < 2:
            continue

        model = _model_pipeline(model_name, config.random_seed + idx, quick)
        model.fit(fit_train[usable_cols], fit_train["target_up"].astype(int))
        raw_cal_prob = model.predict_proba(cal[usable_cols])[:, 1]
        calibrated_cal_prob, _ = _calibrate_probabilities(raw_cal_prob, raw_cal_prob, cal["target_up"].astype(int).to_numpy())
        fit_up = fit_train.loc[fit_train["target_up"] == 1, "target_log_return"]
        fit_down = fit_train.loc[fit_train["target_up"] == 0, "target_log_return"]
        fit_mean_up = float(fit_up.mean()) if len(fit_up) else float(fit_train["target_log_return"].mean())
        fit_mean_down = float(fit_down.mean()) if len(fit_down) else float(fit_train["target_log_return"].mean())
        cal_expected = calibrated_cal_prob * fit_mean_up + (1 - calibrated_cal_prob) * fit_mean_down
        cal_eval = pd.DataFrame(
            {
                "probability_up": calibrated_cal_prob,
                "expected_log_return": cal_expected,
                "actual_return": np.exp(cal["target_log_return"].astype(float)) - 1,
                "actual_up": cal["target_up"].astype(int),
            },
            index=cal.index,
        )
        threshold = tune_threshold_nested(cal_eval, config.transaction_cost_bps)

        raw_test_prob = model.predict_proba(test[usable_cols])[:, 1]
        prob, prob_conf = _calibrate_probabilities(raw_test_prob, raw_cal_prob, cal["target_up"].astype(int).to_numpy())
        prob_up = float(prob[0])
        cal_up = cal.loc[cal["target_up"] == 1, "target_log_return"]
        cal_down = cal.loc[cal["target_up"] == 0, "target_log_return"]
        mean_up = float(cal_up.mean()) if len(cal_up) else float(cal["target_log_return"].mean())
        mean_down = float(cal_down.mean()) if len(cal_down) else float(cal["target_log_return"].mean())
        expected_log = prob_up * mean_up + (1 - prob_up) * mean_down
        actual_log = float(test["target_log_return"].iloc[0])
        row = {
            "date": window.test_date,
            "model": model_name,
            "horizon": window.horizon,
            "window_type": window.window_type,
            "is_official": window.is_official,
            "probability_up": prob_up,
            "probability_confidence": prob_conf,
            "expected_log_return": expected_log,
            "actual_log_return": actual_log,
            "actual_return": math.exp(actual_log) - 1,
            "actual_up": int(actual_log > 0),
            "threshold": threshold,
            "threshold_source": "nested_calibration",
            "predicted_up": int(prob_up >= threshold),
            "position": float(prob_up >= threshold),
            "train_rows": int(len(fit_train)),
            "calibration_rows": int(len(cal)),
            "feature_count": int(len(usable_cols)),
        }
        policy_input = pd.DataFrame([row]).set_index("date")
        policy_row = _add_signal_policy_columns(policy_input, cal_eval, config).iloc[0].to_dict()
        row.update({key: policy_row[key] for key in policy_row if key not in row or key.startswith(("policy_", "risk_", "high_"))})
        rows.append(row)
    return pd.DataFrame(rows).set_index("date").sort_index() if rows else pd.DataFrame()


def asset_specific_candidate_names(asset_id: str) -> List[str]:
    if asset_id == "btc":
        return [
            "all_features",
            "btc_dollar_rates_cycle",
            "core_price_macro_risk",
            "price_plus_dollar_rates",
            "price_plus_stablecoins",
            "btc_derivatives_pack",
            "btc_dollar_rates_cycle_plus_derivatives_pack",
        ]
    if asset_id == "sol":
        return [
            "all_features",
            "sol_rates_risk_price",
            "price_plus_risk_assets",
            "price_plus_macro",
            "sol_dollar_rates_only",
            "price_momentum_only",
        ]
    return ["all_features", "core_price_macro_risk", "price_momentum_only"]


def asset_specific_candidate_models(asset_id: str, candidate_name: str, base_reference_model: str) -> List[str]:
    if candidate_name == "all_features":
        return [base_reference_model] if base_reference_model else ["logistic_linear"]
    if asset_id == "btc":
        return ["logistic_linear"]
    if asset_id == "sol":
        if candidate_name in {"sol_dollar_rates_only", "price_momentum_only"}:
            return ["random_forest"]
        return ["logistic_linear"]
    return ["logistic_linear"]


def _material_worsening(candidate: dict, reference: dict) -> bool:
    if not reference:
        return False
    brier_bad = (
        pd.notna(candidate.get("brier_score"))
        and pd.notna(reference.get("brier_score"))
        and float(candidate["brier_score"]) > float(reference["brier_score"]) + MATERIAL_BRIER_WORSENING
    )
    calibration_bad = (
        pd.notna(candidate.get("calibration_error"))
        and pd.notna(reference.get("calibration_error"))
        and float(candidate["calibration_error"]) > float(reference["calibration_error"]) + MATERIAL_CALIBRATION_WORSENING
    )
    drawdown_bad = (
        pd.notna(candidate.get("max_drawdown"))
        and pd.notna(reference.get("max_drawdown"))
        and float(candidate["max_drawdown"]) < float(reference["max_drawdown"]) - MATERIAL_DRAWDOWN_WORSENING
    )
    return bool(brier_bad or calibration_bad or drawdown_bad)


def _regime_stability_pass(raw: pd.DataFrame, preds: pd.DataFrame) -> bool:
    if preds.empty:
        return False
    passes = 0
    failures = 0
    masks = _period_slice_masks(raw, pd.DatetimeIndex(preds.index))
    for mask in masks.values():
        part = preds.loc[mask.to_numpy()]
        if len(part) < 8:
            continue
        accuracy = float((part["predicted_up"].astype(int) == part["actual_up"].astype(int)).mean())
        net_return = float((1 + _policy_returns_for_predictions(part, 0.0)["position"].astype(float) * part["actual_return"].astype(float)).prod() - 1)
        if accuracy >= 0.55 and net_return > 0:
            passes += 1
        if accuracy < 0.50:
            failures += 1
    return bool(passes >= 2 and failures == 0)


def _asset_feature_rejection_reason(
    *,
    promotion_eligible: bool,
    selection_eligible: bool,
    beats_current_reference: bool,
    material_worsening: bool,
    regime_stability_pass: bool,
    ci_low: float,
    p_value: float,
    n_samples: int,
) -> str:
    if promotion_eligible:
        return "promotion_eligible"
    reasons = []
    if n_samples < MIN_SAMPLE_THRESHOLDS["30d_official"]:
        reasons.append("below_sample_threshold")
    if not selection_eligible:
        reasons.append("failed_official_model_selection_gates")
    if not beats_current_reference:
        reasons.append("did_not_improve_current_all_features_reference")
    if material_worsening:
        reasons.append("materially_worsened_brier_calibration_or_drawdown")
    if not (pd.notna(ci_low) and float(ci_low) > 0.50 and pd.notna(p_value) and float(p_value) <= 0.10):
        reasons.append("failed_bootstrap_or_permutation_stability_check")
    if not regime_stability_pass:
        reasons.append("failed_regime_stability_check")
    return "; ".join(reasons) if reasons else "not_promoted"


def _bootstrap_active_hit_rate(preds: pd.DataFrame, config: ResearchConfig, *, quick: bool) -> Tuple[float, float, float]:
    if preds.empty or "policy_position" not in preds:
        return np.nan, np.nan, np.nan
    active_mask = preds["policy_position"].astype(float).to_numpy() > 0
    if active_mask.sum() == 0:
        return np.nan, np.nan, np.nan
    actual = preds["actual_up"].astype(int).to_numpy()
    observed = float(actual[active_mask].mean())
    iterations = config.quick_bootstrap_iterations if quick else min(config.bootstrap_iterations, 250)
    perm_iterations = config.quick_permutation_iterations if quick else min(config.permutation_iterations, 250)
    rng = np.random.default_rng(config.random_seed + len(preds) + int(active_mask.sum()))
    active_actual = actual[active_mask]
    boot = []
    for _ in range(iterations):
        idx = rng.integers(0, len(active_actual), len(active_actual))
        boot.append(float(active_actual[idx].mean()))
    perm_hits = 0
    for _ in range(perm_iterations):
        if float(rng.permutation(actual)[active_mask].mean()) >= observed:
            perm_hits += 1
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), float((perm_hits + 1) / (perm_iterations + 1))


def signal_policy_report_row(
    run_id: str,
    asset_id: str,
    horizon: int,
    window_type: str,
    candidate_name: str,
    model_name: str,
    preds: pd.DataFrame,
    config: ResearchConfig,
    *,
    quick: bool,
) -> dict:
    if preds.empty:
        return {
            "run_id": run_id,
            "asset_id": asset_id,
            "horizon": horizon,
            "window_type": window_type,
            "candidate_feature_set": candidate_name,
            "model_name": model_name,
            "n_samples": 0,
            "policy_source": "train_calibration_only",
            "long_threshold_median": np.nan,
            "expected_return_min_median": np.nan,
            "threshold_050_used": False,
            "threshold_050_diagnostic_only": True,
            "active_signal_count": 0,
            "active_coverage": 0.0,
            "abstention_rate": 1.0,
            "active_hit_rate": np.nan,
            "missed_up_month_rate": np.nan,
            "after_cost_return": np.nan,
            "max_drawdown": np.nan,
            "bootstrap_ci_low": np.nan,
            "bootstrap_ci_high": np.nan,
            "permutation_p_value": np.nan,
            "risk_off_count": 0,
            "risk_off_rate": 0.0,
            "risk_off_hit_rate": np.nan,
            "risk_off_avg_return": np.nan,
            "risk_off_probability_threshold_median": np.nan,
            "valid_signal_policy": False,
            "high_confidence_policy_eligible": False,
            "promoted_signal_policy": False,
            "rejection_reason": "empty_predictions",
        }
    frame = _policy_returns_for_predictions(preds, config.transaction_cost_bps)
    active = frame[frame["policy_position"].astype(float) > 0]
    actual_up = frame["actual_up"].astype(int)
    risk_mask = frame["risk_off_flag"].astype(bool) if "risk_off_flag" in frame else pd.Series(False, index=frame.index)
    risk_off = frame[risk_mask]
    ci_low, ci_high, p_value = _bootstrap_active_hit_rate(frame, config, quick=quick)
    long_threshold = float(frame["policy_long_threshold"].median()) if "policy_long_threshold" in frame else np.nan
    expected_min = float(frame["policy_expected_return_min"].median()) if "policy_expected_return_min" in frame else np.nan
    active_count = int(len(active))
    active_coverage = float(active_count / len(frame)) if len(frame) else 0.0
    after_cost_return = float(frame["policy_equity"].iloc[-1] - 1) if len(frame) else np.nan
    risk_off_count = int(len(risk_off))
    missed_up = frame[(frame["actual_up"].astype(int) == 1) & (frame["policy_position"].astype(float) <= 0)]
    valid = bool(
        active_count >= active_signal_floor(len(frame))
        and active_coverage >= VALID_ACTIVE_MIN_RATIO
        and pd.notna(after_cost_return)
        and after_cost_return > 0
        and pd.notna(long_threshold)
        and is_official_long_policy_allowed(long_threshold, expected_min if pd.notna(expected_min) else 0.0)
    )
    high_eligible = bool(valid and active_count >= high_confidence_signal_floor(len(frame)))
    active_hit = float(active["actual_up"].astype(int).mean()) if not active.empty else np.nan
    promoted = bool(valid and pd.notna(active_hit) and active_hit > 0.50 and pd.notna(ci_low) and ci_low > 0.50 and pd.notna(p_value) and p_value <= 0.10)
    reasons = []
    if pd.notna(long_threshold) and not is_official_long_policy_allowed(long_threshold, expected_min if pd.notna(expected_min) else 0.0):
        reasons.append("threshold_0_50_is_diagnostic_only")
    if active_count < active_signal_floor(len(frame)):
        reasons.append("below_active_signal_floor")
    if active_coverage < VALID_ACTIVE_MIN_RATIO:
        reasons.append("below_active_coverage_floor")
    if not (pd.notna(after_cost_return) and after_cost_return > 0):
        reasons.append("no_positive_after_cost_return")
    if not (pd.notna(active_hit) and active_hit > 0.50):
        reasons.append("weak_active_hit_rate")
    if not (pd.notna(ci_low) and ci_low > 0.50 and pd.notna(p_value) and p_value <= 0.10):
        reasons.append("failed_bootstrap_or_permutation_check")
    return {
        "run_id": run_id,
        "asset_id": asset_id,
        "horizon": horizon,
        "window_type": window_type,
        "candidate_feature_set": candidate_name,
        "model_name": model_name,
        "n_samples": int(len(frame)),
        "policy_source": "train_calibration_only",
        "long_threshold_median": long_threshold,
        "expected_return_min_median": expected_min,
        "threshold_050_used": bool(pd.notna(long_threshold) and long_threshold <= 0.50),
        "threshold_050_diagnostic_only": True,
        "active_signal_count": active_count,
        "active_coverage": active_coverage,
        "abstention_rate": float(1 - active_coverage),
        "active_hit_rate": active_hit,
        "missed_up_month_rate": float(len(missed_up) / int(actual_up.sum())) if int(actual_up.sum()) else np.nan,
        "after_cost_return": after_cost_return,
        "max_drawdown": float((frame["policy_equity"] / frame["policy_equity"].cummax() - 1).min()) if len(frame) else np.nan,
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "permutation_p_value": p_value,
        "risk_off_count": risk_off_count,
        "risk_off_rate": float(risk_off_count / len(frame)) if len(frame) else 0.0,
        "risk_off_hit_rate": float((risk_off["actual_return"].astype(float) < 0).mean()) if not risk_off.empty else np.nan,
        "risk_off_avg_return": float(risk_off["actual_return"].astype(float).mean()) if not risk_off.empty else np.nan,
        "risk_off_probability_threshold_median": float(frame["risk_off_probability_threshold"].dropna().median()) if "risk_off_probability_threshold" in frame and frame["risk_off_probability_threshold"].notna().any() else np.nan,
        "valid_signal_policy": valid,
        "high_confidence_policy_eligible": high_eligible,
        "promoted_signal_policy": promoted,
        "rejection_reason": "promotion_eligible" if promoted else "; ".join(dict.fromkeys(reasons)),
    }


def _candidate_predictions_and_summaries(
    run_id: str,
    raw: pd.DataFrame,
    feature_result,
    candidate_name: str,
    candidate_cols: Sequence[str],
    config: ResearchConfig,
    *,
    quick: bool,
    model_names_override: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, dict], str, Dict[str, List[int]]]:
    from research_pipeline import BASELINE_MODELS, build_target_frame, build_walk_forward_windows, compute_metrics, decorate_summary, predict_baseline_windows

    if len(candidate_cols) < 5:
        return {}, {}, "", {}
    horizon = 30
    window_type = "official_monthly"
    target_frame = build_target_frame(raw, feature_result.features, candidate_cols, horizon)
    windows = build_walk_forward_windows(target_frame, horizon, window_type, config, quick=quick)
    if quick and len(windows) > 12:
        windows = windows[-12:]
    if not windows:
        return {}, {}, "", {}
    predictions_by_model: Dict[str, pd.DataFrame] = {}
    selected_counts_by_model: Dict[str, List[int]] = {}
    for baseline in config.baseline_models:
        preds = predict_baseline_windows(baseline, target_frame, windows, config)
        if not preds.empty:
            predictions_by_model[baseline] = preds
            selected_counts_by_model[baseline] = [0]
    model_names = list(model_names_override or (config.first_model_set if (not quick or candidate_name == "all_features") else ["logistic_linear"]))
    for model_name in model_names:
        if candidate_name == "all_features":
            preds = _predict_direct_model_windows_with_policy(model_name, target_frame, candidate_cols, windows, config, quick=quick)
            selected_counts = [len(candidate_cols)]
        else:
            preds, selected_counts = _predict_model_windows_nested_pruned(model_name, target_frame, candidate_cols, windows, config, quick=quick)
        if not preds.empty:
            predictions_by_model[model_name] = preds
            selected_counts_by_model[model_name] = selected_counts
    if not predictions_by_model:
        return {}, {}, "", {}
    common_dates = None
    for preds in predictions_by_model.values():
        common_dates = preds.index if common_dates is None else common_dates.intersection(preds.index)
    if common_dates is None or len(common_dates) == 0:
        return {}, {}, "", {}
    predictions_by_model = {model: preds.loc[common_dates].sort_index() for model, preds in predictions_by_model.items()}
    summaries: Dict[str, dict] = {}
    for model, preds in predictions_by_model.items():
        metrics, _ = compute_metrics(preds, horizon, window_type, config.transaction_cost_bps)
        metrics.update(
            {
                "run_id": run_id,
                "horizon": horizon,
                "window_type": window_type,
                "is_official": True,
                "model": model,
                "threshold": float(preds["threshold"].mean()),
                "threshold_source": "baseline_rule" if model in BASELINE_MODELS else "nested_calibration" if candidate_name == "all_features" else "nested_feature_selection_and_calibration",
            }
        )
        summaries[model] = metrics
    baseline_summaries = {name: summaries[name] for name in BASELINE_MODELS if name in summaries}
    decorated = {model: decorate_summary(summary, baseline_summaries, horizon, True, model) for model, summary in summaries.items()}
    return predictions_by_model, decorated, _sample_fingerprint(common_dates), selected_counts_by_model


def asset_feature_set_and_policy_diagnostics(
    run_id: str,
    raw: pd.DataFrame,
    feature_result,
    config: ResearchConfig,
    base_leaderboard: pd.DataFrame,
    *,
    quick: bool,
) -> Dict[str, pd.DataFrame]:
    from research_pipeline import BASELINE_MODELS

    asset_id = _asset_id_from_run(run_id)
    candidate_sets = candidate_feature_sets(feature_result.feature_cols, asset_id)
    requested = asset_specific_candidate_names(asset_id)
    base_30 = base_leaderboard[(base_leaderboard["horizon"] == 30) & (base_leaderboard["window_type"] == "official_monthly")].copy()
    base_ml = base_30[~base_30["model"].isin(BASELINE_MODELS)].copy()
    best_base = (
        base_ml.sort_values(
            ["directional_accuracy", "brier_score", "sharpe", "max_drawdown", "calibration_error"],
            ascending=[False, True, False, False, True],
            na_position="last",
        )
        .head(1)
        .to_dict("records")
    )
    default_reference = best_base[0] if best_base else {}
    base_reference_model = str(default_reference.get("model", "logistic_linear")) if default_reference else "logistic_linear"
    asset_rows = []
    policy_rows = []
    for candidate_name in requested:
        candidate_cols = candidate_sets.get(candidate_name, [])
        model_names = asset_specific_candidate_models(asset_id, candidate_name, base_reference_model)
        if len(candidate_cols) < 5:
            for model_name in model_names:
                asset_rows.append(
                    {
                        "run_id": run_id,
                        "asset_id": asset_id,
                        "horizon": 30,
                        "window_type": "official_monthly",
                        "candidate_feature_set": candidate_name,
                        "model_name": model_name,
                        "feature_selection_method": "nested_train_only_pruned" if candidate_name != "all_features" else "all_features_reference",
                        "n_features": len(candidate_cols),
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
                        "beats_current_reference": False,
                        "material_worsening": False,
                        "regime_stability_pass": False,
                        "bootstrap_ci_low": np.nan,
                        "bootstrap_ci_high": np.nan,
                        "permutation_p_value": np.nan,
                        "promotion_eligible": False,
                        "reliability_label": "Low confidence",
                        "window_fingerprint": "",
                        "rejection_reason": "no_candidate_features_or_below_minimum_feature_count",
                    }
                )
                policy_rows.append(
                    signal_policy_report_row(run_id, asset_id, 30, "official_monthly", candidate_name, model_name, pd.DataFrame(), config, quick=quick)
                )
            continue
        if candidate_name == "all_features":
            for _, base_row in base_ml.iterrows():
                model_name = str(base_row["model"])
                ci_low, ci_high, p_value = np.nan, np.nan, np.nan
                asset_rows.append(
                    {
                        "run_id": run_id,
                        "asset_id": asset_id,
                        "horizon": 30,
                        "window_type": "official_monthly",
                        "candidate_feature_set": candidate_name,
                        "model_name": model_name,
                        "feature_selection_method": "all_features_reference",
                        "n_features": len(candidate_cols),
                        "n_samples": int(base_row.get("sample_count", 0)),
                        "directional_accuracy": base_row.get("directional_accuracy", np.nan),
                        "balanced_accuracy": base_row.get("balanced_accuracy", np.nan),
                        "brier_score": base_row.get("brier_score", np.nan),
                        "calibration_error": base_row.get("calibration_error", np.nan),
                        "sharpe": base_row.get("sharpe", np.nan),
                        "max_drawdown": base_row.get("max_drawdown", np.nan),
                        "net_return": base_row.get("tc_adjusted_return", np.nan),
                        "beats_buy_hold": bool(base_row.get("beats_buy_hold_direction", False)),
                        "beats_momentum_30d": bool(base_row.get("beats_momentum_30d", False)),
                        "beats_momentum_90d": bool(base_row.get("beats_momentum_90d", False)),
                        "beats_random_baseline": bool(base_row.get("beats_random_baseline", False)),
                        "beats_current_reference": False,
                        "material_worsening": False,
                        "regime_stability_pass": False,
                        "bootstrap_ci_low": ci_low,
                        "bootstrap_ci_high": ci_high,
                        "permutation_p_value": p_value,
                        "promotion_eligible": False,
                        "reliability_label": base_row.get("reliability_label", "Low confidence"),
                        "window_fingerprint": "",
                        "rejection_reason": "current_all_features_reference",
                    }
                )
        predictions, summaries, fingerprint, selected_counts = _candidate_predictions_and_summaries(
            run_id,
            raw,
            feature_result,
            candidate_name,
            candidate_cols,
            config,
            quick=quick,
            model_names_override=model_names,
        )
        for model_name in model_names:
            summary = summaries.get(model_name)
            preds = predictions.get(model_name, pd.DataFrame())
            policy_rows.append(
                signal_policy_report_row(run_id, asset_id, 30, "official_monthly", candidate_name, model_name, preds, config, quick=quick)
            )
            if candidate_name == "all_features":
                continue
            if summary is None:
                asset_rows.append(
                    {
                        "run_id": run_id,
                        "asset_id": asset_id,
                        "horizon": 30,
                        "window_type": "official_monthly",
                        "candidate_feature_set": candidate_name,
                        "model_name": model_name,
                        "feature_selection_method": "nested_train_only_pruned" if candidate_name != "all_features" else "all_features_reference",
                        "n_features": len(candidate_cols),
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
                        "beats_current_reference": False,
                        "material_worsening": False,
                        "regime_stability_pass": False,
                        "bootstrap_ci_low": np.nan,
                        "bootstrap_ci_high": np.nan,
                        "permutation_p_value": np.nan,
                        "promotion_eligible": False,
                        "reliability_label": "Low confidence",
                        "window_fingerprint": fingerprint,
                        "rejection_reason": "model_did_not_produce_predictions",
                    }
                )
                continue
            reference = default_reference
            beats_current = bool(
                reference
                and pd.notna(summary.get("directional_accuracy"))
                and float(summary.get("directional_accuracy")) > float(reference.get("directional_accuracy", np.inf))
                and float(summary.get("tc_adjusted_return", -np.inf)) >= float(reference.get("tc_adjusted_return", -np.inf))
            )
            material_bad = _material_worsening(summary, reference)
            ci_low, ci_high, p_value = _bootstrap_signal_quality(preds, config, quick=quick)
            regime_ok = _regime_stability_pass(raw, preds)
            promotion_eligible = bool(
                candidate_name != "all_features"
                and summary.get("selection_eligible", False)
                and summary.get("sample_count", 0) >= MIN_SAMPLE_THRESHOLDS["30d_official"]
                and beats_current
                and not material_bad
                and pd.notna(ci_low)
                and ci_low > 0.50
                and pd.notna(p_value)
                and p_value <= 0.10
                and regime_ok
            )
            asset_rows.append(
                {
                    "run_id": run_id,
                    "asset_id": asset_id,
                    "horizon": 30,
                    "window_type": "official_monthly",
                    "candidate_feature_set": candidate_name,
                    "model_name": model_name,
                    "feature_selection_method": "nested_train_only_pruned" if candidate_name != "all_features" else "all_features_reference",
                    "n_features": int(np.nanmedian(selected_counts.get(model_name, [len(candidate_cols)]))) if model_name in selected_counts else len(candidate_cols),
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
                    "beats_current_reference": beats_current,
                    "material_worsening": material_bad,
                    "regime_stability_pass": regime_ok,
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                    "permutation_p_value": p_value,
                    "promotion_eligible": promotion_eligible,
                    "reliability_label": summary.get("reliability_label", "Low confidence"),
                    "window_fingerprint": fingerprint,
                    "rejection_reason": _asset_feature_rejection_reason(
                        promotion_eligible=promotion_eligible,
                        selection_eligible=bool(summary.get("selection_eligible", False)),
                        beats_current_reference=beats_current,
                        material_worsening=material_bad,
                        regime_stability_pass=regime_ok,
                        ci_low=ci_low,
                        p_value=p_value,
                        n_samples=int(summary.get("sample_count", 0)),
                    ),
                }
            )
    asset_frame = pd.DataFrame(asset_rows)
    if asset_id == "btc" and not asset_frame.empty:
        non_derivative = asset_frame[
            (~asset_frame["candidate_feature_set"].astype(str).str.contains("derivatives_pack", na=False))
            & (asset_frame["candidate_feature_set"] != "all_features")
        ]
        derivative_pack = asset_frame[asset_frame["candidate_feature_set"] == "btc_derivatives_pack"]
        best_non_derivative = float(non_derivative["directional_accuracy"].max()) if not non_derivative.empty else np.nan
        best_derivative_pack = float(derivative_pack["directional_accuracy"].max()) if not derivative_pack.empty else np.nan
        derivative_pack_improved = bool(
            pd.notna(best_derivative_pack)
            and pd.notna(best_non_derivative)
            and best_derivative_pack > best_non_derivative
        )
        if not derivative_pack_improved:
            mask = asset_frame["candidate_feature_set"] == "btc_dollar_rates_cycle_plus_derivatives_pack"
            promoted_mask = mask & (asset_frame["promotion_eligible"] == True)
            asset_frame.loc[promoted_mask, "promotion_eligible"] = False
            asset_frame.loc[mask, "rejection_reason"] = asset_frame.loc[mask, "rejection_reason"].apply(
                lambda reason: (
                    f"{reason}; derivative_pack_did_not_improve_beyond_best_non_derivatives_candidate"
                    if reason and reason != "promotion_eligible"
                    else "derivative_pack_did_not_improve_beyond_best_non_derivatives_candidate"
                )
            )
    return {
        "asset_feature_set_leaderboard.csv": asset_frame,
        "signal_policy_report.csv": pd.DataFrame(policy_rows),
    }


def latest_signal_interpretation_frame(
    run_id: str,
    generated_at: str,
    latest: pd.DataFrame,
    signal_policy: pd.DataFrame,
    asset_feature_sets: pd.DataFrame,
) -> pd.DataFrame:
    if latest.empty:
        return empty_output_frame("latest_signal_interpretation.csv")
    asset_id = _asset_id_from_run(run_id)
    primary = latest[latest["is_primary_objective"] == True].head(1)
    row = primary.iloc[0] if not primary.empty else latest.iloc[0]
    selected_model = str(row.get("selected_model", "no_valid_edge"))
    promoted_features = asset_feature_sets[asset_feature_sets["promotion_eligible"] == True] if not asset_feature_sets.empty else pd.DataFrame()
    selected_feature_set = "all_features"
    asset_feature_promoted = False
    if not promoted_features.empty and selected_model != "no_valid_edge":
        promoted_features = promoted_features.sort_values(
            ["directional_accuracy", "brier_score", "sharpe", "max_drawdown", "calibration_error"],
            ascending=[False, True, False, False, True],
            na_position="last",
        )
        selected_feature_set = str(promoted_features.iloc[0]["candidate_feature_set"])
        asset_feature_promoted = True
    matching_policy = pd.DataFrame()
    if not signal_policy.empty and selected_model != "no_valid_edge":
        matching_policy = signal_policy[
            (signal_policy["candidate_feature_set"] == selected_feature_set)
            & (signal_policy["model_name"] == selected_model)
        ].head(1)
        if matching_policy.empty:
            matching_policy = signal_policy[
                (signal_policy["candidate_feature_set"] == "all_features")
                & (signal_policy["model_name"] == selected_model)
            ].head(1)
    policy_row = matching_policy.iloc[0] if not matching_policy.empty else pd.Series(dtype=object)
    probability_up = float(row.get("predicted_probability_up", 0.5))
    expected_return = float(row.get("expected_return", 0.0))
    long_threshold = policy_row.get("long_threshold_median", np.nan)
    expected_min = policy_row.get("expected_return_min_median", np.nan)
    risk_threshold = policy_row.get("risk_off_probability_threshold_median", np.nan)
    risk_bucket_count = int(policy_row.get("risk_off_count", 0) or 0) if not policy_row.empty else 0
    risk_bucket_avg = policy_row.get("risk_off_avg_return", np.nan)
    signal_policy_promoted = bool(policy_row.get("promoted_signal_policy", False)) if not policy_row.empty else False

    if selected_model == "no_valid_edge":
        selected_feature_set = "none"
        strategy_action = "cash"
        risk_label = "Neutral / no edge"
        reason = "No valid 30d edge; signal policy cannot override no_valid_edge."
    else:
        long_allowed = bool(
            signal_policy_promoted
            and pd.notna(long_threshold)
            and is_official_long_policy_allowed(float(long_threshold), float(expected_min) if pd.notna(expected_min) else 0.0)
            and probability_up >= float(long_threshold)
            and (pd.isna(expected_min) or expected_return >= float(expected_min))
            and str(row.get("reliability_label")) != "Low confidence"
        )
        high_downside = bool(
            pd.notna(risk_threshold)
            and risk_bucket_count >= 8
            and probability_up <= 0.35
            and expected_return < -0.10
        )
        risk_off = bool(pd.notna(risk_threshold) and risk_bucket_count >= 8 and probability_up <= float(risk_threshold))
        if long_allowed:
            strategy_action = "long"
            risk_label = "Long signal"
            reason = "Train-only signal policy is promoted and current probability clears the official long threshold."
        elif high_downside:
            strategy_action = "cash"
            risk_label = "High downside risk"
            reason = "Current probability and expected return fall inside a supported train-derived downside bucket."
        elif risk_off:
            strategy_action = "cash"
            risk_label = "Risk-off / avoid long"
            reason = "Current probability falls inside a train-derived weak-probability bucket; this is not a short signal."
        else:
            strategy_action = "cash"
            risk_label = "Neutral / no edge"
            reason = "Current forecast does not clear the promoted long policy or supported downside-risk bucket."
    return pd.DataFrame(
        [
            {
                "run_id": run_id,
                "generated_at": generated_at,
                "as_of_date": row.get("as_of_date", ""),
                "asset_id": asset_id,
                "horizon": row.get("horizon", 30),
                "selected_model": selected_model,
                "selected_feature_set": selected_feature_set,
                "strategy_action": strategy_action,
                "risk_label": risk_label,
                "probability_up": probability_up,
                "expected_return": expected_return,
                "long_threshold": long_threshold,
                "expected_return_threshold": expected_min,
                "risk_off_threshold": risk_threshold,
                "active_signal_count": int(policy_row.get("active_signal_count", 0) or 0) if not policy_row.empty else 0,
                "active_coverage": policy_row.get("active_coverage", np.nan) if not policy_row.empty else np.nan,
                "risk_off_bucket_count": risk_bucket_count,
                "risk_off_bucket_avg_return": risk_bucket_avg,
                "signal_policy_promoted": signal_policy_promoted,
                "asset_feature_set_promoted": asset_feature_promoted,
                "reliability_label": row.get("reliability_label", "Low confidence"),
                "reason": reason,
            }
        ]
    )


def sol_stability_report(
    run_id: str,
    raw: pd.DataFrame,
    feature_result,
    config: ResearchConfig,
    *,
    quick: bool,
) -> pd.DataFrame:
    from research_pipeline import compute_metrics

    asset_id = _asset_id_from_run(run_id)
    pairs = sol_stability_candidate_pairs(asset_id)
    if not pairs:
        return empty_diagnostic_frame("sol_stability_report.csv")
    candidate_sets = candidate_feature_sets(feature_result.feature_cols, asset_id)
    requested_slices = ["2023-2024", "2024-present", "high-vol", "low-vol", "BTC-up", "BTC-down"]
    rows = []
    for candidate_name, model_name in pairs:
        candidate_cols = candidate_sets.get(candidate_name, [])
        preds = _predict_candidate_model(raw, feature_result, candidate_cols, model_name, config, quick=quick)
        masks = _period_slice_masks(raw, pd.DatetimeIndex(preds.index if not preds.empty else []))
        for period_slice in requested_slices:
            mask = masks.get(period_slice, pd.Series(False, index=preds.index if not preds.empty else []))
            part = preds.loc[mask.to_numpy()] if not preds.empty else pd.DataFrame()
            if part.empty:
                rows.append(
                    {
                        "run_id": run_id,
                        "asset_id": asset_id,
                        "candidate_feature_set": candidate_name,
                        "model_name": model_name,
                        "period_slice": period_slice,
                        "n_samples": 0,
                        "directional_accuracy": np.nan,
                        "brier_score": np.nan,
                        "calibration_error": np.nan,
                        "net_return": np.nan,
                        "sharpe": np.nan,
                        "max_drawdown": np.nan,
                        "bootstrap_ci_low": np.nan,
                        "bootstrap_ci_high": np.nan,
                        "permutation_p_value": np.nan,
                        "reliability_label": "Low confidence",
                        "passes_stability_check": False,
                    }
                )
                continue
            metrics, _ = compute_metrics(part, 30, "official_monthly", config.transaction_cost_bps)
            ci_low, ci_high, p_value = _bootstrap_signal_quality(part, config, quick=quick)
            passes = bool(
                metrics.get("sample_count", 0) >= 8
                and metrics.get("directional_accuracy", 0) >= 0.55
                and metrics.get("tc_adjusted_return", -np.inf) > 0
                and pd.notna(ci_low)
                and ci_low >= 0.50
                and pd.notna(p_value)
                and p_value <= 0.10
            )
            rows.append(
                {
                    "run_id": run_id,
                    "asset_id": asset_id,
                    "candidate_feature_set": candidate_name,
                    "model_name": model_name,
                    "period_slice": period_slice,
                    "n_samples": int(metrics.get("sample_count", 0)),
                    "directional_accuracy": metrics.get("directional_accuracy", np.nan),
                    "brier_score": metrics.get("brier_score", np.nan),
                    "calibration_error": metrics.get("calibration_error", np.nan),
                    "net_return": metrics.get("tc_adjusted_return", np.nan),
                    "sharpe": metrics.get("sharpe", np.nan),
                    "max_drawdown": metrics.get("max_drawdown", np.nan),
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                    "permutation_p_value": p_value,
                    "reliability_label": "Medium confidence" if passes else "Low confidence",
                    "passes_stability_check": passes,
                }
            )
    return pd.DataFrame(rows)


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


def write_feature_pruning_summary(output_dir: Path) -> Path:
    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    latest = pd.read_csv(output_dir / "latest_forecast.csv")
    leaderboard = pd.read_csv(output_dir / "model_leaderboard.csv")
    pruned = pd.read_csv(output_dir / "csv" / "pruned_feature_leaderboard.csv")
    signal = pd.read_csv(output_dir / "csv" / "signal_quality_report.csv")
    factor = pd.read_csv(output_dir / "csv" / "factor_quality_scorecard.csv")
    pruning_report_path = output_dir / "csv" / "feature_pruning_report.csv"
    pruning_report = pd.read_csv(pruning_report_path) if pruning_report_path.exists() else pd.DataFrame()
    asset_name = str(manifest.get("asset_name") or manifest.get("asset_id") or "asset")
    asset_id = str(manifest.get("asset_id") or "").lower()
    primary = latest[latest["is_primary_objective"] == True].iloc[0]
    official_30 = leaderboard[(leaderboard["horizon"] == 30) & (leaderboard["window_type"] == "official_monthly")].copy()
    baseline_models = {"buy_hold_direction", "momentum_30d", "momentum_90d", "random_permutation"}
    base_ml = official_30[~official_30["model"].isin(baseline_models)].sort_values("directional_accuracy", ascending=False).head(1)
    base_baseline = official_30[official_30["model"].isin(baseline_models)].sort_values("directional_accuracy", ascending=False).head(1)
    pruned_30 = pruned[(pruned["horizon"] == 30) & (pruned["window_type"] == "official_monthly")].copy()
    pruned_ml = pruned_30[~pruned_30["model_name"].isin(baseline_models)].sort_values(
        ["promotion_eligible", "directional_accuracy", "brier_score", "sharpe", "max_drawdown", "calibration_error"],
        ascending=[False, False, True, False, False, True],
        na_position="last",
    )
    best_pruned = pruned_ml.head(1)
    promoted = pruned_ml[pruned_ml["promotion_eligible"] == True]
    signal_30 = signal[(signal["horizon"] == 30) & (signal["window_type"] == "official_monthly")]
    best_signal = signal_30.sort_values(["long_hit_rate", "net_return"], ascending=[False, False], na_position="last").head(5)
    factor_30 = factor[(factor["horizon"] == 30) & (factor["window_type"] == "official_monthly")]
    top_factors = factor_30.sort_values("information_coefficient", key=lambda s: s.abs(), ascending=False, na_position="last").head(10)
    dropped = factor_30[factor_30["keep_candidate"] == False]["drop_reason"].value_counts().head(5)

    report = [
        "# Feature Pruning Summary",
        "",
        "## Run Summary",
        f"- Asset: `{asset_name}`",
        f"- Run ID: `{manifest.get('run_id')}`",
        f"- Selected model: `{primary['selected_model']}`",
        f"- Signal: `{primary['signal']}`",
        f"- Reliability: `{primary['reliability_label']}`",
    ]
    if not base_ml.empty:
        row = base_ml.iloc[0]
        report.append(
            f"- Current all-feature best ML: `{row['model']}` at `{_fmt_pct(row['directional_accuracy'])}` accuracy, "
            f"Brier `{_fmt_num(row['brier_score'], 3)}`, net return `{_fmt_pct(row['tc_adjusted_return'])}`."
        )
    if not base_baseline.empty:
        row = base_baseline.iloc[0]
        report.append(f"- Best baseline: `{row['model']}` at `{_fmt_pct(row['directional_accuracy'])}` accuracy.")

    report.extend(["", "## Best Pruned Candidate"])
    if best_pruned.empty:
        report.append("- No pruned candidate produced a valid 30d official ML result.")
    else:
        row = best_pruned.iloc[0]
        detail = pruning_report[
            (pruning_report["candidate_feature_set"] == row["candidate_feature_set"])
            & (pruning_report["model_name"] == row["model_name"])
        ].head(1) if not pruning_report.empty else pd.DataFrame()
        report.append(
            f"- `{row['candidate_feature_set']}` / `{row['model_name']}`: samples `{int(row['n_samples'])}`, "
            f"accuracy `{_fmt_pct(row['directional_accuracy'])}`, Brier `{_fmt_num(row['brier_score'], 3)}`, "
            f"calibration `{_fmt_num(row['calibration_error'], 3)}`, Sharpe `{_fmt_num(row['sharpe'])}`, "
            f"drawdown `{_fmt_pct(row['max_drawdown'])}`, net return `{_fmt_pct(row['net_return'])}`."
        )
        if not detail.empty:
            drow = detail.iloc[0]
            report.append(
                f"- Stability evidence: bootstrap lower bound `{_fmt_pct(drow['bootstrap_ci_low'])}`, "
                f"bootstrap upper bound `{_fmt_pct(drow['bootstrap_ci_high'])}`, "
                f"permutation p-value `{_fmt_num(drow['permutation_p_value'], 3)}`."
            )
        report.append(f"- Promotion eligible: `{bool(row['promotion_eligible'])}`. Reason: `{row['rejection_reason']}`")

    report.extend(["", "## Promotion Decision"])
    if promoted.empty:
        report.append("- No pruned feature set passed the strict promotion rules. The official conclusion is unchanged.")
    else:
        names = [f"{r['candidate_feature_set']} / {r['model_name']}" for _, r in promoted.iterrows()]
        report.append(f"- Promotion candidates found: `{', '.join(names)}`. Review before deploying a changed official signal.")

    report.extend(["", "## Promising But Rejected Candidates"])
    if pruning_report.empty:
        report.append("- Feature pruning rollup was unavailable.")
    else:
        promising = pruning_report[pruning_report["report_label"].isin(["promising_but_unstable", "promising_but_rejected"])].copy()
        promising = promising.sort_values(
            ["report_label", "official_30d_accuracy", "net_return"],
            ascending=[False, False, False],
            na_position="last",
        ).head(8)
        if promising.empty:
            report.append("- No promising-but-rejected candidates were recorded.")
        else:
            for _, row in promising.iterrows():
                report.append(
                    f"- `{row['candidate_feature_set']}` / `{row['model_name']}`: label `{row['report_label']}`, "
                    f"accuracy `{_fmt_pct(row['official_30d_accuracy'])}`, net return `{_fmt_pct(row['net_return'])}`, "
                    f"CI low `{_fmt_pct(row['bootstrap_ci_low'])}`, p-value `{_fmt_num(row['permutation_p_value'], 3)}`, "
                    f"reason `{row['rejection_reason']}`."
                )

    if asset_id == "btc":
        report.extend(["", "## BTC No-Edge Drilldown"])
        drill = pruning_report[
            (pruning_report["candidate_feature_set"] == "btc_dollar_rates_cycle")
            & (pruning_report["model_name"] == "logistic_linear")
        ].head(1) if not pruning_report.empty else pd.DataFrame()
        if drill.empty:
            report.append("- `btc_dollar_rates_cycle` / `logistic_linear` was unavailable in the pruning report.")
        else:
            row = drill.iloc[0]
            bucket = "promising but unstable" if row["report_label"] == "promising_but_unstable" else "bad/noisy or not strong enough"
            report.append(
                f"- `btc_dollar_rates_cycle` / `logistic_linear` is classified as `{bucket}`: "
                f"accuracy `{_fmt_pct(row['official_30d_accuracy'])}`, net return `{_fmt_pct(row['net_return'])}`, "
                f"bootstrap lower bound `{_fmt_pct(row['bootstrap_ci_low'])}`, "
                f"permutation p-value `{_fmt_num(row['permutation_p_value'], 3)}`."
            )
            report.append(f"- Rejection reason: `{row['rejection_reason']}`")

    report.extend(["", "## Signal Quality"])
    if best_signal.empty:
        report.append("- Signal quality rows were unavailable.")
    else:
        for _, row in best_signal.iterrows():
            report.append(
                f"- `{row['candidate_feature_set']}` / `{row['model_name']}`: long signals `{int(row['n_long_signals'])}`, "
                f"abstention `{_fmt_pct(row['abstention_rate'])}`, long hit rate `{_fmt_pct(row['long_hit_rate'])}`, "
                f"long average return `{_fmt_pct(row['long_avg_return'])}`, net return `{_fmt_pct(row['net_return'])}`."
            )

    report.extend(["", "## Strongest 30d Factors"])
    if top_factors.empty:
        report.append("- No factor scorecard rows were available.")
    else:
        for _, row in top_factors.iterrows():
            report.append(
                f"- `{row['feature_name']}` (`{row['feature_group']}`): IC `{_fmt_num(row['information_coefficient'], 3)}`, "
                f"p-value `{_fmt_num(row['ic_p_value'], 3)}`, coverage `{_fmt_pct(row['coverage_pct'])}`, "
                f"keep `{bool(row['keep_candidate'])}`."
            )

    report.extend(["", "## Noisy Or Redundant Factor Reasons"])
    if dropped.empty:
        report.append("- No drop reasons were recorded.")
    else:
        for reason, count in dropped.items():
            report.append(f"- `{reason}`: `{int(count)}` features")

    neutral_justified = str(primary["signal"]) == "neutral" and (
        str(primary["selected_model"]) == "no_valid_edge"
        or str(primary["probability_confidence"]) == "Low confidence"
        or float(primary.get("predicted_probability_up", 0.5)) < 0.55
    )
    report.extend(
        [
            "",
            "## Recommendation",
            f"- Neutral signal justified: `{neutral_justified}`",
            "- Next step: only consider more model complexity after a promoted feature set survives these pruning and signal-quality checks.",
        ]
    )
    path = output_dir / "feature_pruning_summary.md"
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
    pruning = pruned_feature_diagnostics(
        run_id,
        raw,
        feature_result,
        config,
        base_outputs["model_leaderboard.csv"],
        base_outputs.get("confidence_intervals.csv"),
        quick=quick,
    )
    asset_policy = asset_feature_set_and_policy_diagnostics(
        run_id,
        raw,
        feature_result,
        config,
        base_outputs["model_leaderboard.csv"],
        quick=quick,
    )
    generated_at = str(latest["generated_at"].iloc[0]) if "generated_at" in latest and not latest.empty else ""
    latest_interpretation = latest_signal_interpretation_frame(
        run_id,
        generated_at,
        latest,
        asset_policy["signal_policy_report.csv"],
        asset_policy["asset_feature_set_leaderboard.csv"],
    )
    write_schema_csv("latest_signal_interpretation.csv", latest_interpretation, output_dir)
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
        **pruning,
        **asset_policy,
    }
    for filename, frame in diagnostics.items():
        write_diagnostic_csv(filename, frame, output_dir)
    write_full_refresh_diagnostics_report(output_dir, baseline_dir)
    write_derivatives_recovery_report(output_dir)
    write_feature_pruning_summary(output_dir)
    returned = dict(diagnostics)
    returned["latest_signal_interpretation.csv"] = latest_interpretation
    return returned
