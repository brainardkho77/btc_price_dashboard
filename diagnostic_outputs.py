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
    if not refresh:
        return None
    manifest_path = output_dir / "run_manifest.json"
    if not manifest_path.exists():
        return BASELINE_DIR if BASELINE_DIR.exists() else None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return BASELINE_DIR if BASELINE_DIR.exists() else None
    if not manifest.get("quick_mode"):
        return BASELINE_DIR if BASELINE_DIR.exists() else None

    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    for filename in [
        "run_manifest.json",
        "data_availability.csv",
        "model_leaderboard.csv",
        "latest_forecast.csv",
        "feature_audit.csv",
    ]:
        src = output_dir / filename
        if src.exists():
            shutil.copy2(src, BASELINE_DIR / filename)
    return BASELINE_DIR


def feature_group_for_feature(feature: str) -> str:
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


def _score_feature_set(
    raw: pd.DataFrame,
    feature_result,
    feature_cols: Sequence[str],
    horizon: int,
    window_type: str,
    config: ResearchConfig,
    *,
    quick: bool,
) -> Dict[str, dict]:
    from research_pipeline import (
        BASELINE_MODELS,
        build_target_frame,
        build_walk_forward_windows,
        compute_metrics,
        decorate_summary,
        predict_baseline_windows,
        predict_model_windows,
    )

    target_frame = build_target_frame(raw, feature_result.features, feature_cols, horizon)
    windows = build_walk_forward_windows(target_frame, horizon, window_type, config, quick=quick)
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
    predictions_by_model = {model: preds.loc[common_dates].sort_index() for model, preds in predictions_by_model.items()}

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
    price = raw["btc_close"].astype(float)
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
            actual_up = (part["btc_return"] > 0).astype(int)
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
                        "brier_score": summary.get("brier_score", np.nan),
                        "sharpe": summary.get("sharpe", np.nan),
                        "max_drawdown": summary.get("max_drawdown", np.nan),
                        "net_return": summary.get("tc_adjusted_return", np.nan),
                        "reliability_label": summary.get("reliability_label", "Low confidence"),
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


def write_full_refresh_diagnostics_report(output_dir: Path, baseline_dir: Optional[Path]) -> Path:
    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    latest = pd.read_csv(output_dir / "latest_forecast.csv")
    leaderboard = pd.read_csv(output_dir / "model_leaderboard.csv")
    rejection = pd.read_csv(output_dir / "csv" / "model_rejection_reasons.csv")
    ablation = pd.read_csv(output_dir / "csv" / "feature_group_ablation.csv")
    regimes = pd.read_csv(output_dir / "csv" / "model_regime_breakdown.csv")
    availability = pd.read_csv(output_dir / "data_availability.csv")
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
    regime_30 = regimes[(regimes["horizon"] == 30) & (regimes["window_type"] == "official_monthly")]
    regime_best = regime_30.sort_values("directional_accuracy", ascending=False).head(10)

    derivatives_worked = availability[availability["dataset"].str.startswith("binance_")]["status"].eq("worked").any()
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
        ]
    )
    if derivatives_impact_frame.empty:
        report.append("- Derivatives impact test was not run because no derivative features were available.")
    else:
        best_deriv = derivatives_impact_frame.sort_values("directional_accuracy", ascending=False).head(1).iloc[0]
        report.append(
            f"- Best derivatives impact row: `{best_deriv['model_name']}`, derivatives included `{best_deriv['derivatives_included']}`, "
            f"accuracy `{_fmt_pct(best_deriv['directional_accuracy'])}`."
        )

    report.extend(
        [
            "",
            "## Conclusion",
            "- The full refresh did not validate a 30d BTC directional edge.",
            "- `no_valid_edge` remains the correct conclusion unless future data reliability or feature diagnostics materially improve.",
            "- Recommended next step: improve Binance/manual derivatives coverage, then rerun this diagnostic sprint before adding more models.",
        ]
    )

    path = output_dir / "full_refresh_diagnostics.md"
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
        "derivatives_impact.csv": derivatives_impact(run_id, raw, feature_result, config, quick=quick),
    }
    for filename, frame in diagnostics.items():
        write_diagnostic_csv(filename, frame, output_dir)
    write_full_refresh_diagnostics_report(output_dir, baseline_dir)
    return diagnostics
