from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from research_config import get_asset_config
from schemas import (
    REQUIRED_OUTPUT_FILES,
    SchemaError,
    empty_diagnostic_frame,
    validate_frame,
    validate_output_dir,
)


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
ASSET_OUTPUT_DIR = OUTPUT_DIR / "assets"


st.set_page_config(page_title="Crypto Research Validity Report", page_icon="C", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
    [data-testid="stMetricValue"] {font-size: 1.45rem;}
    .small-note {color: #60646c; font-size: 0.86rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_outputs(output_dir: Path, missing_message: str) -> dict:
    try:
        validate_output_dir(output_dir, REQUIRED_OUTPUT_FILES)
    except (SchemaError, FileNotFoundError, pd.errors.EmptyDataError) as exc:
        raise RuntimeError(f"{missing_message}\n\n{exc}") from exc

    diagnostic_dir = output_dir / "csv"
    diagnostic_warnings = []

    def read_diagnostic(filename: str) -> pd.DataFrame:
        path = diagnostic_dir / filename
        if not path.exists():
            diagnostic_warnings.append(f"Missing diagnostic file: outputs/csv/{filename}")
            return empty_diagnostic_frame(filename)
        try:
            frame = pd.read_csv(path)
            validate_frame(filename, frame)
            return frame
        except Exception as exc:
            diagnostic_warnings.append(f"Invalid diagnostic file outputs/csv/{filename}: {exc}")
            return empty_diagnostic_frame(filename)

    data = {
        "leaderboard": pd.read_csv(output_dir / "model_leaderboard.csv"),
        "backtest_summary": pd.read_csv(output_dir / "backtest_summary.csv"),
        "equity_curves": pd.read_csv(output_dir / "equity_curves.csv", parse_dates=["date"]),
        "calibration": pd.read_csv(output_dir / "calibration_table.csv"),
        "confidence": pd.read_csv(output_dir / "confidence_intervals.csv"),
        "regimes": pd.read_csv(output_dir / "regime_slices.csv"),
        "latest": pd.read_csv(output_dir / "latest_forecast.csv"),
        "availability": pd.read_csv(output_dir / "data_availability.csv"),
        "features": pd.read_csv(output_dir / "feature_audit.csv"),
        "rejections": read_diagnostic("model_rejection_reasons.csv"),
        "feature_signal": read_diagnostic("feature_signal_diagnostics.csv"),
        "feature_ablation": read_diagnostic("feature_group_ablation.csv"),
        "regime_breakdown": read_diagnostic("model_regime_breakdown.csv"),
        "derivatives_coverage": read_diagnostic("derivatives_coverage.csv"),
        "derivatives_impact": read_diagnostic("derivatives_impact.csv"),
        "feature_group_stability": read_diagnostic("feature_group_stability.csv"),
    }
    data["manifest"] = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    data["diagnostic_warnings"] = diagnostic_warnings
    return data


def fmt_pct(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.1%}"


def fmt_usd(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"${value:,.0f}"


def output_dir_for_asset(asset_id: str) -> Path:
    asset_dir = ASSET_OUTPUT_DIR / asset_id
    if asset_dir.exists():
        return asset_dir
    if asset_id == "btc":
        return OUTPUT_DIR
    return asset_dir


def equity_chart(equity: pd.DataFrame, horizon: int, window_type: str, model: str, asset_name: str) -> go.Figure:
    frame = equity[
        (equity["horizon"] == horizon)
        & (equity["window_type"] == window_type)
        & (equity["model"] == model)
    ].sort_values("date")
    fig = go.Figure()
    if not frame.empty:
        fig.add_trace(go.Scatter(x=frame["date"], y=frame["equity_tc_adjusted"], name="Model after costs", mode="lines"))
        fig.add_trace(go.Scatter(x=frame["date"], y=frame["equity_buy_hold"], name=f"{asset_name} buy and hold", mode="lines"))
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=25, b=10),
        yaxis_title="Growth of $1",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.05),
    )
    return fig


def calibration_chart(calibration: pd.DataFrame, horizon: int, window_type: str, model: str) -> go.Figure:
    frame = calibration[
        (calibration["horizon"] == horizon)
        & (calibration["window_type"] == window_type)
        & (calibration["model"] == model)
        & (calibration["sample_count"] > 0)
    ].copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Perfect calibration", mode="lines", line=dict(dash="dash")))
    if not frame.empty:
        fig.add_trace(
            go.Scatter(
                x=frame["avg_predicted_prob"],
                y=frame["actual_up_rate"],
                mode="markers+lines",
                name=model,
                marker=dict(size=10),
            )
        )
    fig.update_layout(
        height=330,
        margin=dict(l=10, r=10, t=25, b=10),
        xaxis_title="Average predicted probability",
        yaxis_title="Actual up rate",
        xaxis_tickformat=".0%",
        yaxis_tickformat=".0%",
    )
    return fig


selected_asset = st.sidebar.selectbox(
    "Asset",
    ["btc", "sol"],
    format_func=lambda asset_id: get_asset_config(asset_id).display_name,
)
asset_config = get_asset_config(selected_asset)
selected_output_dir = output_dir_for_asset(selected_asset)
missing_message = f"Run python research_run.py --refresh --asset {selected_asset} first."

try:
    outputs = load_outputs(selected_output_dir, missing_message)
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()

leaderboard = outputs["leaderboard"]
summary = outputs["backtest_summary"]
latest = outputs["latest"]
equity = outputs["equity_curves"]
calibration = outputs["calibration"]
confidence = outputs["confidence"]
regimes = outputs["regimes"]
availability = outputs["availability"]
features = outputs["features"]
rejections = outputs["rejections"]
feature_signal = outputs["feature_signal"]
feature_ablation = outputs["feature_ablation"]
regime_breakdown = outputs["regime_breakdown"]
derivatives_coverage = outputs["derivatives_coverage"]
derivatives_impact = outputs["derivatives_impact"]
feature_group_stability = outputs["feature_group_stability"]
manifest = outputs["manifest"]
asset_name = str(manifest.get("asset_name") or asset_config.display_name)

primary = latest.loc[latest["is_primary_objective"] == True].head(1)
primary_row = primary.iloc[0] if not primary.empty else latest.iloc[0]

st.title(f"{asset_name} Research Validity Report")
st.markdown(
    "<span class='small-note'>Read-only report. All models, thresholds, calibration, and backtests are precomputed by <code>research_run.py</code>.</span>",
    unsafe_allow_html=True,
)

top = st.columns(5)
top[0].metric("Primary objective", "30d direction")
top[1].metric("Selected model", str(primary_row["selected_model"]))
top[2].metric("Signal", str(primary_row["signal"]))
top[3].metric("Reliability", str(primary_row["reliability_label"]))
top[4].metric("Run mode", "Quick" if manifest.get("quick_mode") else "Refresh")

forecast_cols = st.columns(4)
forecast_cols[0].metric(f"{asset_name} spot", fmt_usd(primary_row["current_price"]), str(primary_row["as_of_date"]))
forecast_cols[1].metric("Probability up", fmt_pct(primary_row["predicted_probability_up"]))
forecast_cols[2].metric("Expected return", fmt_pct(primary_row["expected_return"]))
forecast_cols[3].metric("Model-implied forecast price", fmt_usd(primary_row["model_implied_forecast_price"]))

if manifest.get("warnings"):
    with st.expander("Research Warnings", expanded=True):
        for warning in manifest["warnings"]:
            st.warning(warning)

if outputs.get("diagnostic_warnings"):
    with st.expander("Diagnostic Output Warnings", expanded=False):
        for warning in outputs["diagnostic_warnings"]:
            st.warning(warning)

if str(primary_row["selected_model"]) == "no_valid_edge":
    st.info(
        f"The full refresh did not validate a 30d {asset_name} directional edge. "
        "The dashboard is therefore showing a neutral signal instead of forcing a forecast."
    )

quality_tab, no_edge_tab, forecast_tab, calibration_tab, regimes_tab, data_tab, feature_tab = st.tabs(
    ["Model Quality", "No Edge", "Latest Forecast", "Calibration", "Regimes", "Data", "Features"]
)

with quality_tab:
    st.subheader("Apples-to-Apples Leaderboard")
    official = leaderboard[leaderboard["is_official"] == True].copy()
    selected_horizon = st.selectbox("Horizon", [30, 90], format_func=lambda x: f"{x}d official")
    selected_window = "official_monthly" if selected_horizon == 30 else "official_quarterly"
    table = official[(official["horizon"] == selected_horizon) & (official["window_type"] == selected_window)].copy()

    ci_acc = confidence[
        (confidence["horizon"] == selected_horizon)
        & (confidence["window_type"] == selected_window)
        & (confidence["metric"] == "directional_accuracy")
    ][["model", "ci_low", "ci_high", "permutation_p_value"]]
    table = table.merge(ci_acc, on="model", how="left")
    table["confidence_interval"] = table.apply(
        lambda r: f"{fmt_pct(r['ci_low'])} - {fmt_pct(r['ci_high'])}" if pd.notna(r.get("ci_low")) else "n/a",
        axis=1,
    )
    view_cols = [
        "rank",
        "model",
        "sample_count",
        "test_start",
        "test_end",
        "window_type",
        "directional_accuracy",
        "brier_score",
        "sharpe",
        "max_drawdown",
        "tc_adjusted_return",
        "confidence_interval",
        "permutation_p_value",
        "reliability_label",
        "useful_model",
        "notes",
    ]
    st.dataframe(
        table[view_cols].style.format(
            {
                "directional_accuracy": "{:.1%}",
                "brier_score": "{:.3f}",
                "sharpe": "{:.2f}",
                "max_drawdown": "{:.1%}",
                "tc_adjusted_return": "{:.1%}",
                "permutation_p_value": "{:.3f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    chart_model = str(primary_row["selected_model"])
    if chart_model == "no_valid_edge" or chart_model not in set(equity["model"]):
        chart_model = str(table.iloc[0]["model"]) if not table.empty else ""
    if chart_model:
        st.plotly_chart(equity_chart(equity, selected_horizon, selected_window, chart_model, asset_name), use_container_width=True)

with no_edge_tab:
    st.subheader("No Valid Edge Diagnostics")
    official_30 = leaderboard[(leaderboard["horizon"] == 30) & (leaderboard["window_type"] == "official_monthly")].copy()
    baseline_models = ["buy_hold_direction", "momentum_30d", "momentum_90d", "random_permutation"]
    best_baseline = official_30[official_30["model"].isin(baseline_models)].sort_values("directional_accuracy", ascending=False).head(1)
    best_ml = official_30[~official_30["model"].isin(baseline_models)].sort_values("directional_accuracy", ascending=False).head(1)
    c1, c2, c3 = st.columns(3)
    if not best_baseline.empty:
        c1.metric("Best baseline", str(best_baseline.iloc[0]["model"]), fmt_pct(best_baseline.iloc[0]["directional_accuracy"]))
    if not best_ml.empty:
        c2.metric("Best ML model", str(best_ml.iloc[0]["model"]), fmt_pct(best_ml.iloc[0]["directional_accuracy"]))
    c3.metric("Confidence", str(primary_row["reliability_label"]))
    st.caption(str(primary_row["selection_reason"]))

    rejection_30 = rejections[(rejections["horizon"] == 30) & (rejections["window_type"] == "official_monthly")].copy()
    st.dataframe(
        rejection_30[
            [
                "model_name",
                "is_baseline",
                "n_samples",
                "directional_accuracy",
                "beats_buy_hold",
                "beats_momentum_30d",
                "beats_momentum_90d",
                "beats_random_baseline",
                "passes_transaction_cost_check",
                "passes_calibration_check",
                "passes_sample_threshold",
                "passes_reliability_check",
                "final_rejection_reason",
            ]
        ].style.format({"directional_accuracy": "{:.1%}"}),
        width="stretch",
        hide_index=True,
    )

    st.subheader("Feature Group Ablation")
    ablation_30 = feature_ablation[(feature_ablation["horizon"] == 30) & (feature_ablation["window_type"] == "official_monthly")]
    st.dataframe(
        ablation_30.sort_values("directional_accuracy", ascending=False).head(25).style.format(
            {
                "directional_accuracy": "{:.1%}",
                "balanced_accuracy": "{:.1%}",
                "brier_score": "{:.3f}",
                "calibration_error": "{:.3f}",
                "sharpe": "{:.2f}",
                "max_drawdown": "{:.1%}",
                "net_return": "{:.1%}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    st.subheader("Feature Group Watchlist")
    watchlist = ablation_30[
        (ablation_30["feature_group"] == "dollar_rates_only")
        & (ablation_30["model_name"] == "logistic_linear")
    ]
    if watchlist.empty:
        st.warning("No precomputed dollar_rates_only / logistic_linear watchlist row is available.")
    else:
        st.dataframe(
            watchlist.style.format(
                {
                    "directional_accuracy": "{:.1%}",
                    "balanced_accuracy": "{:.1%}",
                    "brier_score": "{:.3f}",
                    "calibration_error": "{:.3f}",
                    "sharpe": "{:.2f}",
                    "max_drawdown": "{:.1%}",
                    "net_return": "{:.1%}",
                }
            ),
            width="stretch",
            hide_index=True,
        )

    st.subheader("Feature Group Stability")
    stability_30 = feature_group_stability[
        (feature_group_stability["horizon"] == 30)
        & (feature_group_stability["window_type"] == "official_monthly")
    ]
    if stability_30.empty:
        st.warning("No precomputed feature group stability file is available.")
    else:
        st.dataframe(
            stability_30.sort_values(["feature_group", "model_name", "period_slice"]).style.format(
                {
                    "directional_accuracy": "{:.1%}",
                    "net_return": "{:.1%}",
                    "sharpe": "{:.2f}",
                    "max_drawdown": "{:.1%}",
                }
            ),
            width="stretch",
            hide_index=True,
        )

with forecast_tab:
    st.subheader("Latest Precomputed Forecast")
    st.dataframe(
        latest[
            [
                "horizon",
                "selected_model",
                "signal",
                "predicted_probability_up",
                "probability_confidence",
                "expected_return",
                "model_implied_forecast_price",
                "reliability_label",
                "selection_reason",
            ]
        ].style.format(
            {
                "predicted_probability_up": "{:.1%}",
                "expected_return": "{:.1%}",
                "model_implied_forecast_price": "${:,.0f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    st.caption("180d is diagnostic only and never drives model selection.")

with calibration_tab:
    st.subheader("Out-of-Sample Calibration")
    c1, c2 = st.columns([1, 1])
    model_options = sorted(calibration["model"].unique())
    selected_model = c1.selectbox("Model", model_options, index=model_options.index(str(primary_row["selected_model"])) if str(primary_row["selected_model"]) in model_options else 0)
    selected_cal_horizon = c2.selectbox("Calibration horizon", [30, 90, 180], format_func=lambda x: f"{x}d")
    selected_cal_window = {
        30: "official_monthly",
        90: "official_quarterly",
        180: "diagnostic_semiannual",
    }[selected_cal_horizon]
    st.plotly_chart(calibration_chart(calibration, selected_cal_horizon, selected_cal_window, selected_model), use_container_width=True)
    st.dataframe(
        calibration[
            (calibration["horizon"] == selected_cal_horizon)
            & (calibration["window_type"] == selected_cal_window)
            & (calibration["model"] == selected_model)
        ].style.format(
            {
                "prob_bin_low": "{:.0%}",
                "prob_bin_high": "{:.0%}",
                "avg_predicted_prob": "{:.1%}",
                "actual_up_rate": "{:.1%}",
                "brier_score": "{:.3f}",
                "calibration_error": "{:.3f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

with regimes_tab:
    st.subheader("Regime Slices")
    st.dataframe(
        regimes.style.format(
            {
                "directional_accuracy": "{:.1%}",
                "balanced_accuracy": "{:.1%}",
                "brier_score": "{:.3f}",
                "calibration_error": "{:.3f}",
                "sharpe": "{:.2f}",
                "max_drawdown": "{:.1%}",
                "tc_adjusted_return": "{:.1%}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

with data_tab:
    st.subheader("Data Availability")
    status_counts = availability["status"].value_counts().rename_axis("status").reset_index(name="count")
    st.dataframe(status_counts, width="stretch", hide_index=True)
    st.dataframe(availability, width="stretch", hide_index=True)

    st.subheader("Derivatives Coverage")
    if derivatives_coverage.empty:
        st.warning("No precomputed derivatives coverage file is available.")
    else:
        st.dataframe(
            derivatives_coverage.style.format({"missing_pct": "{:.1%}"}),
            width="stretch",
            hide_index=True,
        )

    st.subheader("Derivatives Impact")
    if derivatives_impact.empty:
        st.warning("No derivatives impact comparison ran because no valid derivative features entered the model feature set.")
    else:
        st.dataframe(
            derivatives_impact.style.format(
                {
                    "directional_accuracy": "{:.1%}",
                    "balanced_accuracy": "{:.1%}",
                    "brier_score": "{:.3f}",
                    "calibration_error": "{:.3f}",
                    "sharpe": "{:.2f}",
                    "max_drawdown": "{:.1%}",
                    "net_return": "{:.1%}",
                }
            ),
            width="stretch",
            hide_index=True,
        )

    st.subheader("Run Manifest")
    manifest_view = pd.DataFrame(
        [
            {"field": "run_id", "value": manifest.get("run_id", "")},
            {"field": "schema_version", "value": manifest.get("schema_version", "")},
            {"field": "created_at", "value": manifest.get("created_at", "")},
            {"field": "git_commit", "value": manifest.get("git_commit", "")},
            {"field": "start_date", "value": manifest.get("start_date", "")},
            {"field": "end_date", "value": manifest.get("end_date", "")},
            {"field": "config_hash", "value": manifest.get("config_hash", "")},
            {"field": "data_snapshot_hash", "value": manifest.get("data_snapshot_hash", "")},
            {"field": "features_count", "value": str(manifest.get("features_count", ""))},
            {"field": "models_run", "value": ", ".join(manifest.get("models_run", []))},
        ]
    )
    st.dataframe(manifest_view, width="stretch", hide_index=True)

with feature_tab:
    st.subheader("Feature Audit")
    source_filter = st.multiselect("Sources", sorted(features["source"].unique()), default=sorted(features["source"].unique()))
    filtered = features[features["source"].isin(source_filter)]
    st.dataframe(
        filtered.style.format({"missing_pct": "{:.1%}"}),
        width="stretch",
        hide_index=True,
    )
