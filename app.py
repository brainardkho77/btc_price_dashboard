from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from schemas import REQUIRED_OUTPUT_FILES, SchemaError, validate_output_dir


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
MISSING_MESSAGE = "Run python research_run.py --refresh first."


st.set_page_config(page_title="BTC Research Validity Report", page_icon="BTC", layout="wide")

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
def load_outputs(output_dir: Path) -> dict:
    try:
        validate_output_dir(output_dir, REQUIRED_OUTPUT_FILES)
    except (SchemaError, FileNotFoundError, pd.errors.EmptyDataError) as exc:
        raise RuntimeError(f"{MISSING_MESSAGE}\n\n{exc}") from exc

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
    }
    data["manifest"] = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    return data


def fmt_pct(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.1%}"


def fmt_usd(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"${value:,.0f}"


def equity_chart(equity: pd.DataFrame, horizon: int, window_type: str, model: str) -> go.Figure:
    frame = equity[
        (equity["horizon"] == horizon)
        & (equity["window_type"] == window_type)
        & (equity["model"] == model)
    ].sort_values("date")
    fig = go.Figure()
    if not frame.empty:
        fig.add_trace(go.Scatter(x=frame["date"], y=frame["equity_tc_adjusted"], name="Model after costs", mode="lines"))
        fig.add_trace(go.Scatter(x=frame["date"], y=frame["equity_buy_hold"], name="BTC buy and hold", mode="lines"))
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


try:
    outputs = load_outputs(OUTPUT_DIR)
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
manifest = outputs["manifest"]

primary = latest.loc[latest["is_primary_objective"] == True].head(1)
primary_row = primary.iloc[0] if not primary.empty else latest.iloc[0]

st.title("BTC Research Validity Report")
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
forecast_cols[0].metric("BTC spot", fmt_usd(primary_row["current_price"]), str(primary_row["as_of_date"]))
forecast_cols[1].metric("Probability up", fmt_pct(primary_row["predicted_probability_up"]))
forecast_cols[2].metric("Expected return", fmt_pct(primary_row["expected_return"]))
forecast_cols[3].metric("Model-implied forecast price", fmt_usd(primary_row["model_implied_forecast_price"]))

if manifest.get("warnings"):
    with st.expander("Research Warnings", expanded=True):
        for warning in manifest["warnings"]:
            st.warning(warning)

quality_tab, forecast_tab, calibration_tab, regimes_tab, data_tab, feature_tab = st.tabs(
    ["Model Quality", "Latest Forecast", "Calibration", "Regimes", "Data", "Features"]
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
        st.plotly_chart(equity_chart(equity, selected_horizon, selected_window, chart_model), use_container_width=True)

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

    st.subheader("Run Manifest")
    manifest_view = pd.DataFrame(
        [
            {"field": "run_id", "value": manifest.get("run_id", "")},
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
