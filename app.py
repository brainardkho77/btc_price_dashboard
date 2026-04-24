from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_sources import SOURCE_NOTES, cache_metadata, load_all_data, source_coverage
from features import (
    FACTOR_EXPLANATIONS,
    HORIZONS,
    build_training_data,
    current_factor_pressure,
    factor_information_coefficients,
    grouped_pressure,
)
from modeling import MODEL_CHOICES, equity_curves, forecast_horizons, walk_forward_backtest


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


st.set_page_config(page_title="BTC Factor Forecast Dashboard", page_icon="BTC", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
    [data-testid="stMetricValue"] {font-size: 1.55rem;}
    .small-note {color: #60646c; font-size: 0.86rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=60 * 60, show_spinner=False)
def cached_data(force_refresh: bool) -> pd.DataFrame:
    return load_all_data(force=force_refresh)


@st.cache_data(ttl=60 * 60, show_spinner=False)
def cached_forecasts(raw: pd.DataFrame, model_name: str, recompute: bool = False) -> pd.DataFrame:
    path = OUTPUT_DIR / f"latest_forecasts_{model_name}.csv"
    if path.exists() and not recompute:
        frame = pd.read_csv(path, parse_dates=["as_of"])
        return frame

    frame = forecast_horizons(raw, HORIZONS, model_name=model_name)
    frame.to_csv(path, index=False)
    return frame


@st.cache_data(ttl=60 * 60, show_spinner=False)
def cached_training(raw: pd.DataFrame, horizon: int):
    return build_training_data(raw, horizon)


@st.cache_data(ttl=60 * 60, show_spinner=False)
def cached_factor_ic(raw: pd.DataFrame, horizon: int) -> pd.DataFrame:
    return factor_information_coefficients(build_training_data(raw, horizon))


@st.cache_data(ttl=60 * 60, show_spinner=False)
def cached_factor_pressure(raw: pd.DataFrame, horizon: int) -> pd.DataFrame:
    return current_factor_pressure(build_training_data(raw, horizon))


def fmt_pct(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.1%}"


def fmt_usd(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"${value:,.0f}"


def price_chart(raw: pd.DataFrame) -> go.Figure:
    price = raw["btc_close"].dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=price.index, y=price, mode="lines", name="BTC close", line=dict(width=2)))
    for window, color in [(50, "#2f80ed"), (200, "#f2994a")]:
        fig.add_trace(
            go.Scatter(
                x=price.index,
                y=price.rolling(window).mean(),
                mode="lines",
                name=f"{window}d MA",
                line=dict(width=1.2, color=color),
            )
        )
    fig.update_layout(
        height=440,
        margin=dict(l=15, r=15, t=25, b=15),
        yaxis_title="USD",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02),
    )
    fig.update_yaxes(type="log")
    return fig


def forecast_chart(forecasts: pd.DataFrame) -> go.Figure:
    frame = forecasts.sort_values("horizon")
    colors = ["#1a7f64" if v >= 0 else "#c2410c" for v in frame["pred_return"]]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=frame["horizon"].astype(str) + "d",
            y=frame["pred_return"],
            marker_color=colors,
            text=[fmt_pct(v) for v in frame["pred_return"]],
            textposition="outside",
            name="Expected return",
        )
    )
    fig.update_layout(
        height=320,
        margin=dict(l=15, r=15, t=25, b=15),
        yaxis_tickformat=".0%",
        yaxis_title="Expected return",
        showlegend=False,
    )
    return fig


def backtest_summary_from_disk(model_name: str) -> pd.DataFrame:
    path = OUTPUT_DIR / f"backtest_summary_{model_name}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def equity_from_disk(horizon: int, model_name: str) -> pd.DataFrame:
    path = OUTPUT_DIR / f"equity_h{horizon}_{model_name}.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    return frame


def equity_chart(curves: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not curves.empty:
        fig.add_trace(go.Scatter(x=curves.index, y=curves["long_cash_signal"], mode="lines", name="Model long/cash"))
        fig.add_trace(go.Scatter(x=curves.index, y=curves["buy_hold_same_windows"], mode="lines", name="BTC buy/hold windows"))
    fig.update_layout(
        height=370,
        margin=dict(l=15, r=15, t=25, b=15),
        yaxis_title="Growth of $1",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02),
    )
    return fig


def pressure_chart(pressure: pd.DataFrame) -> go.Figure:
    grouped = grouped_pressure(pressure)
    fig = go.Figure()
    if not grouped.empty:
        colors = ["#1a7f64" if v >= 0 else "#c2410c" for v in grouped["pressure"]]
        fig.add_trace(
            go.Bar(
                y=grouped["group"],
                x=grouped["pressure"],
                orientation="h",
                marker_color=colors,
                name="Current pressure",
            )
        )
    fig.update_layout(
        height=330,
        margin=dict(l=15, r=15, t=25, b=15),
        xaxis_title="IC-weighted current z-score",
        yaxis=dict(autorange="reversed"),
        showlegend=False,
    )
    return fig


with st.sidebar:
    st.header("Controls")
    model_name = st.selectbox("Model", MODEL_CHOICES, index=0)
    horizon = st.selectbox("Forecast horizon", HORIZONS, index=2, format_func=lambda x: f"{x} days")
    refresh_data = st.checkbox("Refresh free data sources", value=False)
    recompute_forecasts = st.button(
        "Recompute forecasts",
        help="Uses live cached data and retrains the selected model. This can be slow on Streamlit Community Cloud.",
    )
    st.divider()
    initial_train_days = st.slider("Initial training window", 730, 2555, 1460, step=365)
    step_mode = st.selectbox("Backtest step", ["non-overlapping", "weekly", "daily"], index=0)
    refit_every_days = st.slider("Refit cadence", 7, 90, 14, step=7, help="Backtest retrains at this cadence, using only past data.")
    max_windows = st.slider("Max backtest windows", 0, 1200, 0, step=50, help="0 uses all possible windows.")
    threshold = st.slider("Long/cash threshold", -0.05, 0.05, 0.0, step=0.005, format="%.3f")

st.title("BTC Factor Forecast Dashboard")
st.markdown(
    "<span class='small-note'>Research dashboard only. Forecasts are probabilistic model outputs, not financial advice or a trading instruction.</span>",
    unsafe_allow_html=True,
)

with st.spinner("Loading free market, macro, sentiment, and on-chain data..."):
    raw = cached_data(refresh_data)

with st.spinner("Loading forecasts..."):
    forecasts = cached_forecasts(raw, model_name, recompute_forecasts)

selected = forecasts.loc[forecasts["horizon"] == horizon].iloc[0]
latest_date = pd.Timestamp(selected["as_of"]).date().isoformat()

cols = st.columns(5)
cols[0].metric("BTC spot", fmt_usd(selected["current_price"]), latest_date)
cols[1].metric(f"{horizon}d expected return", fmt_pct(selected["pred_return"]))
cols[2].metric("Probability up", fmt_pct(selected["prob_up"]))
cols[3].metric("Forecast price", fmt_usd(selected["forecast_price"]))
cols[4].metric("68% range", f"{fmt_usd(selected['price_low_68'])} - {fmt_usd(selected['price_high_68'])}")

left, right = st.columns([1.6, 1])
with left:
    st.plotly_chart(price_chart(raw), use_container_width=True)
with right:
    st.plotly_chart(forecast_chart(forecasts), use_container_width=True)
    st.dataframe(
        forecasts[
            [
                "horizon",
                "pred_return",
                "prob_up",
                "forecast_price",
                "price_low_68",
                "price_high_68",
                "train_rows",
                "feature_count",
            ]
        ].style.format(
            {
                "pred_return": "{:.1%}",
                "prob_up": "{:.1%}",
                "forecast_price": "${:,.0f}",
                "price_low_68": "${:,.0f}",
                "price_high_68": "${:,.0f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

forecast_tab, backtest_tab, factors_tab, data_tab = st.tabs(["Forecast", "Backtest", "Factors", "Data"])

with forecast_tab:
    st.subheader("Current Forecast Distribution")
    fcols = st.columns(3)
    fcols[0].metric("90% low", fmt_usd(selected["price_low_90"]))
    fcols[1].metric("Median", fmt_usd(selected["forecast_price"]))
    fcols[2].metric("90% high", fmt_usd(selected["price_high_90"]))
    st.caption(
        "The range uses model residual volatility from the training sample. It is a rough uncertainty band, not a guaranteed confidence interval."
    )

    st.subheader("Factors Used")
    st.dataframe(pd.DataFrame(FACTOR_EXPLANATIONS), use_container_width=True, hide_index=True)

with backtest_tab:
    st.subheader("Walk-Forward Backtest")
    step_days = {"non-overlapping": horizon, "weekly": 7, "daily": 1}[step_mode]
    run_backtest = st.button("Run selected backtest", type="primary")
    disk_summary = backtest_summary_from_disk(model_name)

    if run_backtest:
        with st.spinner("Running walk-forward fit/predict loop..."):
            training = cached_training(raw, horizon)
            preds, summary = walk_forward_backtest(
                training,
                model_name=model_name,
                initial_train_days=initial_train_days,
                step_days=step_days,
                refit_every_days=refit_every_days,
                threshold=threshold,
                max_windows=max_windows or None,
            )
            preds.to_csv(OUTPUT_DIR / f"predictions_h{horizon}_{model_name}.csv")
            curves = equity_curves(preds, threshold=threshold)
            curves.to_csv(OUTPUT_DIR / f"equity_h{horizon}_{model_name}.csv")
            summary_frame = pd.DataFrame([summary])
            st.dataframe(
                summary_frame.style.format(
                    {
                        "mae": "{:.4f}",
                        "rmse": "{:.4f}",
                        "directional_accuracy": "{:.1%}",
                        "strategy_total_return": "{:.1%}",
                        "buyhold_total_return": "{:.1%}",
                        "strategy_ann_return": "{:.1%}",
                        "buyhold_ann_return": "{:.1%}",
                        "strategy_sharpe": "{:.2f}",
                        "buyhold_sharpe": "{:.2f}",
                        "strategy_max_drawdown": "{:.1%}",
                        "buyhold_max_drawdown": "{:.1%}",
                        "exposure": "{:.1%}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
            st.plotly_chart(equity_chart(curves), use_container_width=True)
    else:
        if not disk_summary.empty:
            st.caption("Showing the latest precomputed CLI summary from the local outputs folder.")
            st.dataframe(
                disk_summary.style.format(
                    {
                        "mae": "{:.4f}",
                        "rmse": "{:.4f}",
                        "directional_accuracy": "{:.1%}",
                        "strategy_total_return": "{:.1%}",
                        "buyhold_total_return": "{:.1%}",
                        "strategy_ann_return": "{:.1%}",
                        "buyhold_ann_return": "{:.1%}",
                        "strategy_sharpe": "{:.2f}",
                        "buyhold_sharpe": "{:.2f}",
                        "strategy_max_drawdown": "{:.1%}",
                        "buyhold_max_drawdown": "{:.1%}",
                        "exposure": "{:.1%}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
            curves = equity_from_disk(horizon, model_name)
            st.plotly_chart(equity_chart(curves), use_container_width=True)
        else:
            st.info("Run a selected backtest here, or run `python run_backtest.py --horizons all` from this folder.")

with factors_tab:
    st.subheader(f"{horizon}d Factor Signals")
    pressure = cached_factor_pressure(raw, horizon)
    ic = cached_factor_ic(raw, horizon)
    pc_left, pc_right = st.columns([1, 1])
    with pc_left:
        st.plotly_chart(pressure_chart(pressure), use_container_width=True)
    with pc_right:
        if not pressure.empty:
            st.dataframe(
                pressure.head(20).style.format(
                    {
                        "latest_z": "{:.2f}",
                        "pressure": "{:.3f}",
                        "spearman_ic": "{:.3f}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

    st.subheader("Historical Factor IC")
    pos = ic.sort_values("spearman_ic", ascending=False).head(15)
    neg = ic.sort_values("spearman_ic", ascending=True).head(15)
    ic_cols = st.columns(2)
    ic_cols[0].dataframe(pos.style.format({"spearman_ic": "{:.3f}", "abs_ic": "{:.3f}"}), use_container_width=True, hide_index=True)
    ic_cols[1].dataframe(neg.style.format({"spearman_ic": "{:.3f}", "abs_ic": "{:.3f}"}), use_container_width=True, hide_index=True)

with data_tab:
    st.subheader("Data Sources")
    st.dataframe(
        pd.DataFrame([{"source": key, "usage": value} for key, value in SOURCE_NOTES.items()]),
        use_container_width=True,
        hide_index=True,
    )
    st.subheader("Coverage")
    coverage_cols = [
        "btc_close",
        "eth_close",
        "spx_close",
        "vix_close",
        "dxy_close",
        "cm_mvrv",
        "cm_active_addresses",
        "cm_exchange_inflow_usd",
        "fear_greed_value",
        "us_10y_yield",
        "trade_weighted_usd",
        "m2_money_supply",
    ]
    st.dataframe(source_coverage(raw, coverage_cols), use_container_width=True, hide_index=True)
    meta = cache_metadata()
    if not meta.empty:
        st.subheader("Local Cache")
        st.dataframe(meta, use_container_width=True, hide_index=True)
