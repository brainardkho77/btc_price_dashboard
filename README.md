# BTC Research Validity Dashboard

Offline BTC research pipeline plus a read-only Streamlit report.

Primary objective: 30d BTC directional edge. Secondary objective: 90d BTC directional edge. 180d is diagnostic only and never drives model selection.

This is not financial advice. The outputs are historical research artifacts and probabilistic model outputs.

## Commands

Install the Streamlit viewer dependencies:

```bash
pip install -r requirements.txt
```

Install the offline research dependencies:

```bash
pip install -r requirements-research.txt
```

Run the full offline research pass:

```bash
python research_run.py --refresh
```

Run the faster cached validation pass:

```bash
python research_run.py --quick
```

Open the read-only dashboard:

```bash
streamlit run app.py
```

If the dashboard says `Run python research_run.py --refresh first.`, the required precomputed output files are missing or failed schema validation.

## Output Files

`research_run.py` writes:

- `outputs/model_leaderboard.csv`
- `outputs/backtest_summary.csv`
- `outputs/equity_curves.csv`
- `outputs/calibration_table.csv`
- `outputs/confidence_intervals.csv`
- `outputs/regime_slices.csv`
- `outputs/latest_forecast.csv`
- `outputs/data_availability.csv`
- `outputs/feature_audit.csv`
- `outputs/run_manifest.json`

Streamlit only loads these files. It does not train, tune, fetch heavy data, or run backtests.

## Data Sources

The pipeline uses free/keyless sources where available:

- Coinbase BTC-USD candles with 300-candle chunking.
- Yahoo chart endpoint as a price and market-proxy fallback.
- Coin Metrics Community API for BTC on-chain metrics.
- FRED graph CSV downloads for macro and liquidity series.
- Alternative.me Fear & Greed sentiment.
- Binance USD-M Futures developer endpoints for funding, open interest, long/short ratio, taker buy/sell ratio, and basis when reachable.
- DefiLlama stablecoin supply history.

Unavailable or failed sources are recorded in `outputs/data_availability.csv`. ETF flows and other unavailable metrics are not fabricated.

## Research Controls

The official result for 30d is non-overlapping monthly windows. The official result for 90d is non-overlapping quarterly windows. Weekly 30d and overlapping 90d views are sensitivity checks only.

All models and baselines use identical walk-forward train/test windows for each horizon and window type. Threshold tuning is nested inside each training window, and probabilities are calibrated only on out-of-sample calibration slices before the test window.

The dashboard uses the term model-implied forecast price for model output.
