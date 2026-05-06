from __future__ import annotations

import io
import json
import math
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional
from urllib.parse import quote

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)

USER_AGENT = "btc-factor-dashboard/1.0 (+local research app)"

YAHOO_MARKETS: Dict[str, str] = {
    "BTC-USD": "btc_proxy_close",
    "ETH-USD": "eth_close",
    "^GSPC": "spx_close",
    "^IXIC": "nasdaq_close",
    "^VIX": "vix_close",
    "DX-Y.NYB": "dxy_close",
    "GC=F": "gold_close",
    "TLT": "tlt_close",
}

FRED_SERIES: Dict[str, str] = {
    "DGS10": "us_10y_yield",
    "DGS2": "us_2y_yield",
    "DFII10": "real_10y_yield",
    "DFII5": "real_5y_yield",
    "TB3MS": "us_3m_bill_rate",
    "T10Y2Y": "us_10y_2y_spread",
    "T10YIE": "us_10y_breakeven",
    "T5YIFR": "inflation_expectations_5y5y",
    "STLFSI4": "financial_stress_index",
    "EFFR": "effective_fed_funds_rate",
    "DFF": "fed_funds_rate",
    "WALCL": "fed_balance_sheet",
    "RRPONTSYD": "reverse_repo",
    "WTREGEN": "treasury_general_account",
    "M2SL": "m2_money_supply",
    "DTWEXBGS": "trade_weighted_usd",
}

COINMETRICS_METRICS: Dict[str, str] = {
    "PriceUSD": "cm_price_usd",
    "CapMrktCurUSD": "cm_market_cap_usd",
    "CapMVRVCur": "cm_mvrv",
    "AdrActCnt": "cm_active_addresses",
    "TxCnt": "cm_tx_count",
    "HashRate": "cm_hash_rate",
    "FeeTotNtv": "cm_fees_btc",
    "FlowInExUSD": "cm_exchange_inflow_usd",
    "FlowOutExUSD": "cm_exchange_outflow_usd",
    "SplyCur": "cm_supply",
    "SplyExUSD": "cm_exchange_supply_usd",
    "volume_reported_spot_usd_1d": "cm_spot_volume_usd",
    "ROI30d": "cm_roi_30d",
    "ROI1yr": "cm_roi_1y",
}

SOURCE_NOTES = {
    "Yahoo chart API": "Daily OHLCV for the selected asset plus market proxies: BTC, ETH, S&P 500, Nasdaq, VIX, dollar index, gold, and TLT. Unofficial but keyless.",
    "Coin Metrics Community API": "Daily BTC on-chain metrics. Community endpoint is free without an API key for non-commercial use.",
    "FRED CSV downloads": "Daily and lower-frequency macro series from St. Louis Fed graph CSV downloads. No API key is needed for graph CSV files.",
    "Alternative.me Fear & Greed": "Daily crypto fear/greed sentiment values from the public Alternative.me endpoint.",
}


class DataSourceError(RuntimeError):
    pass


def _read_dotenv_secret(name: str) -> str:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return ""
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            if key.strip() == name:
                return value.strip().strip("'\"")
    except OSError:
        return ""
    return ""


def _read_streamlit_secret(name: str) -> str:
    try:
        import streamlit as st  # type: ignore

        return str(st.secrets.get(name, "")).strip()
    except Exception:
        return ""


def fred_api_key() -> str:
    return (
        os.environ.get("FRED_API_KEY", "").strip()
        or _read_dotenv_secret("FRED_API_KEY")
        or _read_streamlit_secret("FRED_API_KEY")
    )


def fred_api_key_configured() -> bool:
    return bool(fred_api_key())


def sanitize_fred_error(message: object) -> str:
    text = str(message)
    key = fred_api_key()
    if key:
        text = text.replace(key, "[redacted]")
    text = re.sub(r"([?&]api_key=)[^&\\s]+", r"\1[redacted]", text)
    text = re.sub(r"(Authorization:\\s*Bearer\\s+)[^\\s]+", r"\1[redacted]", text, flags=re.IGNORECASE)
    return text[:500]


def _cache_path(name: str) -> Path:
    safe = name.replace("/", "_").replace(":", "_")
    return CACHE_DIR / f"{safe}.csv"


def _meta_path(name: str) -> Path:
    return CACHE_DIR / f"{name.replace('/', '_').replace(':', '_')}.json"


def _is_fresh(path: Path, max_age_hours: float) -> bool:
    if not path.exists():
        return False
    age_seconds = time.time() - path.stat().st_mtime
    return age_seconds <= max_age_hours * 3600


def _read_cached(name: str) -> Optional[pd.DataFrame]:
    path = _cache_path(name)
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    return df.set_index("date").sort_index()


def _write_cached(name: str, df: pd.DataFrame, meta: Optional[dict] = None) -> pd.DataFrame:
    path = _cache_path(name)
    out = df.copy()
    out.index.name = "date"
    out.to_csv(path)
    meta_payload = {
        "cached_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(out)),
        "start": out.index.min().date().isoformat() if len(out) else None,
        "end": out.index.max().date().isoformat() if len(out) else None,
    }
    if meta:
        meta_payload.update(meta)
    _meta_path(name).write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")
    return df


def fetch_with_cache(
    name: str,
    fetcher: Callable[[], pd.DataFrame],
    *,
    force: bool = False,
    max_age_hours: float = 12,
) -> pd.DataFrame:
    path = _cache_path(name)
    if not force:
        cached = _read_cached(name)
        if cached is not None and not cached.empty:
            return cached
    if not force and _is_fresh(path, max_age_hours):
        cached = _read_cached(name)
        if cached is not None and not cached.empty:
            return cached

    try:
        df = fetcher()
        if df.empty:
            raise DataSourceError(f"{name} returned no rows")
        return _write_cached(name, df)
    except Exception:
        cached = _read_cached(name)
        if cached is not None and not cached.empty:
            return cached
        raise


def _get_json(url: str, params: Optional[dict] = None) -> dict:
    response = requests.get(
        url,
        params=params,
        timeout=30,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()
    return response.json()


def _unix_day(value: str) -> int:
    dt = pd.Timestamp(value, tz="UTC")
    return int(dt.timestamp())


def fetch_yahoo_chart(symbol: str, start: str = "2014-01-01", end: Optional[str] = None) -> pd.DataFrame:
    end_ts = _unix_day(end) if end else int(pd.Timestamp.utcnow().timestamp())
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(symbol, safe='')}"
    data = _get_json(
        url,
        params={
            "period1": _unix_day(start),
            "period2": end_ts,
            "interval": "1d",
            "events": "history",
        },
    )
    chart = data.get("chart", {})
    if chart.get("error"):
        raise DataSourceError(f"Yahoo chart error for {symbol}: {chart['error']}")
    result = chart.get("result") or []
    if not result:
        raise DataSourceError(f"Yahoo chart returned no data for {symbol}")

    payload = result[0]
    timestamps = payload.get("timestamp") or []
    quote_payload = (payload.get("indicators", {}).get("quote") or [{}])[0]
    frame = pd.DataFrame(quote_payload)
    frame["date"] = pd.to_datetime(timestamps, unit="s", utc=True).date
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.set_index("date").sort_index()
    frame = frame.rename(columns={c: c.lower() for c in frame.columns})
    return frame[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce")


def fetch_btc_price(force: bool = False) -> pd.DataFrame:
    def _fetch() -> pd.DataFrame:
        btc = fetch_yahoo_chart("BTC-USD")
        return btc.rename(
            columns={
                "open": "btc_open",
                "high": "btc_high",
                "low": "btc_low",
                "close": "btc_close",
                "volume": "btc_volume",
            }
        )

    return fetch_with_cache("yahoo_btc_usd", _fetch, force=force, max_age_hours=2)


def fetch_market_proxies(force: bool = False) -> pd.DataFrame:
    def _fetch() -> pd.DataFrame:
        frames = []
        for symbol, column in YAHOO_MARKETS.items():
            chart = fetch_yahoo_chart(symbol)
            frames.append(chart[["close"]].rename(columns={"close": column}))
            time.sleep(0.15)
        return pd.concat(frames, axis=1).sort_index()

    return fetch_with_cache("yahoo_market_proxies", _fetch, force=force, max_age_hours=12)


def _coinmetrics_page(url: str, params: Optional[dict] = None) -> dict:
    response = requests.get(
        url,
        params=params,
        timeout=45,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()
    data = response.json()
    if "error" in data:
        raise DataSourceError(f"Coin Metrics error: {data['error']}")
    return data


def fetch_coinmetrics(force: bool = False, start: str = "2014-01-01") -> pd.DataFrame:
    def _fetch() -> pd.DataFrame:
        metrics = ",".join(COINMETRICS_METRICS.keys())
        params = {
            "assets": "btc",
            "metrics": metrics,
            "frequency": "1d",
            "start_time": start,
            "paging_from": "start",
            "page_size": 10000,
        }
        url = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
        rows = []
        page = _coinmetrics_page(url, params=params)
        rows.extend(page.get("data", []))

        while page.get("next_page_url"):
            time.sleep(0.7)
            page = _coinmetrics_page(page["next_page_url"])
            rows.extend(page.get("data", []))

        if not rows:
            raise DataSourceError("Coin Metrics returned no rows")
        frame = pd.DataFrame(rows)
        frame["date"] = pd.to_datetime(frame["time"], utc=True).dt.tz_localize(None).dt.normalize()
        frame = frame.set_index("date").sort_index()
        keep = [c for c in COINMETRICS_METRICS if c in frame.columns]
        frame = frame[keep].rename(columns=COINMETRICS_METRICS)
        for col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        return frame

    return fetch_with_cache("coinmetrics_btc_daily", _fetch, force=force, max_age_hours=12)


def _parse_fred_observations(frame: pd.DataFrame, series_id: str) -> pd.Series:
    if frame.empty or len(frame.columns) < 2:
        raise DataSourceError(f"FRED returned no data for {series_id}")
    frame = frame.rename(columns={"observation_date": "date", series_id: "value"})
    frame["date"] = pd.to_datetime(frame["date"])
    frame["value"] = pd.to_numeric(frame["value"].replace(".", math.nan), errors="coerce")
    return frame.set_index("date")["value"].sort_index()


def fetch_fred_graph_csv_series(series_id: str) -> pd.Series:
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    response = requests.get(url, params={"id": series_id}, timeout=30, headers={"User-Agent": USER_AGENT})
    response.raise_for_status()
    frame = pd.read_csv(io.StringIO(response.text))
    return _parse_fred_observations(frame, series_id)


def fetch_fred_api_series(series_id: str, start: str = "2014-01-01") -> pd.Series:
    api_key = fred_api_key()
    if not api_key:
        raise DataSourceError("FRED_API_KEY is not set.")
    response = requests.get(
        "https://api.stlouisfed.org/fred/series/observations",
        params={
            "series_id": series_id,
            "file_type": "json",
            "observation_start": start,
            "api_key": api_key,
        },
        headers={
            "User-Agent": USER_AGENT,
            "Authorization": f"Bearer {api_key}",
        },
        timeout=30,
    )
    if response.status_code >= 400:
        raise DataSourceError(f"FRED API request failed for {series_id} with HTTP {response.status_code}.")
    payload = response.json()
    rows = payload.get("observations") or []
    if not rows:
        raise DataSourceError(f"FRED API returned no observations for {series_id}")
    frame = pd.DataFrame(rows).rename(columns={"value": series_id, "date": "observation_date"})
    return _parse_fred_observations(frame[["observation_date", series_id]], series_id)


def _fred_series_detail(series_id: str, column: str, status: str, series: Optional[pd.Series] = None, failure_reason: str = "") -> dict:
    valid = series.dropna() if series is not None else pd.Series(dtype=float)
    return {
        "series_id": series_id,
        "series_name": column,
        "status": status,
        "rows": int(len(valid)),
        "first_date": valid.index.min().date().isoformat() if len(valid) else "",
        "last_date": valid.index.max().date().isoformat() if len(valid) else "",
        "failure_reason": sanitize_fred_error(failure_reason),
    }


def _fred_detail_path(name: str) -> Path:
    return _meta_path(f"{name}_series_status")


def _read_fred_details(name: str, frame: pd.DataFrame) -> list[dict]:
    path = _fred_detail_path(name)
    if path.exists():
        try:
            details = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(details, list):
                return details
        except json.JSONDecodeError:
            pass
    details = []
    for series_id, column in FRED_SERIES.items():
        if column in frame:
            details.append(_fred_series_detail(series_id, column, "worked", frame[column]))
        else:
            details.append(_fred_series_detail(series_id, column, "failed", failure_reason="Series not present in cached FRED API data."))
    return details


def fetch_fred_api_detailed(force: bool = False) -> tuple[pd.DataFrame, list[dict]]:
    cache_name = "fred_macro_api"
    if not force:
        cached = _read_cached(cache_name)
        if cached is not None and not cached.empty:
            return cached, _read_fred_details(cache_name, cached)

    frames = []
    details = []
    for series_id, column in FRED_SERIES.items():
        try:
            series = fetch_fred_api_series(series_id).rename(column)
            frames.append(series.to_frame())
            details.append(_fred_series_detail(series_id, column, "worked", series))
        except Exception as exc:
            details.append(_fred_series_detail(series_id, column, "failed", failure_reason=exc))
        time.sleep(0.1)

    if not frames:
        _fred_detail_path(cache_name).write_text(json.dumps(details, indent=2), encoding="utf-8")
        raise DataSourceError("FRED API returned no usable series.")

    frame = pd.concat(frames, axis=1).sort_index()
    _write_cached(cache_name, frame)
    _fred_detail_path(cache_name).write_text(json.dumps(details, indent=2), encoding="utf-8")
    return frame, details


def _fetch_fred_series_map(series_fetcher: Callable[[str], pd.Series]) -> pd.DataFrame:
    frames = []
    failures = []
    for series_id, column in FRED_SERIES.items():
        try:
            frames.append(series_fetcher(series_id).rename(column).to_frame())
        except Exception as exc:
            failures.append(f"{series_id}: {exc}")
        time.sleep(0.1)
    if not frames:
        detail = "; ".join(failures[:3])
        raise DataSourceError(f"FRED returned no usable series. {detail}")
    return pd.concat(frames, axis=1).sort_index()


def fetch_fred_api(force: bool = False) -> pd.DataFrame:
    frame, _ = fetch_fred_api_detailed(force=force)
    return frame


def fetch_fred(force: bool = False) -> pd.DataFrame:
    def _fetch() -> pd.DataFrame:
        return _fetch_fred_series_map(fetch_fred_graph_csv_series)

    return fetch_with_cache("fred_macro", _fetch, force=force, max_age_hours=24)


def fetch_fear_greed(force: bool = False) -> pd.DataFrame:
    def _fetch() -> pd.DataFrame:
        data = _get_json(
            "https://api.alternative.me/fng/",
            params={"limit": 0, "format": "json"},
        )
        rows = data.get("data") or []
        if not rows:
            raise DataSourceError("Alternative.me returned no fear/greed rows")
        frame = pd.DataFrame(rows)
        frame["date"] = pd.to_datetime(pd.to_numeric(frame["timestamp"]), unit="s", utc=True)
        frame["date"] = frame["date"].dt.tz_localize(None).dt.normalize()
        frame["fear_greed_value"] = pd.to_numeric(frame["value"], errors="coerce")
        frame["fear_greed_label"] = frame["value_classification"].astype(str)
        return frame.set_index("date")[["fear_greed_value", "fear_greed_label"]].sort_index()

    return fetch_with_cache("alternative_fear_greed", _fetch, force=force, max_age_hours=12)


def load_all_data(force: bool = False) -> pd.DataFrame:
    btc = fetch_btc_price(force=force)
    markets = fetch_market_proxies(force=force)
    cm = fetch_coinmetrics(force=force)
    fred = fetch_fred(force=force)
    fear_greed = fetch_fear_greed(force=force)

    frame = pd.concat([btc, markets, cm, fred, fear_greed], axis=1).sort_index()
    frame = frame.loc[frame.index >= pd.Timestamp("2014-09-17")]

    if "btc_close" in frame and "cm_price_usd" in frame:
        frame["btc_close"] = frame["btc_close"].combine_first(frame["cm_price_usd"])
    elif "cm_price_usd" in frame:
        frame["btc_close"] = frame["cm_price_usd"]

    numeric_cols = frame.select_dtypes(include=["number"]).columns
    frame[numeric_cols] = frame[numeric_cols].ffill()
    if "fear_greed_label" in frame:
        frame["fear_greed_label"] = frame["fear_greed_label"].ffill()

    return frame.dropna(subset=["btc_close"]).sort_index()


def cache_metadata() -> pd.DataFrame:
    rows = []
    for meta_file in sorted(CACHE_DIR.glob("*.json")):
        try:
            payload = json.loads(meta_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        payload["cache"] = meta_file.stem
        rows.append(payload)
    return pd.DataFrame(rows)


def source_coverage(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    rows = []
    for col in columns:
        if col not in frame:
            continue
        valid = frame[col].dropna()
        rows.append(
            {
                "column": col,
                "first_valid": valid.index.min().date().isoformat() if len(valid) else None,
                "last_valid": valid.index.max().date().isoformat() if len(valid) else None,
                "coverage_pct": round(float(valid.size / len(frame) * 100), 1) if len(frame) else 0,
            }
        )
    return pd.DataFrame(rows)
