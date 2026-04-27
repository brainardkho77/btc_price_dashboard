from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from xml.etree import ElementTree

import numpy as np
import pandas as pd
import requests

from data_sources import (
    CACHE_DIR,
    USER_AGENT,
    fetch_btc_price,
    fetch_coinmetrics,
    fetch_fear_greed,
    fetch_fred,
    fetch_market_proxies,
)
from features import make_features
from research_config import MIN_SAMPLE_THRESHOLDS, SOURCE_SPECS, ResearchConfig, SourceSpec
from schemas import empty_output_frame, write_schema_csv


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
MANUAL_DATA_DIR = ROOT / "data" / "manual"
BINANCE_RAW_CACHE_DIR = CACHE_DIR / "binance_raw"
BINANCE_ARCHIVE_CACHE_DIR = CACHE_DIR / "binance_archive_metrics"
BINANCE_ARCHIVE_BUCKET_URL = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
BINANCE_ARCHIVE_DOWNLOAD_URL = "https://data.binance.vision"
BINANCE_ARCHIVE_METRICS_PREFIX = "data/futures/um/daily/metrics/BTCUSDT/"
SCHEMA_VERSION = "1.0"

BASELINE_MODELS = {"buy_hold_direction", "momentum_30d", "momentum_90d", "random_permutation"}
LEAKAGE_TOKENS = ("target", "future", "forward", "label", "shifted_target", "target_shift", "future_return", "forward_return")
_BINANCE_HEALTH_ERROR: Optional[str] = None
_BINANCE_HEALTH_OK = False


@dataclass(frozen=True)
class ResearchWindow:
    horizon: int
    window_type: str
    is_official: bool
    test_date: pd.Timestamp
    train_start: pd.Timestamp
    train_end: pd.Timestamp


@dataclass
class DataLayerResult:
    raw: pd.DataFrame
    availability: pd.DataFrame
    warnings: List[str]


@dataclass
class FeatureBuildResult:
    features: pd.DataFrame
    feature_cols: List[str]
    feature_audit: pd.DataFrame
    released_raw: pd.DataFrame


@dataclass(frozen=True)
class ManualDerivativeSpec:
    dataset: str
    metric: str
    filename: str
    required_columns: Tuple[str, ...]
    rename_map: Dict[str, str]
    positive_columns: Tuple[str, ...] = ()
    nonnegative_columns: Tuple[str, ...] = ()
    bounded_columns: Tuple[Tuple[str, float, float], ...] = ()


@dataclass
class ManualCsvResult:
    spec: ManualDerivativeSpec
    status: str
    frame: Optional[pd.DataFrame]
    failure_reason: str
    missing_pct: float


MANUAL_DERIVATIVE_SPECS: Dict[str, ManualDerivativeSpec] = {
    "binance_funding_rate": ManualDerivativeSpec(
        dataset="binance_funding_rate",
        metric="funding_rate",
        filename="btc_funding.csv",
        required_columns=("date", "funding_rate"),
        rename_map={"funding_rate": "binance_funding_rate"},
        bounded_columns=(("funding_rate", -1.0, 1.0),),
    ),
    "binance_open_interest": ManualDerivativeSpec(
        dataset="binance_open_interest",
        metric="open_interest",
        filename="btc_open_interest.csv",
        required_columns=("date", "sum_open_interest", "sum_open_interest_value"),
        rename_map={
            "sum_open_interest": "binance_sum_open_interest",
            "sum_open_interest_value": "binance_sum_open_interest_value",
        },
        nonnegative_columns=("sum_open_interest", "sum_open_interest_value"),
    ),
    "binance_long_short_ratio": ManualDerivativeSpec(
        dataset="binance_long_short_ratio",
        metric="long_short_ratio",
        filename="btc_long_short_ratio.csv",
        required_columns=("date", "long_short_ratio"),
        rename_map={
            "long_short_ratio": "binance_long_short_ratio",
        },
        positive_columns=("long_short_ratio",),
    ),
    "binance_taker_buy_sell_ratio": ManualDerivativeSpec(
        dataset="binance_taker_buy_sell_ratio",
        metric="taker_buy_sell_ratio",
        filename="btc_taker_buy_sell_ratio.csv",
        required_columns=("date", "taker_buy_sell_ratio"),
        rename_map={
            "taker_buy_sell_ratio": "binance_taker_buy_sell_ratio",
        },
        positive_columns=("taker_buy_sell_ratio",),
    ),
    "binance_basis": ManualDerivativeSpec(
        dataset="binance_basis",
        metric="basis",
        filename="btc_basis.csv",
        required_columns=("date", "basis", "basis_rate", "annualized_basis_rate"),
        rename_map={
            "basis": "binance_basis",
            "basis_rate": "binance_basis_rate",
            "annualized_basis_rate": "binance_annualized_basis_rate",
        },
        bounded_columns=(("basis_rate", -1.0, 1.0), ("annualized_basis_rate", -10.0, 10.0)),
    ),
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def date_str(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return pd.Timestamp(value).date().isoformat()


def _cache_path(dataset: str) -> Path:
    return CACHE_DIR / f"{dataset.replace('/', '_').replace(':', '_')}.csv"


def _read_cached_dataset(dataset: str) -> Optional[pd.DataFrame]:
    path = _cache_path(dataset)
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path, parse_dates=["date"])
    except Exception:
        return None
    if "date" not in frame.columns or frame.empty:
        return None
    return frame.set_index("date").sort_index()


def _write_cached_dataset(dataset: str, frame: pd.DataFrame) -> pd.DataFrame:
    CACHE_DIR.mkdir(exist_ok=True)
    out = frame.copy()
    out.index.name = "date"
    out.to_csv(_cache_path(dataset))
    return frame


def _availability_record(
    run_id: str,
    generated_at: str,
    spec: SourceSpec,
    status: str,
    frame: Optional[pd.DataFrame] = None,
    failure_reason: str = "",
    used_override: Optional[bool] = None,
) -> dict:
    rows = 0
    first_date = ""
    last_date = ""
    available_fields: List[str] = []
    if frame is not None and not frame.empty:
        rows = int(len(frame))
        first_date = date_str(frame.index.min())
        last_date = date_str(frame.index.max())
        available_fields = [str(c) for c in frame.columns]

    return {
        "run_id": run_id,
        "generated_at": generated_at,
        "source": spec.source,
        "endpoint": spec.endpoint,
        "dataset": spec.dataset,
        "status": status,
        "requested_fields": "|".join(spec.requested_fields),
        "available_fields": "|".join(available_fields),
        "rows": rows,
        "first_date": first_date,
        "last_date": last_date,
        "failure_reason": failure_reason,
        "is_used_in_model": bool(spec.is_used_in_model if used_override is None else used_override),
        "revision_warning": spec.revision_warning,
    }


def initial_data_availability(run_id: str, generated_at: str) -> pd.DataFrame:
    rows = []
    for spec in SOURCE_SPECS:
        status = "unavailable" if spec.dataset == "spot_btc_etf_flows" else "skipped"
        reason = "No free source configured; not fabricated." if spec.dataset == "spot_btc_etf_flows" else "Not attempted yet."
        rows.append(_availability_record(run_id, generated_at, spec, status, failure_reason=reason, used_override=False))
    return pd.DataFrame(rows)


def write_initial_data_availability(output_dir: Path, run_id: str, generated_at: str) -> None:
    write_schema_csv("data_availability.csv", initial_data_availability(run_id, generated_at), output_dir)


def _source_spec(dataset: str) -> SourceSpec:
    for spec in SOURCE_SPECS:
        if spec.dataset == dataset:
            return spec
    raise KeyError(dataset)


def _load_source(
    run_id: str,
    generated_at: str,
    dataset: str,
    fetcher: Callable[[], pd.DataFrame],
    *,
    quick: bool,
    allow_quick_fetch: bool,
    used_override: Optional[bool] = None,
) -> Tuple[Optional[pd.DataFrame], dict]:
    spec = _source_spec(dataset)
    cached = _read_cached_dataset(dataset)
    if quick and cached is not None and not cached.empty:
        return cached, _availability_record(
            run_id,
            generated_at,
            spec,
            "worked",
            cached,
            failure_reason="Loaded cached data in quick mode.",
            used_override=used_override,
        )
    if quick and not allow_quick_fetch:
        return None, _availability_record(
            run_id,
            generated_at,
            spec,
            "skipped",
            failure_reason="Skipped in quick mode because no cache was available.",
            used_override=False,
        )

    try:
        frame = fetcher()
        if frame is None or frame.empty:
            raise ValueError("source returned no rows")
        return frame, _availability_record(run_id, generated_at, spec, "worked", frame, used_override=used_override)
    except Exception as exc:
        if cached is not None and not cached.empty:
            return cached, _availability_record(
                run_id,
                generated_at,
                spec,
                "worked",
                cached,
                failure_reason=f"Live fetch failed, loaded cached data: {exc}",
                used_override=used_override,
            )
        return None, _availability_record(
            run_id,
            generated_at,
            spec,
            "failed",
            failure_reason=str(exc)[:500],
            used_override=False,
        )


def _load_derivative_source(
    run_id: str,
    generated_at: str,
    dataset: str,
    fetcher: Callable[[], pd.DataFrame],
    *,
    quick: bool,
) -> Tuple[Optional[pd.DataFrame], List[dict]]:
    api_spec = _source_spec(dataset)
    manual_spec = _manual_source_spec(dataset)
    records: List[dict] = []

    cached = _read_cached_dataset(dataset)
    if quick and cached is not None and not cached.empty:
        records.append(
            _availability_record(
                run_id,
                generated_at,
                api_spec,
                "worked",
                cached,
                failure_reason="Loaded cached derivatives data in quick mode.",
                used_override=True,
            )
        )
        manual_result = validate_manual_derivative_dataset(dataset)
        records.append(
            _availability_record(
                run_id,
                generated_at,
                manual_spec,
                manual_result.status,
                manual_result.frame,
                failure_reason=manual_result.failure_reason,
                used_override=False,
            )
        )
        return cached, records

    if quick:
        records.append(
            _availability_record(
                run_id,
                generated_at,
                api_spec,
                "skipped",
                failure_reason="Skipped live Binance fetch in quick mode because no cache was available.",
                used_override=False,
            )
        )
        manual_result = validate_manual_derivative_dataset(dataset)
        manual_used = manual_result.status == "worked" and manual_result.frame is not None
        records.append(
            _availability_record(
                run_id,
                generated_at,
                manual_spec,
                manual_result.status,
                manual_result.frame,
                failure_reason=manual_result.failure_reason,
                used_override=manual_used,
            )
        )
        return (manual_result.frame if manual_used else None), records

    api_frame: Optional[pd.DataFrame] = None
    api_error = ""
    try:
        fetched = fetcher()
        if fetched is None or fetched.empty:
            raise ValueError("Binance endpoint returned no rows")
        api_frame = fetched
    except Exception as exc:
        api_error = str(exc)[:500]

    manual_result = validate_manual_derivative_dataset(dataset)
    manual_valid = manual_result.status == "worked" and manual_result.frame is not None
    manual_used = False
    if manual_valid and api_frame is None:
        manual_used = True
    elif manual_valid and api_frame is not None:
        api_first = api_frame.index.min()
        manual_first = manual_result.frame.index.min()
        manual_rows = len(manual_result.frame)
        api_rows = len(api_frame)
        manual_used = bool(manual_rows > max(api_rows * 3, api_rows + 365) or manual_first < api_first - pd.Timedelta(days=180))

    if api_frame is not None:
        records.append(
            _availability_record(
                run_id,
                generated_at,
                api_spec,
                "worked",
                api_frame,
                failure_reason="Valid manual CSV had broader historical coverage and was used in model." if manual_used else "",
                used_override=not manual_used,
            )
        )
    else:
        records.append(
            _availability_record(
                run_id,
                generated_at,
                api_spec,
                "failed",
                failure_reason=api_error,
                used_override=False,
            )
        )
    records.append(
        _availability_record(
            run_id,
            generated_at,
            manual_spec,
            manual_result.status,
            manual_result.frame,
            failure_reason=manual_result.failure_reason,
            used_override=manual_used,
        )
    )
    if manual_used:
        print(f"[binance] using validated manual fallback {manual_result.spec.filename}")
        _write_cached_dataset(dataset, manual_result.frame)
        return manual_result.frame, records
    if api_frame is not None:
        return api_frame, records
    return None, records


def fetch_coinbase_btc_usd(start_date: str, end_date: Optional[str]) -> pd.DataFrame:
    start = pd.Timestamp(start_date, tz="UTC")
    end = pd.Timestamp(end_date, tz="UTC") if end_date else pd.Timestamp.utcnow().tz_convert("UTC").normalize()
    cursor = start
    rows: List[list] = []
    while cursor < end:
        chunk_end = min(cursor + pd.Timedelta(days=299), end)
        response = requests.get(
            "https://api.exchange.coinbase.com/products/BTC-USD/candles",
            params={
                "granularity": 86400,
                "start": cursor.isoformat(),
                "end": chunk_end.isoformat(),
            },
            headers={"User-Agent": USER_AGENT},
            timeout=25,
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict) and payload.get("message"):
            raise RuntimeError(payload["message"])
        rows.extend(payload)
        cursor = chunk_end + pd.Timedelta(days=1)
        time.sleep(0.05)

    if not rows:
        raise RuntimeError("Coinbase returned no candles")
    frame = pd.DataFrame(rows, columns=["timestamp", "low", "high", "open", "close", "volume"])
    frame["date"] = pd.to_datetime(frame["timestamp"], unit="s", utc=True).dt.tz_localize(None).dt.normalize()
    frame = frame.drop_duplicates("date").set_index("date").sort_index()
    frame = frame[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce")
    frame = frame.rename(
        columns={
            "open": "btc_open",
            "high": "btc_high",
            "low": "btc_low",
            "close": "btc_close",
            "volume": "btc_volume",
        }
    )
    return _write_cached_dataset("coinbase_btc_usd_candles", frame)


def fetch_coingecko_btc_usd(start_date: str, end_date: Optional[str]) -> pd.DataFrame:
    start_ts = int(pd.Timestamp(start_date, tz="UTC").timestamp())
    end_ts = int((pd.Timestamp(end_date, tz="UTC") if end_date else pd.Timestamp.utcnow()).timestamp())
    response = requests.get(
        "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range",
        params={"vs_currency": "usd", "from": start_ts, "to": end_ts},
        headers={"User-Agent": USER_AGENT},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    prices = data.get("prices") or []
    if not prices:
        raise RuntimeError("CoinGecko returned no prices")
    frame = pd.DataFrame(prices, columns=["timestamp_ms", "btc_close"])
    frame["date"] = pd.to_datetime(frame["timestamp_ms"], unit="ms", utc=True).dt.tz_localize(None).dt.normalize()
    frame = frame.groupby("date", as_index=True)["btc_close"].last().to_frame().sort_index()
    return _write_cached_dataset("coingecko_btc_usd", frame)


def fetch_defillama_stablecoins() -> pd.DataFrame:
    response = requests.get(
        "https://stablecoins.llama.fi/stablecoincharts/all",
        headers={"User-Agent": USER_AGENT},
        timeout=45,
    )
    response.raise_for_status()
    rows = response.json()
    if not rows:
        raise RuntimeError("DefiLlama returned no stablecoin rows")
    parsed = []
    for row in rows:
        date = pd.to_datetime(int(row["date"]), unit="s", utc=True).tz_localize(None).normalize()
        total = (row.get("totalCirculatingUSD") or {}).get("peggedUSD")
        if total is None:
            total = (row.get("totalCirculating") or {}).get("peggedUSD")
        parsed.append({"date": date, "stablecoin_total_circulating_usd": total})
    frame = pd.DataFrame(parsed).set_index("date").sort_index()
    frame["stablecoin_total_circulating_usd"] = pd.to_numeric(frame["stablecoin_total_circulating_usd"], errors="coerce")
    return _write_cached_dataset("defillama_stablecoins", frame)


def _request_json_with_retries(endpoint: str, params: Optional[dict], *, attempts: int = 3, timeout: int = 25) -> object:
    last_error: Optional[Exception] = None
    for attempt in range(attempts):
        try:
            response = requests.get(endpoint, params=params, headers={"User-Agent": USER_AGENT}, timeout=timeout)
            if response.status_code in {418, 429} and attempt < attempts - 1:
                retry_after = response.headers.get("Retry-After")
                sleep_seconds = float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() else min(15, 2 ** (attempt + 2))
                print(f"[binance] rate limited, retrying in {sleep_seconds:.1f}s: {endpoint}")
                time.sleep(sleep_seconds)
                continue
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
            if attempt < attempts - 1:
                sleep_seconds = min(15, 2 ** (attempt + 1))
                print(f"[binance] request failed, retrying in {sleep_seconds}s: {endpoint} {exc}")
                time.sleep(sleep_seconds)
    raise RuntimeError(str(last_error))


def _binance_health_message() -> str:
    global _BINANCE_HEALTH_ERROR, _BINANCE_HEALTH_OK
    if _BINANCE_HEALTH_OK:
        return "ok"
    if _BINANCE_HEALTH_ERROR:
        return _BINANCE_HEALTH_ERROR
    try:
        payload = _request_json_with_retries("https://fapi.binance.com/fapi/v1/time", None, attempts=2, timeout=20)
        if not isinstance(payload, dict) or "serverTime" not in payload:
            raise RuntimeError(f"unexpected Binance health response: {payload}")
        _BINANCE_HEALTH_OK = True
        return "ok"
    except Exception as exc:
        _BINANCE_HEALTH_ERROR = f"Binance health check failed: {exc}"
        return _BINANCE_HEALTH_ERROR


def _write_binance_raw_cache(dataset: str, params: dict, payload: object) -> None:
    BINANCE_RAW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(json.dumps(params, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]
    path = BINANCE_RAW_CACHE_DIR / f"{dataset}_{key}.json"
    path.write_text(
        json.dumps(
            {
                "cached_at": utc_now_iso(),
                "dataset": dataset,
                "params": params,
                "payload": payload,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


def _list_binance_archive_metric_keys() -> List[str]:
    keys: List[str] = []
    continuation: Optional[str] = None
    while True:
        params = {
            "list-type": "2",
            "prefix": BINANCE_ARCHIVE_METRICS_PREFIX,
            "max-keys": 1000,
        }
        if continuation:
            params["continuation-token"] = continuation
        response = requests.get(BINANCE_ARCHIVE_BUCKET_URL, params=params, headers={"User-Agent": USER_AGENT}, timeout=60)
        response.raise_for_status()
        root = ElementTree.fromstring(response.text)
        namespace = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
        for key_node in root.findall("s3:Contents/s3:Key", namespace):
            key = key_node.text or ""
            if key.endswith(".zip") and "/BTCUSDT-metrics-" in key:
                keys.append(key)
        truncated = (root.findtext("s3:IsTruncated", default="false", namespaces=namespace) or "false").lower() == "true"
        continuation = root.findtext("s3:NextContinuationToken", default="", namespaces=namespace) or None
        if not truncated or not continuation:
            break
    return sorted(set(keys))


def _download_binance_archive_key(key: str, *, refresh: bool) -> Path:
    BINANCE_ARCHIVE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = BINANCE_ARCHIVE_CACHE_DIR / Path(key).name
    # Binance archive files are immutable daily snapshots; keep local copies and
    # only download files that are not already cached.
    if path.exists() and path.stat().st_size > 0:
        return path
    url = f"{BINANCE_ARCHIVE_DOWNLOAD_URL}/{key}"
    last_error: Optional[Exception] = None
    for attempt in range(4):
        try:
            response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=60)
            if response.status_code in {418, 429} and attempt < 3:
                retry_after = response.headers.get("Retry-After")
                sleep_seconds = float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() else min(30, 2 ** (attempt + 2))
                print(f"[binance-archive] rate limited, retrying in {sleep_seconds:.1f}s: {Path(key).name}")
                time.sleep(sleep_seconds)
                continue
            response.raise_for_status()
            path.write_bytes(response.content)
            return path
        except Exception as exc:
            last_error = exc
            if attempt < 3:
                sleep_seconds = min(30, 2 ** (attempt + 1))
                print(f"[binance-archive] download retry in {sleep_seconds}s: {Path(key).name} {exc}")
                time.sleep(sleep_seconds)
    raise RuntimeError(f"Binance archive download failed for {key}: {last_error}")


def _parse_binance_archive_zip(path: Path) -> pd.DataFrame:
    try:
        with zipfile.ZipFile(path) as archive:
            names = [name for name in archive.namelist() if name.endswith(".csv")]
            if not names:
                return pd.DataFrame()
            with archive.open(names[0]) as handle:
                frame = pd.read_csv(handle)
    except zipfile.BadZipFile:
        path.unlink(missing_ok=True)
        raise
    if frame.empty or "create_time" not in frame:
        return pd.DataFrame()
    frame["date"] = pd.to_datetime(frame["create_time"], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
    numeric_columns = [
        "sum_open_interest",
        "sum_open_interest_value",
        "count_long_short_ratio",
        "sum_taker_long_short_vol_ratio",
    ]
    available = [col for col in numeric_columns if col in frame.columns]
    if not available:
        return pd.DataFrame()
    for col in available:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.groupby("date", as_index=True)[available].mean().sort_index()


def fetch_binance_archive_metrics(*, refresh: bool) -> pd.DataFrame:
    cached = _read_cached_dataset("binance_archive_metrics_daily")
    if cached is not None and not cached.empty and not refresh:
        return cached
    keys = _list_binance_archive_metric_keys()
    if not keys:
        raise RuntimeError("Binance public data archive returned no BTCUSDT metric keys.")

    frames: List[pd.DataFrame] = []
    failures: List[str] = []
    total = len(keys)
    print(f"[binance-archive] found {total} daily BTCUSDT metrics files")
    max_workers = 8
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_key = {executor.submit(_download_binance_archive_key, key, refresh=refresh): key for key in keys}
        for idx, future in enumerate(as_completed(future_to_key), start=1):
            key = future_to_key[future]
            try:
                path = future.result()
                parsed = _parse_binance_archive_zip(path)
                if not parsed.empty:
                    frames.append(parsed)
            except Exception as exc:
                failures.append(f"{Path(key).name}: {exc}")
            if idx == 1 or idx % 250 == 0 or idx == total:
                print(f"[binance-archive] processed {idx}/{total} files")

    if not frames:
        detail = "; ".join(failures[:3])
        raise RuntimeError(f"Binance public data archive metrics produced no rows. {detail}")
    if failures:
        print(f"[binance-archive] skipped {len(failures)} files with parse/download failures")
    daily = pd.concat(frames).sort_index()
    daily = daily[~daily.index.duplicated(keep="last")]
    return _write_cached_dataset("binance_archive_metrics_daily", daily)


def recover_binance_archive_manual_csvs(*, refresh: bool) -> List[str]:
    metrics = fetch_binance_archive_metrics(refresh=refresh)
    warnings: List[str] = []
    conversions = {
        "binance_open_interest": {
            "columns": {
                "sum_open_interest": "sum_open_interest",
                "sum_open_interest_value": "sum_open_interest_value",
            },
        },
        "binance_long_short_ratio": {
            "columns": {
                "count_long_short_ratio": "long_short_ratio",
            },
        },
        "binance_taker_buy_sell_ratio": {
            "columns": {
                "sum_taker_long_short_vol_ratio": "taker_buy_sell_ratio",
            },
        },
    }
    for dataset, conversion in conversions.items():
        spec = MANUAL_DERIVATIVE_SPECS[dataset]
        source_columns = list(conversion["columns"].keys())
        if not all(col in metrics.columns for col in source_columns):
            warnings.append(f"Binance archive lacked required columns for {dataset}; manual CSV was not updated.")
            continue
        out = metrics[source_columns].rename(columns=conversion["columns"]).dropna(how="all").copy()
        if out.empty:
            warnings.append(f"Binance archive had no usable rows for {dataset}; manual CSV was not updated.")
            continue
        out.index.name = "date"
        manual_path = MANUAL_DATA_DIR / spec.filename
        MANUAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
        out.reset_index().to_csv(manual_path, index=False)
        result = validate_manual_derivative_csv(manual_path, spec)
        if result.status != "worked":
            warnings.append(f"Recovered {spec.filename} failed validation: {result.failure_reason}")
        else:
            warnings.append(
                f"Recovered {spec.filename} from Binance public data archive with {len(result.frame)} daily rows "
                f"from {date_str(result.frame.index.min())} to {date_str(result.frame.index.max())}."
            )
    for dataset in ["binance_funding_rate", "binance_basis"]:
        spec = MANUAL_DERIVATIVE_SPECS[dataset]
        result = validate_manual_derivative_dataset(dataset)
        if result.status != "worked":
            warnings.append(f"Binance public data archive did not provide {spec.metric}; {spec.filename} remains {result.status}.")
    return warnings


def _fetch_binance_json(endpoint: str, params: dict, *, dataset: str) -> list:
    health = _binance_health_message()
    attempts = 4 if health == "ok" else 1
    timeout = 45 if health == "ok" else 12
    try:
        payload = _request_json_with_retries(endpoint, params, attempts=attempts, timeout=timeout)
    except Exception as exc:
        raise RuntimeError(f"{exc}; health={health}") from exc
    _write_binance_raw_cache(dataset, params, payload)
    if isinstance(payload, dict) and payload.get("code"):
        raise RuntimeError(f"{payload.get('msg', payload)}; health={health}")
    if not isinstance(payload, list):
        raise RuntimeError(f"unexpected Binance response: {payload}")
    return payload


def _fetch_binance_chunked(
    endpoint: str,
    base_params: dict,
    *,
    dataset: str,
    start: Optional[str] = "2019-01-01",
    recent_days: Optional[int] = None,
    chunk_days: int = 30,
    limit: int = 500,
) -> list:
    rows: List[dict] = []
    end = pd.Timestamp.utcnow().tz_convert("UTC").normalize()
    if recent_days is not None:
        cursor = end - pd.Timedelta(days=recent_days)
    else:
        cursor = pd.Timestamp(start, tz="UTC") if start else end - pd.Timedelta(days=30)
    while cursor <= end:
        chunk_end = min(cursor + pd.Timedelta(days=chunk_days), end)
        params = {
            **base_params,
            "startTime": int(cursor.timestamp() * 1000),
            "endTime": int(chunk_end.timestamp() * 1000),
            "limit": limit,
        }
        chunk = _fetch_binance_json(endpoint, params, dataset=dataset)
        rows.extend(chunk)
        cursor = chunk_end + pd.Timedelta(milliseconds=1)
        time.sleep(0.35)
    return rows


def validate_manual_derivative_csv(path: Path, spec: ManualDerivativeSpec) -> ManualCsvResult:
    if not path.exists():
        return ManualCsvResult(spec, "skipped", None, "Manual CSV file does not exist.", 1.0)
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        return ManualCsvResult(spec, "failed", None, f"Manual CSV could not be read: {exc}", 1.0)
    missing = [col for col in spec.required_columns if col not in frame.columns]
    if missing:
        return ManualCsvResult(spec, "failed", None, f"Manual CSV missing required columns: {', '.join(missing)}", 1.0)
    if frame.empty:
        return ManualCsvResult(spec, "skipped", None, "Manual CSV is an empty template.", 1.0)

    work = frame.loc[:, list(spec.required_columns)].copy()
    parsed_dates = pd.to_datetime(work["date"], errors="coerce", utc=True)
    if parsed_dates.isna().any():
        return ManualCsvResult(spec, "failed", None, "Manual CSV contains unparsable dates.", 1.0)
    normalized_dates = parsed_dates.dt.tz_localize(None).dt.normalize()
    if normalized_dates.duplicated().any():
        return ManualCsvResult(spec, "failed", None, "Manual CSV contains duplicate dates.", 1.0)
    work["date"] = normalized_dates
    work = work.sort_values("date")
    if not work["date"].is_monotonic_increasing:
        return ManualCsvResult(spec, "failed", None, "Manual CSV dates are not monotonic after sorting.", 1.0)

    numeric_columns = [col for col in spec.required_columns if col != "date"]
    for col in numeric_columns:
        numeric = pd.to_numeric(work[col], errors="coerce")
        if numeric.isna().any() and work[col].notna().any():
            return ManualCsvResult(spec, "failed", None, f"Manual CSV column {col} contains non-numeric values.", 1.0)
        work[col] = numeric

    for col in spec.positive_columns:
        if (work[col].dropna() <= 0).any():
            return ManualCsvResult(spec, "failed", None, f"Manual CSV column {col} contains non-positive values.", 1.0)
    for col in spec.nonnegative_columns:
        if (work[col].dropna() < 0).any():
            return ManualCsvResult(spec, "failed", None, f"Manual CSV column {col} contains negative values.", 1.0)
    for col, low, high in spec.bounded_columns:
        valid = work[col].dropna()
        if ((valid < low) | (valid > high)).any():
            return ManualCsvResult(spec, "failed", None, f"Manual CSV column {col} has values outside [{low}, {high}].", 1.0)

    out = work.set_index("date").rename(columns=spec.rename_map)
    missing_pct = float(out.isna().mean().mean()) if not out.empty else 1.0
    out = out.dropna(how="all").sort_index()
    if out.empty:
        return ManualCsvResult(spec, "skipped", None, "Manual CSV has no rows with numeric data.", missing_pct)
    return ManualCsvResult(spec, "worked", out, "", missing_pct)


def validate_manual_derivative_dataset(dataset: str) -> ManualCsvResult:
    spec = MANUAL_DERIVATIVE_SPECS[dataset]
    return validate_manual_derivative_csv(MANUAL_DATA_DIR / spec.filename, spec)


def _manual_source_spec(dataset: str) -> SourceSpec:
    spec = MANUAL_DERIVATIVE_SPECS[dataset]
    return SourceSpec(
        source="manual_csv",
        endpoint=str(MANUAL_DATA_DIR / spec.filename),
        dataset=dataset,
        requested_fields=[col for col in spec.required_columns if col != "date"],
        source_group="manual_csv",
        is_used_in_model=False,
    )


def fetch_binance_funding_rate() -> pd.DataFrame:
    rows = _fetch_binance_chunked(
        "https://fapi.binance.com/fapi/v1/fundingRate",
        {"symbol": "BTCUSDT"},
        dataset="binance_funding_rate",
        chunk_days=30,
        limit=1000,
    )
    if not rows:
        raise RuntimeError("Binance funding returned no rows")
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["fundingTime"], unit="ms", utc=True).dt.tz_localize(None).dt.normalize()
    frame["binance_funding_rate"] = pd.to_numeric(frame["fundingRate"], errors="coerce")
    daily = frame.groupby("date", as_index=True)["binance_funding_rate"].mean().to_frame().sort_index()
    return _write_cached_dataset("binance_funding_rate", daily)


def fetch_binance_open_interest() -> pd.DataFrame:
    rows = _fetch_binance_chunked(
        "https://fapi.binance.com/futures/data/openInterestHist",
        {"symbol": "BTCUSDT", "period": "1d"},
        dataset="binance_open_interest",
        recent_days=29,
        chunk_days=7,
        limit=500,
    )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("Binance open interest returned no rows")
    frame["date"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True).dt.tz_localize(None).dt.normalize()
    out = pd.DataFrame(index=frame["date"])
    out["binance_sum_open_interest"] = pd.to_numeric(frame.get("sumOpenInterest"), errors="coerce").to_numpy()
    out["binance_sum_open_interest_value"] = pd.to_numeric(frame.get("sumOpenInterestValue"), errors="coerce").to_numpy()
    return _write_cached_dataset("binance_open_interest", out.sort_index())


def fetch_binance_long_short_ratio() -> pd.DataFrame:
    rows = _fetch_binance_chunked(
        "https://fapi.binance.com/futures/data/globalLongShortAccountRatio",
        {"symbol": "BTCUSDT", "period": "1d"},
        dataset="binance_long_short_ratio",
        recent_days=29,
        chunk_days=7,
        limit=500,
    )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("Binance long/short returned no rows")
    frame["date"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True).dt.tz_localize(None).dt.normalize()
    out = pd.DataFrame(index=frame["date"])
    out["binance_long_short_ratio"] = pd.to_numeric(frame.get("longShortRatio"), errors="coerce").to_numpy()
    out["binance_long_account"] = pd.to_numeric(frame.get("longAccount"), errors="coerce").to_numpy()
    out["binance_short_account"] = pd.to_numeric(frame.get("shortAccount"), errors="coerce").to_numpy()
    return _write_cached_dataset("binance_long_short_ratio", out.sort_index())


def fetch_binance_taker_ratio() -> pd.DataFrame:
    rows = _fetch_binance_chunked(
        "https://fapi.binance.com/futures/data/takerlongshortRatio",
        {"symbol": "BTCUSDT", "period": "1d"},
        dataset="binance_taker_buy_sell_ratio",
        recent_days=29,
        chunk_days=7,
        limit=500,
    )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("Binance taker ratio returned no rows")
    frame["date"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True).dt.tz_localize(None).dt.normalize()
    out = pd.DataFrame(index=frame["date"])
    out["binance_taker_buy_sell_ratio"] = pd.to_numeric(frame.get("buySellRatio"), errors="coerce").to_numpy()
    out["binance_taker_buy_volume"] = pd.to_numeric(frame.get("buyVol"), errors="coerce").to_numpy()
    out["binance_taker_sell_volume"] = pd.to_numeric(frame.get("sellVol"), errors="coerce").to_numpy()
    return _write_cached_dataset("binance_taker_buy_sell_ratio", out.sort_index())


def fetch_binance_basis() -> pd.DataFrame:
    rows = _fetch_binance_chunked(
        "https://fapi.binance.com/futures/data/basis",
        {"pair": "BTCUSDT", "contractType": "PERPETUAL", "period": "1d"},
        dataset="binance_basis",
        recent_days=29,
        chunk_days=7,
        limit=500,
    )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("Binance basis returned no rows")
    frame["date"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True).dt.tz_localize(None).dt.normalize()
    out = pd.DataFrame(index=frame["date"])
    out["binance_basis"] = pd.to_numeric(frame.get("basis"), errors="coerce").to_numpy()
    out["binance_basis_rate"] = pd.to_numeric(frame.get("basisRate"), errors="coerce").to_numpy()
    out["binance_annualized_basis_rate"] = pd.to_numeric(frame.get("annualizedBasisRate"), errors="coerce").to_numpy()
    return _write_cached_dataset("binance_basis", out.sort_index())


def load_research_data(
    config: ResearchConfig,
    *,
    run_id: str,
    generated_at: str,
    refresh: bool,
    quick: bool,
    output_dir: Path,
) -> DataLayerResult:
    warnings: List[str] = []
    records: List[dict] = []
    effective_start = config.quick_start_date if quick else config.start_date

    coinbase, record = _load_source(
        run_id,
        generated_at,
        "coinbase_btc_usd_candles",
        lambda: fetch_coinbase_btc_usd(effective_start, config.end_date),
        quick=quick and not refresh,
        allow_quick_fetch=True,
    )
    records.append(record)

    yahoo_btc, record = _load_source(
        run_id,
        generated_at,
        "yahoo_btc_usd",
        lambda: fetch_btc_price(force=refresh),
        quick=quick and not refresh,
        allow_quick_fetch=True,
    )
    records.append(record)

    if coinbase is not None and len(coinbase.dropna(subset=["btc_close"])) >= 365:
        price = coinbase.copy()
        if yahoo_btc is not None:
            price = price.combine_first(yahoo_btc)
    elif yahoo_btc is not None:
        price = yahoo_btc.copy()
        warnings.append("Coinbase price was unavailable or sparse; Yahoo BTC-USD was used as fallback.")
    else:
        coingecko, record = _load_source(
            run_id,
            generated_at,
            "coingecko_btc_usd",
            lambda: fetch_coingecko_btc_usd(effective_start, config.end_date),
            quick=quick and not refresh,
            allow_quick_fetch=True,
            used_override=True,
        )
        records.append(record)
        if coingecko is None:
            raise RuntimeError("No BTC price source worked; cannot build research dataset.")
        price = coingecko.copy()

    if "coingecko_btc_usd" not in [r["dataset"] for r in records]:
        records.append(
            _availability_record(
                run_id,
                generated_at,
                _source_spec("coingecko_btc_usd"),
                "skipped",
                failure_reason="Skipped because Coinbase/Yahoo price data was available.",
                used_override=False,
            )
        )

    loaders: List[Tuple[str, Callable[[], pd.DataFrame], bool]] = [
        ("yahoo_market_proxies", lambda: fetch_market_proxies(force=refresh), True),
        ("fred_macro", lambda: fetch_fred(force=refresh), True),
        ("coinmetrics_btc_daily", lambda: fetch_coinmetrics(force=refresh), True),
        ("alternative_fear_greed", lambda: fetch_fear_greed(force=refresh), True),
        ("defillama_stablecoins", fetch_defillama_stablecoins, True),
    ]

    frames: List[pd.DataFrame] = [price]
    for dataset, fetcher, allow_quick_fetch in loaders:
        frame, record = _load_source(
            run_id,
            generated_at,
            dataset,
            fetcher,
            quick=quick and not refresh,
            allow_quick_fetch=allow_quick_fetch,
        )
        records.append(record)
        if frame is not None and not frame.empty:
            frames.append(frame)

    if refresh and not quick:
        try:
            warnings.extend(recover_binance_archive_manual_csvs(refresh=refresh))
        except Exception as exc:
            warnings.append(f"Binance public data archive recovery failed: {exc}")

    derivative_loaders: List[Tuple[str, Callable[[], pd.DataFrame]]] = [
        ("binance_funding_rate", fetch_binance_funding_rate),
        ("binance_open_interest", fetch_binance_open_interest),
        ("binance_long_short_ratio", fetch_binance_long_short_ratio),
        ("binance_taker_buy_sell_ratio", fetch_binance_taker_ratio),
        ("binance_basis", fetch_binance_basis),
    ]
    for dataset, fetcher in derivative_loaders:
        frame, derivative_records = _load_derivative_source(
            run_id,
            generated_at,
            dataset,
            fetcher,
            quick=quick and not refresh,
        )
        records.extend(derivative_records)
        if frame is not None and not frame.empty:
            frames.append(frame)

    manual_records = [record for record in records if record["source"] == "manual_csv" and record["dataset"].startswith("binance_")]
    if manual_records:
        manual_status = "worked" if any(record["status"] == "worked" for record in manual_records) else "failed" if any(record["status"] == "failed" for record in manual_records) else "skipped"
        manual_reason = "Manual derivative CSV status is recorded per metric."
    else:
        manual_status = "skipped"
        manual_reason = "No manual CSV inputs were configured for this run."
    records.append(
        _availability_record(
            run_id,
            generated_at,
            _source_spec("manual_csv"),
            manual_status,
            failure_reason=manual_reason,
            used_override=any(record["is_used_in_model"] for record in manual_records),
        )
    )
    records.append(
        _availability_record(
            run_id,
            generated_at,
            _source_spec("spot_btc_etf_flows"),
            "unavailable",
            failure_reason="No free source configured; ETF flows were not fabricated.",
            used_override=False,
        )
    )

    raw = pd.concat(frames, axis=1).sort_index()
    raw = raw.loc[raw.index >= pd.Timestamp(effective_start)]
    if config.end_date:
        raw = raw.loc[raw.index <= pd.Timestamp(config.end_date)]

    if "btc_close" not in raw and "cm_price_usd" in raw:
        raw["btc_close"] = raw["cm_price_usd"]
    elif "btc_close" in raw and "cm_price_usd" in raw:
        raw["btc_close"] = raw["btc_close"].combine_first(raw["cm_price_usd"])

    numeric_cols = raw.select_dtypes(include=["number"]).columns
    raw[numeric_cols] = raw[numeric_cols].ffill()
    if "fear_greed_label" in raw:
        raw["fear_greed_label"] = raw["fear_greed_label"].ffill()
    raw = raw.dropna(subset=["btc_close"]).sort_index()

    availability = pd.DataFrame(records)
    write_schema_csv("data_availability.csv", availability, output_dir)
    if any(availability["revision_warning"].astype(str).str.len() > 0):
        warnings.append("FRED and Coin Metrics are revised historical data unless point-in-time vintage data is added.")
    return DataLayerResult(raw=raw, availability=availability, warnings=warnings)


def raw_column_source_group(column: str) -> str:
    if column.startswith("btc_") or column.endswith("_close") or column in {"open", "high", "low", "close", "volume"}:
        return "price"
    if column.startswith("binance_"):
        return "binance_derivatives"
    if column.startswith("stablecoin_"):
        return "stablecoins"
    if column.startswith("cm_"):
        return "coinmetrics_onchain"
    if column in {"m2_money_supply", "fed_balance_sheet"}:
        return "fred_macro_monthly"
    if column in {
        "us_10y_yield",
        "us_10y_2y_spread",
        "us_10y_breakeven",
        "fed_funds_rate",
        "reverse_repo",
        "trade_weighted_usd",
    }:
        return "fred_macro_daily"
    if column.startswith("fear_greed"):
        return "manual_csv"
    return "manual_csv"


def apply_release_lags(raw: pd.DataFrame, config: ResearchConfig) -> pd.DataFrame:
    released = raw.copy()
    for column in released.columns:
        group = raw_column_source_group(column)
        delay = int(config.release_delays_days.get(group, 1))
        if delay > 0:
            released[column] = released[column].shift(delay)
    return released


def _safe_log(series: pd.Series) -> pd.Series:
    return np.log(series.where(series > 0))


def _zscore(series: pd.Series, window: int = 365, min_periods: int = 90) -> pd.Series:
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std()
    return (series - mean) / std.replace(0, np.nan)


def add_research_external_features(released_raw: pd.DataFrame, out: pd.DataFrame) -> None:
    for column in [
        "binance_funding_rate",
        "binance_sum_open_interest",
        "binance_sum_open_interest_value",
        "binance_long_short_ratio",
        "binance_long_account",
        "binance_short_account",
        "binance_taker_buy_sell_ratio",
        "binance_taker_buy_volume",
        "binance_taker_sell_volume",
        "binance_basis",
        "binance_basis_rate",
        "binance_annualized_basis_rate",
    ]:
        if column not in released_raw:
            continue
        series = pd.to_numeric(released_raw[column], errors="coerce")
        prefix = f"derivatives_{column.replace('binance_', '')}"
        out[prefix] = series
        out[f"{prefix}_chg_7d"] = series.diff(7)
        out[f"{prefix}_z_90d"] = _zscore(series, 90, 30)

    if "stablecoin_total_circulating_usd" in released_raw:
        supply = pd.to_numeric(released_raw["stablecoin_total_circulating_usd"], errors="coerce")
        out["stablecoins_supply_log"] = _safe_log(supply)
        out["stablecoins_supply_chg_30d"] = _safe_log(supply).diff(30)
        out["stablecoins_supply_chg_90d"] = _safe_log(supply).diff(90)
        out["stablecoins_supply_z_365d"] = _zscore(supply, 365, 120)


def is_leaky_feature_name(column: str) -> bool:
    lowered = column.lower()
    return any(token in lowered for token in LEAKAGE_TOKENS)


def select_feature_columns(features: pd.DataFrame, config: ResearchConfig) -> List[str]:
    cols: List[str] = []
    for col in features.columns:
        if is_leaky_feature_name(col):
            continue
        if not pd.api.types.is_numeric_dtype(features[col]):
            continue
        valid_ratio = float(features[col].notna().mean())
        required_ratio = (
            config.min_derivative_feature_valid_ratio
            if col.startswith("derivatives_")
            else config.min_feature_valid_ratio
        )
        if valid_ratio >= required_ratio:
            cols.append(col)
    return cols


def _feature_source(feature: str) -> Tuple[str, str, str, int]:
    if feature.startswith("derivatives_"):
        raw = "binance_" + feature.replace("derivatives_", "").split("_chg_")[0].split("_z_")[0]
        return "binance_derivatives", raw, "derivatives transform", 1
    if feature.startswith("stablecoins_"):
        return "stablecoins", "stablecoin_total_circulating_usd", "stablecoin transform", 1
    if feature.startswith("macro_m2_money_supply") or feature.startswith("macro_fed_balance_sheet"):
        return "fred_macro_monthly", feature.replace("macro_", "").split("_ret_")[0].split("_z_")[0], "macro transform", 30
    if feature.startswith("macro_"):
        return "fred_macro_daily", feature.replace("macro_", "").split("_chg_")[0].split("_z_")[0], "macro transform", 2
    if feature.startswith("onchain_"):
        return "coinmetrics_onchain", feature.replace("onchain_", "cm_").split("_chg_")[0].split("_z_")[0], "on-chain transform", 2
    if feature.startswith("sentiment_"):
        return "manual_csv", "fear_greed_value", "sentiment transform", 1
    if feature.startswith("cross_asset_"):
        return "price", feature.replace("cross_asset_", "").split("_ret_")[0], "cross-asset price transform", 0
    return "price", "btc_close", "price/technical transform", 0


def build_feature_audit(features: pd.DataFrame, feature_cols: Sequence[str], config: ResearchConfig) -> pd.DataFrame:
    used = set(feature_cols)
    rows = []
    for feature in features.columns:
        source, raw_metric, transform, default_delay = _feature_source(feature)
        delay = int(config.release_delays_days.get(source, default_delay))
        valid = features[feature].dropna()
        rows.append(
            {
                "feature_name": feature,
                "source": source,
                "raw_metric": raw_metric,
                "transform": transform,
                "lag_days": delay,
                "release_delay_days": delay,
                "first_date": date_str(valid.index.min()) if len(valid) else "",
                "last_date": date_str(valid.index.max()) if len(valid) else "",
                "missing_pct": float(1 - features[feature].notna().mean()) if len(features) else 1.0,
                "used_in_model": feature in used,
            }
        )
    return pd.DataFrame(rows)


def build_features(raw: pd.DataFrame, config: ResearchConfig) -> FeatureBuildResult:
    released_raw = apply_release_lags(raw, config)
    features = make_features(released_raw)
    add_research_external_features(released_raw, features)
    features = features.replace([np.inf, -np.inf], np.nan).sort_index()
    feature_cols = select_feature_columns(features, config)
    audit = build_feature_audit(features, feature_cols, config)
    return FeatureBuildResult(features=features, feature_cols=feature_cols, feature_audit=audit, released_raw=released_raw)


def build_target_frame(raw: pd.DataFrame, features: pd.DataFrame, feature_cols: Sequence[str], horizon: int) -> pd.DataFrame:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    price = raw["btc_close"].astype(float)
    target_log_return = np.log(price.shift(-horizon) / price)
    frame = features.copy()
    frame["btc_close"] = price
    frame["future_close"] = price.shift(-horizon)
    frame["target_log_return"] = target_log_return
    frame["target_up"] = np.where(target_log_return.notna(), (target_log_return > 0).astype(float), np.nan)
    for col in feature_cols:
        if is_leaky_feature_name(col):
            raise ValueError(f"Leaky feature column selected: {col}")
    return frame.sort_index()


def _period_last_dates(index: pd.DatetimeIndex, freq: str) -> List[pd.Timestamp]:
    series = pd.Series(index, index=index)
    dates: List[pd.Timestamp] = []
    for period_end, values in series.groupby(pd.Grouper(freq=freq)):
        if values.empty:
            continue
        last_date = pd.Timestamp(values.max())
        if last_date.normalize() < pd.Timestamp(period_end).normalize():
            continue
        dates.append(last_date)
    return dates


def _step_dates(index: pd.DatetimeIndex, first_allowed: pd.Timestamp, step_days: int) -> List[pd.Timestamp]:
    dates = []
    cursor = first_allowed
    while cursor <= index.max():
        loc = index[index >= cursor]
        if len(loc) == 0:
            break
        date = pd.Timestamp(loc[0])
        dates.append(date)
        cursor = date + pd.Timedelta(days=step_days)
    return dates


def build_walk_forward_windows(
    frame: pd.DataFrame,
    horizon: int,
    window_type: str,
    config: ResearchConfig,
    *,
    quick: bool,
) -> List[ResearchWindow]:
    valid = frame.dropna(subset=["target_log_return", "target_up", "btc_close"]).copy()
    initial_days = config.quick_initial_train_days if quick else config.initial_train_days
    if valid.empty:
        return []
    first_allowed = valid.index.min() + pd.Timedelta(days=initial_days)
    available = valid.loc[valid.index >= first_allowed]
    if available.empty:
        return []

    if horizon == 30 and window_type == "official_monthly":
        test_dates = _period_last_dates(available.index, "ME")
        is_official = True
    elif horizon == 90 and window_type == "official_quarterly":
        test_dates = _period_last_dates(available.index, "QE")
        is_official = True
    elif horizon == 30 and window_type == "sensitivity_weekly":
        test_dates = _period_last_dates(available.index, "W-SUN")
        is_official = False
    elif horizon == 90 and window_type == "sensitivity_overlapping_monthly":
        test_dates = _period_last_dates(available.index, "ME")
        is_official = False
    elif horizon == 180 and window_type == "diagnostic_semiannual":
        test_dates = _step_dates(available.index, first_allowed, 180)
        is_official = False
    else:
        raise ValueError(f"Unsupported window_type {window_type} for horizon {horizon}")

    windows = []
    for date in test_dates:
        if date not in valid.index:
            continue
        train = valid.loc[valid.index < date]
        if (date - train.index.min()).days < initial_days or len(train) < 365:
            continue
        windows.append(
            ResearchWindow(
                horizon=horizon,
                window_type=window_type,
                is_official=is_official,
                test_date=pd.Timestamp(date),
                train_start=pd.Timestamp(train.index.min()),
                train_end=pd.Timestamp(train.index.max()),
            )
        )
    return windows


def window_fingerprint(windows: Sequence[ResearchWindow]) -> str:
    payload = "|".join(f"{w.horizon}:{w.window_type}:{w.test_date.date().isoformat()}" for w in windows)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _balanced_accuracy(actual: np.ndarray, pred: np.ndarray) -> float:
    positives = actual == 1
    negatives = actual == 0
    tpr = np.mean(pred[positives] == 1) if positives.any() else np.nan
    tnr = np.mean(pred[negatives] == 0) if negatives.any() else np.nan
    if pd.isna(tpr) and pd.isna(tnr):
        return np.nan
    if pd.isna(tpr):
        return float(tnr)
    if pd.isna(tnr):
        return float(tpr)
    return float((tpr + tnr) / 2)


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return np.nan
    running = equity.cummax()
    return float((equity / running - 1).min())


def _periods_per_year(horizon: int, window_type: str) -> float:
    if "weekly" in window_type:
        return 52.0
    if "quarterly" in window_type or horizon == 90:
        return 4.0 if "official" in window_type else 12.0
    if "semiannual" in window_type or horizon >= 180:
        return 2.0
    return 12.0


def _returns_from_predictions(preds: pd.DataFrame, transaction_cost_bps: float) -> pd.DataFrame:
    frame = preds.copy()
    cost_rate = transaction_cost_bps / 10000.0
    frame["position"] = frame["position"].astype(float)
    frame["turnover"] = frame["position"].diff().abs().fillna(frame["position"].abs())
    frame["trade_flag"] = (frame["turnover"] > 0).astype(int)
    frame["gross_strategy_return"] = frame["position"] * frame["actual_return"]
    frame["tc_adjusted_strategy_return"] = frame["gross_strategy_return"] - frame["turnover"] * cost_rate
    frame["equity_gross"] = (1 + frame["gross_strategy_return"]).cumprod()
    frame["equity_tc_adjusted"] = (1 + frame["tc_adjusted_strategy_return"]).cumprod()
    frame["equity_buy_hold"] = (1 + frame["actual_return"]).cumprod()
    return frame


def compute_metrics(preds: pd.DataFrame, horizon: int, window_type: str, transaction_cost_bps: float) -> Tuple[dict, pd.DataFrame]:
    if preds.empty:
        return {}, pd.DataFrame()
    with_returns = _returns_from_predictions(preds, transaction_cost_bps)
    actual_up = with_returns["actual_up"].astype(int).to_numpy()
    pred_up = with_returns["predicted_up"].astype(int).to_numpy()
    prob = with_returns["probability_up"].astype(float).clip(0, 1).to_numpy()
    expected = with_returns["expected_log_return"].astype(float).to_numpy()
    actual_log = with_returns["actual_log_return"].astype(float).to_numpy()
    ret = with_returns["tc_adjusted_strategy_return"].astype(float)
    periods_per_year = _periods_per_year(horizon, window_type)
    ret_std = float(ret.std(ddof=1))
    sharpe = float((ret.mean() / ret_std) * math.sqrt(periods_per_year)) if ret_std > 0 else np.nan
    gross_return = float(with_returns["equity_gross"].iloc[-1] - 1)
    tc_return = float(with_returns["equity_tc_adjusted"].iloc[-1] - 1)
    return (
        {
            "test_start": date_str(with_returns.index.min()),
            "test_end": date_str(with_returns.index.max()),
            "sample_count": int(len(with_returns)),
            "directional_accuracy": float(np.mean(pred_up == actual_up)),
            "balanced_accuracy": _balanced_accuracy(actual_up, pred_up),
            "brier_score": float(np.mean((prob - actual_up) ** 2)),
            "calibration_error": calibration_error(with_returns),
            "mae": float(np.mean(np.abs(expected - actual_log))),
            "rmse": float(math.sqrt(np.mean((expected - actual_log) ** 2))),
            "gross_return": gross_return,
            "tc_adjusted_return": tc_return,
            "transaction_cost_bps": float(transaction_cost_bps),
            "sharpe": sharpe,
            "max_drawdown": _max_drawdown(with_returns["equity_tc_adjusted"]),
            "num_trades": int(with_returns["trade_flag"].sum()),
            "turnover": float(with_returns["turnover"].mean()),
        },
        with_returns,
    )


def calibration_error(preds: pd.DataFrame, bins: int = 5) -> float:
    if preds.empty:
        return np.nan
    edges = np.linspace(0, 1, bins + 1)
    total = 0
    weighted_error = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        if high == 1:
            mask = (preds["probability_up"] >= low) & (preds["probability_up"] <= high)
        else:
            mask = (preds["probability_up"] >= low) & (preds["probability_up"] < high)
        part = preds.loc[mask]
        if part.empty:
            continue
        total += len(part)
        weighted_error += len(part) * abs(float(part["probability_up"].mean()) - float(part["actual_up"].mean()))
    return float(weighted_error / total) if total else np.nan


def calibration_table(run_id: str, preds: pd.DataFrame, horizon: int, window_type: str, model: str, probability_confidence: str) -> pd.DataFrame:
    rows = []
    edges = np.linspace(0, 1, 6)
    for low, high in zip(edges[:-1], edges[1:]):
        if high == 1:
            part = preds[(preds["probability_up"] >= low) & (preds["probability_up"] <= high)]
        else:
            part = preds[(preds["probability_up"] >= low) & (preds["probability_up"] < high)]
        rows.append(
            {
                "run_id": run_id,
                "horizon": horizon,
                "window_type": window_type,
                "model": model,
                "prob_bin_low": low,
                "prob_bin_high": high,
                "sample_count": int(len(part)),
                "avg_predicted_prob": float(part["probability_up"].mean()) if len(part) else np.nan,
                "actual_up_rate": float(part["actual_up"].mean()) if len(part) else np.nan,
                "brier_score": float(np.mean((part["probability_up"] - part["actual_up"]) ** 2)) if len(part) else np.nan,
                "calibration_error": calibration_error(part) if len(part) else np.nan,
                "probability_confidence": probability_confidence,
            }
        )
    return pd.DataFrame(rows)


def _threshold_returns(prob: pd.Series, actual_return: pd.Series, threshold: float, transaction_cost_bps: float) -> float:
    position = (prob >= threshold).astype(float)
    turnover = position.diff().abs().fillna(position.abs())
    returns = position * actual_return - turnover * (transaction_cost_bps / 10000.0)
    if returns.empty:
        return -np.inf
    return float((1 + returns).prod() - 1)


def tune_threshold_nested(calibration: pd.DataFrame, transaction_cost_bps: float) -> float:
    candidates = [0.40, 0.45, 0.50, 0.55, 0.60]
    best_threshold = 0.50
    best_score = -np.inf
    for threshold in candidates:
        score = _threshold_returns(
            calibration["probability_up"],
            calibration["actual_return"],
            threshold,
            transaction_cost_bps,
        )
        if score > best_score:
            best_threshold = threshold
            best_score = score
    return float(best_threshold)


def _split_fit_calibration(train: pd.DataFrame, config: ResearchConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cal_rows = max(config.calibration_min_rows, int(len(train) * config.calibration_fraction))
    cal_rows = min(cal_rows, max(30, len(train) - 365))
    if cal_rows <= 0:
        raise ValueError("Not enough train rows for nested calibration")
    return train.iloc[:-cal_rows].copy(), train.iloc[-cal_rows:].copy()


def _model_pipeline(model_name: str, random_state: int, quick: bool):
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if model_name == "logistic_linear":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler()),
                ("model", LogisticRegression(C=0.5, max_iter=1000, class_weight="balanced", random_state=random_state)),
            ]
        )
    if model_name == "hgb":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=45 if quick else 90,
                        learning_rate=0.05,
                        max_leaf_nodes=15,
                        min_samples_leaf=25,
                        l2_regularization=0.08,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    if model_name == "random_forest":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=70 if quick else 150,
                        min_samples_leaf=18,
                        max_features="sqrt",
                        class_weight="balanced_subsample",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unknown model: {model_name}")


def _calibrate_probabilities(raw_prob: np.ndarray, cal_prob: np.ndarray, cal_y: np.ndarray) -> Tuple[np.ndarray, str]:
    if len(np.unique(cal_y)) < 2 or len(cal_y) < 50:
        return np.clip(raw_prob, 0.01, 0.99), "Low confidence"
    try:
        from sklearn.isotonic import IsotonicRegression

        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(cal_prob, cal_y)
        calibrated = calibrator.transform(raw_prob)
        return np.clip(calibrated, 0.01, 0.99), "Medium confidence"
    except Exception:
        return np.clip(raw_prob, 0.01, 0.99), "Low confidence"


def predict_model_windows(
    model_name: str,
    frame: pd.DataFrame,
    feature_cols: Sequence[str],
    windows: Sequence[ResearchWindow],
    config: ResearchConfig,
    *,
    quick: bool,
) -> pd.DataFrame:
    rows = []
    valid = frame.dropna(subset=["target_log_return", "target_up", "btc_close"]).copy()
    for idx, window in enumerate(windows):
        train = valid.loc[valid.index < window.test_date].copy()
        test = valid.loc[[window.test_date]].copy()
        if train.empty or test.empty:
            continue
        fit_train, cal = _split_fit_calibration(train, config)
        usable_cols = [c for c in feature_cols if fit_train[c].notna().any()]
        if len(usable_cols) < 5 or fit_train["target_up"].nunique() < 2 or cal["target_up"].nunique() < 2:
            continue

        model = _model_pipeline(model_name, config.random_seed + idx, quick)
        model.fit(fit_train[usable_cols], fit_train["target_up"].astype(int))

        raw_cal_prob = model.predict_proba(cal[usable_cols])[:, 1]
        calibrated_cal_prob, _ = _calibrate_probabilities(raw_cal_prob, raw_cal_prob, cal["target_up"].astype(int).to_numpy())
        cal_eval = pd.DataFrame(
            {
                "probability_up": calibrated_cal_prob,
                "actual_return": np.exp(cal["target_log_return"].astype(float)) - 1,
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
        rows.append(
            {
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
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("date").sort_index()


def predict_baseline_windows(
    model_name: str,
    frame: pd.DataFrame,
    windows: Sequence[ResearchWindow],
    config: ResearchConfig,
) -> pd.DataFrame:
    rows = []
    valid = frame.dropna(subset=["target_log_return", "target_up", "btc_close"]).copy()
    stable_offset = int(hashlib.sha256(model_name.encode("utf-8")).hexdigest()[:8], 16) % 10000
    rng = np.random.default_rng(config.random_seed + stable_offset)
    price = frame["btc_close"]
    ret_30 = np.log(price / price.shift(30))
    ret_90 = np.log(price / price.shift(90))
    for window in windows:
        if window.test_date not in valid.index:
            continue
        test = valid.loc[[window.test_date]]
        actual_log = float(test["target_log_return"].iloc[0])
        if model_name == "buy_hold_direction":
            prob_up = 0.99
            expected_log = float(valid.loc[valid.index < window.test_date, "target_log_return"].mean())
        elif model_name == "momentum_30d":
            signal = float(ret_30.loc[window.test_date]) if window.test_date in ret_30.index else 0.0
            prob_up = 0.60 if signal > 0 else 0.40
            expected_log = signal
        elif model_name == "momentum_90d":
            signal = float(ret_90.loc[window.test_date]) if window.test_date in ret_90.index else 0.0
            prob_up = 0.60 if signal > 0 else 0.40
            expected_log = signal
        elif model_name == "random_permutation":
            prob_up = float(rng.uniform(0.35, 0.65))
            expected_log = (prob_up - 0.5) * 0.10
        else:
            raise ValueError(model_name)
        predicted_up = int(prob_up >= 0.5)
        rows.append(
            {
                "date": window.test_date,
                "model": model_name,
                "horizon": window.horizon,
                "window_type": window.window_type,
                "is_official": window.is_official,
                "probability_up": prob_up,
                "probability_confidence": "Low confidence",
                "expected_log_return": expected_log,
                "actual_log_return": actual_log,
                "actual_return": math.exp(actual_log) - 1,
                "actual_up": int(actual_log > 0),
                "threshold": 0.5,
                "threshold_source": "baseline_rule",
                "predicted_up": predicted_up,
                "position": float(predicted_up),
                "train_rows": int((valid.index < window.test_date).sum()),
                "calibration_rows": 0,
                "feature_count": 0,
            }
        )
    return pd.DataFrame(rows).set_index("date").sort_index() if rows else pd.DataFrame()


def _min_samples_for(horizon: int, is_official: bool) -> int:
    if horizon == 30 and is_official:
        return MIN_SAMPLE_THRESHOLDS["30d_official"]
    if horizon == 90 and is_official:
        return MIN_SAMPLE_THRESHOLDS["90d_official"]
    if horizon == 180:
        return MIN_SAMPLE_THRESHOLDS["180d_diagnostic"]
    return 12


def _probability_confidence(summary: dict) -> str:
    if summary.get("sample_count", 0) < 24 or pd.isna(summary.get("calibration_error")):
        return "Low confidence"
    if summary["calibration_error"] <= 0.08 and summary["brier_score"] <= 0.23:
        return "Medium confidence"
    return "Low confidence"


def decorate_summary(summary: dict, baselines: Dict[str, dict], horizon: int, is_official: bool, model: str) -> dict:
    buy_hold = baselines.get("buy_hold_direction", {})
    momentum_30 = baselines.get("momentum_30d", {})
    momentum_90 = baselines.get("momentum_90d", {})
    random = baselines.get("random_permutation", {})
    sample_ok = summary.get("sample_count", 0) >= _min_samples_for(horizon, is_official)
    beats_buy_hold_direction = summary.get("directional_accuracy", -np.inf) > buy_hold.get("directional_accuracy", np.inf)
    beats_momentum_30d = summary.get("directional_accuracy", -np.inf) > momentum_30.get("directional_accuracy", np.inf)
    beats_momentum_90d = summary.get("directional_accuracy", -np.inf) > momentum_90.get("directional_accuracy", np.inf)
    beats_random = summary.get("directional_accuracy", -np.inf) > random.get("directional_accuracy", np.inf)
    beats_required_momentum = beats_momentum_30d and beats_momentum_90d
    best_baseline_return = max(
        [
            buy_hold.get("tc_adjusted_return", -np.inf),
            momentum_30.get("tc_adjusted_return", -np.inf),
            momentum_90.get("tc_adjusted_return", -np.inf),
            random.get("tc_adjusted_return", -np.inf),
        ]
    )
    beats_cost_return = summary.get("tc_adjusted_return", -np.inf) > best_baseline_return and summary.get("tc_adjusted_return", 0) > 0
    useful = (
        model not in BASELINE_MODELS
        and horizon != 180
        and sample_ok
        and beats_buy_hold_direction
        and beats_required_momentum
        and beats_random
        and beats_cost_return
        and summary.get("calibration_error", 1) <= 0.15
    )
    reliability = "Low confidence"
    if useful:
        # Keep high confidence unavailable until regime-stability checks are strong enough
        # to support that label. This avoids overstating a short BTC backtest.
        reliability = "Medium confidence"
    selection_eligible = bool(useful and horizon == 30 and is_official and reliability != "Low confidence")
    notes = []
    if not sample_ok:
        notes.append("below minimum official sample threshold")
    if model in BASELINE_MODELS:
        notes.append("baseline")
    if horizon == 180:
        notes.append("diagnostic only")
    if model not in BASELINE_MODELS and not beats_cost_return:
        notes.append("does not beat baselines after transaction costs")
    if model not in BASELINE_MODELS and summary.get("calibration_error", 1) > 0.15:
        notes.append("weak probability calibration")

    summary.update(
        {
            "buy_hold_directional_accuracy": buy_hold.get("directional_accuracy", np.nan),
            "buy_hold_tc_adjusted_return": buy_hold.get("tc_adjusted_return", np.nan),
            "momentum_30d_directional_accuracy": momentum_30.get("directional_accuracy", np.nan),
            "momentum_30d_tc_adjusted_return": momentum_30.get("tc_adjusted_return", np.nan),
            "momentum_90d_directional_accuracy": momentum_90.get("directional_accuracy", np.nan),
            "momentum_90d_tc_adjusted_return": momentum_90.get("tc_adjusted_return", np.nan),
            "random_directional_accuracy": random.get("directional_accuracy", np.nan),
            "random_tc_adjusted_return": random.get("tc_adjusted_return", np.nan),
            "beats_buy_hold_direction": bool(beats_buy_hold_direction),
            "beats_momentum_30d": bool(beats_momentum_30d),
            "beats_momentum_90d": bool(beats_momentum_90d),
            "beats_random_baseline": bool(beats_random),
            "useful_model": bool(useful),
            "reliability_label": reliability,
            "selection_eligible": selection_eligible,
            "notes": "; ".join(notes),
        }
    )
    return summary


def confidence_intervals(
    run_id: str,
    preds: pd.DataFrame,
    summary: dict,
    horizon: int,
    window_type: str,
    model: str,
    config: ResearchConfig,
    *,
    quick: bool,
) -> pd.DataFrame:
    iterations = config.quick_bootstrap_iterations if quick else config.bootstrap_iterations
    perm_iterations = config.quick_permutation_iterations if quick else config.permutation_iterations
    if preds.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(config.random_seed + len(model) + horizon)
    n = len(preds)
    metrics = {"directional_accuracy": [], "tc_adjusted_return": [], "sharpe": []}
    for _ in range(iterations):
        sample = preds.iloc[rng.integers(0, n, n)].copy()
        metric, _ = compute_metrics(sample, horizon, window_type, config.transaction_cost_bps)
        for key in metrics:
            metrics[key].append(metric.get(key, np.nan))

    actual = preds["actual_up"].astype(int).to_numpy()
    pred = preds["predicted_up"].astype(int).to_numpy()
    observed_acc = float(np.mean(actual == pred))
    perm_hits = 0
    for _ in range(perm_iterations):
        shuffled = rng.permutation(actual)
        if float(np.mean(shuffled == pred)) >= observed_acc:
            perm_hits += 1
    p_value = (perm_hits + 1) / (perm_iterations + 1)

    rows = []
    for metric, values in metrics.items():
        arr = np.array(values, dtype=float)
        arr = arr[~np.isnan(arr)]
        rows.append(
            {
                "run_id": run_id,
                "horizon": horizon,
                "window_type": window_type,
                "model": model,
                "metric": metric,
                "estimate": summary.get(metric, np.nan),
                "ci_low": float(np.percentile(arr, 2.5)) if len(arr) else np.nan,
                "ci_high": float(np.percentile(arr, 97.5)) if len(arr) else np.nan,
                "ci_method": "bootstrap_rows",
                "permutation_p_value": p_value if metric == "directional_accuracy" else np.nan,
                "bootstrap_iterations": iterations,
                "sample_count": n,
            }
        )
    return pd.DataFrame(rows)


def add_regime_columns(preds: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    out = preds.copy()
    price = raw["btc_close"].astype(float)
    sma_200 = price.rolling(200, min_periods=100).mean()
    vol_90 = np.log(price).diff().rolling(90, min_periods=30).std()
    vol_median = vol_90.expanding(min_periods=180).median()
    out["regime_trend"] = ["above_200d_sma" if price.get(idx, np.nan) >= sma_200.get(idx, np.nan) else "below_200d_sma" for idx in out.index]
    out["regime_volatility"] = ["high_vol" if vol_90.get(idx, np.nan) >= vol_median.get(idx, np.nan) else "low_vol" for idx in out.index]
    return out


def regime_slices(
    run_id: str,
    preds: pd.DataFrame,
    raw: pd.DataFrame,
    baseline_summary: dict,
    horizon: int,
    window_type: str,
    model: str,
    config: ResearchConfig,
) -> pd.DataFrame:
    if preds.empty:
        return pd.DataFrame()
    frame = add_regime_columns(preds, raw)
    rows = []
    baseline_acc = baseline_summary.get("directional_accuracy", np.nan)
    for regime_name in ["regime_trend", "regime_volatility"]:
        for regime_value, part in frame.groupby(regime_name):
            if len(part) < 4:
                continue
            metrics, _ = compute_metrics(part, horizon, window_type, config.transaction_cost_bps)
            beats = bool(metrics.get("directional_accuracy", -np.inf) > baseline_acc)
            rows.append(
                {
                    "run_id": run_id,
                    "horizon": horizon,
                    "window_type": window_type,
                    "model": model,
                    "regime_name": regime_name.replace("regime_", ""),
                    "regime_value": regime_value,
                    "sample_count": metrics.get("sample_count", 0),
                    "directional_accuracy": metrics.get("directional_accuracy", np.nan),
                    "balanced_accuracy": metrics.get("balanced_accuracy", np.nan),
                    "brier_score": metrics.get("brier_score", np.nan),
                    "calibration_error": metrics.get("calibration_error", np.nan),
                    "sharpe": metrics.get("sharpe", np.nan),
                    "max_drawdown": metrics.get("max_drawdown", np.nan),
                    "tc_adjusted_return": metrics.get("tc_adjusted_return", np.nan),
                    "beats_baselines": beats,
                    "stable_regime_result": bool(beats and metrics.get("sample_count", 0) >= 8),
                }
            )
    return pd.DataFrame(rows)


def _equity_output(run_id: str, returns_frame: pd.DataFrame, horizon: int, window_type: str, model: str) -> pd.DataFrame:
    if returns_frame.empty:
        return pd.DataFrame()
    out = returns_frame.copy()
    out["run_id"] = run_id
    out["date"] = out.index
    out["horizon"] = horizon
    out["window_type"] = window_type
    out["model"] = model
    out["signal"] = np.where(out["predicted_up"] == 1, "long", "cash")
    out["btc_return"] = out["actual_return"]
    return out[
        [
            "run_id",
            "date",
            "horizon",
            "window_type",
            "model",
            "signal",
            "position",
            "btc_return",
            "gross_strategy_return",
            "tc_adjusted_strategy_return",
            "equity_gross",
            "equity_tc_adjusted",
            "equity_buy_hold",
            "trade_flag",
            "turnover",
        ]
    ]


def run_backtests(
    raw: pd.DataFrame,
    feature_result: FeatureBuildResult,
    config: ResearchConfig,
    *,
    run_id: str,
    generated_at: str,
    quick: bool,
) -> Dict[str, pd.DataFrame]:
    model_rows: List[dict] = []
    equity_rows: List[pd.DataFrame] = []
    calibration_rows: List[pd.DataFrame] = []
    ci_rows: List[pd.DataFrame] = []
    regime_rows: List[pd.DataFrame] = []

    horizon_windows = {
        30: ["official_monthly", "sensitivity_weekly"],
        90: ["official_quarterly", "sensitivity_overlapping_monthly"],
        180: ["diagnostic_semiannual"],
    }
    for horizon, window_types in horizon_windows.items():
        target_frame = build_target_frame(raw, feature_result.features, feature_result.feature_cols, horizon)
        for window_type in window_types:
            windows = build_walk_forward_windows(target_frame, horizon, window_type, config, quick=quick)
            if quick and window_type in {"sensitivity_weekly"} and len(windows) > 96:
                windows = windows[-96:]
            predictions_by_model: Dict[str, pd.DataFrame] = {}

            for baseline in config.baseline_models:
                preds = predict_baseline_windows(baseline, target_frame, windows, config)
                if preds.empty:
                    continue
                predictions_by_model[baseline] = preds

            for model_name in config.first_model_set:
                preds = predict_model_windows(
                    model_name,
                    target_frame,
                    feature_result.feature_cols,
                    windows,
                    config,
                    quick=quick,
                )
                if preds.empty:
                    continue
                predictions_by_model[model_name] = preds

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
                metrics, returns_frame = compute_metrics(preds, horizon, window_type, config.transaction_cost_bps)
                metrics.update(
                    {
                        "run_id": run_id,
                        "horizon": horizon,
                        "window_type": window_type,
                        "is_official": bool(windows[0].is_official if windows else False),
                        "model": model,
                        "threshold": float(preds["threshold"].mean()),
                        "threshold_source": "baseline_rule" if model in BASELINE_MODELS else "nested_calibration",
                    }
                )
                summaries[model] = metrics
                equity_rows.append(_equity_output(run_id, returns_frame, horizon, window_type, model))

            baseline_summaries = {name: summaries[name] for name in BASELINE_MODELS if name in summaries}
            buyhold_summary = baseline_summaries.get("buy_hold_direction", {})
            for model, summary in summaries.items():
                decorated = decorate_summary(summary, baseline_summaries, horizon, summary["is_official"], model)
                model_rows.append(decorated)
                prob_conf = _probability_confidence(decorated)
                calibration_rows.append(calibration_table(run_id, predictions_by_model[model], horizon, window_type, model, prob_conf))
                ci_rows.append(
                    confidence_intervals(
                        run_id,
                        predictions_by_model[model],
                        decorated,
                        horizon,
                        window_type,
                        model,
                        config,
                        quick=quick,
                    )
                )
                regime_rows.append(
                    regime_slices(
                        run_id,
                        predictions_by_model[model],
                        raw,
                        buyhold_summary,
                        horizon,
                        window_type,
                        model,
                        config,
                    )
                )

    summary = pd.DataFrame(model_rows)
    if summary.empty:
        summary = empty_output_frame("backtest_summary.csv")
    leaderboard = build_leaderboard(summary, generated_at)
    return {
        "backtest_summary.csv": summary,
        "model_leaderboard.csv": leaderboard,
        "equity_curves.csv": pd.concat(equity_rows, ignore_index=True) if equity_rows else empty_output_frame("equity_curves.csv"),
        "calibration_table.csv": pd.concat(calibration_rows, ignore_index=True) if calibration_rows else empty_output_frame("calibration_table.csv"),
        "confidence_intervals.csv": pd.concat(ci_rows, ignore_index=True) if ci_rows else empty_output_frame("confidence_intervals.csv"),
        "regime_slices.csv": pd.concat(regime_rows, ignore_index=True) if regime_rows else empty_output_frame("regime_slices.csv"),
    }


def build_leaderboard(summary: pd.DataFrame, generated_at: str) -> pd.DataFrame:
    if summary.empty:
        return empty_output_frame("model_leaderboard.csv")
    rows = []
    sort_cols = ["directional_accuracy", "brier_score", "sharpe", "max_drawdown", "calibration_error"]
    ascending = [False, True, False, False, True]
    for (_, _), group in summary.groupby(["horizon", "window_type"], dropna=False):
        ranked = group.sort_values(sort_cols, ascending=ascending, na_position="last").copy()
        ranked["rank"] = range(1, len(ranked) + 1)
        ranked["generated_at"] = generated_at
        rows.append(ranked)
    frame = pd.concat(rows, ignore_index=True)
    columns = [
        "run_id",
        "generated_at",
        "horizon",
        "window_type",
        "is_official",
        "model",
        "rank",
        "test_start",
        "test_end",
        "sample_count",
        "directional_accuracy",
        "balanced_accuracy",
        "brier_score",
        "calibration_error",
        "mae",
        "rmse",
        "sharpe",
        "max_drawdown",
        "num_trades",
        "turnover",
        "tc_adjusted_return",
        "beats_buy_hold_direction",
        "beats_momentum_30d",
        "beats_momentum_90d",
        "beats_random_baseline",
        "useful_model",
        "reliability_label",
        "selection_eligible",
        "notes",
    ]
    return frame.reindex(columns=columns)


def select_primary_model(leaderboard: pd.DataFrame) -> Tuple[str, str, str]:
    required = {"horizon", "window_type", "selection_eligible", "model"}
    if leaderboard.empty or not required.issubset(set(leaderboard.columns)):
        return "no_valid_edge", "No 30d official model beat baselines after transaction costs with sufficient reliability.", "Low confidence"
    official = leaderboard[
        (leaderboard["horizon"] == 30)
        & (leaderboard["window_type"] == "official_monthly")
        & (leaderboard["selection_eligible"] == True)
        & (~leaderboard["model"].isin(BASELINE_MODELS))
    ].copy()
    if official.empty:
        return "no_valid_edge", "No 30d official model beat baselines after transaction costs with sufficient reliability.", "Low confidence"
    official = official.sort_values(
        ["directional_accuracy", "brier_score", "sharpe", "max_drawdown", "calibration_error"],
        ascending=[False, True, False, False, True],
        na_position="last",
    )
    row = official.iloc[0]
    return str(row["model"]), "Selected from 30d official results only.", str(row["reliability_label"])


def fit_latest_forecast(
    model_name: str,
    raw: pd.DataFrame,
    feature_result: FeatureBuildResult,
    horizon: int,
    config: ResearchConfig,
    *,
    quick: bool,
) -> Tuple[float, float, str]:
    target_frame = build_target_frame(raw, feature_result.features, feature_result.feature_cols, horizon)
    valid = target_frame.dropna(subset=["target_log_return", "target_up"]).copy()
    if len(valid) < 500 or model_name == "no_valid_edge":
        return 0.5, 0.0, "Low confidence"
    fit_train, cal = _split_fit_calibration(valid, config)
    usable_cols = [c for c in feature_result.feature_cols if fit_train[c].notna().any()]
    latest_x = target_frame[usable_cols].tail(1)
    if len(usable_cols) < 5 or fit_train["target_up"].nunique() < 2 or cal["target_up"].nunique() < 2:
        return 0.5, 0.0, "Low confidence"
    model = _model_pipeline(model_name, config.random_seed + horizon, quick)
    model.fit(fit_train[usable_cols], fit_train["target_up"].astype(int))
    raw_cal_prob = model.predict_proba(cal[usable_cols])[:, 1]
    raw_latest_prob = model.predict_proba(latest_x)[:, 1]
    prob, confidence = _calibrate_probabilities(raw_latest_prob, raw_cal_prob, cal["target_up"].astype(int).to_numpy())
    prob_up = float(prob[0])
    mean_up = float(cal.loc[cal["target_up"] == 1, "target_log_return"].mean())
    mean_down = float(cal.loc[cal["target_up"] == 0, "target_log_return"].mean())
    if pd.isna(mean_up) or pd.isna(mean_down):
        expected_log = 0.0
    else:
        expected_log = prob_up * mean_up + (1 - prob_up) * mean_down
    return prob_up, float(math.exp(expected_log) - 1), confidence


def build_latest_forecast(
    run_id: str,
    generated_at: str,
    raw: pd.DataFrame,
    feature_result: FeatureBuildResult,
    leaderboard: pd.DataFrame,
    config: ResearchConfig,
    *,
    quick: bool,
) -> pd.DataFrame:
    selected_model, reason, reliability = select_primary_model(leaderboard)
    current_price = float(raw["btc_close"].dropna().iloc[-1])
    as_of_date = date_str(raw["btc_close"].dropna().index[-1])
    rows = []
    for horizon in [30, 90, 180]:
        if selected_model == "no_valid_edge":
            prob_up, expected_return, prob_confidence = 0.5, 0.0, "Low confidence"
            direction = "neutral"
            signal = "neutral"
            model_used = "no_valid_edge"
        else:
            prob_up, expected_return, prob_confidence = fit_latest_forecast(
                selected_model,
                raw,
                feature_result,
                horizon,
                config,
                quick=quick,
            )
            direction = "up" if prob_up >= 0.5 else "down"
            signal = "long" if prob_up >= 0.55 and reliability != "Low confidence" else "neutral"
            model_used = selected_model
        rows.append(
            {
                "run_id": run_id,
                "generated_at": generated_at,
                "as_of_date": as_of_date,
                "horizon": horizon,
                "model": model_used,
                "selected_model": selected_model,
                "selection_reason": reason,
                "current_price": current_price,
                "predicted_direction": direction,
                "signal": signal,
                "predicted_probability_up": prob_up,
                "probability_confidence": prob_confidence if reliability != "Low confidence" else "Low confidence",
                "expected_return": expected_return,
                "model_implied_forecast_price": current_price * (1 + expected_return),
                "reliability_label": reliability,
                "is_primary_objective": horizon == config.primary_horizon,
                "is_secondary_objective": horizon == config.secondary_horizon,
                "is_diagnostic": horizon == config.diagnostic_horizon,
            }
        )
    return pd.DataFrame(rows)


def dataframe_snapshot_hash(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "empty"
    sample = frame.tail(min(len(frame), 1000)).copy()
    payload = sample.to_csv(index=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return ""


def package_versions() -> Dict[str, str]:
    packages = ["pandas", "numpy", "requests", "streamlit", "plotly", "sklearn"]
    versions: Dict[str, str] = {}
    for package in packages:
        try:
            mod = __import__(package)
            versions[package] = str(getattr(mod, "__version__", "unknown"))
        except Exception:
            versions[package] = "not_installed"
    return versions


def write_manifest(
    output_dir: Path,
    *,
    run_id: str,
    created_at: str,
    config: ResearchConfig,
    raw: pd.DataFrame,
    models_run: Sequence[str],
    features_count: int,
    quick_mode: bool,
    warnings: Sequence[str],
) -> Path:
    manifest = {
        "run_id": run_id,
        "schema_version": SCHEMA_VERSION,
        "created_at": created_at,
        "git_commit": git_commit(),
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "package_versions": package_versions(),
        "start_date": date_str(raw.index.min()) if len(raw) else "",
        "end_date": date_str(raw.index.max()) if len(raw) else "",
        "config_hash": config.hash(),
        "data_snapshot_hash": dataframe_snapshot_hash(raw),
        "models_run": list(models_run),
        "features_count": int(features_count),
        "quick_mode": bool(quick_mode),
        "warnings": list(warnings),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path


def run_research(*, refresh: bool = False, quick: bool = False, output_dir: Path = OUTPUT_DIR) -> Dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    created_at = utc_now_iso()
    run_id = f"btc_research_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    config = ResearchConfig()
    from diagnostic_outputs import preserve_quick_baseline

    baseline_dir = preserve_quick_baseline(output_dir, refresh)
    write_initial_data_availability(output_dir, run_id, created_at)

    data = load_research_data(
        config,
        run_id=run_id,
        generated_at=created_at,
        refresh=refresh,
        quick=quick,
        output_dir=output_dir,
    )
    feature_result = build_features(data.raw, config)
    write_schema_csv("feature_audit.csv", feature_result.feature_audit, output_dir)

    outputs = run_backtests(
        data.raw,
        feature_result,
        config,
        run_id=run_id,
        generated_at=created_at,
        quick=quick,
    )
    latest = build_latest_forecast(
        run_id,
        created_at,
        data.raw,
        feature_result,
        outputs["model_leaderboard.csv"],
        config,
        quick=quick,
    )
    outputs["latest_forecast.csv"] = latest
    outputs["data_availability.csv"] = data.availability
    outputs["feature_audit.csv"] = feature_result.feature_audit

    for filename in [
        "model_leaderboard.csv",
        "backtest_summary.csv",
        "equity_curves.csv",
        "calibration_table.csv",
        "confidence_intervals.csv",
        "regime_slices.csv",
        "latest_forecast.csv",
        "data_availability.csv",
        "feature_audit.csv",
    ]:
        write_schema_csv(filename, outputs[filename], output_dir)

    write_manifest(
        output_dir,
        run_id=run_id,
        created_at=created_at,
        config=config,
        raw=data.raw,
        models_run=config.baseline_models + config.first_model_set,
        features_count=len(feature_result.feature_cols),
        quick_mode=quick,
        warnings=data.warnings,
    )
    from diagnostic_outputs import write_diagnostics

    diagnostic_outputs = write_diagnostics(
        output_dir=output_dir,
        run_id=run_id,
        raw=data.raw,
        feature_result=feature_result,
        config=config,
        base_outputs=outputs,
        latest=latest,
        baseline_dir=baseline_dir,
        quick=quick,
    )
    outputs.update(diagnostic_outputs)
    return outputs
