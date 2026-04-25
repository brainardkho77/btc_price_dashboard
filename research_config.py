from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


RELEASE_DELAYS_DAYS: Dict[str, int] = {
    "price": 0,
    "binance_derivatives": 1,
    "stablecoins": 1,
    "fred_macro_monthly": 30,
    "fred_macro_daily": 2,
    "coinmetrics_onchain": 2,
    "manual_csv": 1,
}

MIN_SAMPLE_THRESHOLDS = {
    "30d_official": 24,
    "90d_official": 12,
    "180d_diagnostic": 8,
}


@dataclass(frozen=True)
class SourceSpec:
    source: str
    endpoint: str
    dataset: str
    requested_fields: List[str]
    source_group: str
    revision_warning: str = ""
    is_used_in_model: bool = True


@dataclass(frozen=True)
class ResearchConfig:
    start_date: str = "2015-01-01"
    end_date: Optional[str] = None
    quick_start_date: str = "2018-01-01"
    primary_horizon: int = 30
    secondary_horizon: int = 90
    diagnostic_horizon: int = 180
    initial_train_days: int = 1095
    quick_initial_train_days: int = 730
    calibration_min_rows: int = 120
    calibration_fraction: float = 0.25
    transaction_cost_bps: float = 10.0
    random_seed: int = 42
    bootstrap_iterations: int = 500
    quick_bootstrap_iterations: int = 150
    permutation_iterations: int = 500
    quick_permutation_iterations: int = 150
    min_feature_valid_ratio: float = 0.55
    release_delays_days: Dict[str, int] = field(default_factory=lambda: dict(RELEASE_DELAYS_DAYS))
    first_model_set: List[str] = field(default_factory=lambda: ["logistic_linear", "hgb", "random_forest"])
    baseline_models: List[str] = field(
        default_factory=lambda: ["buy_hold_direction", "momentum_30d", "momentum_90d", "random_permutation"]
    )

    def to_dict(self) -> dict:
        return asdict(self)

    def hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


SOURCE_SPECS: List[SourceSpec] = [
    SourceSpec(
        source="Coinbase Exchange",
        endpoint="https://api.exchange.coinbase.com/products/BTC-USD/candles",
        dataset="coinbase_btc_usd_candles",
        requested_fields=["open", "high", "low", "close", "volume"],
        source_group="price",
    ),
    SourceSpec(
        source="Yahoo chart API",
        endpoint="https://query1.finance.yahoo.com/v8/finance/chart/BTC-USD",
        dataset="yahoo_btc_usd",
        requested_fields=["open", "high", "low", "close", "volume"],
        source_group="price",
    ),
    SourceSpec(
        source="CoinGecko API",
        endpoint="https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range",
        dataset="coingecko_btc_usd",
        requested_fields=["price"],
        source_group="price",
        is_used_in_model=False,
    ),
    SourceSpec(
        source="Yahoo chart API",
        endpoint="https://query1.finance.yahoo.com/v8/finance/chart",
        dataset="yahoo_market_proxies",
        requested_fields=["eth_close", "spx_close", "nasdaq_close", "vix_close", "dxy_close", "gold_close", "tlt_close"],
        source_group="price",
    ),
    SourceSpec(
        source="FRED CSV downloads",
        endpoint="https://fred.stlouisfed.org/graph/fredgraph.csv",
        dataset="fred_macro",
        requested_fields=[
            "us_10y_yield",
            "us_10y_2y_spread",
            "us_10y_breakeven",
            "fed_funds_rate",
            "fed_balance_sheet",
            "reverse_repo",
            "m2_money_supply",
            "trade_weighted_usd",
        ],
        source_group="fred_macro_daily",
        revision_warning="FRED graph downloads are revised historical data, not point-in-time vintages.",
    ),
    SourceSpec(
        source="Coin Metrics Community API",
        endpoint="https://community-api.coinmetrics.io/v4/timeseries/asset-metrics",
        dataset="coinmetrics_btc_daily",
        requested_fields=[
            "cm_price_usd",
            "cm_market_cap_usd",
            "cm_mvrv",
            "cm_active_addresses",
            "cm_tx_count",
            "cm_hash_rate",
            "cm_fees_btc",
            "cm_exchange_inflow_usd",
            "cm_exchange_outflow_usd",
            "cm_supply",
            "cm_exchange_supply_usd",
            "cm_spot_volume_usd",
            "cm_roi_30d",
            "cm_roi_1y",
        ],
        source_group="coinmetrics_onchain",
        revision_warning="Coin Metrics Community data is revised historical data unless point-in-time vintages are added.",
    ),
    SourceSpec(
        source="Alternative.me Fear & Greed",
        endpoint="https://api.alternative.me/fng/",
        dataset="alternative_fear_greed",
        requested_fields=["fear_greed_value", "fear_greed_label"],
        source_group="manual_csv",
    ),
    SourceSpec(
        source="Binance USD-M Futures",
        endpoint="https://fapi.binance.com/fapi/v1/fundingRate",
        dataset="binance_funding_rate",
        requested_fields=["funding_rate"],
        source_group="binance_derivatives",
    ),
    SourceSpec(
        source="Binance USD-M Futures",
        endpoint="https://fapi.binance.com/futures/data/openInterestHist",
        dataset="binance_open_interest",
        requested_fields=["sum_open_interest", "sum_open_interest_value"],
        source_group="binance_derivatives",
    ),
    SourceSpec(
        source="Binance USD-M Futures",
        endpoint="https://fapi.binance.com/futures/data/globalLongShortAccountRatio",
        dataset="binance_long_short_ratio",
        requested_fields=["long_short_ratio", "long_account", "short_account"],
        source_group="binance_derivatives",
    ),
    SourceSpec(
        source="Binance USD-M Futures",
        endpoint="https://fapi.binance.com/futures/data/takerlongshortRatio",
        dataset="binance_taker_buy_sell_ratio",
        requested_fields=["taker_buy_sell_ratio", "taker_buy_volume", "taker_sell_volume"],
        source_group="binance_derivatives",
    ),
    SourceSpec(
        source="Binance USD-M Futures",
        endpoint="https://fapi.binance.com/futures/data/basis",
        dataset="binance_basis",
        requested_fields=["basis", "basis_rate", "annualized_basis_rate"],
        source_group="binance_derivatives",
    ),
    SourceSpec(
        source="DefiLlama Stablecoins API",
        endpoint="https://stablecoins.llama.fi/stablecoincharts/all",
        dataset="defillama_stablecoins",
        requested_fields=["stablecoin_total_circulating_usd"],
        source_group="stablecoins",
    ),
    SourceSpec(
        source="Manual CSV",
        endpoint="local manual csv folder",
        dataset="manual_csv",
        requested_fields=[],
        source_group="manual_csv",
        is_used_in_model=False,
    ),
    SourceSpec(
        source="ETF flows",
        endpoint="not configured",
        dataset="spot_btc_etf_flows",
        requested_fields=["etf_flow_usd"],
        source_group="manual_csv",
        is_used_in_model=False,
    ),
]
