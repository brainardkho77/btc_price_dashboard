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
    "polymarket_prediction_markets": 1,
    "solana_ecosystem": 1,
}

MIN_SAMPLE_THRESHOLDS = {
    "30d_official": 24,
    "90d_official": 12,
    "180d_diagnostic": 8,
}


@dataclass(frozen=True)
class AssetConfig:
    asset_id: str
    display_name: str
    coinbase_product: Optional[str]
    yahoo_symbol: str
    coingecko_id: Optional[str]
    coingecko_dataset: str
    coinmetrics_asset: str
    binance_symbol: str
    enable_derivatives: bool
    enable_onchain: bool
    enable_coinbase: bool = True
    enable_coingecko: bool = True
    enable_polymarket: bool = True
    enable_crypto_sentiment: bool = True
    enable_stablecoins: bool = True


ASSET_CONFIGS: Dict[str, AssetConfig] = {
    "btc": AssetConfig(
        asset_id="btc",
        display_name="BTC",
        coinbase_product="BTC-USD",
        yahoo_symbol="BTC-USD",
        coingecko_id="bitcoin",
        coingecko_dataset="coingecko_btc_usd",
        coinmetrics_asset="btc",
        binance_symbol="BTCUSDT",
        enable_derivatives=True,
        enable_onchain=True,
    ),
    "sol": AssetConfig(
        asset_id="sol",
        display_name="SOL",
        coinbase_product="SOL-USD",
        yahoo_symbol="SOL-USD",
        coingecko_id="solana",
        coingecko_dataset="coingecko_sol_usd",
        coinmetrics_asset="sol",
        binance_symbol="SOLUSDT",
        enable_derivatives=False,
        enable_onchain=False,
    ),
    "spx": AssetConfig(
        asset_id="spx",
        display_name="S&P 500 / SPX",
        coinbase_product=None,
        yahoo_symbol="SPY",
        coingecko_id=None,
        coingecko_dataset="coingecko_spx_usd",
        coinmetrics_asset="",
        binance_symbol="",
        enable_derivatives=False,
        enable_onchain=False,
        enable_coinbase=False,
        enable_coingecko=False,
        enable_polymarket=False,
        enable_crypto_sentiment=False,
        enable_stablecoins=False,
    ),
}


def get_asset_config(asset_id: str) -> AssetConfig:
    key = asset_id.lower()
    if key not in ASSET_CONFIGS:
        raise ValueError(f"Unsupported asset: {asset_id}")
    return ASSET_CONFIGS[key]


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
    min_derivative_feature_valid_ratio: float = 0.45
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


def build_source_specs(asset: AssetConfig) -> List[SourceSpec]:
    coinmetrics_dataset = f"coinmetrics_{asset.asset_id}_daily"
    etf_dataset = f"spot_{asset.asset_id}_etf_flows"
    coinbase_endpoint = (
        f"https://api.exchange.coinbase.com/products/{asset.coinbase_product}/candles"
        if asset.enable_coinbase and asset.coinbase_product
        else "not configured for this asset"
    )
    coingecko_endpoint = (
        f"https://api.coingecko.com/api/v3/coins/{asset.coingecko_id}/market_chart/range"
        if asset.enable_coingecko and asset.coingecko_id
        else "not configured for this asset"
    )
    polymarket_endpoint = (
        "https://gamma-api.polymarket.com/public-search | https://clob.polymarket.com/prices-history"
        if asset.enable_polymarket
        else "not configured for this asset"
    )
    requested_fred_fields = [
        "us_10y_yield",
        "us_2y_yield",
        "real_10y_yield",
        "real_5y_yield",
        "us_3m_bill_rate",
        "us_10y_2y_spread",
        "us_10y_breakeven",
        "inflation_expectations_5y5y",
        "financial_stress_index",
        "effective_fed_funds_rate",
        "fed_funds_rate",
        "fed_balance_sheet",
        "reverse_repo",
        "treasury_general_account",
        "m2_money_supply",
        "trade_weighted_usd",
    ]
    specs = [
        SourceSpec(
            source="Coinbase Exchange",
            endpoint=coinbase_endpoint,
            dataset=f"coinbase_{asset.asset_id}_usd_candles",
            requested_fields=["open", "high", "low", "close", "volume"],
            source_group="price",
            is_used_in_model=asset.enable_coinbase,
        ),
        SourceSpec(
            source="Yahoo chart API",
            endpoint=f"https://query1.finance.yahoo.com/v8/finance/chart/{asset.yahoo_symbol}",
            dataset=f"yahoo_{asset.asset_id}_usd",
            requested_fields=["open", "high", "low", "close", "volume"],
            source_group="price",
        ),
        SourceSpec(
            source="CoinGecko API",
            endpoint=coingecko_endpoint,
            dataset=asset.coingecko_dataset,
            requested_fields=["price"],
            source_group="price",
            is_used_in_model=False,
        ),
        SourceSpec(
            source="Yahoo chart API",
            endpoint="https://query1.finance.yahoo.com/v8/finance/chart",
            dataset="yahoo_market_proxies",
            requested_fields=[
                "btc_proxy_close",
                "eth_close",
                "spx_close",
                "nasdaq_close",
                "vix_close",
                "dxy_close",
                "gold_close",
                "tlt_close",
            ],
            source_group="price",
        ),
        SourceSpec(
            source="FRED API",
            endpoint="https://api.stlouisfed.org/fred/series/observations",
            dataset="fred_macro_api",
            requested_fields=requested_fred_fields,
            source_group="fred_macro_daily",
            revision_warning="FRED API data is revised historical data, not point-in-time vintages.",
        ),
        SourceSpec(
            source="FRED CSV downloads",
            endpoint="https://fred.stlouisfed.org/graph/fredgraph.csv",
            dataset="fred_macro",
            requested_fields=requested_fred_fields,
            source_group="fred_macro_daily",
            revision_warning="FRED graph downloads are revised historical data, not point-in-time vintages.",
        ),
        SourceSpec(
            source="Coin Metrics Community API",
            endpoint="https://community-api.coinmetrics.io/v4/timeseries/asset-metrics",
            dataset=coinmetrics_dataset,
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
            is_used_in_model=asset.enable_onchain,
            revision_warning="Coin Metrics Community data is revised historical data unless point-in-time vintages are added.",
        ),
        SourceSpec(
            source="Alternative.me Fear & Greed",
            endpoint="https://api.alternative.me/fng/",
            dataset="alternative_fear_greed",
            requested_fields=["fear_greed_value", "fear_greed_label"],
            source_group="manual_csv",
            is_used_in_model=asset.enable_crypto_sentiment,
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
            endpoint="https://fapi.binance.com/futures/data/topLongShortAccountRatio",
            dataset="binance_top_trader_account_ratio",
            requested_fields=["top_trader_long_short_account_ratio", "top_trader_long_account", "top_trader_short_account"],
            source_group="binance_derivatives",
        ),
        SourceSpec(
            source="Binance USD-M Futures",
            endpoint="https://fapi.binance.com/futures/data/topLongShortPositionRatio",
            dataset="binance_top_trader_position_ratio",
            requested_fields=["top_trader_long_short_position_ratio", "top_trader_long_account", "top_trader_short_account"],
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
            is_used_in_model=asset.enable_stablecoins,
        ),
        SourceSpec(
            source="Polymarket Gamma/CLOB APIs",
            endpoint=polymarket_endpoint,
            dataset=f"polymarket_{asset.asset_id}_monthly_ladders",
            requested_fields=[
                "implied_median_price",
                "upside_probability",
                "downside_probability",
                "ladder_skew",
                "ladder_width",
                "market_count",
            ],
            source_group="polymarket_prediction_markets",
            is_used_in_model=False,
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
            dataset=etf_dataset,
            requested_fields=["etf_flow_usd"],
            source_group="manual_csv",
            is_used_in_model=False,
        ),
    ]
    if asset.asset_id == "sol":
        specs.extend(
            [
                SourceSpec(
                    source="DefiLlama Solana Stablecoins API",
                    endpoint="https://stablecoins.llama.fi/stablecoincharts/Solana",
                    dataset="defillama_solana_stablecoins",
                    requested_fields=["solana_stablecoin_circulating_usd"],
                    source_group="solana_ecosystem",
                ),
                SourceSpec(
                    source="DefiLlama Solana TVL API",
                    endpoint="https://api.llama.fi/v2/historicalChainTvl/Solana",
                    dataset="defillama_solana_tvl",
                    requested_fields=["solana_tvl_usd"],
                    source_group="solana_ecosystem",
                ),
                SourceSpec(
                    source="DefiLlama Solana DEX Volume API",
                    endpoint="https://api.llama.fi/overview/dexs/Solana",
                    dataset="defillama_solana_dex_volume",
                    requested_fields=["solana_dex_volume_usd"],
                    source_group="solana_ecosystem",
                ),
                SourceSpec(
                    source="DefiLlama Solana Fees API",
                    endpoint="https://api.llama.fi/overview/fees/Solana",
                    dataset="defillama_solana_fees_revenue",
                    requested_fields=["solana_fees_revenue_usd"],
                    source_group="solana_ecosystem",
                ),
            ]
        )
    return specs


SOURCE_SPECS: List[SourceSpec] = build_source_specs(ASSET_CONFIGS["btc"])
