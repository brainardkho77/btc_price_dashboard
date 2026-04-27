from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from research_pipeline import OUTPUT_DIR, run_research
from research_config import ASSET_CONFIGS, get_asset_config
from schemas import REQUIRED_OUTPUT_FILES, validate_diagnostic_output_dir, validate_output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the offline multi-asset crypto research pipeline.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--refresh", action="store_true", help="Refresh free data sources and run the full offline research pass.")
    mode.add_argument("--quick", action="store_true", help="Use cached data where possible and run the smaller validation pass.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Directory for precomputed output files.")
    parser.add_argument("--asset", default="btc", choices=["btc", "sol", "all"], help="Asset to run: btc, sol, or all.")
    return parser.parse_args()


def asset_output_dir(base_output_dir: Path, asset_id: str) -> Path:
    return base_output_dir / "assets" / asset_id


def mirror_btc_outputs(base_output_dir: Path, btc_output_dir: Path) -> None:
    for filename in REQUIRED_OUTPUT_FILES:
        src = btc_output_dir / filename
        if src.exists():
            shutil.copy2(src, base_output_dir / filename)
    src_csv = btc_output_dir / "csv"
    dst_csv = base_output_dir / "csv"
    if src_csv.exists():
        if dst_csv.exists():
            shutil.rmtree(dst_csv)
        shutil.copytree(src_csv, dst_csv)
    for filename in ["full_refresh_report.md", "full_refresh_diagnostics.md", "derivatives_recovery_report.md"]:
        src = btc_output_dir / filename
        if src.exists():
            shutil.copy2(src, base_output_dir / filename)


def run_one_asset(asset_id: str, *, base_output_dir: Path, refresh: bool, quick: bool) -> tuple[dict, Path]:
    output_dir = asset_output_dir(base_output_dir, asset_id)
    outputs = run_research(refresh=refresh, quick=quick, output_dir=output_dir, asset=asset_id)
    validate_output_dir(output_dir, REQUIRED_OUTPUT_FILES)
    validate_diagnostic_output_dir(output_dir)
    if asset_id == "btc":
        base_output_dir.mkdir(parents=True, exist_ok=True)
        mirror_btc_outputs(base_output_dir, output_dir)
    return outputs, output_dir


def print_asset_summary(asset_id: str, outputs: dict, output_dir: Path) -> None:
    asset = get_asset_config(asset_id)
    latest = outputs["latest_forecast.csv"]
    leaderboard = outputs["model_leaderboard.csv"]
    primary = latest.loc[latest["is_primary_objective"] == True].head(1)
    print(f"Wrote {asset.display_name} outputs to: {output_dir.resolve()}")
    if not primary.empty:
        row = primary.iloc[0]
        print(
            f"{asset.display_name} primary 30d result: "
            f"selected_model={row['selected_model']} signal={row['signal']} "
            f"reliability={row['reliability_label']}"
        )

    official_30 = leaderboard[(leaderboard["horizon"] == 30) & (leaderboard["window_type"] == "official_monthly")]
    if not official_30.empty:
        cols = ["model", "rank", "sample_count", "directional_accuracy", "brier_score", "tc_adjusted_return", "reliability_label"]
        print(f"\n{asset.display_name} 30d official leaderboard:")
        print(official_30[cols].to_string(index=False))


def main() -> None:
    args = parse_args()
    base_output_dir = Path(args.output_dir)
    quick = bool(args.quick)
    refresh = bool(args.refresh)
    asset_ids = list(ASSET_CONFIGS) if args.asset == "all" else [args.asset]
    for asset_id in asset_ids:
        outputs, output_dir = run_one_asset(asset_id, base_output_dir=base_output_dir, refresh=refresh, quick=quick)
        print_asset_summary(asset_id, outputs, output_dir)


if __name__ == "__main__":
    main()
