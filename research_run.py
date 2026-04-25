from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from research_pipeline import OUTPUT_DIR, run_research
from schemas import REQUIRED_OUTPUT_FILES, validate_diagnostic_output_dir, validate_output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the offline BTC research pipeline.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--refresh", action="store_true", help="Refresh free data sources and run the full offline research pass.")
    mode.add_argument("--quick", action="store_true", help="Use cached data where possible and run the smaller validation pass.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Directory for precomputed output files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    quick = bool(args.quick)
    refresh = bool(args.refresh)
    outputs = run_research(refresh=refresh, quick=quick, output_dir=output_dir)
    validate_output_dir(output_dir, REQUIRED_OUTPUT_FILES)
    validate_diagnostic_output_dir(output_dir)

    latest = outputs["latest_forecast.csv"]
    leaderboard = outputs["model_leaderboard.csv"]
    primary = latest.loc[latest["is_primary_objective"] == True].head(1)
    print(f"Wrote outputs to: {output_dir.resolve()}")
    if not primary.empty:
        row = primary.iloc[0]
        print(
            "Primary 30d result: "
            f"selected_model={row['selected_model']} signal={row['signal']} "
            f"reliability={row['reliability_label']}"
        )

    official_30 = leaderboard[(leaderboard["horizon"] == 30) & (leaderboard["window_type"] == "official_monthly")]
    if not official_30.empty:
        cols = ["model", "rank", "sample_count", "directional_accuracy", "brier_score", "tc_adjusted_return", "reliability_label"]
        print("\n30d official leaderboard:")
        print(official_30[cols].to_string(index=False))


if __name__ == "__main__":
    main()
