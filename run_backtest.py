from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from data_sources import load_all_data
from features import HORIZONS, build_training_data
from modeling import MODEL_CHOICES, equity_curves, walk_forward_backtest


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def parse_horizons(value: str) -> List[int]:
    if value.lower() == "all":
        return HORIZONS
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BTC factor-model walk-forward backtests.")
    parser.add_argument("--horizons", default="all", help="Comma-separated horizons in days, or 'all'.")
    parser.add_argument("--model", default="ensemble", choices=MODEL_CHOICES)
    parser.add_argument("--force-refresh", action="store_true", help="Refresh all free data-source caches.")
    parser.add_argument("--initial-train-days", type=int, default=1460)
    parser.add_argument("--step-days", type=int, default=0, help="0 means use each horizon as the non-overlap step.")
    parser.add_argument("--refit-every-days", type=int, default=0, help="0 means max(7, step_days).")
    parser.add_argument("--max-windows", type=int, default=0, help="0 means use all possible walk-forward windows.")
    parser.add_argument("--threshold", type=float, default=0.0, help="Predicted log-return threshold for long/cash strategy.")
    args = parser.parse_args()

    raw = load_all_data(force=args.force_refresh)
    summary_rows = []

    for horizon in parse_horizons(args.horizons):
        step_days = args.step_days or max(1, horizon)
        max_windows = args.max_windows or None
        training = build_training_data(raw, horizon)
        preds, summary = walk_forward_backtest(
            training,
            model_name=args.model,
            initial_train_days=args.initial_train_days,
            step_days=step_days,
            refit_every_days=args.refit_every_days or None,
            threshold=args.threshold,
            max_windows=max_windows,
        )
        pred_path = OUTPUT_DIR / f"predictions_h{horizon}_{args.model}.csv"
        curve_path = OUTPUT_DIR / f"equity_h{horizon}_{args.model}.csv"
        preds.to_csv(pred_path)
        equity_curves(preds, threshold=args.threshold).to_csv(curve_path)
        summary_rows.append(summary)
        print(
            f"h={horizon:>3} windows={summary['windows']:>4} "
            f"dir_acc={summary['directional_accuracy']:.1%} "
            f"strategy={summary['strategy_total_return']:.1%} "
            f"buyhold={summary['buyhold_total_return']:.1%} "
            f"sharpe={summary['strategy_sharpe']:.2f}"
        )

    summary_frame = pd.DataFrame(summary_rows).sort_values("horizon")
    summary_path = OUTPUT_DIR / f"backtest_summary_{args.model}.csv"
    summary_frame.to_csv(summary_path, index=False)
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
