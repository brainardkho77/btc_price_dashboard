from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


OUTPUT_SCHEMAS: Dict[str, List[str]] = {
    "model_leaderboard.csv": [
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
    ],
    "backtest_summary.csv": [
        "run_id",
        "horizon",
        "window_type",
        "is_official",
        "model",
        "test_start",
        "test_end",
        "sample_count",
        "threshold",
        "threshold_source",
        "directional_accuracy",
        "balanced_accuracy",
        "brier_score",
        "calibration_error",
        "mae",
        "rmse",
        "gross_return",
        "tc_adjusted_return",
        "transaction_cost_bps",
        "sharpe",
        "max_drawdown",
        "num_trades",
        "turnover",
        "buy_hold_directional_accuracy",
        "buy_hold_tc_adjusted_return",
        "momentum_30d_directional_accuracy",
        "momentum_30d_tc_adjusted_return",
        "momentum_90d_directional_accuracy",
        "momentum_90d_tc_adjusted_return",
        "random_directional_accuracy",
        "random_tc_adjusted_return",
        "beats_buy_hold_direction",
        "beats_momentum_30d",
        "beats_momentum_90d",
        "beats_random_baseline",
        "useful_model",
        "reliability_label",
        "selection_eligible",
        "notes",
    ],
    "equity_curves.csv": [
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
    ],
    "calibration_table.csv": [
        "run_id",
        "horizon",
        "window_type",
        "model",
        "prob_bin_low",
        "prob_bin_high",
        "sample_count",
        "avg_predicted_prob",
        "actual_up_rate",
        "brier_score",
        "calibration_error",
        "probability_confidence",
    ],
    "confidence_intervals.csv": [
        "run_id",
        "horizon",
        "window_type",
        "model",
        "metric",
        "estimate",
        "ci_low",
        "ci_high",
        "ci_method",
        "permutation_p_value",
        "bootstrap_iterations",
        "sample_count",
    ],
    "regime_slices.csv": [
        "run_id",
        "horizon",
        "window_type",
        "model",
        "regime_name",
        "regime_value",
        "sample_count",
        "directional_accuracy",
        "balanced_accuracy",
        "brier_score",
        "calibration_error",
        "sharpe",
        "max_drawdown",
        "tc_adjusted_return",
        "beats_baselines",
        "stable_regime_result",
    ],
    "latest_forecast.csv": [
        "run_id",
        "generated_at",
        "as_of_date",
        "horizon",
        "model",
        "selected_model",
        "selection_reason",
        "current_price",
        "predicted_direction",
        "signal",
        "predicted_probability_up",
        "probability_confidence",
        "expected_return",
        "model_implied_forecast_price",
        "reliability_label",
        "is_primary_objective",
        "is_secondary_objective",
        "is_diagnostic",
    ],
    "data_availability.csv": [
        "run_id",
        "generated_at",
        "source",
        "endpoint",
        "dataset",
        "status",
        "requested_fields",
        "available_fields",
        "rows",
        "first_date",
        "last_date",
        "failure_reason",
        "is_used_in_model",
        "revision_warning",
    ],
    "feature_audit.csv": [
        "feature_name",
        "source",
        "raw_metric",
        "transform",
        "lag_days",
        "release_delay_days",
        "first_date",
        "last_date",
        "missing_pct",
        "used_in_model",
    ],
}

REQUIRED_CSV_OUTPUTS = list(OUTPUT_SCHEMAS.keys())
REQUIRED_OUTPUT_FILES = REQUIRED_CSV_OUTPUTS + ["run_manifest.json"]

ALLOWED_DATA_STATUSES = {"worked", "failed", "skipped", "unavailable"}
ALLOWED_RELIABILITY_LABELS = {"High confidence", "Medium confidence", "Low confidence"}


class SchemaError(ValueError):
    pass


def empty_output_frame(filename: str) -> pd.DataFrame:
    return pd.DataFrame(columns=OUTPUT_SCHEMAS[filename])


def validate_frame(filename: str, frame: pd.DataFrame) -> None:
    if filename not in OUTPUT_SCHEMAS:
        raise SchemaError(f"Unknown output schema: {filename}")
    missing = [col for col in OUTPUT_SCHEMAS[filename] if col not in frame.columns]
    if missing:
        raise SchemaError(f"{filename} is missing required columns: {', '.join(missing)}")

    if filename == "data_availability.csv":
        invalid = sorted(set(frame["status"].dropna().astype(str)) - ALLOWED_DATA_STATUSES)
        if invalid:
            raise SchemaError(f"data_availability.csv has invalid statuses: {', '.join(invalid)}")

    if "reliability_label" in frame.columns:
        invalid = sorted(set(frame["reliability_label"].dropna().astype(str)) - ALLOWED_RELIABILITY_LABELS)
        if invalid:
            raise SchemaError(f"{filename} has invalid reliability labels: {', '.join(invalid)}")


def write_schema_csv(filename: str, frame: pd.DataFrame, output_dir: Path) -> Path:
    validate_frame(filename, frame)
    ordered = frame.reindex(columns=OUTPUT_SCHEMAS[filename])
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    ordered.to_csv(path, index=False)
    return path


def validate_output_file(output_dir: Path, filename: str) -> None:
    path = output_dir / filename
    if not path.exists():
        raise SchemaError(f"Missing required output file: {path}")
    if filename.endswith(".csv"):
        validate_frame(filename, pd.read_csv(path))


def validate_output_dir(output_dir: Path, required: Iterable[str] = REQUIRED_OUTPUT_FILES) -> None:
    for filename in required:
        validate_output_file(output_dir, filename)
