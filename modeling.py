from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import HORIZONS, TrainingData, build_training_data


MODEL_CHOICES = ["ensemble", "hgb", "random_forest", "extra_trees", "elastic_net"]


@dataclass
class FitResult:
    model_name: str
    models: List[Tuple[str, object]]
    residual_sigma: float
    train_rows: int
    feature_cols: List[str]


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _base_models(random_state: int = 42) -> Dict[str, object]:
    return {
        "hgb": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_iter=90,
                        learning_rate=0.045,
                        l2_regularization=0.08,
                        max_leaf_nodes=17,
                        min_samples_leaf=28,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=70,
                        min_samples_leaf=22,
                        max_features="sqrt",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "extra_trees": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=80,
                        min_samples_leaf=24,
                        max_features="sqrt",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "elastic_net": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "model",
                    ElasticNet(
                        alpha=0.0015,
                        l1_ratio=0.15,
                        max_iter=20000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def _selected_models(model_name: str, random_state: int = 42) -> List[Tuple[str, object]]:
    models = _base_models(random_state)
    if model_name == "ensemble":
        return [(name, clone(models[name])) for name in ["hgb", "extra_trees", "elastic_net"]]
    if model_name not in models:
        raise ValueError(f"Unknown model_name '{model_name}'. Choose one of {MODEL_CHOICES}.")
    return [(model_name, clone(models[model_name]))]


def _clean_xy(training: TrainingData) -> Tuple[pd.DataFrame, pd.Series]:
    frame = training.frame.dropna(subset=[training.target_col]).copy()
    x = frame[training.feature_cols]
    y = frame[training.target_col].astype(float)
    return x, y


def fit_forecaster(
    training: TrainingData,
    *,
    model_name: str = "ensemble",
    random_state: int = 42,
    max_train_rows: Optional[int] = None,
) -> FitResult:
    x, y = _clean_xy(training)
    if max_train_rows and len(x) > max_train_rows:
        x = x.tail(max_train_rows)
        y = y.tail(max_train_rows)
    if len(x) < 365:
        raise ValueError("Need at least 365 rows for a BTC model fit.")

    fitted = []
    preds = []
    for name, model in _selected_models(model_name, random_state):
        model.fit(x, y)
        fitted.append((name, model))
        preds.append(model.predict(x))

    pred = np.mean(np.vstack(preds), axis=0)
    residual = y.to_numpy() - pred
    residual_sigma = float(np.nanstd(residual))
    target_sigma = float(np.nanstd(y.to_numpy()))
    residual_sigma = max(residual_sigma, target_sigma * 0.35, 1e-6)

    return FitResult(
        model_name=model_name,
        models=fitted,
        residual_sigma=residual_sigma,
        train_rows=len(x),
        feature_cols=list(training.feature_cols),
    )


def predict_fit(fit: FitResult, x: pd.DataFrame) -> np.ndarray:
    preds = []
    for _, model in fit.models:
        preds.append(model.predict(x[fit.feature_cols]))
    return np.mean(np.vstack(preds), axis=0)


def forecast_latest(
    raw: pd.DataFrame,
    horizon: int,
    *,
    model_name: str = "ensemble",
    random_state: int = 42,
) -> dict:
    training = build_training_data(raw, horizon)
    fit = fit_forecaster(training, model_name=model_name, random_state=random_state)
    latest_x = training.frame[fit.feature_cols].tail(1)
    pred_log = float(predict_fit(fit, latest_x)[0])
    current_price = float(training.frame["btc_close"].iloc[-1])
    sigma = fit.residual_sigma
    prob_up = _normal_cdf(pred_log / sigma)
    return {
        "horizon": horizon,
        "model": model_name,
        "as_of": training.frame.index[-1],
        "current_price": current_price,
        "pred_log_return": pred_log,
        "pred_return": math.exp(pred_log) - 1,
        "prob_up": max(0.01, min(0.99, prob_up)),
        "forecast_price": current_price * math.exp(pred_log),
        "price_low_68": current_price * math.exp(pred_log - sigma),
        "price_high_68": current_price * math.exp(pred_log + sigma),
        "price_low_90": current_price * math.exp(pred_log - 1.645 * sigma),
        "price_high_90": current_price * math.exp(pred_log + 1.645 * sigma),
        "residual_sigma": sigma,
        "train_rows": fit.train_rows,
        "feature_count": len(fit.feature_cols),
    }


def forecast_horizons(
    raw: pd.DataFrame,
    horizons: Iterable[int] = HORIZONS,
    *,
    model_name: str = "ensemble",
    random_state: int = 42,
) -> pd.DataFrame:
    rows = []
    for horizon in horizons:
        rows.append(forecast_latest(raw, int(horizon), model_name=model_name, random_state=random_state))
    return pd.DataFrame(rows)


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return np.nan
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    return float(drawdown.min())


def _annualized_return(total_return: float, periods: int, periods_per_year: float) -> float:
    if periods <= 0 or total_return <= -1:
        return np.nan
    return (1 + total_return) ** (periods_per_year / periods) - 1


def score_prediction_frame(preds: pd.DataFrame, step_days: int, threshold: float = 0.0) -> dict:
    if preds.empty:
        return {}
    pred = preds["pred_log_return"].to_numpy()
    actual = preds["actual_log_return"].to_numpy()
    actual_simple = np.exp(actual) - 1
    strategy_returns = np.where(pred > threshold, actual_simple, 0.0)
    buyhold_returns = actual_simple

    strategy_equity = pd.Series((1 + strategy_returns).cumprod(), index=preds.index)
    buyhold_equity = pd.Series((1 + buyhold_returns).cumprod(), index=preds.index)
    periods_per_year = 365 / max(1, step_days)
    strat_total = float(strategy_equity.iloc[-1] - 1)
    bh_total = float(buyhold_equity.iloc[-1] - 1)
    strat_std = float(np.std(strategy_returns, ddof=1))
    bh_std = float(np.std(buyhold_returns, ddof=1))

    return {
        "windows": int(len(preds)),
        "start": preds.index.min().date().isoformat(),
        "end": preds.index.max().date().isoformat(),
        "mae": float(mean_absolute_error(actual, pred)),
        "rmse": float(math.sqrt(mean_squared_error(actual, pred))),
        "directional_accuracy": float(np.mean(np.sign(pred) == np.sign(actual))),
        "avg_pred_return": float(np.mean(np.exp(pred) - 1)),
        "avg_actual_return": float(np.mean(actual_simple)),
        "strategy_total_return": strat_total,
        "buyhold_total_return": bh_total,
        "strategy_ann_return": _annualized_return(strat_total, len(preds), periods_per_year),
        "buyhold_ann_return": _annualized_return(bh_total, len(preds), periods_per_year),
        "strategy_sharpe": float((np.mean(strategy_returns) / strat_std) * math.sqrt(periods_per_year)) if strat_std > 0 else np.nan,
        "buyhold_sharpe": float((np.mean(buyhold_returns) / bh_std) * math.sqrt(periods_per_year)) if bh_std > 0 else np.nan,
        "strategy_max_drawdown": _max_drawdown(strategy_equity),
        "buyhold_max_drawdown": _max_drawdown(buyhold_equity),
        "exposure": float(np.mean(pred > threshold)),
    }


def walk_forward_backtest(
    training: TrainingData,
    *,
    model_name: str = "ensemble",
    initial_train_days: int = 1460,
    step_days: Optional[int] = None,
    refit_every_days: Optional[int] = None,
    threshold: float = 0.0,
    max_windows: Optional[int] = None,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, dict]:
    valid = training.frame.dropna(subset=[training.target_col]).copy()
    if len(valid) < initial_train_days + 30:
        raise ValueError("Not enough rows for the requested walk-forward backtest.")

    if step_days is None:
        step_days = max(1, training.horizon)
    if refit_every_days is None:
        refit_every_days = max(7, step_days)

    positions = list(range(initial_train_days, len(valid), step_days))
    if max_windows and len(positions) > max_windows:
        positions = positions[-max_windows:]

    rows = []
    fit = None
    last_refit_pos = None
    refit_count = 0
    for idx, pos in enumerate(positions):
        test = valid.iloc[[pos]]
        should_refit = fit is None or last_refit_pos is None or (pos - last_refit_pos) >= refit_every_days
        if should_refit:
            train = valid.iloc[:pos]
            temp_training = TrainingData(
                frame=train,
                feature_cols=training.feature_cols,
                target_col=training.target_col,
                horizon=training.horizon,
            )
            fit = fit_forecaster(
                temp_training,
                model_name=model_name,
                random_state=random_state + refit_count,
                max_train_rows=2500,
            )
            last_refit_pos = pos
            refit_count += 1
        pred_log = float(predict_fit(fit, test[training.feature_cols])[0])
        actual_log = float(test[training.target_col].iloc[0])
        date = test.index[0]
        rows.append(
            {
                "date": date,
                "pred_log_return": pred_log,
                "pred_return": math.exp(pred_log) - 1,
                "actual_log_return": actual_log,
                "actual_return": math.exp(actual_log) - 1,
                "prob_up": max(0.01, min(0.99, _normal_cdf(pred_log / fit.residual_sigma))),
                "btc_close": float(test["btc_close"].iloc[0]),
                "future_close": float(test["future_close"].iloc[0]),
                "train_rows": fit.train_rows,
            }
        )

    pred_frame = pd.DataFrame(rows).set_index("date").sort_index()
    summary = score_prediction_frame(pred_frame, step_days, threshold=threshold)
    summary.update(
        {
            "horizon": training.horizon,
            "model": model_name,
            "step_days": step_days,
            "threshold": threshold,
            "initial_train_days": initial_train_days,
            "feature_count": len(training.feature_cols),
            "refit_every_days": refit_every_days,
            "refits": refit_count,
        }
    )
    return pred_frame, summary


def equity_curves(preds: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    if preds.empty:
        return pd.DataFrame()
    actual_simple = np.exp(preds["actual_log_return"]) - 1
    strategy_returns = np.where(preds["pred_log_return"] > threshold, actual_simple, 0.0)
    curves = pd.DataFrame(index=preds.index)
    curves["long_cash_signal"] = (1 + pd.Series(strategy_returns, index=preds.index)).cumprod()
    curves["buy_hold_same_windows"] = (1 + actual_simple).cumprod()
    return curves
