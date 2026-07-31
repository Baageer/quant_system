"""Train a T+1-aligned return and downside model for trade ranking.

Unlike the legacy tradability classifier, this module does not require a
future ``up`` event.  It predicts the forward cross-sectional excess return
and maximum adverse excursion (MAE) already exported by ``tradability_labels``.
The selected validation configuration produces one causal ``entry_score`` for
each date × symbol observation:

``predicted_excess_return - risk_penalty * max(-predicted_mae, 0)``.

Example::

    python -m research.market_regime.trade_outcome_model \
        --dataset-id hs300_probability_dataset --trend-model-id hs300_logistic_v2 \
        --run-id hs300_trade_outcome_v1 --horizon 10
"""

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from research.market_regime.logistic_model import fit_preprocessor, transform_features
from research.market_regime.tradability_labels import normalize_a_share_symbols
from research.market_regime.tradability_model import load_trend_feature_columns


def target_columns(horizon: int) -> Tuple[str, str]:
    """Return the T+1-aligned return and downside target names for ``horizon``."""

    if int(horizon) < 1:
        raise ValueError("horizon must be positive")
    return (
        "future_excess_return_{}d".format(int(horizon)),
        "future_mae_{}d".format(int(horizon)),
    )


def load_outcome_split(
    dataset_dir: Path,
    dataset_id: str,
    split: str,
    feature_columns: Sequence[str],
    horizon: int,
) -> pd.DataFrame:
    """Load rows with direct, available T+1 outcome labels.

    Event weights are intentionally not used: the direct return target should
    represent every eligible date × symbol decision rather than trend-event
    windows only.
    """

    excess_column, mae_column = target_columns(horizon)
    path = dataset_dir / "trend_probability_{}_tradability_samples_{}.csv".format(split, dataset_id)
    required = ["date", "symbol", "auxiliary_label_eligible", excess_column, mae_column]
    usecols = list(dict.fromkeys(required + list(feature_columns)))
    samples = pd.read_csv(path, usecols=usecols, dtype={"symbol": str})
    samples["date"] = pd.to_datetime(samples["date"], errors="coerce")
    samples["symbol"] = normalize_a_share_symbols(samples["symbol"])
    eligible = samples["auxiliary_label_eligible"].astype(str).str.lower().eq("true")
    outcomes = samples.loc[:, [excess_column, mae_column]].apply(pd.to_numeric, errors="coerce")
    valid = eligible & outcomes.notnull().all(axis=1) & np.isfinite(outcomes).all(axis=1)
    return samples.loc[valid].dropna(subset=["date", "symbol"]).sort_values(["date", "symbol"]).reset_index(drop=True)


def _target_values(samples: pd.DataFrame, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    excess_column, mae_column = target_columns(horizon)
    return (
        np.asarray(pd.to_numeric(samples[excess_column], errors="raise"), dtype=float),
        np.asarray(pd.to_numeric(samples[mae_column], errors="raise"), dtype=float),
    )


def decision_score(predicted_excess_return: np.ndarray, predicted_mae: np.ndarray, risk_penalty: float) -> np.ndarray:
    """Combine expected excess return and expected downside into a rank score."""

    if float(risk_penalty) < 0:
        raise ValueError("risk_penalty must be non-negative")
    downside = np.maximum(-np.asarray(predicted_mae, dtype=float), 0.0)
    return np.asarray(predicted_excess_return, dtype=float) - float(risk_penalty) * downside


def _daily_rank_ic(dates: pd.Series, actual: np.ndarray, predicted: np.ndarray) -> float:
    values = []
    frame = pd.DataFrame({"date": pd.to_datetime(dates), "actual": actual, "predicted": predicted})
    for _, group in frame.groupby("date", sort=False):
        if len(group) < 2 or group["actual"].nunique() < 2 or group["predicted"].nunique() < 2:
            continue
        values.append(float(group["actual"].rank().corr(group["predicted"].rank())))
    return float(np.mean(values)) if values else np.nan


def _top_k_mask(dates: pd.Series, scores: np.ndarray, top_k: float) -> np.ndarray:
    if not 0 < float(top_k) <= 1:
        raise ValueError("top_k must be in (0, 1]")
    frame = pd.DataFrame({"date": pd.to_datetime(dates), "score": scores})
    ranks = frame.groupby("date", sort=False)["score"].rank(method="first", ascending=False)
    counts = frame.groupby("date", sort=False)["score"].transform("size")
    return np.asarray(ranks <= np.ceil(counts * float(top_k)), dtype=bool)


def evaluate_outcome_predictions(
    samples: pd.DataFrame,
    predicted_excess_return: np.ndarray,
    predicted_mae: np.ndarray,
    horizon: int,
    risk_penalty: float,
    top_k: float,
    split: str,
    model: str,
) -> Dict[str, object]:
    """Evaluate prediction errors and equal-weight daily Top-K outcomes."""

    actual_excess, actual_mae = _target_values(samples, horizon)
    predicted_excess = np.asarray(predicted_excess_return, dtype=float)
    predicted_mae = np.asarray(predicted_mae, dtype=float)
    if len(samples) != len(predicted_excess) or len(samples) != len(predicted_mae):
        raise ValueError("Prediction lengths must match samples")
    scores = decision_score(predicted_excess, predicted_mae, risk_penalty)
    selected = _top_k_mask(samples["date"], scores, top_k)
    selected_excess = actual_excess[selected]
    selected_mae = actual_mae[selected]
    universe_mean = float(np.mean(actual_excess))
    return {
        "split": split,
        "model": model,
        "horizon": int(horizon),
        "risk_penalty": float(risk_penalty),
        "top_k": float(top_k),
        "sample_count": int(len(samples)),
        "date_count": int(pd.Series(samples["date"]).nunique()),
        "excess_return_rmse": float(np.sqrt(np.mean((predicted_excess - actual_excess) ** 2))),
        "mae_rmse": float(np.sqrt(np.mean((predicted_mae - actual_mae) ** 2))),
        "excess_return_rank_ic": _daily_rank_ic(samples["date"], actual_excess, predicted_excess),
        "mae_rank_ic": _daily_rank_ic(samples["date"], actual_mae, predicted_mae),
        "selected_count": int(np.sum(selected)),
        "selected_mean_excess_return": float(np.mean(selected_excess)),
        "selected_median_excess_return": float(np.median(selected_excess)),
        "selected_positive_return_rate": float(np.mean(selected_excess > 0)),
        "selected_mean_mae": float(np.mean(selected_mae)),
        "universe_mean_excess_return": universe_mean,
        "selected_excess_over_universe": float(np.mean(selected_excess) - universe_mean),
    }


def _fit_models(features: np.ndarray, excess_return: np.ndarray, mae: np.ndarray, alpha: float) -> Tuple[Ridge, Ridge]:
    if float(alpha) < 0:
        raise ValueError("All alpha values must be non-negative")
    return Ridge(alpha=float(alpha)).fit(features, excess_return), Ridge(alpha=float(alpha)).fit(features, mae)


def train_trade_outcome_experiment(
    train_samples: pd.DataFrame,
    validation_samples: pd.DataFrame,
    test_samples: pd.DataFrame,
    feature_columns: Sequence[str],
    horizon: int = 10,
    alpha_values: Sequence[float] = (0.1, 1.0, 10.0, 100.0),
    risk_penalties: Sequence[float] = (0.0, 0.25, 0.5, 1.0),
    top_k: float = 0.05,
) -> Dict[str, object]:
    """Fit direct outcome heads and choose their decision score on validation."""

    if not alpha_values or not risk_penalties:
        raise ValueError("alpha_values and risk_penalties cannot be empty")
    preprocessor = fit_preprocessor(train_samples, feature_columns)
    train_features = transform_features(train_samples, preprocessor)
    validation_features = transform_features(validation_samples, preprocessor)
    test_features = transform_features(test_samples, preprocessor)
    train_excess, train_mae = _target_values(train_samples, horizon)

    candidates = []
    validation_rows = []
    for alpha in alpha_values:
        excess_model, mae_model = _fit_models(train_features, train_excess, train_mae, float(alpha))
        validation_excess = excess_model.predict(validation_features)
        validation_mae = mae_model.predict(validation_features)
        for risk_penalty in risk_penalties:
            metrics = evaluate_outcome_predictions(
                validation_samples, validation_excess, validation_mae, horizon, float(risk_penalty), top_k,
                "validation", "trade_outcome_ridge",
            )
            metrics["alpha"] = float(alpha)
            validation_rows.append(metrics)
        candidates.append((float(alpha), excess_model, mae_model, validation_excess, validation_mae))

    validation_grid = pd.DataFrame(validation_rows).sort_values(
        ["selected_excess_over_universe", "selected_mean_mae", "excess_return_rank_ic", "alpha", "risk_penalty"],
        ascending=[False, False, False, True, True],
    ).reset_index(drop=True)
    selected = validation_grid.iloc[0]
    selected_alpha = float(selected["alpha"])
    selected_risk_penalty = float(selected["risk_penalty"])
    _, excess_model, mae_model, validation_excess, validation_mae = next(
        candidate for candidate in candidates if candidate[0] == selected_alpha
    )
    test_excess = excess_model.predict(test_features)
    test_mae = mae_model.predict(test_features)
    selected_validation_metrics = evaluate_outcome_predictions(
        validation_samples, validation_excess, validation_mae, horizon, selected_risk_penalty, top_k,
        "validation", "trade_outcome_ridge_selected",
    )
    selected_test_metrics = evaluate_outcome_predictions(
        test_samples, test_excess, test_mae, horizon, selected_risk_penalty, top_k,
        "test", "trade_outcome_ridge_selected",
    )
    return {
        "horizon": int(horizon),
        "top_k": float(top_k),
        "selected_alpha": selected_alpha,
        "selected_risk_penalty": selected_risk_penalty,
        "feature_columns": list(feature_columns),
        "preprocessor": preprocessor,
        "excess_model": excess_model,
        "mae_model": mae_model,
        "validation_grid": validation_grid,
        "metrics": pd.DataFrame([selected_validation_metrics, selected_test_metrics]),
        "validation_predictions": build_outcome_predictions(
            validation_samples, validation_excess, validation_mae, horizon, selected_risk_penalty,
            "trade_outcome_ridge_selected",
        ),
        "test_predictions": build_outcome_predictions(
            test_samples, test_excess, test_mae, horizon, selected_risk_penalty,
            "trade_outcome_ridge_selected",
        ),
    }


def build_outcome_predictions(
    samples: pd.DataFrame,
    predicted_excess_return: np.ndarray,
    predicted_mae: np.ndarray,
    horizon: int,
    risk_penalty: float,
    model: str,
) -> pd.DataFrame:
    """Build a causal score table for ranking and downstream backtests."""

    excess_column, mae_column = target_columns(horizon)
    result = samples.loc[:, ["date", "symbol", excess_column, mae_column]].copy()
    result["predicted_excess_return_{}d".format(horizon)] = np.asarray(predicted_excess_return, dtype=float)
    result["predicted_mae_{}d".format(horizon)] = np.asarray(predicted_mae, dtype=float)
    result["entry_score"] = decision_score(predicted_excess_return, predicted_mae, risk_penalty)
    result["risk_penalty"] = float(risk_penalty)
    result["model"] = model
    return result.sort_values(["date", "symbol"]).reset_index(drop=True)


def export_trade_outcome_experiment(
    report: Mapping[str, object], output_dir: Path, run_id: str, metadata: Mapping[str, object]
) -> Dict[str, str]:
    """Export scores, validation selection diagnostics, and serialized models."""

    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "validation_grid": report["validation_grid"],
        "metrics": report["metrics"],
        "validation_predictions": report["validation_predictions"],
        "test_predictions": report["test_predictions"],
    }
    paths = {}
    for name, table in tables.items():
        path = output_dir / "trade_outcome_{}_{}.csv".format(name, run_id)
        table.to_csv(path, index=False, encoding="utf-8-sig")
        paths[name] = str(path)
    model_path = output_dir / "trade_outcome_model_{}.pkl".format(run_id)
    with model_path.open("wb") as handle:
        pickle.dump(
            {key: report[key] for key in ("feature_columns", "preprocessor", "excess_model", "mae_model")},
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    paths["model"] = str(model_path)
    manifest = {
        "analysis_id": run_id,
        "created_at": datetime.now().isoformat(),
        "model_type": "dual_ridge_regression",
        "target_definition": "T+1 entry forward excess return and maximum adverse excursion",
        "horizon": report["horizon"],
        "top_k": report["top_k"],
        "selected_alpha": report["selected_alpha"],
        "selected_risk_penalty": report["selected_risk_penalty"],
        "entry_score": "predicted_excess_return - risk_penalty * max(-predicted_mae, 0)",
        "sample_weighting": "uniform eligible date × symbol rows; trend-event sample weights are not used",
        "feature_columns": report["feature_columns"],
        "paths": paths,
        **dict(metadata),
    }
    manifest_path = output_dir / "trade_outcome_manifest_{}.json".format(run_id)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    paths["manifest"] = str(manifest_path)
    return paths


def _parse_float_values(value: str, argument: str, minimum: float = 0.0) -> List[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item < minimum for item in values):
        raise ValueError("{} must contain values >= {}".format(argument, minimum))
    return values


def main(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="output/datasets")
    parser.add_argument("--dataset-id", default="hs300_probability_dataset")
    parser.add_argument("--trend-model-id", default="hs300_logistic_v2")
    parser.add_argument("--run-id", default="hs300_trade_outcome_v1")
    parser.add_argument("--output-dir", default="output/datasets")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--alphas", default="0.1,1,10,100")
    parser.add_argument("--risk-penalties", default="0,0.25,0.5,1")
    parser.add_argument("--top-k", type=float, default=0.05)
    args = parser.parse_args(argv)
    dataset_dir = Path(args.dataset_dir)
    features = load_trend_feature_columns(dataset_dir, args.trend_model_id)
    splits = [
        load_outcome_split(dataset_dir, args.dataset_id, split, features, args.horizon)
        for split in ("train", "validation", "test")
    ]
    report = train_trade_outcome_experiment(
        *splits,
        feature_columns=features,
        horizon=args.horizon,
        alpha_values=_parse_float_values(args.alphas, "alphas"),
        risk_penalties=_parse_float_values(args.risk_penalties, "risk-penalties"),
        top_k=args.top_k,
    )
    paths = export_trade_outcome_experiment(
        report,
        Path(args.output_dir),
        args.run_id,
        {
            "dataset_id": args.dataset_id,
            "trend_model_id": args.trend_model_id,
            "alpha_values": _parse_float_values(args.alphas, "alphas"),
            "risk_penalties": _parse_float_values(args.risk_penalties, "risk-penalties"),
            "selection_objective": "validation_top_k_excess_over_universe_then_lower_downside",
        },
    )
    test_metrics = report["metrics"].query("split == 'test'").iloc[0]
    print(
        "Selected alpha={alpha}; risk_penalty={penalty}; test Top-K excess={excess:.4%}; manifest={manifest}".format(
            alpha=report["selected_alpha"],
            penalty=report["selected_risk_penalty"],
            excess=test_metrics["selected_excess_over_universe"],
            manifest=paths["manifest"],
        )
    )


if __name__ == "__main__":
    main()
