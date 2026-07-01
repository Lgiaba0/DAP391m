import argparse
import importlib.util
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

ML_CORE_ROOT = Path(__file__).resolve().parents[3]
V2_SCRIPT_PATH = ML_CORE_ROOT / "scripts/reg/v2/train_price_vnd_models_v2.py"
spec = importlib.util.spec_from_file_location("price_regression_v2", V2_SCRIPT_PATH)
base = importlib.util.module_from_spec(spec)
sys.modules["price_regression_v2"] = base
spec.loader.exec_module(base)


RAW_PATH = Path("data/raw/vietnam_rooms_properties_merged.csv")
PROCESSED_PATH = Path("data/processed/reg/v4/vietnam_price_vnd_modeling_v4.csv")
REPORT_PATH = Path("reports/reg/v4/price_vnd_model_evaluation_v4.json")
PREDICTIONS_PATH = Path("reports/reg/v4/price_vnd_predictions_v4.csv")
MODEL_DIR = Path("models/reg/v4")
RANDOM_STATE = 42
SEGMENT_LABELS = ["<10M", "10M-30M", ">30M"]
BINARY_ROUTE_LABELS = ["<10M", ">=10M"]


def assign_v4_segment(values):
    values = pd.Series(values, dtype="float64")
    labels = pd.Series(index=values.index, dtype="string")
    labels[values < 10_000_000] = "<10M"
    labels[(values >= 10_000_000) & (values < 30_000_000)] = "10M-30M"
    labels[values >= 30_000_000] = ">30M"
    return labels.fillna("Unknown")


def add_luxury_route_score(df):
    df = df.copy()
    feature_defaults = {
        "has_presidential": 0,
        "has_penthouse": 0,
        "has_private_pool": 0,
        "has_four_bedroom_plus": 0,
        "has_villa": 0,
        "has_suite": 0,
        "has_residence": 0,
        "has_private_beach": 0,
        "has_beachfront": 0,
        "has_ocean_front": 0,
        "name_has_resort": 0,
        "name_has_villa": 0,
        "name_has_luxury": 0,
        "luxury_amenity_score": 0,
    }
    for col, default in feature_defaults.items():
        if col not in df.columns:
            df[col] = default
    star_5 = (pd.to_numeric(df.get("star_rating_clean"), errors="coerce") >= 5).astype(int)
    luxury_score = pd.to_numeric(df["luxury_amenity_score"], errors="coerce").fillna(0)
    df["luxury_route_score"] = (
        2 * pd.to_numeric(df["has_presidential"], errors="coerce").fillna(0)
        + 2 * pd.to_numeric(df["has_penthouse"], errors="coerce").fillna(0)
        + 2 * pd.to_numeric(df["has_private_pool"], errors="coerce").fillna(0)
        + 2 * pd.to_numeric(df["has_four_bedroom_plus"], errors="coerce").fillna(0)
        + pd.to_numeric(df["has_villa"], errors="coerce").fillna(0)
        + pd.to_numeric(df["has_suite"], errors="coerce").fillna(0)
        + pd.to_numeric(df["has_residence"], errors="coerce").fillna(0)
        + pd.to_numeric(df["has_private_beach"], errors="coerce").fillna(0)
        + pd.to_numeric(df["has_beachfront"], errors="coerce").fillna(0)
        + pd.to_numeric(df["has_ocean_front"], errors="coerce").fillna(0)
        + pd.to_numeric(df["name_has_resort"], errors="coerce").fillna(0)
        + pd.to_numeric(df["name_has_villa"], errors="coerce").fillna(0)
        + pd.to_numeric(df["name_has_luxury"], errors="coerce").fillna(0)
        + star_5
        + (luxury_score >= 6).astype(int)
    ).astype(int)
    return df


def make_classifier():
    try:
        from xgboost import XGBClassifier

        return "XGBoostClassifier", XGBClassifier(
            n_estimators=500,
            learning_rate=0.04,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
        )
    except Exception:
        return "RandomForestClassifier", RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )


def make_xgb_regressor():
    try:
        from xgboost import XGBRegressor

        return "XGBoostRegressor", XGBRegressor(
            n_estimators=700,
            learning_rate=0.04,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
        )
    except Exception:
        return "RandomForestRegressor", RandomForestRegressor(
            n_estimators=400,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )


def make_xgb_regressor_with_objective(objective):
    try:
        from xgboost import XGBRegressor

        return f"XGBoostRegressor_{objective.replace(':', '_')}", XGBRegressor(
            n_estimators=700,
            learning_rate=0.04,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            objective=objective,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
        )
    except Exception:
        return None, None


def make_lgbm_regressor_with_objective(objective):
    try:
        from lightgbm import LGBMRegressor

        params = {
            "n_estimators": 700,
            "learning_rate": 0.04,
            "num_leaves": 31,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "objective": objective,
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbose": -1,
        }
        if objective == "huber":
            params["alpha"] = 0.9
        return f"LightGBMRegressor_{objective}", LGBMRegressor(**params)
    except Exception:
        return None, None


def make_lgbm_quantile(alpha):
    try:
        from lightgbm import LGBMRegressor

        return f"LightGBMRegressor_quantile_{alpha:.2f}", LGBMRegressor(
            n_estimators=700,
            learning_rate=0.04,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="quantile",
            alpha=alpha,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
    except Exception:
        return None, None


def safe_r2(y_true, y_pred):
    return None if len(y_true) < 2 else round(float(r2_score(y_true, y_pred)), 6)


def regression_metrics(y_true_vnd, y_pred_vnd):
    y_true_vnd = np.asarray(y_true_vnd, dtype=float)
    y_pred_vnd = np.clip(np.asarray(y_pred_vnd, dtype=float), 0, None)
    abs_error = np.abs(y_true_vnd - y_pred_vnd)
    ape = np.where(y_true_vnd > 0, abs_error / y_true_vnd, np.nan)
    return {
        "row_count": int(len(y_true_vnd)),
        "MAE": round(float(mean_absolute_error(y_true_vnd, y_pred_vnd)), 2),
        "RMSE": round(float(math.sqrt(mean_squared_error(y_true_vnd, y_pred_vnd))), 2),
        "MAPE_percent": round(float(np.nanmean(ape) * 100), 4),
        "RMSLE": round(float(math.sqrt(mean_squared_error(np.log1p(y_true_vnd), np.log1p(y_pred_vnd)))), 6),
        "R2": safe_r2(y_true_vnd, y_pred_vnd),
        "Median_Absolute_Error": round(float(np.median(abs_error)), 2),
        "P90_Absolute_Error": round(float(np.percentile(abs_error, 90)), 2),
    }


def empty_metrics():
    return {"row_count": 0}


def underprediction_metrics(y_true_vnd, y_pred_vnd):
    y_true_vnd = np.asarray(y_true_vnd, dtype=float)
    y_pred_vnd = np.clip(np.asarray(y_pred_vnd, dtype=float), 0, None)
    pred_to_actual = np.where(y_true_vnd > 0, y_pred_vnd / y_true_vnd, np.nan)
    under_error = np.maximum(y_true_vnd - y_pred_vnd, 0)
    return {
        "row_count": int(len(y_true_vnd)),
        "underprediction_rate": round(float(np.mean(y_pred_vnd < y_true_vnd)), 6),
        "mean_pred_minus_actual": round(float(np.mean(y_pred_vnd - y_true_vnd)), 2),
        "median_pred_to_actual_ratio": round(float(np.nanmedian(pred_to_actual)), 6),
        "p10_pred_to_actual_ratio": round(float(np.nanpercentile(pred_to_actual, 10)), 6),
        "p90_underprediction_error": round(float(np.percentile(under_error, 90)), 2),
        "severe_underprediction_count_ratio_below_0_5": int(np.nansum(pred_to_actual < 0.5)),
    }


def metric_by_labels(y_true_vnd, y_pred_vnd, labels, label_order):
    labels = np.asarray(labels)
    y_true_vnd = np.asarray(y_true_vnd)
    y_pred_vnd = np.asarray(y_pred_vnd)
    result = {}
    for label in label_order:
        mask = labels == label
        result[label] = empty_metrics() if mask.sum() == 0 else regression_metrics(y_true_vnd[mask], y_pred_vnd[mask])
    return result


def underprediction_by_labels(y_true_vnd, y_pred_vnd, labels, label_order):
    labels = np.asarray(labels)
    y_true_vnd = np.asarray(y_true_vnd)
    y_pred_vnd = np.asarray(y_pred_vnd)
    result = {}
    for label in label_order:
        mask = labels == label
        result[label] = empty_metrics() if mask.sum() == 0 else underprediction_metrics(y_true_vnd[mask], y_pred_vnd[mask])
    return result


def classifier_metrics_binary(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "precision_high_price": round(float(precision[1]), 6),
        "recall_high_price": round(float(recall[1]), 6),
        "f1_high_price": round(float(f1[1]), 6),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 6),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 6),
        "per_class": {
            "<10M": {
                "precision": round(float(precision[0]), 6),
                "recall": round(float(recall[0]), 6),
                "f1": round(float(f1[0]), 6),
                "support": int(support[0]),
            },
            ">=10M": {
                "precision": round(float(precision[1]), 6),
                "recall": round(float(recall[1]), 6),
                "f1": round(float(f1[1]), 6),
                "support": int(support[1]),
            },
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).astype(int).tolist(),
        "false_negative_high_price_count": int(((y_true == 1) & (y_pred == 0)).sum()),
        "false_positive_high_price_count": int(((y_true == 0) & (y_pred == 1)).sum()),
    }


def confusion_dict(y_true_labels, y_pred_labels, labels):
    return {
        "labels": labels,
        "matrix": confusion_matrix(y_true_labels, y_pred_labels, labels=labels).astype(int).tolist(),
        "rows_are_actual_columns_are_predicted": True,
    }


def fit_regressor(modeling_df, config, train_idx, train_mask, model_name, model, sample_weight=None):
    feature_cols = [
        col
        for col in modeling_df.columns
        if col not in {"price_vnd", "hotel_id", "true_segment_v4", "true_is_high_price"}
    ]
    train_positions = np.asarray(train_idx)[train_mask]
    X_train = modeling_df.iloc[train_positions][feature_cols]
    y_train_vnd = modeling_df.iloc[train_positions]["price_vnd"].astype(float)
    y_train = np.log1p(y_train_vnd)
    pipeline = Pipeline([("preprocessor", base.make_preprocessor(config)), ("model", model)])
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["model__sample_weight"] = sample_weight
    pipeline.fit(X_train, y_train, **fit_kwargs)
    return model_name, pipeline, feature_cols


def predict_pipeline(pipeline, df, features):
    pred_log = np.asarray(pipeline.predict(df[features]), dtype=float)
    pred_log = np.nan_to_num(pred_log, nan=0.0, posinf=np.log1p(1_000_000_000), neginf=0.0)
    pred_log = np.clip(pred_log, 0.0, np.log1p(1_000_000_000))
    return np.clip(np.expm1(pred_log), 0, None)


def compute_calibration(high_pipeline, high_df, high_features, train_idx):
    train = high_df.iloc[train_idx].copy()
    actual = train["price_vnd"].astype(float).to_numpy()
    high_mask = actual >= 10_000_000
    mid_mask = (actual >= 10_000_000) & (actual < 30_000_000)
    over_mask = actual >= 30_000_000
    calibration = {
        "high_price_multiplier": 1.0,
        "luxury_multiplier": 1.0,
        "q10_actual_10m_30m_train": None,
        "q25_actual_over_30m_train": None,
    }
    if high_mask.sum() > 0:
        pred = np.clip(predict_pipeline(high_pipeline, train.loc[high_mask], high_features), 1, None)
        multiplier = float(np.nanmedian(actual[high_mask] / pred))
        calibration["high_price_multiplier"] = round(float(np.clip(multiplier, 1.0, 3.0)), 6)
    if mid_mask.sum() > 0:
        calibration["q10_actual_10m_30m_train"] = round(float(np.quantile(actual[mid_mask], 0.10)), 2)
    if over_mask.sum() > 0:
        pred = np.clip(predict_pipeline(high_pipeline, train.loc[over_mask], high_features), 1, None)
        multiplier = float(np.nanmedian(actual[over_mask] / pred))
        calibration["luxury_multiplier"] = round(float(np.clip(multiplier, 1.0, 3.0)), 6)
        calibration["q25_actual_over_30m_train"] = round(float(np.quantile(actual[over_mask], 0.25)), 2)
    return calibration


def apply_calibration(raw_pred, routes, luxury_like, calibration):
    pred = np.asarray(raw_pred, dtype=float).copy()
    groups = np.where(routes == ">=10M", "route_>=10M", "route_<10M").astype(object)
    high_mask = routes == ">=10M"
    pred[high_mask] *= calibration["high_price_multiplier"]
    if calibration["q10_actual_10m_30m_train"] is not None:
        pred[high_mask] = np.maximum(pred[high_mask], calibration["q10_actual_10m_30m_train"])
    luxury_mask = high_mask & luxury_like
    groups[luxury_mask] = "luxury_like"
    pred[luxury_mask] *= calibration["luxury_multiplier"]
    if calibration["q25_actual_over_30m_train"] is not None:
        pred[luxury_mask] = np.maximum(pred[luxury_mask], calibration["q25_actual_over_30m_train"])
    return pred, groups


def evaluate_routing(y_true, y_pred, true_segments, predicted_routes, y_true_high, y_pred_high):
    focus_mask = np.asarray(y_true_high, dtype=bool)
    mid_mask = np.asarray(true_segments) == "10M-30M"
    over_mask = np.asarray(true_segments) == ">30M"
    low_mask = np.asarray(true_segments) == "<10M"
    payload = {
        "overall_metrics": regression_metrics(y_true, y_pred),
        "segment_metrics_by_true_segment": metric_by_labels(y_true, y_pred, true_segments, SEGMENT_LABELS),
        "segment_metrics_by_predicted_route": metric_by_labels(y_true, y_pred, predicted_routes, BINARY_ROUTE_LABELS),
        "underprediction_metrics": {
            "overall": underprediction_metrics(y_true, y_pred),
            "by_true_segment": underprediction_by_labels(y_true, y_pred, true_segments, SEGMENT_LABELS),
            "by_predicted_route": underprediction_by_labels(y_true, y_pred, predicted_routes, BINARY_ROUTE_LABELS),
        },
        "high_price_focus": {
            "true_over_10m_row_count": int(focus_mask.sum()),
            "route_recall_over_10m": round(float(((y_true_high == 1) & (y_pred_high == 1)).sum() / max((y_true_high == 1).sum(), 1)), 6),
            "route_precision_over_10m": round(float(((y_true_high == 1) & (y_pred_high == 1)).sum() / max((y_pred_high == 1).sum(), 1)), 6),
            "false_negative_over_10m_count": int(((y_true_high == 1) & (y_pred_high == 0)).sum()),
            "false_positive_over_10m_count": int(((y_true_high == 0) & (y_pred_high == 1)).sum()),
            "metrics_over_10m": empty_metrics() if focus_mask.sum() == 0 else regression_metrics(np.asarray(y_true)[focus_mask], np.asarray(y_pred)[focus_mask]),
            "metrics_10m_30m": empty_metrics() if mid_mask.sum() == 0 else regression_metrics(np.asarray(y_true)[mid_mask], np.asarray(y_pred)[mid_mask]),
            "metrics_over_30m": empty_metrics() if over_mask.sum() == 0 else regression_metrics(np.asarray(y_true)[over_mask], np.asarray(y_pred)[over_mask]),
            "metrics_under_10m": empty_metrics() if low_mask.sum() == 0 else regression_metrics(np.asarray(y_true)[low_mask], np.asarray(y_pred)[low_mask]),
            "underprediction_over_10m": empty_metrics() if focus_mask.sum() == 0 else underprediction_metrics(np.asarray(y_true)[focus_mask], np.asarray(y_pred)[focus_mask]),
            "underprediction_10m_30m": empty_metrics() if mid_mask.sum() == 0 else underprediction_metrics(np.asarray(y_true)[mid_mask], np.asarray(y_pred)[mid_mask]),
            "underprediction_over_30m": empty_metrics() if over_mask.sum() == 0 else underprediction_metrics(np.asarray(y_true)[over_mask], np.asarray(y_pred)[over_mask]),
        },
    }
    return payload


def build_raw_predictions(under_pipeline, high_pipeline, under_features, high_features, under_test, high_test, routes):
    pred = np.zeros(len(routes), dtype=float)
    regressor_names = np.empty(len(routes), dtype=object)
    low_mask = routes == "<10M"
    high_mask = routes == ">=10M"
    if low_mask.sum():
        pred[low_mask] = predict_pipeline(under_pipeline, under_test.loc[low_mask], under_features)
        regressor_names[low_mask] = "under_10m"
    if high_mask.sum():
        pred[high_mask] = predict_pipeline(high_pipeline, high_test.loc[high_mask], high_features)
        regressor_names[high_mask] = "high_price"
    return pred, regressor_names


def load_previous_comparisons():
    comparisons = {}
    for version, path in [
        ("v1", Path("reports/reg/v1/price_vnd_model_evaluation.json")),
        ("v2", Path("reports/reg/v2/price_vnd_model_evaluation_v2.json")),
        ("v3", Path("reports/reg/v3/price_vnd_model_evaluation_v3.json")),
    ]:
        if path.exists():
            try:
                comparisons[version] = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                comparisons[version] = {"read_error": str(exc)}
    return comparisons


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-path", type=Path, default=RAW_PATH)
    parser.add_argument("--processed-path", type=Path, default=PROCESSED_PATH)
    parser.add_argument("--report-path", type=Path, default=REPORT_PATH)
    parser.add_argument("--predictions-path", type=Path, default=PREDICTIONS_PATH)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    args.processed_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_path.parent.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    df, amenity_features = base.load_and_preprocess(args.raw_path)
    df = add_luxury_route_score(df)
    baseline_config = base.make_feature_config(df, amenity_features, "baseline")
    v2_config = base.make_feature_config(df, amenity_features, "v2")
    if "luxury_route_score" not in v2_config["numeric_features"]:
        v2_config["numeric_features"] = v2_config["numeric_features"] + ["luxury_route_score"]
    baseline_df = base.make_modeling_df(df, baseline_config)
    v2_df = base.make_modeling_df(df, v2_config)
    v2_df["true_segment_v4"] = assign_v4_segment(v2_df["price_vnd"]).astype(str)
    v2_df["true_is_high_price"] = (v2_df["price_vnd"].astype(float) >= 10_000_000).astype(int)
    baseline_df["true_segment_v4"] = v2_df["true_segment_v4"].to_numpy()
    baseline_df["true_is_high_price"] = v2_df["true_is_high_price"].to_numpy()

    v2_df.to_csv(args.processed_path, index=False, encoding="utf-8-sig")

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    y_split = np.log1p(v2_df["price_vnd"])
    train_idx, test_idx = next(splitter.split(v2_df, y_split, groups=v2_df["hotel_id"]))
    train_segments = v2_df.iloc[train_idx]["true_segment_v4"].to_numpy()
    y_train_vnd = v2_df.iloc[train_idx]["price_vnd"].astype(float).to_numpy()

    low_model_name, low_model = make_xgb_regressor()
    under_name, under_pipeline, under_features = fit_regressor(
        baseline_df,
        baseline_config,
        train_idx,
        train_segments == "<10M",
        low_model_name,
        low_model,
    )

    high_regressors = {}
    for over30_weight in [4, 6, 8, 10]:
        name, model = make_xgb_regressor()
        variant = f"high_price_xgboost_log_mse_weight_{over30_weight}"
        high_mask_train = y_train_vnd >= 10_000_000
        weights = np.where(y_train_vnd[high_mask_train] >= 30_000_000, float(over30_weight), 1.0)
        model_name, pipeline, features = fit_regressor(
            v2_df,
            v2_config,
            train_idx,
            high_mask_train,
            name,
            model,
            sample_weight=weights,
        )
        high_regressors[variant] = {
            "model_name": model_name,
            "pipeline": pipeline,
            "features": features,
            "sample_weight": f"{over30_weight}.0 for >30M rows, 1.0 for 10M-30M rows",
            "calibration": compute_calibration(pipeline, v2_df, features, train_idx),
        }
    for alpha in [0.60, 0.70, 0.80]:
        name, model = make_lgbm_quantile(alpha)
        if model is None:
            continue
        variant = f"high_price_lightgbm_quantile_{str(alpha).replace('.', '_')}"
        high_mask_train = y_train_vnd >= 10_000_000
        model_name, pipeline, features = fit_regressor(
            v2_df,
            v2_config,
            train_idx,
            high_mask_train,
            name,
            model,
        )
        high_regressors[variant] = {
            "model_name": model_name,
            "pipeline": pipeline,
            "features": features,
            "sample_weight": "none",
            "calibration": compute_calibration(pipeline, v2_df, features, train_idx),
        }
    for objective in ["reg:absoluteerror", "reg:pseudohubererror"]:
        name, model = make_xgb_regressor_with_objective(objective)
        if model is None:
            continue
        variant = f"high_price_xgboost_log_{objective.replace(':', '_')}_weight_8"
        high_mask_train = y_train_vnd >= 10_000_000
        weights = np.where(y_train_vnd[high_mask_train] >= 30_000_000, 8.0, 1.0)
        model_name, pipeline, features = fit_regressor(
            v2_df,
            v2_config,
            train_idx,
            high_mask_train,
            name,
            model,
            sample_weight=weights,
        )
        high_regressors[variant] = {
            "model_name": model_name,
            "pipeline": pipeline,
            "features": features,
            "sample_weight": "8.0 for >30M rows, 1.0 for 10M-30M rows",
            "calibration": compute_calibration(pipeline, v2_df, features, train_idx),
        }
    for objective in ["regression_l1", "huber"]:
        name, model = make_lgbm_regressor_with_objective(objective)
        if model is None:
            continue
        variant = f"high_price_lightgbm_log_{objective}"
        high_mask_train = y_train_vnd >= 10_000_000
        model_name, pipeline, features = fit_regressor(
            v2_df,
            v2_config,
            train_idx,
            high_mask_train,
            name,
            model,
        )
        high_regressors[variant] = {
            "model_name": model_name,
            "pipeline": pipeline,
            "features": features,
            "sample_weight": "none",
            "calibration": compute_calibration(pipeline, v2_df, features, train_idx),
        }

    classifier_feature_cols = [
        col
        for col in v2_df.columns
        if col not in {"price_vnd", "hotel_id", "true_segment_v4", "true_is_high_price"}
    ]
    classifier_weight_configs = {
        "classifier_weight_config_a": {"<10M": 1.0, "10M-30M": 4.0, ">30M": 8.0},
        "classifier_weight_config_b": {"<10M": 1.0, "10M-30M": 5.0, ">30M": 10.0},
        "classifier_weight_config_c": {"<10M": 1.0, "10M-30M": 6.0, ">30M": 12.0},
    }
    classifiers = {}
    y_train_binary = v2_df.iloc[train_idx]["true_is_high_price"].astype(int).to_numpy()
    for config_name, weights in classifier_weight_configs.items():
        clf_name, classifier = make_classifier()
        pipeline = Pipeline([("preprocessor", base.make_preprocessor(v2_config)), ("model", classifier)])
        sample_weight = pd.Series(train_segments).map(weights).astype(float).to_numpy()
        pipeline.fit(
            v2_df.iloc[train_idx][classifier_feature_cols],
            y_train_binary,
            model__sample_weight=sample_weight,
        )
        classifiers[config_name] = {"model_name": clf_name, "pipeline": pipeline, "weights": weights}

    under_test = baseline_df.iloc[test_idx].reset_index(drop=True)
    high_test = v2_df.iloc[test_idx].reset_index(drop=True)
    actual_test_vnd = high_test["price_vnd"].astype(float).to_numpy()
    true_segments = high_test["true_segment_v4"].astype(str).to_numpy()
    y_true_high = high_test["true_is_high_price"].astype(int).to_numpy()
    luxury_scores = high_test["luxury_route_score"].astype(int).to_numpy()

    experiments = {}
    prediction_frames = []
    best_name = None
    best_score = None
    best_artifacts = None
    best_overall_name = None
    best_overall_mae = float("inf")

    for clf_config_name, clf_payload in classifiers.items():
        probabilities = clf_payload["pipeline"].predict_proba(high_test[classifier_feature_cols])[:, 1]
        for high_variant, reg_payload in high_regressors.items():
            oracle_routes = np.where(y_true_high == 1, ">=10M", "<10M")
            raw_oracle, _ = build_raw_predictions(
                under_pipeline,
                reg_payload["pipeline"],
                under_features,
                reg_payload["features"],
                under_test,
                high_test,
                oracle_routes,
            )
            oracle_luxury_like = true_segments == ">30M"
            oracle_calibrated, _ = apply_calibration(
                raw_oracle,
                oracle_routes,
                oracle_luxury_like,
                reg_payload["calibration"],
            )
            oracle_eval = evaluate_routing(
                actual_test_vnd,
                oracle_calibrated,
                true_segments,
                oracle_routes,
                y_true_high,
                y_true_high,
            )

            for threshold in [0.10, 0.15, 0.20, 0.25, 0.30]:
                for luxury_threshold in [2, 3, 4]:
                    threshold_name = f"threshold_{str(threshold).replace('.', '_')}"
                    experiment_name = (
                        f"{threshold_name}_{clf_config_name}_luxury_score_{luxury_threshold}_{high_variant}"
                    )
                    classifier_route = probabilities >= threshold
                    rule_override = (~classifier_route) & (luxury_scores >= luxury_threshold)
                    predicted_high = (classifier_route | rule_override).astype(int)
                    predicted_routes = np.where(predicted_high == 1, ">=10M", "<10M")
                    raw_pred, regressor_names = build_raw_predictions(
                        under_pipeline,
                        reg_payload["pipeline"],
                        under_features,
                        reg_payload["features"],
                        under_test,
                        high_test,
                        predicted_routes,
                    )
                    luxury_like = (luxury_scores >= luxury_threshold) & (predicted_high == 1)
                    calibrated_pred, calibration_groups = apply_calibration(
                        raw_pred,
                        predicted_routes,
                        luxury_like,
                        reg_payload["calibration"],
                    )
                    raw_eval = evaluate_routing(
                        actual_test_vnd,
                        raw_pred,
                        true_segments,
                        predicted_routes,
                        y_true_high,
                        predicted_high,
                    )
                    predicted_eval = evaluate_routing(
                        actual_test_vnd,
                        calibrated_pred,
                        true_segments,
                        predicted_routes,
                        y_true_high,
                        predicted_high,
                    )
                    cls_metrics = classifier_metrics_binary(y_true_high, predicted_high)
                    true_route_labels = np.where(y_true_high == 1, ">=10M", "<10M")
                    report = {
                        "classifier": {
                            "model": clf_payload["model_name"],
                            "feature_config": "v2_expanded_features",
                            "weight_config": clf_config_name,
                            "sample_weights": clf_payload["weights"],
                            "high_price_threshold": threshold,
                            "luxury_route_threshold": luxury_threshold,
                            "metrics": cls_metrics,
                            "confusion_matrix": confusion_dict(
                                true_route_labels,
                                predicted_routes,
                                BINARY_ROUTE_LABELS,
                            ),
                        },
                        "regressors": {
                            "under_10m": {
                                "model": under_name,
                                "feature_config": "baseline_features",
                                "train_rows": int((train_segments == "<10M").sum()),
                            },
                            "high_price": {
                                "model": reg_payload["model_name"],
                                "feature_config": "v2_expanded_features",
                                "variant": high_variant,
                                "train_rows": int((train_segments != "<10M").sum()),
                                "sample_weight": reg_payload["sample_weight"],
                            },
                        },
                        "calibration": reg_payload["calibration"],
                        "oracle_routing": oracle_eval,
                        "oracle_luxury_calibration": oracle_eval,
                        "predicted_routing_no_calibration": raw_eval,
                        "predicted_routing": predicted_eval,
                        "high_price_focus": predicted_eval["high_price_focus"],
                        "routing_confusion_matrix": confusion_dict(
                            true_route_labels,
                            predicted_routes,
                            BINARY_ROUTE_LABELS,
                        ),
                    }
                    experiments[experiment_name] = report

                    metadata_cols = [col for col in base.PREDICTION_COLUMNS if col in high_test.columns]
                    predictions = high_test[metadata_cols].copy()
                    predictions["actual_price_vnd"] = actual_test_vnd
                    predictions["true_segment_v4"] = true_segments
                    predictions["true_is_high_price"] = y_true_high
                    predictions["classifier_p_high_price"] = probabilities
                    predictions["classifier_route_high_price"] = classifier_route.astype(int)
                    predictions["luxury_route_score"] = luxury_scores
                    predictions["rule_override_high_price"] = rule_override.astype(int)
                    predictions["routing_strategy"] = "probability_threshold_plus_luxury_score_override"
                    predictions["predicted_route"] = predicted_routes
                    predictions["regressor_name"] = regressor_names
                    predictions["high_price_regressor_variant"] = high_variant
                    predictions["raw_pred_price_vnd"] = raw_pred
                    predictions["calibrated_pred_price_vnd"] = calibrated_pred
                    predictions["calibration_group"] = calibration_groups
                    predictions["calibration_mode"] = "calibrated"
                    predictions["absolute_error"] = np.abs(actual_test_vnd - calibrated_pred)
                    predictions["absolute_percentage_error"] = np.where(
                        actual_test_vnd > 0,
                        predictions["absolute_error"] / actual_test_vnd * 100,
                        np.nan,
                    )
                    predictions["pred_minus_actual"] = calibrated_pred - actual_test_vnd
                    predictions["pred_to_actual_ratio"] = calibrated_pred / actual_test_vnd
                    predictions["is_underprediction"] = calibrated_pred < actual_test_vnd
                    predictions["is_severe_underprediction_ratio_below_0_5"] = predictions["pred_to_actual_ratio"] < 0.5
                    predictions["experiment_name"] = experiment_name
                    predictions["threshold_name"] = threshold_name
                    prediction_frames.append(predictions)

                    for calibration_mode, candidate_eval in [
                        ("uncalibrated", raw_eval),
                        ("calibrated", predicted_eval),
                    ]:
                        focus = candidate_eval["high_price_focus"]
                        low_mape = focus["metrics_under_10m"].get("MAPE_percent", 9999)
                        over30_ratio = focus["underprediction_over_30m"].get("median_pred_to_actual_ratio", 0)
                        severe_over30 = focus["underprediction_over_30m"].get(
                            "severe_underprediction_count_ratio_below_0_5",
                            10**9,
                        )
                        within_low_price_band = low_mape <= 38.5
                        score = (
                            int(within_low_price_band),
                            focus["route_recall_over_10m"],
                            -focus["false_negative_over_10m_count"],
                            over30_ratio,
                            -severe_over30,
                            -low_mape,
                            -candidate_eval["overall_metrics"]["MAE"],
                        )
                        if best_score is None or score > best_score:
                            best_score = score
                            best_name = experiment_name
                            best_artifacts = {
                                "classifier_pipeline": clf_payload["pipeline"],
                                "under_10m_pipeline": under_pipeline,
                                "high_price_pipeline": reg_payload["pipeline"],
                                "under_features": under_features,
                                "high_features": reg_payload["features"],
                                "classifier_features": classifier_feature_cols,
                                "high_price_threshold": threshold,
                                "luxury_route_threshold": luxury_threshold,
                                "calibration": reg_payload["calibration"],
                                "calibration_mode": calibration_mode,
                                "classifier_weight_config": clf_config_name,
                                "high_price_regressor_variant": high_variant,
                            }
                    overall_mae = predicted_eval["overall_metrics"]["MAE"]
                    if overall_mae < best_overall_mae:
                        best_overall_mae = overall_mae
                        best_overall_name = experiment_name

    pd.concat(prediction_frames, ignore_index=True).to_csv(args.predictions_path, index=False, encoding="utf-8-sig")
    model_path = args.model_dir / "price_vnd_v4_routed_model.joblib"
    joblib.dump({"experiment_name": best_name, **best_artifacts}, model_path)

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "target": "price_vnd",
        "task": "High-price-focused routed price prediction",
        "raw_path": str(args.raw_path),
        "processed_path": str(args.processed_path),
        "predictions_path": str(args.predictions_path),
        "model_path": str(model_path),
        "split": {
            "method": "GroupShuffleSplit",
            "group_column": "hotel_id",
            "test_size": 0.2,
            "random_state": RANDOM_STATE,
            "train_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
            "train_hotels": int(v2_df.iloc[train_idx]["hotel_id"].nunique()),
            "test_hotels": int(v2_df.iloc[test_idx]["hotel_id"].nunique()),
        },
        "segment_definition": {
            "<10M": "price_vnd < 10000000",
            "10M-30M": "10000000 <= price_vnd < 30000000",
            ">30M": "price_vnd >= 30000000",
            "high_price": "price_vnd >= 10000000",
        },
        "preserved_v3_low_price_path": {
            "model": under_name,
            "feature_config": "baseline_features",
            "training_rows": "price_vnd < 10000000",
        },
        "data": {
            "rows_after_target_filter": int(len(v2_df)),
            "unique_hotels": int(v2_df["hotel_id"].nunique()),
            "v4_segment_counts": v2_df["true_segment_v4"].value_counts().to_dict(),
            "v4_segment_counts_train": v2_df.iloc[train_idx]["true_segment_v4"].value_counts().to_dict(),
            "v4_segment_counts_test": v2_df.iloc[test_idx]["true_segment_v4"].value_counts().to_dict(),
        },
        "dropped_columns": {
            "id": [col for col in base.ID_DROP_COLS if col in df.columns],
            "leakage": [col for col in base.LEAKAGE_COLS if col in df.columns],
            "constant": [col for col in base.CONSTANT_COLS if col in df.columns],
            "all_missing": [col for col in base.ALL_MISSING_COLS if col in df.columns],
            "long_text_or_unused": [col for col in base.TEXT_DROP_COLS if col in df.columns],
        },
        "experiments": experiments,
        "best_experiment_by_high_price_focus": best_name,
        "best_calibration_mode_by_high_price_focus": best_artifacts["calibration_mode"],
        "best_experiment_by_overall_mae": best_overall_name,
        "comparison_to_previous_versions": load_previous_comparisons(),
    }
    args.report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    best = experiments[best_name]
    best_eval_key = (
        "predicted_routing_no_calibration"
        if best_artifacts["calibration_mode"] == "uncalibrated"
        else "predicted_routing"
    )
    summary = {
        "best_experiment_by_high_price_focus": best_name,
        "best_calibration_mode_by_high_price_focus": best_artifacts["calibration_mode"],
        "best_experiment_by_overall_mae": best_overall_name,
        "best_high_price_focus": best[best_eval_key]["high_price_focus"],
        "best_overall_metrics": best[best_eval_key]["overall_metrics"],
        "best_classifier_metrics": best["classifier"]["metrics"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Processed data saved to: {args.processed_path}")
    print(f"Prediction errors saved to: {args.predictions_path}")
    print(f"Evaluation report saved to: {args.report_path}")
    print(f"Best model artifact saved to: {model_path}")


if __name__ == "__main__":
    main()
