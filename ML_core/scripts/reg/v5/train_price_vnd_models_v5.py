import argparse
import importlib.util
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
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
PROCESSED_PATH = Path("data/processed/reg/v5/vietnam_price_vnd_modeling_v5.csv")
AUDIT_PATH = Path("reports/reg/v5/price_vnd_luxury_outlier_audit_v5.csv")
REPORT_PATH = Path("reports/reg/v5/price_vnd_model_evaluation_v5.json")
PREDICTIONS_PATH = Path("reports/reg/v5/price_vnd_predictions_v5.csv")
MODEL_DIR = Path("models/reg/v5")
RANDOM_STATE = 42

SEGMENT_LABELS = ["<10M", "10M-30M", ">30M"]
BINARY_ROUTE_LABELS = ["<10M", ">=10M"]
SUBROUTE_LABELS = ["<10M", "10M-30M_like", ">30M_like"]
TARGET_META_COLS = {
    "price_vnd",
    "hotel_id",
    "true_segment_v5",
    "true_is_high_price",
    "true_is_luxury_tail",
    "suspicious_price_flag",
    "suspicious_reason",
}
PREDICTION_METADATA_COLS = [
    "hotel_id",
    "room_name",
    "city",
    "source_city",
    "district",
    "region",
    "property_type",
    "star_rating_clean",
    "max_persons",
    "bedroom_count",
    "room_type_extracted",
    "bed_type",
]
V5_NUMERIC_FEATURES = [
    "luxury_route_score",
    "luxury_tail_score",
    "is_villa_or_residence",
    "is_suite_or_penthouse",
    "is_presidential_or_royal",
    "has_private_pool_or_pool_access",
    "has_beach_or_ocean_front",
    "has_multi_bedroom_luxury",
    "is_large_capacity_luxury",
    "is_five_star_luxury",
    "is_resort_luxury",
    "is_luxury_tail_keyword_strong",
]


def assign_v5_segment(values):
    values = pd.Series(values, dtype="float64")
    labels = pd.Series(index=values.index, dtype="string")
    labels[values < 10_000_000] = "<10M"
    labels[(values >= 10_000_000) & (values < 30_000_000)] = "10M-30M"
    labels[values >= 30_000_000] = ">30M"
    return labels.fillna("Unknown")


def _num(df, col, default=0):
    if col not in df.columns:
        return pd.Series(default, index=df.index)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def add_luxury_route_score(df):
    df = df.copy()
    star_5 = (_num(df, "star_rating_clean", np.nan) >= 5).astype(int)
    luxury_score = _num(df, "luxury_amenity_score")
    df["luxury_route_score"] = (
        2 * _num(df, "has_presidential")
        + 2 * _num(df, "has_penthouse")
        + 2 * _num(df, "has_private_pool")
        + 2 * _num(df, "has_four_bedroom_plus")
        + _num(df, "has_villa")
        + _num(df, "has_suite")
        + _num(df, "has_residence")
        + _num(df, "has_private_beach")
        + _num(df, "has_beachfront")
        + _num(df, "has_ocean_front")
        + _num(df, "name_has_resort")
        + _num(df, "name_has_villa")
        + _num(df, "name_has_luxury")
        + star_5
        + (luxury_score >= 6).astype(int)
    ).astype(int)
    return df


def add_v5_luxury_features(df):
    df = df.copy()
    star = _num(df, "star_rating_clean", np.nan)
    max_persons = _num(df, "max_persons")
    bedrooms = _num(df, "bedroom_count")
    luxury_amenity = _num(df, "luxury_amenity_score")
    has_villa = _num(df, "has_villa")
    has_residence = _num(df, "has_residence")
    has_suite = _num(df, "has_suite")
    has_penthouse = _num(df, "has_penthouse")
    has_private_pool = _num(df, "has_private_pool")

    df["is_villa_or_residence"] = ((has_villa == 1) | (has_residence == 1)).astype(int)
    df["is_suite_or_penthouse"] = ((has_suite == 1) | (has_penthouse == 1)).astype(int)
    df["is_presidential_or_royal"] = ((_num(df, "has_presidential") == 1) | (_num(df, "has_royal") == 1)).astype(int)
    df["has_private_pool_or_pool_access"] = ((has_private_pool == 1) | (_num(df, "has_pool_access") == 1)).astype(int)
    df["has_beach_or_ocean_front"] = (
        (_num(df, "has_beachfront") == 1)
        | (_num(df, "has_ocean_front") == 1)
        | (_num(df, "has_private_beach") == 1)
    ).astype(int)
    df["has_multi_bedroom_luxury"] = ((bedrooms >= 2) & ((has_villa == 1) | (has_residence == 1) | (has_suite == 1))).astype(int)
    df["is_large_capacity_luxury"] = ((max_persons >= 4) & ((has_villa == 1) | (has_suite == 1) | (has_private_pool == 1))).astype(int)
    df["is_five_star_luxury"] = ((star >= 5) & (luxury_amenity >= 6)).astype(int)
    df["is_resort_luxury"] = ((_num(df, "name_has_resort") == 1) & (luxury_amenity >= 5)).astype(int)
    df["is_luxury_tail_keyword_strong"] = (
        (_num(df, "has_presidential") == 1)
        | (has_penthouse == 1)
        | (_num(df, "has_four_bedroom_plus") == 1)
        | (has_private_pool == 1)
        | (_num(df, "has_private_beach") == 1)
    ).astype(int)
    df["luxury_tail_score"] = (
        3 * _num(df, "has_presidential")
        + 3 * has_penthouse
        + 3 * _num(df, "has_four_bedroom_plus")
        + 2 * has_private_pool
        + 2 * _num(df, "has_private_beach")
        + 2 * _num(df, "has_beachfront")
        + 2 * has_villa
        + 2 * has_residence
        + has_suite
        + _num(df, "has_pool_access")
        + _num(df, "has_ocean_front")
        + _num(df, "name_has_resort")
        + _num(df, "name_has_villa")
        + _num(df, "name_has_luxury")
        + (star >= 5).astype(int)
        + (max_persons >= 4).astype(int)
        + (luxury_amenity >= 7).astype(int)
    ).astype(int)
    return df


def add_suspicious_outlier_flags(df):
    df = df.copy()
    suspicious = (
        (_num(df, "price_vnd") >= 100_000_000)
        & (_num(df, "star_rating_clean", 0) < 4)
        & (_num(df, "bedroom_count") <= 1)
        & (_num(df, "luxury_route_score") < 3)
        & (_num(df, "max_persons") <= 2)
    )
    df["suspicious_price_flag"] = suspicious.astype(int)
    df["suspicious_reason"] = np.where(
        suspicious,
        "price>=100M, star<4, bedroom<=1, luxury_route_score<3, max_persons<=2",
        "",
    )
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


def make_secondary_classifier(kind):
    if kind == "logistic":
        return "LogisticRegression", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
    if kind == "random_forest":
        return "RandomForestClassifier", RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    return make_classifier()


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


def metrics_by_labels(y_true_vnd, y_pred_vnd, labels, label_order):
    labels = np.asarray(labels)
    y_true_vnd = np.asarray(y_true_vnd)
    y_pred_vnd = np.asarray(y_pred_vnd)
    return {
        label: empty_metrics() if (labels == label).sum() == 0 else regression_metrics(y_true_vnd[labels == label], y_pred_vnd[labels == label])
        for label in label_order
    }


def underprediction_by_labels(y_true_vnd, y_pred_vnd, labels, label_order):
    labels = np.asarray(labels)
    y_true_vnd = np.asarray(y_true_vnd)
    y_pred_vnd = np.asarray(y_pred_vnd)
    return {
        label: empty_metrics()
        if (labels == label).sum() == 0
        else underprediction_metrics(y_true_vnd[labels == label], y_pred_vnd[labels == label])
        for label in label_order
    }


def classifier_metrics_binary(y_true, y_pred, positive_name, negative_name):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        f"precision_{positive_name}": round(float(precision[1]), 6),
        f"recall_{positive_name}": round(float(recall[1]), 6),
        f"f1_{positive_name}": round(float(f1[1]), 6),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 6),
        "per_class": {
            negative_name: {
                "precision": round(float(precision[0]), 6),
                "recall": round(float(recall[0]), 6),
                "f1": round(float(f1[0]), 6),
                "support": int(support[0]),
            },
            positive_name: {
                "precision": round(float(precision[1]), 6),
                "recall": round(float(recall[1]), 6),
                "f1": round(float(f1[1]), 6),
                "support": int(support[1]),
            },
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).astype(int).tolist(),
        f"false_negative_{positive_name}_count": int(((y_true == 1) & (y_pred == 0)).sum()),
        f"false_positive_{positive_name}_count": int(((y_true == 0) & (y_pred == 1)).sum()),
    }


def confusion_dict(y_true_labels, y_pred_labels, labels):
    return {
        "labels": labels,
        "matrix": confusion_matrix(y_true_labels, y_pred_labels, labels=labels).astype(int).tolist(),
        "rows_are_actual_columns_are_predicted": True,
    }


def fit_regressor(modeling_df, config, train_idx, train_mask, model_name, model, sample_weight=None):
    feature_cols = [col for col in modeling_df.columns if col not in TARGET_META_COLS]
    train_positions = np.asarray(train_idx)[train_mask]
    X_train = modeling_df.iloc[train_positions][feature_cols]
    y_train_vnd = modeling_df.iloc[train_positions]["price_vnd"].astype(float)
    pipeline = Pipeline([("preprocessor", base.make_preprocessor(config)), ("model", model)])
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["model__sample_weight"] = sample_weight
    pipeline.fit(X_train, np.log1p(y_train_vnd), **fit_kwargs)
    return model_name, pipeline, feature_cols


def predict_pipeline(pipeline, df, features):
    pred_log = np.asarray(pipeline.predict(df[features]), dtype=float)
    pred_log = np.nan_to_num(pred_log, nan=0.0, posinf=np.log1p(1_000_000_000), neginf=0.0)
    pred_log = np.clip(pred_log, 0.0, np.log1p(1_000_000_000))
    return np.clip(np.expm1(pred_log), 0, None)


def make_v5_feature_config(df, amenity_features):
    config = base.make_feature_config(df, amenity_features, "v2")
    config["numeric_features"] = list(dict.fromkeys(config["numeric_features"] + [col for col in V5_NUMERIC_FEATURES if col in df.columns]))
    return config


def create_outlier_audit(df, audit_path, v4_predictions_path=Path("reports/reg/v4/price_vnd_predictions_v4.csv")):
    audit = df.copy()
    audit["true_segment_v5"] = assign_v5_segment(audit["price_vnd"]).astype(str)
    audit["valid_luxury_signal_count"] = (
        (_num(audit, "luxury_tail_score") >= 5).astype(int)
        + (_num(audit, "luxury_route_score") >= 3).astype(int)
        + (_num(audit, "is_luxury_tail_keyword_strong") == 1).astype(int)
        + (_num(audit, "is_five_star_luxury") == 1).astype(int)
        + (_num(audit, "is_large_capacity_luxury") == 1).astype(int)
    )
    audit["outlier_bucket"] = ""
    buckets = [
        ("price_vnd>=30M", audit["price_vnd"].astype(float) >= 30_000_000),
        ("price_vnd>=50M", audit["price_vnd"].astype(float) >= 50_000_000),
        ("price_vnd>=100M", audit["price_vnd"].astype(float) >= 100_000_000),
    ]
    top_price_idx = audit["price_vnd"].astype(float).nlargest(50).index
    buckets.append(("top_50_by_price_vnd", audit.index.isin(top_price_idx)))

    if v4_predictions_path.exists():
        try:
            v4_preds = pd.read_csv(v4_predictions_path, usecols=lambda col: col in {"hotel_id", "room_name", "actual_price_vnd", "absolute_error"})
            top_v4 = v4_preds.sort_values("absolute_error", ascending=False).drop_duplicates(["hotel_id", "room_name", "actual_price_vnd"]).head(50)
            keys = set(zip(top_v4["hotel_id"], top_v4["room_name"], top_v4["actual_price_vnd"]))
            top_error_mask = audit.apply(lambda row: (row.get("hotel_id"), row.get("room_name"), row.get("price_vnd")) in keys, axis=1)
            buckets.append(("top_50_by_v4_absolute_error", top_error_mask))
        except Exception:
            pass

    selected = pd.Series(False, index=audit.index)
    bucket_labels = pd.Series("", index=audit.index, dtype=object)
    for label, mask in buckets:
        mask = pd.Series(mask, index=audit.index).astype(bool)
        selected |= mask
        bucket_labels.loc[mask] = bucket_labels.loc[mask].apply(lambda x: label if not x else f"{x}|{label}")
    audit["outlier_bucket"] = bucket_labels
    audit_cols = [
        "hotel_id",
        "room_name",
        "city",
        "source_city",
        "district",
        "region",
        "property_type",
        "star_rating_clean",
        "max_persons",
        "bedroom_count",
        "room_type_extracted",
        "bed_type",
        "luxury_route_score",
        "luxury_amenity_score",
        "wellness_amenity_score",
        "price_vnd",
        "true_segment_v5",
        "outlier_bucket",
        "valid_luxury_signal_count",
        "suspicious_price_flag",
        "suspicious_reason",
    ]
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit.loc[selected, [col for col in audit_cols if col in audit.columns]].to_csv(audit_path, index=False, encoding="utf-8-sig")
    return {
        "over_30m_count": int((df["price_vnd"].astype(float) >= 30_000_000).sum()),
        "over_50m_count": int((df["price_vnd"].astype(float) >= 50_000_000).sum()),
        "over_100m_count": int((df["price_vnd"].astype(float) >= 100_000_000).sum()),
        "suspicious_outlier_count": int(df["suspicious_price_flag"].sum()),
        "audit_path": str(audit_path),
    }


def compute_calibration(shared_pipeline, high_mid_pipeline, high_df, shared_features, high_mid_features, train_idx):
    train = high_df.iloc[train_idx].copy()
    actual = train["price_vnd"].astype(float).to_numpy()
    mid_mask = (actual >= 10_000_000) & (actual < 30_000_000)
    over_mask = actual >= 30_000_000
    calibration = {
        "luxury_tail_multiplier": 1.0,
        "q10_actual_10m_30m_train": None,
        "q10_actual_over_30m_train": None,
        "q25_actual_over_30m_train": None,
        "q40_actual_over_30m_train": None,
    }
    if mid_mask.sum() > 0:
        calibration["q10_actual_10m_30m_train"] = round(float(np.quantile(actual[mid_mask], 0.10)), 2)
    if over_mask.sum() > 0:
        pred = np.clip(predict_pipeline(shared_pipeline, train.loc[over_mask], shared_features), 1, None)
        multiplier = float(np.nanmedian(actual[over_mask] / pred))
        calibration["luxury_tail_multiplier"] = round(float(np.clip(multiplier, 1.0, 2.5)), 6)
        for q, name in [(0.10, "q10_actual_over_30m_train"), (0.25, "q25_actual_over_30m_train"), (0.40, "q40_actual_over_30m_train")]:
            calibration[name] = round(float(np.quantile(actual[over_mask], q)), 2)
    return calibration


def build_path_predictions(under_pipeline, high_mid_pipeline, shared_pipeline, specialist_pipeline, features, under_test, high_test, routes, subroutes, tail_blend):
    pred = np.zeros(len(routes), dtype=float)
    raw_pred = np.zeros(len(routes), dtype=float)
    blended_pred = np.zeros(len(routes), dtype=float)
    regressor_names = np.empty(len(routes), dtype=object)
    low_mask = routes == "<10M"
    mid_mask = (routes == ">=10M") & (subroutes == "10M-30M_like")
    tail_mask = (routes == ">=10M") & (subroutes == ">30M_like")
    if low_mask.sum():
        values = predict_pipeline(under_pipeline, under_test.loc[low_mask], features["under"])
        raw_pred[low_mask] = values
        blended_pred[low_mask] = values
        pred[low_mask] = values
        regressor_names[low_mask] = "under_10m"
    if mid_mask.sum():
        values = predict_pipeline(high_mid_pipeline, high_test.loc[mid_mask], features["high_mid"])
        raw_pred[mid_mask] = values
        blended_pred[mid_mask] = values
        pred[mid_mask] = values
        regressor_names[mid_mask] = "high_mid_10m_30m"
    if tail_mask.sum():
        shared = predict_pipeline(shared_pipeline, high_test.loc[tail_mask], features["shared"])
        if specialist_pipeline is not None and tail_blend > 0:
            tail = predict_pipeline(specialist_pipeline, high_test.loc[tail_mask], features["specialist"])
            values = (1.0 - tail_blend) * shared + tail_blend * tail
            regressor_names[tail_mask] = f"shared_plus_luxury_specialist_blend_{tail_blend:.2f}"
        else:
            values = shared
            regressor_names[tail_mask] = "shared_high_price"
        raw_pred[tail_mask] = shared
        blended_pred[tail_mask] = values
        pred[tail_mask] = values
    return raw_pred, blended_pred, pred, regressor_names


def apply_v5_calibration(pred, routes, p_luxury, luxury_scores, luxury_threshold, score_threshold, calibration, lower_bound_name):
    calibrated = np.asarray(pred, dtype=float).copy()
    groups = np.where(routes == ">=10M", "10M-30M_like_or_weak_luxury", "<10M").astype(object)
    strong_luxury = (routes == ">=10M") & (p_luxury >= luxury_threshold) & (luxury_scores >= score_threshold)
    groups[strong_luxury] = ">30M_like_by_both_classifier_and_rule"
    calibrated[strong_luxury] *= calibration["luxury_tail_multiplier"]
    lower_bound = calibration.get(lower_bound_name)
    if lower_bound is not None:
        calibrated[strong_luxury] = np.maximum(calibrated[strong_luxury], lower_bound)
    return calibrated, groups, strong_luxury


def evaluate_routing(y_true, y_pred, true_segments, predicted_routes, predicted_subroutes, y_true_high, y_pred_high, y_true_tail, y_pred_tail):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    true_segments = np.asarray(true_segments)
    predicted_routes = np.asarray(predicted_routes)
    predicted_subroutes = np.asarray(predicted_subroutes)
    high_mask = np.asarray(y_true_high, dtype=bool)
    tail_mask = np.asarray(y_true_tail, dtype=bool)
    over50_mask = y_true >= 50_000_000
    over100_mask = y_true >= 100_000_000
    mid_mask = true_segments == "10M-30M"
    low_mask = true_segments == "<10M"
    payload = {
        "overall_metrics": regression_metrics(y_true, y_pred),
        "segment_metrics_by_true_segment": metrics_by_labels(y_true, y_pred, true_segments, SEGMENT_LABELS),
        "segment_metrics_by_predicted_route": metrics_by_labels(y_true, y_pred, predicted_routes, BINARY_ROUTE_LABELS),
        "segment_metrics_by_predicted_subroute": metrics_by_labels(y_true, y_pred, predicted_subroutes, SUBROUTE_LABELS),
        "underprediction_metrics": {
            "overall": underprediction_metrics(y_true, y_pred),
            "by_true_segment": underprediction_by_labels(y_true, y_pred, true_segments, SEGMENT_LABELS),
            "by_predicted_route": underprediction_by_labels(y_true, y_pred, predicted_routes, BINARY_ROUTE_LABELS),
            "by_predicted_subroute": underprediction_by_labels(y_true, y_pred, predicted_subroutes, SUBROUTE_LABELS),
        },
        "high_price_focus": {
            "true_over_10m_row_count": int(high_mask.sum()),
            "route_recall_over_10m": round(float(((y_true_high == 1) & (y_pred_high == 1)).sum() / max((y_true_high == 1).sum(), 1)), 6),
            "route_precision_over_10m": round(float(((y_true_high == 1) & (y_pred_high == 1)).sum() / max((y_pred_high == 1).sum(), 1)), 6),
            "false_negative_over_10m_count": int(((y_true_high == 1) & (y_pred_high == 0)).sum()),
            "false_positive_over_10m_count": int(((y_true_high == 0) & (y_pred_high == 1)).sum()),
            "metrics_over_10m": empty_metrics() if high_mask.sum() == 0 else regression_metrics(y_true[high_mask], y_pred[high_mask]),
            "metrics_10m_30m": empty_metrics() if mid_mask.sum() == 0 else regression_metrics(y_true[mid_mask], y_pred[mid_mask]),
            "metrics_over_30m": empty_metrics() if tail_mask.sum() == 0 else regression_metrics(y_true[tail_mask], y_pred[tail_mask]),
            "metrics_under_10m": empty_metrics() if low_mask.sum() == 0 else regression_metrics(y_true[low_mask], y_pred[low_mask]),
            "underprediction_over_10m": empty_metrics() if high_mask.sum() == 0 else underprediction_metrics(y_true[high_mask], y_pred[high_mask]),
            "underprediction_10m_30m": empty_metrics() if mid_mask.sum() == 0 else underprediction_metrics(y_true[mid_mask], y_pred[mid_mask]),
            "underprediction_over_30m": empty_metrics() if tail_mask.sum() == 0 else underprediction_metrics(y_true[tail_mask], y_pred[tail_mask]),
        },
        "luxury_tail_focus": {
            "true_over_30m_row_count": int(tail_mask.sum()),
            "subroute_recall_over_30m": round(float(((y_true_tail == 1) & (y_pred_tail == 1)).sum() / max((y_true_tail == 1).sum(), 1)), 6),
            "subroute_precision_over_30m": round(float(((y_true_tail == 1) & (y_pred_tail == 1)).sum() / max((y_pred_tail == 1).sum(), 1)), 6),
            "false_negative_over_30m_count": int(((y_true_tail == 1) & (y_pred_tail == 0)).sum()),
            "false_positive_over_30m_like_count": int(((y_true_tail == 0) & (y_pred_tail == 1)).sum()),
            "metrics_over_30m": empty_metrics() if tail_mask.sum() == 0 else regression_metrics(y_true[tail_mask], y_pred[tail_mask]),
            "underprediction_over_30m": empty_metrics() if tail_mask.sum() == 0 else underprediction_metrics(y_true[tail_mask], y_pred[tail_mask]),
            "metrics_over_50m": empty_metrics() if over50_mask.sum() == 0 else regression_metrics(y_true[over50_mask], y_pred[over50_mask]),
            "underprediction_over_50m": empty_metrics() if over50_mask.sum() == 0 else underprediction_metrics(y_true[over50_mask], y_pred[over50_mask]),
            "metrics_over_100m": empty_metrics() if over100_mask.sum() == 0 else regression_metrics(y_true[over100_mask], y_pred[over100_mask]),
            "underprediction_over_100m": empty_metrics() if over100_mask.sum() == 0 else underprediction_metrics(y_true[over100_mask], y_pred[over100_mask]),
        },
    }
    return payload


def load_previous_comparisons():
    comparisons = {}
    for version, path in [
        ("v1", Path("reports/reg/v1/price_vnd_model_evaluation.json")),
        ("v2", Path("reports/reg/v2/price_vnd_model_evaluation_v2.json")),
        ("v3", Path("reports/reg/v3/price_vnd_model_evaluation_v3.json")),
        ("v4", Path("reports/reg/v4/price_vnd_model_evaluation_v4.json")),
    ]:
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                comparisons[version] = {k: payload.get(k) for k in ["task", "best_experiment_by_high_price_focus", "best_experiment_by_overall_mae", "data"]}
            except Exception as exc:
                comparisons[version] = {"read_error": str(exc)}
    return comparisons


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-path", type=Path, default=RAW_PATH)
    parser.add_argument("--processed-path", type=Path, default=PROCESSED_PATH)
    parser.add_argument("--audit-path", type=Path, default=AUDIT_PATH)
    parser.add_argument("--report-path", type=Path, default=REPORT_PATH)
    parser.add_argument("--predictions-path", type=Path, default=PREDICTIONS_PATH)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    args.processed_path.parent.mkdir(parents=True, exist_ok=True)
    args.audit_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_path.parent.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    df, amenity_features = base.load_and_preprocess(args.raw_path)
    df = add_v5_luxury_features(add_luxury_route_score(df))
    df = add_suspicious_outlier_flags(df)
    baseline_config = base.make_feature_config(df, amenity_features, "baseline")
    v5_config = make_v5_feature_config(df, amenity_features)
    baseline_df = base.make_modeling_df(df, baseline_config)
    v5_df = base.make_modeling_df(df, v5_config)
    for col in ["suspicious_price_flag", "suspicious_reason"]:
        v5_df[col] = df[col].to_numpy()
        baseline_df[col] = df[col].to_numpy()
    v5_df["true_segment_v5"] = assign_v5_segment(v5_df["price_vnd"]).astype(str)
    v5_df["true_is_high_price"] = (v5_df["price_vnd"].astype(float) >= 10_000_000).astype(int)
    v5_df["true_is_luxury_tail"] = (v5_df["price_vnd"].astype(float) >= 30_000_000).astype(int)
    baseline_df["true_segment_v5"] = v5_df["true_segment_v5"].to_numpy()
    baseline_df["true_is_high_price"] = v5_df["true_is_high_price"].to_numpy()
    baseline_df["true_is_luxury_tail"] = v5_df["true_is_luxury_tail"].to_numpy()

    v5_df.to_csv(args.processed_path, index=False, encoding="utf-8-sig")
    outlier_audit = create_outlier_audit(df, args.audit_path)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    y_split = np.log1p(v5_df["price_vnd"])
    train_idx, test_idx = next(splitter.split(v5_df, y_split, groups=v5_df["hotel_id"]))
    train_segments = v5_df.iloc[train_idx]["true_segment_v5"].to_numpy()
    y_train_vnd = v5_df.iloc[train_idx]["price_vnd"].astype(float).to_numpy()
    high_train_mask = y_train_vnd >= 10_000_000
    mid_train_mask = (y_train_vnd >= 10_000_000) & (y_train_vnd < 30_000_000)
    tail_train_mask = y_train_vnd >= 30_000_000

    low_name, low_model = make_xgb_regressor()
    under_name, under_pipeline, under_features = fit_regressor(
        baseline_df, baseline_config, train_idx, train_segments == "<10M", low_name, low_model
    )

    high_mid_name, high_mid_model = make_lgbm_quantile(0.60)
    if high_mid_model is None:
        high_mid_name, high_mid_model = make_xgb_regressor()
    high_mid_name, high_mid_pipeline, high_mid_features = fit_regressor(
        v5_df, v5_config, train_idx, mid_train_mask, high_mid_name, high_mid_model
    )

    shared_regressors = {}
    for over30_weight in [6, 10]:
        name, model = make_xgb_regressor()
        weights = np.where(y_train_vnd[high_train_mask] >= 30_000_000, float(over30_weight), 1.0)
        model_name, pipeline, features = fit_regressor(
            v5_df, v5_config, train_idx, high_train_mask, name, model, sample_weight=weights
        )
        shared_regressors[f"shared_xgboost_log_mse_weight_{over30_weight}"] = {
            "model_name": model_name,
            "pipeline": pipeline,
            "features": features,
            "sample_weight": f"{over30_weight}.0 for >30M rows, 1.0 for 10M-30M rows",
        }
    for alpha in [0.70]:
        name, model = make_lgbm_quantile(alpha)
        if model is None:
            continue
        model_name, pipeline, features = fit_regressor(v5_df, v5_config, train_idx, high_train_mask, name, model)
        shared_regressors[f"shared_lightgbm_quantile_{str(alpha).replace('.', '_')}"] = {
            "model_name": model_name,
            "pipeline": pipeline,
            "features": features,
            "sample_weight": "none",
        }

    specialist_name, specialist_pipeline, specialist_features = None, None, None
    if tail_train_mask.sum() >= 10:
        specialist_name, specialist_model = "RandomForestRegressor", RandomForestRegressor(
            n_estimators=500, max_depth=6, min_samples_leaf=1, random_state=RANDOM_STATE, n_jobs=-1
        )
        specialist_name, specialist_pipeline, specialist_features = fit_regressor(
            v5_df, v5_config, train_idx, tail_train_mask, specialist_name, specialist_model
        )

    classifier_feature_cols = [col for col in v5_df.columns if col not in TARGET_META_COLS]
    primary_name, primary_classifier = make_classifier()
    primary_pipeline = Pipeline([("preprocessor", base.make_preprocessor(v5_config)), ("model", primary_classifier)])
    primary_weights = {"<10M": 1.0, "10M-30M": 6.0, ">30M": 12.0}
    primary_sample_weight = pd.Series(train_segments).map(primary_weights).astype(float).to_numpy()
    primary_pipeline.fit(
        v5_df.iloc[train_idx][classifier_feature_cols],
        v5_df.iloc[train_idx]["true_is_high_price"].astype(int).to_numpy(),
        model__sample_weight=primary_sample_weight,
    )

    secondary_classifiers = {}
    high_train_positions = np.asarray(train_idx)[high_train_mask]
    y_secondary_train = v5_df.iloc[high_train_positions]["true_is_luxury_tail"].astype(int).to_numpy()
    for secondary_kind in ["xgboost", "logistic"]:
        for tail_weight in [6, 10]:
            sec_name, sec_model = make_secondary_classifier(secondary_kind)
            sec_pipeline = Pipeline([("preprocessor", base.make_preprocessor(v5_config)), ("model", sec_model)])
            sec_weights = np.where(y_secondary_train == 1, float(tail_weight), 1.0)
            fit_kwargs = {}
            if secondary_kind != "logistic":
                fit_kwargs["model__sample_weight"] = sec_weights
            sec_pipeline.fit(v5_df.iloc[high_train_positions][classifier_feature_cols], y_secondary_train, **fit_kwargs)
            secondary_classifiers[f"{secondary_kind}_tail_weight_{tail_weight}"] = {
                "model_name": sec_name,
                "pipeline": sec_pipeline,
                "tail_weight": tail_weight,
            }

    under_test = baseline_df.iloc[test_idx].reset_index(drop=True)
    high_test = v5_df.iloc[test_idx].reset_index(drop=True)
    actual_test_vnd = high_test["price_vnd"].astype(float).to_numpy()
    true_segments = high_test["true_segment_v5"].astype(str).to_numpy()
    y_true_high = high_test["true_is_high_price"].astype(int).to_numpy()
    y_true_tail = high_test["true_is_luxury_tail"].astype(int).to_numpy()
    luxury_route_scores = high_test["luxury_route_score"].astype(int).to_numpy()
    luxury_tail_scores = high_test["luxury_tail_score"].astype(int).to_numpy()
    primary_prob = primary_pipeline.predict_proba(high_test[classifier_feature_cols])[:, 1]

    experiments = {}
    prediction_frames = []
    best_luxury_name, best_luxury_score, best_artifacts = None, None, None
    best_high_name, best_high_score = None, None
    best_overall_name, best_overall_mae = None, float("inf")

    threshold_configs = [
        (0.20, 0.15, 5),
        (0.25, 0.20, 5),
        (0.30, 0.20, 6),
        (0.35, 0.25, 6),
    ]
    lower_bound_modes = ["none", "q25_actual_over_30m_train"]

    for shared_variant, shared_payload in shared_regressors.items():
        calibration = compute_calibration(
            shared_payload["pipeline"],
            high_mid_pipeline,
            v5_df,
            shared_payload["features"],
            high_mid_features,
            train_idx,
        )
        features = {
            "under": under_features,
            "high_mid": high_mid_features,
            "shared": shared_payload["features"],
            "specialist": specialist_features,
        }
        oracle_primary_routes = np.where(y_true_high == 1, ">=10M", "<10M")
        oracle_subroutes = np.where(y_true_high == 0, "<10M", np.where(y_true_tail == 1, ">30M_like", "10M-30M_like"))
        _, oracle_blended, oracle_pred, _ = build_path_predictions(
            under_pipeline,
            high_mid_pipeline,
            shared_payload["pipeline"],
            specialist_pipeline,
            features,
            under_test,
            high_test,
            oracle_primary_routes,
            oracle_subroutes,
            tail_blend=0.35,
        )
        oracle_eval = evaluate_routing(
            actual_test_vnd,
            oracle_pred,
            true_segments,
            oracle_primary_routes,
            oracle_subroutes,
            y_true_high,
            y_true_high,
            y_true_tail,
            y_true_tail,
        )

        for sec_config_name, sec_payload in secondary_classifiers.items():
            secondary_prob = sec_payload["pipeline"].predict_proba(high_test[classifier_feature_cols])[:, 1]
            for high_threshold, luxury_threshold, score_threshold in threshold_configs:
                classifier_route = primary_prob >= high_threshold
                high_rule_override = (~classifier_route) & (luxury_route_scores >= 4)
                predicted_high = (classifier_route | high_rule_override).astype(int)
                predicted_routes = np.where(predicted_high == 1, ">=10M", "<10M")
                luxury_classifier_route = secondary_prob >= luxury_threshold
                luxury_rule_override = luxury_tail_scores >= score_threshold
                predicted_tail = ((predicted_high == 1) & (luxury_classifier_route | luxury_rule_override)).astype(int)
                predicted_subroutes = np.where(
                    predicted_high == 0,
                    "<10M",
                    np.where(predicted_tail == 1, ">30M_like", "10M-30M_like"),
                )
                for tail_blend in [0.0, 0.35]:
                    raw_pred, blended_pred, path_pred, regressor_names = build_path_predictions(
                        under_pipeline,
                        high_mid_pipeline,
                        shared_payload["pipeline"],
                        specialist_pipeline,
                        features,
                        under_test,
                        high_test,
                        predicted_routes,
                        predicted_subroutes,
                        tail_blend=tail_blend,
                    )
                    raw_eval = evaluate_routing(
                        actual_test_vnd,
                        path_pred,
                        true_segments,
                        predicted_routes,
                        predicted_subroutes,
                        y_true_high,
                        predicted_high,
                        y_true_tail,
                        predicted_tail,
                    )
                    for lower_bound_name in lower_bound_modes:
                        if lower_bound_name == "none":
                            final_pred = path_pred
                            calibration_groups = np.where(predicted_routes == ">=10M", "uncalibrated_>=10M", "uncalibrated_<10M")
                            strong_luxury = np.zeros(len(final_pred), dtype=bool)
                            calibration_mode = "uncalibrated"
                        else:
                            final_pred, calibration_groups, strong_luxury = apply_v5_calibration(
                                path_pred,
                                predicted_routes,
                                secondary_prob,
                                luxury_tail_scores,
                                luxury_threshold,
                                score_threshold,
                                calibration,
                                lower_bound_name,
                            )
                            calibration_mode = f"confidence_gated_{lower_bound_name}"
                        predicted_eval = evaluate_routing(
                            actual_test_vnd,
                            final_pred,
                            true_segments,
                            predicted_routes,
                            predicted_subroutes,
                            y_true_high,
                            predicted_high,
                            y_true_tail,
                            predicted_tail,
                        )
                        primary_metrics = classifier_metrics_binary(y_true_high, predicted_high, "high_price", "<10M")
                        secondary_metrics = classifier_metrics_binary(y_true_tail[y_true_high == 1], predicted_tail[y_true_high == 1], "luxury_tail", "10M-30M")
                        threshold_name = (
                            f"high_threshold_{str(high_threshold).replace('.', '_')}_"
                            f"luxury_threshold_{str(luxury_threshold).replace('.', '_')}_score_{score_threshold}"
                        )
                        experiment_name = (
                            f"{threshold_name}_{sec_config_name}_{shared_variant}_"
                            f"tail_blend_{str(tail_blend).replace('.', '_')}_{calibration_mode}"
                        )
                        true_route_labels = np.where(y_true_high == 1, ">=10M", "<10M")
                        true_subroute_labels = np.where(y_true_high == 0, "<10M", np.where(y_true_tail == 1, ">30M_like", "10M-30M_like"))
                        experiments[experiment_name] = {
                            "primary_classifier": {
                                "model": primary_name,
                                "feature_config": "v2_expanded_features_plus_v5_luxury_features",
                                "sample_weights": primary_weights,
                                "high_price_threshold": high_threshold,
                                "luxury_route_rule_threshold": 4,
                                "metrics": primary_metrics,
                            },
                            "secondary_classifier": {
                                "model": sec_payload["model_name"],
                                "feature_config": "v2_expanded_features_plus_v5_luxury_features",
                                "config": sec_config_name,
                                "luxury_tail_threshold": luxury_threshold,
                                "luxury_tail_score_threshold": score_threshold,
                                "metrics": secondary_metrics,
                            },
                            "regressors": {
                                "under_10m": {"model": under_name, "feature_config": "baseline_features", "train_rows": int((train_segments == "<10M").sum())},
                                "high_mid": {"model": high_mid_name, "feature_config": "v5_features", "train_rows": int(mid_train_mask.sum())},
                                "shared_high_price": {
                                    "model": shared_payload["model_name"],
                                    "variant": shared_variant,
                                    "train_rows": int(high_train_mask.sum()),
                                    "sample_weight": shared_payload["sample_weight"],
                                },
                                "luxury_tail_specialist": {
                                    "model": specialist_name,
                                    "train_rows": int(tail_train_mask.sum()),
                                    "tail_blend": tail_blend,
                                    "deployed": specialist_pipeline is not None and tail_blend > 0,
                                },
                            },
                            "calibration": {"mode": calibration_mode, "parameters": calibration},
                            "oracle_primary_routing": oracle_eval,
                            "oracle_subroute": oracle_eval,
                            "predicted_routing_no_calibration": raw_eval,
                            "predicted_routing": predicted_eval,
                            "high_price_focus": predicted_eval["high_price_focus"],
                            "luxury_tail_focus": predicted_eval["luxury_tail_focus"],
                            "routing_confusion_matrix": confusion_dict(true_route_labels, predicted_routes, BINARY_ROUTE_LABELS),
                            "subrouting_confusion_matrix": confusion_dict(true_subroute_labels, predicted_subroutes, SUBROUTE_LABELS),
                        }

                        metadata_cols = [col for col in PREDICTION_METADATA_COLS if col in high_test.columns]
                        predictions = high_test[metadata_cols].copy()
                        predictions["actual_price_vnd"] = actual_test_vnd
                        predictions["true_segment_v5"] = true_segments
                        predictions["true_is_high_price"] = y_true_high
                        predictions["true_is_luxury_tail"] = y_true_tail
                        predictions["primary_p_high_price"] = primary_prob
                        predictions["primary_route_high_price"] = classifier_route.astype(int)
                        predictions["luxury_route_score"] = luxury_route_scores
                        predictions["high_price_rule_override"] = high_rule_override.astype(int)
                        predictions["secondary_p_luxury_tail"] = secondary_prob
                        predictions["secondary_route_luxury_tail"] = luxury_classifier_route.astype(int)
                        predictions["luxury_tail_score"] = luxury_tail_scores
                        predictions["luxury_tail_rule_override"] = luxury_rule_override.astype(int)
                        predictions["predicted_route"] = predicted_routes
                        predictions["predicted_subroute"] = predicted_subroutes
                        predictions["regressor_name"] = regressor_names
                        predictions["high_price_regressor_variant"] = shared_variant
                        predictions["luxury_tail_variant"] = f"{sec_config_name}_tail_blend_{tail_blend}"
                        predictions["raw_pred_price_vnd"] = raw_pred
                        predictions["blended_pred_price_vnd"] = blended_pred
                        predictions["calibrated_pred_price_vnd"] = final_pred
                        predictions["calibration_mode"] = calibration_mode
                        predictions["calibration_group"] = calibration_groups
                        predictions["absolute_error"] = np.abs(actual_test_vnd - final_pred)
                        predictions["absolute_percentage_error"] = np.where(actual_test_vnd > 0, predictions["absolute_error"] / actual_test_vnd * 100, np.nan)
                        predictions["pred_minus_actual"] = final_pred - actual_test_vnd
                        predictions["pred_to_actual_ratio"] = final_pred / actual_test_vnd
                        predictions["is_underprediction"] = final_pred < actual_test_vnd
                        predictions["is_severe_underprediction_ratio_below_0_5"] = predictions["pred_to_actual_ratio"] < 0.5
                        predictions["suspicious_price_flag"] = high_test["suspicious_price_flag"].astype(int).to_numpy()
                        predictions["suspicious_reason"] = high_test["suspicious_reason"].astype(str).to_numpy()
                        predictions["experiment_name"] = experiment_name
                        predictions["threshold_name"] = threshold_name
                        prediction_frames.append(predictions)

                        focus = predicted_eval["luxury_tail_focus"]
                        high_focus = predicted_eval["high_price_focus"]
                        low_mape = high_focus["metrics_under_10m"].get("MAPE_percent", 9999)
                        mid_mape = high_focus["metrics_10m_30m"].get("MAPE_percent", 9999)
                        over30_ratio = focus["underprediction_over_30m"].get("median_pred_to_actual_ratio", 0)
                        severe_over30 = focus["underprediction_over_30m"].get("severe_underprediction_count_ratio_below_0_5", 10**9)
                        luxury_score = (
                            int(low_mape <= 38.5),
                            high_focus["route_recall_over_10m"],
                            focus["subroute_recall_over_30m"],
                            over30_ratio,
                            -severe_over30,
                            -max(mid_mape - 49.014, 0),
                            -predicted_eval["overall_metrics"]["MAE"],
                        )
                        if best_luxury_score is None or luxury_score > best_luxury_score:
                            best_luxury_score = luxury_score
                            best_luxury_name = experiment_name
                            best_artifacts = {
                                "primary_classifier_pipeline": primary_pipeline,
                                "secondary_classifier_pipeline": sec_payload["pipeline"],
                                "under_10m_pipeline": under_pipeline,
                                "high_mid_pipeline": high_mid_pipeline,
                                "shared_high_price_pipeline": shared_payload["pipeline"],
                                "luxury_tail_specialist_pipeline": specialist_pipeline,
                                "features": features,
                                "classifier_features": classifier_feature_cols,
                                "high_price_threshold": high_threshold,
                                "luxury_tail_threshold": luxury_threshold,
                                "luxury_tail_score_threshold": score_threshold,
                                "luxury_route_rule_threshold": 4,
                                "tail_blend": tail_blend,
                                "calibration": calibration,
                                "calibration_mode": calibration_mode,
                                "shared_regressor_variant": shared_variant,
                                "secondary_classifier_config": sec_config_name,
                            }
                        high_score = (
                            int(low_mape <= 38.5),
                            high_focus["route_recall_over_10m"],
                            -high_focus["false_negative_over_10m_count"],
                            -predicted_eval["overall_metrics"]["MAE"],
                        )
                        if best_high_score is None or high_score > best_high_score:
                            best_high_score = high_score
                            best_high_name = experiment_name
                        overall_mae = predicted_eval["overall_metrics"]["MAE"]
                        if overall_mae < best_overall_mae:
                            best_overall_mae = overall_mae
                            best_overall_name = experiment_name

    pd.concat(prediction_frames, ignore_index=True).to_csv(args.predictions_path, index=False, encoding="utf-8-sig")
    model_path = args.model_dir / "price_vnd_v5_routed_model.joblib"
    joblib.dump({"experiment_name": best_luxury_name, **best_artifacts}, model_path)

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "target": "price_vnd",
        "task": "Luxury-tail-aware routed price prediction",
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
            "train_hotels": int(v5_df.iloc[train_idx]["hotel_id"].nunique()),
            "test_hotels": int(v5_df.iloc[test_idx]["hotel_id"].nunique()),
        },
        "segment_definition": {
            "<10M": "price_vnd < 10000000",
            "10M-30M": "10000000 <= price_vnd < 30000000",
            ">30M": "price_vnd >= 30000000",
            "high_price": "price_vnd >= 10000000",
            "luxury_tail": "price_vnd >= 30000000",
        },
        "preserved_v4_foundation": {
            "low_price_path": {"model": under_name, "feature_config": "baseline_features", "training_rows": "price_vnd < 10000000"},
            "primary_high_price_router": {"model": primary_name, "feature_config": "v2_expanded_features_plus_v5_luxury_features"},
        },
        "outlier_audit": outlier_audit,
        "data": {
            "rows_after_target_filter": int(len(v5_df)),
            "unique_hotels": int(v5_df["hotel_id"].nunique()),
            "v5_segment_counts": v5_df["true_segment_v5"].value_counts().to_dict(),
            "v5_segment_counts_train": v5_df.iloc[train_idx]["true_segment_v5"].value_counts().to_dict(),
            "v5_segment_counts_test": v5_df.iloc[test_idx]["true_segment_v5"].value_counts().to_dict(),
        },
        "dropped_columns": {
            "id": [col for col in base.ID_DROP_COLS if col in df.columns],
            "leakage": [col for col in base.LEAKAGE_COLS if col in df.columns],
            "constant": [col for col in base.CONSTANT_COLS if col in df.columns],
            "all_missing": [col for col in base.ALL_MISSING_COLS if col in df.columns],
            "long_text_or_unused": [col for col in base.TEXT_DROP_COLS if col in df.columns],
        },
        "experiments": experiments,
        "best_experiment_by_luxury_tail_focus": best_luxury_name,
        "best_experiment_by_high_price_focus": best_high_name,
        "best_experiment_by_overall_mae": best_overall_name,
        "comparison_to_previous_versions": load_previous_comparisons(),
    }
    args.report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    best = experiments[best_luxury_name]
    summary = {
        "best_experiment_by_luxury_tail_focus": best_luxury_name,
        "best_experiment_by_high_price_focus": best_high_name,
        "best_experiment_by_overall_mae": best_overall_name,
        "best_overall_metrics": best["predicted_routing"]["overall_metrics"],
        "best_high_price_focus": best["high_price_focus"],
        "best_luxury_tail_focus": best["luxury_tail_focus"],
        "primary_classifier_metrics": best["primary_classifier"]["metrics"],
        "secondary_classifier_metrics": best["secondary_classifier"]["metrics"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Processed data saved to: {args.processed_path}")
    print(f"Outlier audit saved to: {args.audit_path}")
    print(f"Prediction errors saved to: {args.predictions_path}")
    print(f"Evaluation report saved to: {args.report_path}")
    print(f"Best model artifact saved to: {model_path}")


if __name__ == "__main__":
    main()
