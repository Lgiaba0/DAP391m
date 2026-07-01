import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.feature_extraction.text import TfidfVectorizer


SOURCE_PATH = Path("data/processed/classify/v1/vietnam_price_classification_v1.csv")
FALLBACK_SOURCE_PATH = Path("data/processed/reg/v5/vietnam_price_vnd_modeling_v5.csv")
OUTPUT_DATA_PATH = Path("data/processed/classify/v2/vietnam_price_classification_v2.csv")
REPORT_PATH = Path("reports/classify/v2/price_classification_evaluation_v2.json")
PREDICTIONS_PATH = Path("reports/classify/v2/price_classification_predictions_v2.csv")
ERROR_ANALYSIS_PATH = Path("reports/classify/v2/price_classification_error_analysis_v2.csv")
MODEL_PATH = Path("models/classify/v2/price_classification_v2_model.joblib")
V1_REPORT_PATH = Path("reports/classify/v1/price_classification_evaluation_v1.json")
RANDOM_STATE = 42

PRICE_CLASS_LABELS = {
    0: "budget",
    1: "economy",
    2: "mid_range",
    3: "upscale",
    4: "premium_luxury",
}
CLASS_IDS = list(PRICE_CLASS_LABELS.keys())
CLASS_NAMES = [PRICE_CLASS_LABELS[class_id] for class_id in CLASS_IDS]
PROBA_COLS = [f"pred_proba_{label}" for label in CLASS_NAMES]

LEAKAGE_COLS = [
    "price_vnd",
    "price_usd",
    "cheapest_room_usd",
    "most_expensive_usd",
    "cheapest_room_vnd",
    "price_range_usd",
    "price_to_star",
    "value_index",
    "price_category",
]
ID_DROP_COLS = [
    "hotel_id",
    "block_id",
    "source_url",
    "hotel_name",
    "street_address",
]
TARGET_DERIVED_COLS = [
    "true_segment_v5",
    "true_is_high_price",
    "true_is_luxury_tail",
    "price_class_id",
    "price_class_label",
    "suspicious_price_flag",
    "suspicious_reason",
]
TEXT_FEATURES = ["room_name", "description"]
BASELINE_CATEGORICAL_FEATURES = [
    "city",
    "source_city",
    "district",
    "region",
    "property_type",
    "room_type_extracted",
    "bed_type",
]
METADATA_COLS = [
    "hotel_id",
    "room_name",
    "city",
    "source_city",
    "district",
    "region",
    "property_type",
    "star_rating_clean",
]
MARKET_CONTEXT_SPECS = {
    "city_median_class_train": ["city"],
    "city_mean_class_train": ["city"],
    "source_city_median_class_train": ["source_city"],
    "region_median_class_train": ["region"],
    "property_type_median_class_train": ["property_type"],
    "star_rating_clean_median_class_train": ["star_rating_clean"],
    "city_property_type_median_class_train": ["city", "property_type"],
    "source_city_property_type_median_class_train": ["source_city", "property_type"],
    "city_star_median_class_train": ["city", "star_rating_clean"],
    "property_type_star_median_class_train": ["property_type", "star_rating_clean"],
}
V1_BASELINE = {
    "macro_f1": 0.654782,
    "mid_range_f1": 0.506108,
    "mid_range_recall": 0.469256,
    "balanced_accuracy": 0.650007,
    "adjacent_band_accuracy": 0.981050,
    "severe_misclassification_rate": 0.018950,
    "premium_luxury_recall": 0.699301,
}


def assign_price_class(price_vnd):
    if price_vnd < 500_000:
        return 0
    if price_vnd < 1_000_000:
        return 1
    if price_vnd < 2_000_000:
        return 2
    if price_vnd < 5_000_000:
        return 3
    return 4


def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def class_distribution(series):
    counts = pd.Series(series).value_counts().reindex(CLASS_IDS, fill_value=0).astype(int)
    total = max(int(counts.sum()), 1)
    return {
        PRICE_CLASS_LABELS[class_id]: {
            "class_id": int(class_id),
            "rows": int(counts.loc[class_id]),
            "share": round(float(counts.loc[class_id] / total), 6),
        }
        for class_id in CLASS_IDS
    }


def safe_numeric(df, col, default=0.0):
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def safe_text(df, col):
    if col in df.columns:
        return df[col].astype("string").fillna("").str.lower()
    return pd.Series("", index=df.index, dtype="string")


def add_mid_market_features(df):
    out = df.copy()
    property_type = safe_text(out, "property_type")
    room_name = safe_text(out, "room_name")
    star = safe_numeric(out, "star_rating_clean")
    room_tier = safe_numeric(out, "room_quality_tier")
    luxury_tail = safe_numeric(out, "luxury_tail_score")
    luxury_route = safe_numeric(out, "luxury_route_score")
    luxury_amenity = safe_numeric(out, "luxury_amenity_score")
    amenity_count = safe_numeric(out, "amenities_count")
    max_persons = safe_numeric(out, "max_persons")
    bedroom_count = safe_numeric(out, "bedroom_count")
    has_pool = safe_numeric(out, "amenity_outdoor_pool") + safe_numeric(out, "amenity_indoor_pool")
    has_restaurant = safe_numeric(out, "amenity_restaurant")
    has_fitness = safe_numeric(out, "amenity_fitness_center")
    has_spa = safe_numeric(out, "amenity_spa")

    mid_property_types = {"hotel", "apartment", "resort", "homestay"}
    room_quality_mid = room_tier.between(1, 3, inclusive="both")
    deluxe_or_superior = room_name.str.contains("deluxe|superior|view", regex=True, na=False)

    out["is_standard_hotel_room"] = ((property_type == "hotel") & room_quality_mid).astype(int)
    out["is_deluxe_without_luxury_tail"] = (deluxe_or_superior & (luxury_tail <= 1)).astype(int)
    out["is_3_to_4_star_full_service"] = (
        star.between(3, 4, inclusive="both")
        & amenity_count.between(12, 45, inclusive="both")
        & luxury_amenity.between(2, 6, inclusive="both")
    ).astype(int)
    out["is_midscale_property_type"] = (property_type.isin(mid_property_types) & (luxury_tail <= 2)).astype(int)
    out["has_midscale_amenity_profile"] = (luxury_amenity.between(3, 5, inclusive="both") & (luxury_tail <= 2)).astype(int)
    out["midscale_amenity_score"] = (
        luxury_amenity.clip(0, 6)
        + has_pool.clip(0, 1)
        + has_restaurant.clip(0, 1)
        + has_fitness.clip(0, 1)
        - (2 * luxury_tail.clip(0, 4))
        - has_spa.clip(0, 1)
    )
    out["luxury_signal_without_premium_capacity"] = (
        (luxury_route >= 2) & (max_persons <= 2) & (bedroom_count <= 1)
    ).astype(int)
    out["budget_like_low_signal"] = (
        (star <= 2.5)
        & (luxury_amenity <= 1)
        & (luxury_tail == 0)
        & (amenity_count <= 18)
    ).astype(int)
    out["economy_mid_boundary_signal"] = (
        star.between(2.5, 3.5, inclusive="both")
        & (luxury_amenity <= 3)
        & (luxury_tail <= 1)
        & (max_persons <= 3)
    ).astype(int)
    out["mid_upscale_boundary_signal"] = (
        star.between(3.5, 4.5, inclusive="both")
        & luxury_amenity.between(4, 7, inclusive="both")
        & (luxury_tail <= 2)
    ).astype(int)
    return out


class MarketContextEncoder:
    def __init__(self, specs):
        self.specs = specs
        self.global_median_ = None
        self.maps_ = {}
        self.methods_ = {}

    @staticmethod
    def _key_frame(df, cols):
        key_df = pd.DataFrame(index=df.index)
        for col in cols:
            if col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    key_df[col] = pd.to_numeric(df[col], errors="coerce").round(2).astype("string").fillna("Unknown")
                else:
                    key_df[col] = df[col].astype("string").fillna("Unknown")
            else:
                key_df[col] = "Unknown"
        return key_df.astype(str).agg("||".join, axis=1)

    def fit(self, df, y):
        target = pd.Series(y, index=df.index, name="target").astype(float)
        self.global_median_ = float(target.median())
        for feature_name, cols in self.specs.items():
            method = "mean" if feature_name.endswith("_mean_class_train") else "median"
            keys = self._key_frame(df, cols)
            grouped = target.groupby(keys).mean() if method == "mean" else target.groupby(keys).median()
            self.maps_[feature_name] = grouped.to_dict()
            self.methods_[feature_name] = method
        return self

    def transform(self, df):
        out = pd.DataFrame(index=df.index)
        for feature_name, cols in self.specs.items():
            keys = self._key_frame(df, cols)
            out[feature_name] = keys.map(self.maps_[feature_name]).fillna(self.global_median_).astype(float)
        return out


def add_market_context_features(train_df, test_df, y_train):
    encoder = MarketContextEncoder(MARKET_CONTEXT_SPECS).fit(train_df, y_train)
    train_out = train_df.copy()
    test_out = test_df.copy()
    train_context = encoder.transform(train_df)
    test_context = encoder.transform(test_df)
    for col in train_context.columns:
        train_out[col] = train_context[col]
        test_out[col] = test_context[col]
    return train_out, test_out, encoder


def build_feature_config(df, feature_set):
    excluded = set(LEAKAGE_COLS + ID_DROP_COLS + TARGET_DERIVED_COLS)
    text_features = [col for col in TEXT_FEATURES if col in df.columns]

    if feature_set not in {"expanded_luxury_features", "market_context_features", "mid_market_features"}:
        raise ValueError(f"Unknown feature set: {feature_set}")

    candidate_cols = [col for col in df.columns if col not in excluded and col not in text_features]
    categorical_features = [
        col
        for col in BASELINE_CATEGORICAL_FEATURES + ["local_currency", "score_tier"]
        if col in candidate_cols
    ]
    numeric_features = []
    extra_categorical = []
    for col in candidate_cols:
        if col in categorical_features:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_features.append(col)
        else:
            extra_categorical.append(col)
    categorical_features = categorical_features + extra_categorical

    return {
        "feature_set": feature_set,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "text_features": text_features,
    }


def make_preprocessor(config):
    transformers = []
    if config["numeric_features"]:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler(with_mean=False)),
                    ]
                ),
                config["numeric_features"],
            )
        )
    if config["categorical_features"]:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                        ("onehot", make_one_hot_encoder()),
                    ]
                ),
                config["categorical_features"],
            )
        )
    if "room_name" in config["text_features"]:
        transformers.append(
            ("room_name_tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=3, max_features=300), "room_name")
        )
    if "description" in config["text_features"]:
        transformers.append(
            ("description_tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=10, max_features=500), "description")
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def prepare_features(df, feature_cols):
    X = df[feature_cols].copy()
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors="coerce")
        else:
            X[col] = X[col].astype("string").fillna("")
    return X


def get_lgbm_classifier(class_weight=None, objective=None):
    from lightgbm import LGBMClassifier

    params = {
        "n_estimators": 500,
        "learning_rate": 0.04,
        "num_leaves": 31,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbose": -1,
    }
    if class_weight is not None:
        params["class_weight"] = class_weight
    if objective is not None:
        params["objective"] = objective
    return LGBMClassifier(**params)


def get_xgb_classifier(binary=False):
    from xgboost import XGBClassifier

    params = {
        "n_estimators": 500,
        "learning_rate": 0.04,
        "max_depth": 5,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "eval_metric": "logloss" if binary else "mlogloss",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "tree_method": "hist",
    }
    if not binary:
        params["objective"] = "multi:softprob"
    else:
        params["objective"] = "binary:logistic"
    return XGBClassifier(**params)


def compute_weighted_samples(y_train, config_name):
    weights = compute_sample_weight(class_weight="balanced", y=y_train).astype(float)
    multipliers = {
        "weight_config_baseline": {},
        "weight_config_mid_1_5": {2: 1.5},
        "weight_config_mid_2_0": {2: 2.0},
        "weight_config_mid_1_5_premium_1_25": {2: 1.5, 4: 1.25},
    }[config_name]
    for class_id, multiplier in multipliers.items():
        weights[np.asarray(y_train) == class_id] *= multiplier
    return weights


def build_direct_candidates():
    candidates = {
        "dummy_most_frequent": {
            "model_factory": lambda: DummyClassifier(strategy="most_frequent"),
            "model_family": "dummy",
            "sample_weight": None,
        },
        "random_forest_weight_config_mid_1_5": {
            "model_factory": lambda: RandomForestClassifier(
                n_estimators=500,
                class_weight=None,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "model_family": "direct_multiclass_random_forest",
            "sample_weight": "weight_config_mid_1_5",
        },
        "extra_trees_weight_config_mid_1_5": {
            "model_factory": lambda: ExtraTreesClassifier(
                n_estimators=700,
                class_weight=None,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "model_family": "direct_multiclass_extra_trees",
            "sample_weight": "weight_config_mid_1_5",
        }
    }
    try:
        get_lgbm_classifier()
        for weight_config in [
            "weight_config_baseline",
            "weight_config_mid_1_5",
            "weight_config_mid_2_0",
            "weight_config_mid_1_5_premium_1_25",
        ]:
            candidates[f"lightgbm_{weight_config}"] = {
                "model_factory": lambda: get_lgbm_classifier(class_weight=None),
                "model_family": "direct_multiclass_lightgbm",
                "sample_weight": weight_config,
            }
    except Exception as exc:
        candidates["lightgbm_unavailable"] = {"unavailable_reason": str(exc)}

    try:
        get_xgb_classifier()
        for weight_config in [
            "weight_config_baseline",
            "weight_config_mid_1_5",
            "weight_config_mid_2_0",
            "weight_config_mid_1_5_premium_1_25",
        ]:
            candidates[f"xgboost_{weight_config}"] = {
                "model_factory": lambda: get_xgb_classifier(binary=False),
                "model_family": "direct_multiclass_xgboost",
                "sample_weight": weight_config,
            }
    except Exception as exc:
        candidates["xgboost_unavailable"] = {"unavailable_reason": str(exc)}
    return candidates


def near_boundary_info(price_vnd, class_id):
    boundaries = np.array([500_000, 1_000_000, 2_000_000, 5_000_000], dtype=float)
    distances = np.abs(boundaries - float(price_vnd))
    nearest_boundary = float(boundaries[int(np.argmin(distances))])
    distance = float(np.min(distances))
    band_width = {
        0: 500_000,
        1: 500_000,
        2: 1_000_000,
        3: 3_000_000,
        4: 1_000_000,
    }[int(class_id)]
    is_near = distance <= 0.15 * band_width
    if int(class_id) == 4:
        is_near = abs(float(price_vnd) - 5_000_000) <= 1_000_000
    return nearest_boundary, distance, bool(is_near)


def add_boundary_columns(predictions):
    boundary_values = predictions.apply(
        lambda row: near_boundary_info(row["actual_price_vnd"], row["actual_price_class_id"]),
        axis=1,
        result_type="expand",
    )
    predictions["nearest_boundary_vnd"] = boundary_values[0].astype(float)
    predictions["distance_to_nearest_boundary_vnd"] = boundary_values[1].astype(float)
    predictions["is_near_boundary"] = boundary_values[2].astype(int)
    return predictions


def proba_to_top_metrics(proba):
    if proba.size == 0:
        return np.zeros(0), np.zeros(0)
    sorted_proba = np.sort(proba, axis=1)
    top1 = sorted_proba[:, -1]
    top2 = sorted_proba[:, -2] if proba.shape[1] > 1 else np.zeros(len(proba))
    return top1, top1 - top2


def evaluate_classifier(y_true, y_pred, prices=None):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=CLASS_IDS,
        zero_division=0,
    )
    abs_error = np.abs(y_pred - y_true)
    per_class = {
        PRICE_CLASS_LABELS[class_id]: {
            "class_id": int(class_id),
            "precision": round(float(precision[i]), 6),
            "recall": round(float(recall[i]), 6),
            "f1": round(float(f1[i]), 6),
            "support": int(support[i]),
        }
        for i, class_id in enumerate(CLASS_IDS)
    }
    class_mae = {}
    for class_id in CLASS_IDS:
        mask = y_true == class_id
        class_mae[f"mae_class_error_{PRICE_CLASS_LABELS[class_id]}"] = round(float(np.mean(abs_error[mask])), 6) if mask.any() else None

    near_boundary_rate = None
    far_boundary_rate = None
    if prices is not None:
        near = np.array([near_boundary_info(price, class_id)[2] for price, class_id in zip(prices, y_true)], dtype=bool)
        error = y_true != y_pred
        near_boundary_rate = round(float(error[near].mean()), 6) if near.any() else None
        far_boundary_rate = round(float(error[~near].mean()), 6) if (~near).any() else None

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 6),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 6),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 6),
        "per_class": per_class,
        "confusion_matrix": {
            "labels": CLASS_NAMES,
            "matrix": confusion_matrix(y_true, y_pred, labels=CLASS_IDS).astype(int).tolist(),
            "rows_are_actual_columns_are_predicted": True,
        },
        "ordinal_metrics": {
            "mean_absolute_class_error": round(float(np.mean(abs_error)), 6),
            "median_absolute_class_error": round(float(np.median(abs_error)), 6),
            "adjacent_band_accuracy": round(float(np.mean(abs_error <= 1)), 6),
            "severe_misclassification_rate": round(float(np.mean(abs_error >= 2)), 6),
        },
        "business_metrics": {
            "premium_luxury_recall": round(float(per_class["premium_luxury"]["recall"]), 6),
            "premium_luxury_precision": round(float(per_class["premium_luxury"]["precision"]), 6),
            "budget_recall": round(float(per_class["budget"]["recall"]), 6),
            "high_to_low_confusion_count": int(((y_true == 4) & (y_pred <= 1)).sum()),
            "low_to_high_confusion_count": int(((y_true <= 1) & (y_pred == 4)).sum()),
        },
        "v2_boundary_metrics": {
            "mid_range_recall": round(float(per_class["mid_range"]["recall"]), 6),
            "mid_range_precision": round(float(per_class["mid_range"]["precision"]), 6),
            "mid_range_f1": round(float(per_class["mid_range"]["f1"]), 6),
            "economy_mid_boundary_error_count": int((((y_true == 1) & (y_pred == 2)) | ((y_true == 2) & (y_pred == 1))).sum()),
            "mid_upscale_boundary_error_count": int((((y_true == 2) & (y_pred == 3)) | ((y_true == 3) & (y_pred == 2))).sum()),
            "budget_economy_boundary_error_count": int((((y_true == 0) & (y_pred == 1)) | ((y_true == 1) & (y_pred == 0))).sum()),
            "upscale_premium_boundary_error_count": int((((y_true == 3) & (y_pred == 4)) | ((y_true == 4) & (y_pred == 3))).sum()),
            "adjacent_error_count": int((abs_error == 1).sum()),
            "non_adjacent_error_count": int((abs_error >= 2).sum()),
            "near_boundary_error_rate": near_boundary_rate,
            "far_from_boundary_error_rate": far_boundary_rate,
            **class_mae,
        },
    }


def selection_key(metrics):
    ordinal = metrics["ordinal_metrics"]
    business = metrics["business_metrics"]
    v2 = metrics["v2_boundary_metrics"]
    return (
        metrics["macro_f1"],
        v2["mid_range_f1"],
        v2["mid_range_recall"],
        metrics["balanced_accuracy"],
        ordinal["adjacent_band_accuracy"],
        -ordinal["severe_misclassification_rate"],
        business["premium_luxury_recall"],
        metrics["weighted_f1"],
        metrics["accuracy"],
    )


def ordinal_safety_key(metrics):
    ordinal = metrics["ordinal_metrics"]
    return (
        ordinal["adjacent_band_accuracy"],
        -ordinal["severe_misclassification_rate"],
        metrics["macro_f1"],
        metrics["v2_boundary_metrics"]["mid_range_f1"],
    )


def business_safety_key(metrics):
    business = metrics["business_metrics"]
    ordinal = metrics["ordinal_metrics"]
    return (
        -ordinal["severe_misclassification_rate"],
        -business["high_to_low_confusion_count"],
        -business["low_to_high_confusion_count"],
        business["premium_luxury_recall"],
    )


def predict_proba_array(pipeline, X):
    if not hasattr(pipeline, "predict_proba"):
        return np.zeros((len(X), len(CLASS_IDS)), dtype=float)
    proba = pipeline.predict_proba(X)
    model_classes = list(pipeline.named_steps["model"].classes_)
    full = np.zeros((len(X), len(CLASS_IDS)), dtype=float)
    for source_idx, class_id in enumerate(model_classes):
        full[:, CLASS_IDS.index(int(class_id))] = proba[:, source_idx]
    return full


def threshold_probas_to_class_probas(threshold_probas):
    p_ge = np.asarray(threshold_probas, dtype=float)
    p_ge = np.maximum.accumulate(p_ge[:, ::-1], axis=1)[:, ::-1]
    p_ge = np.clip(p_ge, 0, 1)
    class_proba = np.column_stack(
        [
            1 - p_ge[:, 0],
            p_ge[:, 0] - p_ge[:, 1],
            p_ge[:, 1] - p_ge[:, 2],
            p_ge[:, 2] - p_ge[:, 3],
            p_ge[:, 3],
        ]
    )
    return np.clip(class_proba, 0, 1)


def threshold_preds_from_probas(threshold_probas, thresholds):
    p_ge = np.asarray(threshold_probas, dtype=float)
    p_ge = np.maximum.accumulate(p_ge[:, ::-1], axis=1)[:, ::-1]
    thresholds = np.asarray(thresholds, dtype=float)
    return (p_ge >= thresholds).sum(axis=1).astype(int)


def fit_direct_experiment(name, spec, feature_config, X_train, y_train, X_test, y_test, test_prices):
    start = time.perf_counter()
    pipeline = Pipeline(
        [
            ("preprocessor", make_preprocessor(feature_config)),
            ("model", spec["model_factory"]()),
        ]
    )
    fit_kwargs = {}
    sample_weight_note = "none"
    if spec["sample_weight"]:
        fit_kwargs["model__sample_weight"] = compute_weighted_samples(y_train, spec["sample_weight"])
        sample_weight_note = spec["sample_weight"]
    pipeline.fit(X_train, y_train, **fit_kwargs)
    y_pred = pipeline.predict(X_test).astype(int)
    proba = predict_proba_array(pipeline, X_test)
    metrics = evaluate_classifier(y_test, y_pred, prices=test_prices)
    return {
        "name": name,
        "pipeline": pipeline,
        "pred": y_pred,
        "proba": proba,
        "threshold_probas": None,
        "thresholds": None,
        "experiment": {
            "model": pipeline.named_steps["model"].__class__.__name__,
            "model_family": spec["model_family"],
            "feature_set": feature_config["feature_set"],
            "sample_weight": sample_weight_note,
            "fit_predict_seconds": round(float(time.perf_counter() - start), 3),
            "metrics": metrics,
        },
    }


def tune_thresholds(probas, y_true, prices):
    grid = [0.4, 0.45, 0.5, 0.55, 0.6]
    best_thresholds = (0.5, 0.5, 0.5, 0.5)
    best_key = None
    for t0 in grid:
        for t1 in grid:
            for t2 in grid:
                for t3 in grid:
                    thresholds = (t0, t1, t2, t3)
                    pred = threshold_preds_from_probas(probas, thresholds)
                    metrics = evaluate_classifier(y_true, pred, prices=prices)
                    ordinal = metrics["ordinal_metrics"]
                    v2 = metrics["v2_boundary_metrics"]
                    key = (
                        v2["mid_range_f1"],
                        v2["mid_range_recall"],
                        ordinal["adjacent_band_accuracy"],
                        -ordinal["severe_misclassification_rate"],
                        metrics["macro_f1"],
                    )
                    if best_key is None or key > best_key:
                        best_key = key
                        best_thresholds = thresholds
    return best_thresholds


def fit_ordinal_threshold_experiment(feature_config, X_train, y_train, train_df, X_test, y_test, test_prices):
    try:
        get_lgbm_classifier(objective="binary")
    except Exception as exc:
        return None, {"ordinal_threshold_lightgbm_unavailable": {"unavailable_reason": str(exc)}}

    start = time.perf_counter()
    train_groups = train_df["hotel_id"] if "hotel_id" in train_df.columns else np.arange(len(train_df))
    inner_splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE + 11)
    inner_fit_idx, inner_val_idx = next(inner_splitter.split(X_train, y_train, groups=train_groups))
    X_inner_fit, X_inner_val = X_train.iloc[inner_fit_idx], X_train.iloc[inner_val_idx]
    y_inner_fit, y_inner_val = y_train[inner_fit_idx], y_train[inner_val_idx]
    val_prices = train_df.iloc[inner_val_idx]["price_vnd"].astype(float).to_numpy()
    threshold_names = ["threshold_500k", "threshold_1m", "threshold_2m", "threshold_5m"]
    threshold_levels = [1, 2, 3, 4]

    val_threshold_probas = []
    for level in threshold_levels:
        binary_y = (y_inner_fit >= level).astype(int)
        pipeline = Pipeline(
            [
                ("preprocessor", make_preprocessor(feature_config)),
                ("model", get_lgbm_classifier(objective="binary")),
            ]
        )
        weights = compute_sample_weight(class_weight="balanced", y=binary_y).astype(float)
        pipeline.fit(X_inner_fit, binary_y, model__sample_weight=weights)
        val_threshold_probas.append(pipeline.predict_proba(X_inner_val)[:, 1])
    val_threshold_probas = np.column_stack(val_threshold_probas)
    tuned_thresholds = tune_thresholds(val_threshold_probas, y_inner_val, val_prices)

    final_models = {}
    test_threshold_probas = []
    for name, level in zip(threshold_names, threshold_levels):
        binary_y = (y_train >= level).astype(int)
        pipeline = Pipeline(
            [
                ("preprocessor", make_preprocessor(feature_config)),
                ("model", get_lgbm_classifier(objective="binary")),
            ]
        )
        weights = compute_sample_weight(class_weight="balanced", y=binary_y).astype(float)
        pipeline.fit(X_train, binary_y, model__sample_weight=weights)
        final_models[name] = pipeline
        test_threshold_probas.append(pipeline.predict_proba(X_test)[:, 1])

    test_threshold_probas = np.column_stack(test_threshold_probas)
    y_pred = threshold_preds_from_probas(test_threshold_probas, tuned_thresholds)
    class_proba = threshold_probas_to_class_probas(test_threshold_probas)
    metrics = evaluate_classifier(y_test, y_pred, prices=test_prices)
    result = {
        "name": "ordinal_threshold_lightgbm_tuned",
        "pipeline": final_models,
        "pred": y_pred,
        "proba": class_proba,
        "threshold_probas": test_threshold_probas,
        "thresholds": tuned_thresholds,
        "experiment": {
            "model": "four_binary_LGBMClassifier",
            "model_family": "ordinal_threshold_lightgbm",
            "feature_set": feature_config["feature_set"],
            "sample_weight": "balanced binary threshold sample weights",
            "thresholds": {
                "t_500k": tuned_thresholds[0],
                "t_1m": tuned_thresholds[1],
                "t_2m": tuned_thresholds[2],
                "t_5m": tuned_thresholds[3],
            },
            "monotonicity_postprocessing": "cumulative threshold probabilities clipped so higher thresholds cannot exceed lower thresholds",
            "fit_predict_seconds": round(float(time.perf_counter() - start), 3),
            "metrics": metrics,
        },
    }
    return result, {}


def make_predictions(test_df, y_pred, proba, experiment_name, model_family, threshold_probas=None):
    metadata_cols = [col for col in METADATA_COLS if col in test_df.columns]
    predictions = test_df[metadata_cols].copy()
    predictions["actual_price_vnd"] = test_df["price_vnd"].astype(float)
    predictions["actual_price_class_id"] = test_df["price_class_id"].astype(int)
    predictions["actual_price_class_label"] = test_df["price_class_label"].astype(str)
    predictions["pred_price_class_id"] = y_pred.astype(int)
    predictions["pred_price_class_label"] = pd.Series(y_pred).map(PRICE_CLASS_LABELS).to_numpy()
    proba_df = pd.DataFrame(proba, columns=PROBA_COLS, index=predictions.index)
    predictions = pd.concat([predictions, proba_df], axis=1)
    if threshold_probas is None:
        for col in ["threshold_p_ge_500k", "threshold_p_ge_1m", "threshold_p_ge_2m", "threshold_p_ge_5m"]:
            predictions[col] = np.nan
    else:
        predictions["threshold_p_ge_500k"] = threshold_probas[:, 0]
        predictions["threshold_p_ge_1m"] = threshold_probas[:, 1]
        predictions["threshold_p_ge_2m"] = threshold_probas[:, 2]
        predictions["threshold_p_ge_5m"] = threshold_probas[:, 3]
    predictions["abs_class_error"] = np.abs(predictions["pred_price_class_id"] - predictions["actual_price_class_id"])
    predictions["is_adjacent_or_exact"] = (predictions["abs_class_error"] <= 1).astype(int)
    predictions["is_severe_misclassification"] = (predictions["abs_class_error"] >= 2).astype(int)
    predictions = add_boundary_columns(predictions)
    top1, gap = proba_to_top_metrics(proba)
    predictions["top1_probability"] = top1
    predictions["top2_probability_gap"] = gap
    predictions["model_family"] = model_family
    predictions["experiment_name"] = experiment_name
    predictions["suspicious_price_flag"] = test_df.get("suspicious_price_flag", pd.Series(0, index=test_df.index)).fillna(0).astype(int).to_numpy()
    predictions["suspicious_reason"] = test_df.get("suspicious_reason", pd.Series("", index=test_df.index)).fillna("").astype(str).to_numpy()
    return predictions


def top_values(series, limit=5):
    if series.empty:
        return ""
    return "; ".join([f"{idx}:{count}" for idx, count in series.astype(str).value_counts().head(limit).items()])


def build_error_analysis(predictions, experiment_name):
    rows = []
    for actual_id in CLASS_IDS:
        actual_mask = predictions["actual_price_class_id"] == actual_id
        actual_total = int(actual_mask.sum())
        for pred_id in CLASS_IDS:
            pair = predictions[actual_mask & (predictions["pred_price_class_id"] == pred_id)]
            if pair.empty:
                continue
            rows.append(
                {
                    "actual_price_class_label": PRICE_CLASS_LABELS[actual_id],
                    "pred_price_class_label": PRICE_CLASS_LABELS[pred_id],
                    "rows": int(len(pair)),
                    "error_rate_within_actual_class": round(float((actual_id != pred_id) * len(pair) / max(actual_total, 1)), 6),
                    "median_price_vnd": round(float(pair["actual_price_vnd"].median()), 2),
                    "mean_price_vnd": round(float(pair["actual_price_vnd"].mean()), 2),
                    "near_lower_boundary_rate": round(float(pair["is_near_boundary"].mean()), 6),
                    "near_upper_boundary_rate": round(float(pair["is_near_boundary"].mean()), 6),
                    "near_any_boundary_rate": round(float(pair["is_near_boundary"].mean()), 6),
                    "median_pred_top1_probability": round(float(pair["top1_probability"].median()), 6),
                    "median_pred_top2_gap": round(float(pair["top2_probability_gap"].median()), 6),
                    "median_star_rating_clean": round(float(pd.to_numeric(pair.get("star_rating_clean"), errors="coerce").median()), 6) if "star_rating_clean" in pair else np.nan,
                    "median_luxury_route_score": np.nan,
                    "median_luxury_tail_score": np.nan,
                    "median_luxury_amenity_score": np.nan,
                    "median_max_persons": np.nan,
                    "median_bedroom_count": np.nan,
                    "top_source_city_values": top_values(pair["source_city"]) if "source_city" in pair else "",
                    "top_property_type_values": top_values(pair["property_type"]) if "property_type" in pair else "",
                    "experiment_name": experiment_name,
                }
            )
    return pd.DataFrame(rows)


def enrich_error_analysis_from_test(error_analysis, predictions, test_df):
    if error_analysis.empty:
        return error_analysis
    enriched_predictions = predictions.copy()
    for col in ["luxury_route_score", "luxury_tail_score", "luxury_amenity_score", "max_persons", "bedroom_count"]:
        if col in test_df.columns:
            enriched_predictions[col] = pd.to_numeric(test_df[col].to_numpy(), errors="coerce")
    rows = []
    for _, row in error_analysis.iterrows():
        mask = (
            (enriched_predictions["actual_price_class_label"] == row["actual_price_class_label"])
            & (enriched_predictions["pred_price_class_label"] == row["pred_price_class_label"])
        )
        pair = enriched_predictions[mask]
        updated = row.to_dict()
        for col in ["luxury_route_score", "luxury_tail_score", "luxury_amenity_score", "max_persons", "bedroom_count"]:
            target_col = f"median_{col}"
            if col in pair.columns:
                updated[target_col] = round(float(pair[col].median()), 6) if pair[col].notna().any() else np.nan
        rows.append(updated)
    return pd.DataFrame(rows)


def load_v1_report():
    if not V1_REPORT_PATH.exists():
        return V1_BASELINE
    try:
        report = json.loads(V1_REPORT_PATH.read_text(encoding="utf-8"))
        best = report.get("best_experiment_by_macro_f1")
        metrics = report.get("experiments", {}).get(best, {}).get("metrics", {})
        return {
            "best_experiment": best,
            "macro_f1": metrics.get("macro_f1", V1_BASELINE["macro_f1"]),
            "mid_range_f1": metrics.get("per_class", {}).get("mid_range", {}).get("f1", V1_BASELINE["mid_range_f1"]),
            "mid_range_recall": metrics.get("per_class", {}).get("mid_range", {}).get("recall", V1_BASELINE["mid_range_recall"]),
            "balanced_accuracy": metrics.get("balanced_accuracy", V1_BASELINE["balanced_accuracy"]),
            "adjacent_band_accuracy": metrics.get("ordinal_metrics", {}).get("adjacent_band_accuracy", V1_BASELINE["adjacent_band_accuracy"]),
            "severe_misclassification_rate": metrics.get("ordinal_metrics", {}).get("severe_misclassification_rate", V1_BASELINE["severe_misclassification_rate"]),
            "premium_luxury_recall": metrics.get("business_metrics", {}).get("premium_luxury_recall", V1_BASELINE["premium_luxury_recall"]),
        }
    except Exception:
        return V1_BASELINE


def compare_to_v1(metrics, v1_baseline):
    return {
        "macro_f1_delta": round(float(metrics["macro_f1"] - v1_baseline["macro_f1"]), 6),
        "mid_range_f1_delta": round(float(metrics["v2_boundary_metrics"]["mid_range_f1"] - v1_baseline["mid_range_f1"]), 6),
        "mid_range_recall_delta": round(float(metrics["v2_boundary_metrics"]["mid_range_recall"] - v1_baseline["mid_range_recall"]), 6),
        "balanced_accuracy_delta": round(float(metrics["balanced_accuracy"] - v1_baseline["balanced_accuracy"]), 6),
        "adjacent_band_accuracy_delta": round(float(metrics["ordinal_metrics"]["adjacent_band_accuracy"] - v1_baseline["adjacent_band_accuracy"]), 6),
        "severe_misclassification_rate_delta": round(float(metrics["ordinal_metrics"]["severe_misclassification_rate"] - v1_baseline["severe_misclassification_rate"]), 6),
        "premium_luxury_recall_delta": round(float(metrics["business_metrics"]["premium_luxury_recall"] - v1_baseline["premium_luxury_recall"]), 6),
        "minimum_v2_target_met": bool(
            metrics["v2_boundary_metrics"]["mid_range_f1"] >= 0.55
            and metrics["v2_boundary_metrics"]["mid_range_recall"] >= 0.55
            and metrics["macro_f1"] >= v1_baseline["macro_f1"] - 0.01
            and metrics["ordinal_metrics"]["adjacent_band_accuracy"] >= 0.975
            and metrics["ordinal_metrics"]["severe_misclassification_rate"] <= 0.025
            and metrics["business_metrics"]["premium_luxury_recall"] >= 0.65
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-path", type=Path, default=SOURCE_PATH)
    parser.add_argument("--fallback-source-path", type=Path, default=FALLBACK_SOURCE_PATH)
    parser.add_argument("--output-data-path", type=Path, default=OUTPUT_DATA_PATH)
    parser.add_argument("--report-path", type=Path, default=REPORT_PATH)
    parser.add_argument("--predictions-path", type=Path, default=PREDICTIONS_PATH)
    parser.add_argument("--error-analysis-path", type=Path, default=ERROR_ANALYSIS_PATH)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--skip-xgboost", action="store_true")
    parser.add_argument("--skip-ordinal", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    for path in [args.output_data_path, args.report_path, args.predictions_path, args.error_analysis_path, args.model_path]:
        path.parent.mkdir(parents=True, exist_ok=True)

    source_path = args.source_path if args.source_path.exists() else args.fallback_source_path
    df = pd.read_csv(source_path)
    df["price_vnd"] = pd.to_numeric(df["price_vnd"], errors="coerce")
    df = df[df["price_vnd"].notna() & (df["price_vnd"] > 0)].copy()
    df["price_class_id"] = df["price_vnd"].apply(assign_price_class).astype(int)
    df["price_class_label"] = df["price_class_id"].map(PRICE_CLASS_LABELS)
    df = add_mid_market_features(df)
    df.to_csv(args.output_data_path, index=False, encoding="utf-8-sig")

    y = df["price_class_id"].astype(int).to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(df, y, groups=df["hotel_id"]))
    raw_train_df = df.iloc[train_idx].reset_index(drop=True)
    raw_test_df = df.iloc[test_idx].reset_index(drop=True)
    y_train = raw_train_df["price_class_id"].astype(int).to_numpy()
    y_test = raw_test_df["price_class_id"].astype(int).to_numpy()
    train_df, test_df, market_encoder = add_market_context_features(raw_train_df, raw_test_df, y_train)

    feature_config = build_feature_config(train_df, "market_context_features")
    feature_cols = feature_config["numeric_features"] + feature_config["categorical_features"] + feature_config["text_features"]
    X_train = prepare_features(train_df, feature_cols)
    X_test = prepare_features(test_df, feature_cols)
    test_prices = test_df["price_vnd"].astype(float).to_numpy()

    experiments = {}
    unavailable = {}
    fitted = {}
    candidate_specs = build_direct_candidates()
    if args.skip_xgboost:
        candidate_specs = {name: spec for name, spec in candidate_specs.items() if not name.startswith("xgboost")}

    for name, spec in candidate_specs.items():
        if "model_factory" not in spec:
            unavailable[name] = spec
            continue
        result = fit_direct_experiment(name, spec, feature_config, X_train, y_train, X_test, y_test, test_prices)
        experiments[name] = result["experiment"]
        fitted[name] = result
        print(
            json.dumps(
                {
                    "experiment": name,
                    "macro_f1": result["experiment"]["metrics"]["macro_f1"],
                    "mid_range_f1": result["experiment"]["metrics"]["v2_boundary_metrics"]["mid_range_f1"],
                    "mid_range_recall": result["experiment"]["metrics"]["v2_boundary_metrics"]["mid_range_recall"],
                    "severe_misclassification_rate": result["experiment"]["metrics"]["ordinal_metrics"]["severe_misclassification_rate"],
                },
                ensure_ascii=False,
            )
        )

    if not args.skip_ordinal:
        ordinal_result, ordinal_unavailable = fit_ordinal_threshold_experiment(feature_config, X_train, y_train, train_df, X_test, y_test, test_prices)
        unavailable.update(ordinal_unavailable)
        if ordinal_result is not None:
            experiments[ordinal_result["name"]] = ordinal_result["experiment"]
            fitted[ordinal_result["name"]] = ordinal_result
            print(
                json.dumps(
                    {
                        "experiment": ordinal_result["name"],
                        "macro_f1": ordinal_result["experiment"]["metrics"]["macro_f1"],
                        "mid_range_f1": ordinal_result["experiment"]["metrics"]["v2_boundary_metrics"]["mid_range_f1"],
                        "mid_range_recall": ordinal_result["experiment"]["metrics"]["v2_boundary_metrics"]["mid_range_recall"],
                        "thresholds": ordinal_result["experiment"]["thresholds"],
                    },
                    ensure_ascii=False,
                )
            )

    best_macro_name = max(experiments, key=lambda name: selection_key(experiments[name]["metrics"]))
    best_mid_name = max(
        experiments,
        key=lambda name: (
            experiments[name]["metrics"]["v2_boundary_metrics"]["mid_range_f1"],
            experiments[name]["metrics"]["v2_boundary_metrics"]["mid_range_recall"],
            experiments[name]["metrics"]["macro_f1"],
        ),
    )
    best_ordinal_name = max(experiments, key=lambda name: ordinal_safety_key(experiments[name]["metrics"]))
    best_business_name = max(experiments, key=lambda name: business_safety_key(experiments[name]["metrics"]))
    selected_name = best_macro_name
    selected = fitted[selected_name]
    selected_metrics = experiments[selected_name]["metrics"]
    selected_model_family = experiments[selected_name]["model_family"]

    predictions = make_predictions(
        test_df,
        selected["pred"],
        selected["proba"],
        selected_name,
        selected_model_family,
        selected["threshold_probas"],
    )
    predictions.to_csv(args.predictions_path, index=False, encoding="utf-8-sig")

    error_analysis = build_error_analysis(predictions, selected_name)
    error_analysis = enrich_error_analysis_from_test(error_analysis, predictions, test_df)
    error_analysis.to_csv(args.error_analysis_path, index=False, encoding="utf-8-sig")

    joblib.dump(
        {
            "experiment_name": selected_name,
            "model_family": selected_model_family,
            "model": selected["pipeline"],
            "feature_config": feature_config,
            "feature_cols": feature_cols,
            "price_class_labels": PRICE_CLASS_LABELS,
            "market_context_encoder": market_encoder,
            "thresholds": selected["thresholds"],
            "leakage_controls": {
                "dropped_columns": {
                    "leakage": LEAKAGE_COLS,
                    "id": ID_DROP_COLS,
                    "derived_from_target": TARGET_DERIVED_COLS,
                },
                "train_fold_aggregate_features": MARKET_CONTEXT_SPECS,
            },
        },
        args.model_path,
    )

    v1_baseline = load_v1_report()
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "price_vnd_band_classification",
        "target": "price_class_id",
        "source_data": str(source_path),
        "output_data": str(args.output_data_path),
        "predictions_path": str(args.predictions_path),
        "error_analysis_path": str(args.error_analysis_path),
        "model_path": str(args.model_path),
        "class_definition": {
            "budget": "price_vnd < 500000",
            "economy": "500000 <= price_vnd < 1000000",
            "mid_range": "1000000 <= price_vnd < 2000000",
            "upscale": "2000000 <= price_vnd < 5000000",
            "premium_luxury": "price_vnd >= 5000000",
        },
        "v1_baseline": v1_baseline,
        "class_distribution": class_distribution(df["price_class_id"]),
        "split": {
            "method": "GroupShuffleSplit",
            "group_column": "hotel_id",
            "test_size": 0.2,
            "random_state": RANDOM_STATE,
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "train_hotels": int(train_df["hotel_id"].nunique()),
            "test_hotels": int(test_df["hotel_id"].nunique()),
            "train_class_distribution": class_distribution(y_train),
            "test_class_distribution": class_distribution(y_test),
        },
        "feature_sets": {
            "expanded_luxury_features": "V1-compatible luxury feature foundation retained.",
            "mid_market_features": [
                "is_standard_hotel_room",
                "is_deluxe_without_luxury_tail",
                "is_3_to_4_star_full_service",
                "is_midscale_property_type",
                "has_midscale_amenity_profile",
                "midscale_amenity_score",
                "luxury_signal_without_premium_capacity",
                "budget_like_low_signal",
                "economy_mid_boundary_signal",
                "mid_upscale_boundary_signal",
            ],
            "market_context_features": list(MARKET_CONTEXT_SPECS.keys()),
            "selected_feature_config": feature_config,
        },
        "leakage_controls": {
            "dropped_columns": {
                "leakage": [col for col in LEAKAGE_COLS if col in df.columns],
                "id": [col for col in ID_DROP_COLS if col in df.columns],
                "derived_from_target": [col for col in TARGET_DERIVED_COLS if col in df.columns],
            },
            "train_fold_aggregate_features": {
                "note": "Market context features are target-derived supervised encodings fitted on train fold only, with global train median fallback for unseen groups.",
                "specs": MARKET_CONTEXT_SPECS,
                "global_train_median_class": market_encoder.global_median_,
            },
        },
        "experiments": experiments,
        "unavailable_candidates": unavailable,
        "best_experiment_by_macro_f1": best_macro_name,
        "best_experiment_by_mid_range_f1": best_mid_name,
        "best_experiment_by_ordinal_safety": best_ordinal_name,
        "best_experiment_by_business_safety": best_business_name,
        "selected_v2_model": selected_name,
        "comparison_to_v1": compare_to_v1(selected_metrics, v1_baseline),
        "recommended_v3_direction": {
            "ordinal_threshold_next_step": "Tune threshold calibration more carefully if ordinal threshold is competitive.",
            "market_context_next_step": "Make the train-fold target encoder cross-validated if market context remains helpful.",
            "mid_range_next_step": "If mid_range remains weak, test a two-stage hierarchy around low/economy, middle, and high/luxury.",
            "six_class_split": "Only test after premium_luxury stability remains acceptable.",
        },
        "optional_v2_items_not_run": {
            "boundary_specialist_classifiers": "Skipped by default to keep V2 focused on direct and ordinal-aware comparisons.",
            "probability_calibration": "Skipped by default because group-aware manual calibration would require another nested holdout; selected model still reports raw probabilities.",
        },
    }
    args.report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "selected_v2_model": selected_name,
        "best_experiment_by_macro_f1": best_macro_name,
        "best_experiment_by_mid_range_f1": best_mid_name,
        "best_experiment_by_ordinal_safety": best_ordinal_name,
        "best_experiment_by_business_safety": best_business_name,
        "selected_metrics": selected_metrics,
        "comparison_to_v1": report["comparison_to_v1"],
        "unavailable_candidates": unavailable,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Processed classification data saved to: {args.output_data_path}")
    print(f"Prediction-level output saved to: {args.predictions_path}")
    print(f"Error analysis output saved to: {args.error_analysis_path}")
    print(f"Evaluation JSON saved to: {args.report_path}")
    print(f"Best model artifact saved to: {args.model_path}")


if __name__ == "__main__":
    main()
