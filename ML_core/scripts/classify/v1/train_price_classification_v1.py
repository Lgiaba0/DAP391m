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
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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


SOURCE_PATH = Path("data/processed/reg/v5/vietnam_price_vnd_modeling_v5.csv")
OUTPUT_DATA_PATH = Path("data/processed/classify/v1/vietnam_price_classification_v1.csv")
REPORT_PATH = Path("reports/classify/v1/price_classification_evaluation_v1.json")
PREDICTIONS_PATH = Path("reports/classify/v1/price_classification_predictions_v1.csv")
MODEL_PATH = Path("models/classify/v1/price_classification_v1_model.joblib")
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


def build_feature_config(df, feature_set):
    excluded = set(LEAKAGE_COLS + ID_DROP_COLS + TARGET_DERIVED_COLS)
    text_features = [col for col in TEXT_FEATURES if col in df.columns]

    if feature_set == "baseline":
        candidate_cols = [
            col
            for col in df.columns
            if col not in excluded and col not in text_features and not col.startswith("amenity_") and not col.startswith("name_has_")
        ]
    elif feature_set == "expanded_luxury":
        candidate_cols = [col for col in df.columns if col not in excluded and col not in text_features]
    else:
        raise ValueError(f"Unknown feature set: {feature_set}")

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


def build_models():
    models = {
        "dummy_most_frequent": {
            "model": DummyClassifier(strategy="most_frequent"),
            "sample_weight": False,
        },
        "logistic_regression_balanced": {
            "model": LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                solver="saga",
                n_jobs=-1,
            ),
            "sample_weight": False,
        },
        "random_forest_balanced": {
            "model": RandomForestClassifier(
                n_estimators=400,
                class_weight="balanced",
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "sample_weight": False,
        },
    }
    try:
        from xgboost import XGBClassifier

        models["xgboost_weighted"] = {
            "model": XGBClassifier(
                n_estimators=500,
                learning_rate=0.04,
                max_depth=5,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="multi:softprob",
                eval_metric="mlogloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                tree_method="hist",
            ),
            "sample_weight": True,
        }
    except Exception as exc:
        models["xgboost_weighted_unavailable"] = {"unavailable_reason": str(exc)}

    try:
        from lightgbm import LGBMClassifier

        models["lightgbm_balanced"] = {
            "model": LGBMClassifier(
                n_estimators=500,
                learning_rate=0.04,
                num_leaves=31,
                subsample=0.9,
                colsample_bytree=0.9,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=-1,
            ),
            "sample_weight": False,
        }
    except Exception as exc:
        models["lightgbm_balanced_unavailable"] = {"unavailable_reason": str(exc)}

    return models


def evaluate_classifier(y_true, y_pred):
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
    }


def selection_key(metrics):
    ordinal = metrics["ordinal_metrics"]
    business = metrics["business_metrics"]
    return (
        metrics["macro_f1"],
        metrics["balanced_accuracy"],
        business["premium_luxury_recall"],
        ordinal["adjacent_band_accuracy"],
        -ordinal["severe_misclassification_rate"],
        metrics["weighted_f1"],
        metrics["accuracy"],
    )


def ordinal_safety_key(metrics):
    ordinal = metrics["ordinal_metrics"]
    business = metrics["business_metrics"]
    return (
        ordinal["adjacent_band_accuracy"],
        -ordinal["severe_misclassification_rate"],
        metrics["macro_f1"],
        business["premium_luxury_recall"],
    )


def predict_proba_frame(pipeline, X):
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(X)
        model_classes = list(pipeline.named_steps["model"].classes_)
        full = np.zeros((len(X), len(CLASS_IDS)), dtype=float)
        for source_idx, class_id in enumerate(model_classes):
            full[:, CLASS_IDS.index(int(class_id))] = proba[:, source_idx]
    else:
        full = np.zeros((len(X), len(CLASS_IDS)), dtype=float)
    return pd.DataFrame(full, columns=[f"pred_proba_{label}" for label in CLASS_NAMES])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-path", type=Path, default=SOURCE_PATH)
    parser.add_argument("--output-data-path", type=Path, default=OUTPUT_DATA_PATH)
    parser.add_argument("--report-path", type=Path, default=REPORT_PATH)
    parser.add_argument("--predictions-path", type=Path, default=PREDICTIONS_PATH)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    return parser.parse_args()


def main():
    args = parse_args()
    for path in [args.output_data_path, args.report_path, args.predictions_path, args.model_path]:
        path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.source_path)
    df["price_vnd"] = pd.to_numeric(df["price_vnd"], errors="coerce")
    df = df[df["price_vnd"].notna() & (df["price_vnd"] > 0)].copy()
    df["price_class_id"] = df["price_vnd"].apply(assign_price_class).astype(int)
    df["price_class_label"] = df["price_class_id"].map(PRICE_CLASS_LABELS)
    df.to_csv(args.output_data_path, index=False, encoding="utf-8-sig")

    feature_config = build_feature_config(df, "expanded_luxury")
    feature_cols = (
        feature_config["numeric_features"]
        + feature_config["categorical_features"]
        + feature_config["text_features"]
    )
    X = prepare_features(df, feature_cols)
    y = df["price_class_id"].astype(int).to_numpy()

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(df, y, groups=df["hotel_id"]))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    balanced_weights = compute_sample_weight(class_weight="balanced", y=y_train).astype(float)
    premium_weighted = balanced_weights.copy()
    premium_weighted[y_train == 4] *= 1.5

    experiments = {}
    pipelines = {}
    best_name = None
    best_key = None
    best_ordinal_name = None
    best_ordinal_key = None
    unavailable = {}

    for name, spec in build_models().items():
        if "model" not in spec:
            unavailable[name] = spec
            continue
        start = time.perf_counter()
        pipeline = Pipeline(
            [
                ("preprocessor", make_preprocessor(feature_config)),
                ("model", spec["model"]),
            ]
        )
        fit_kwargs = {}
        sample_weight_note = "none"
        if spec["sample_weight"]:
            fit_kwargs["model__sample_weight"] = premium_weighted
            sample_weight_note = "balanced sample weights; premium_luxury multiplied by 1.5"
        pipeline.fit(X_train, y_train, **fit_kwargs)
        y_pred = pipeline.predict(X_test).astype(int)
        metrics = evaluate_classifier(y_test, y_pred)
        elapsed = round(float(time.perf_counter() - start), 3)
        experiments[name] = {
            "model": spec["model"].__class__.__name__,
            "feature_set": feature_config["feature_set"],
            "sample_weight": sample_weight_note,
            "fit_predict_seconds": elapsed,
            "metrics": metrics,
        }
        pipelines[name] = pipeline

        key = selection_key(metrics)
        if best_key is None or key > best_key:
            best_key = key
            best_name = name
        ordinal_key = ordinal_safety_key(metrics)
        if best_ordinal_key is None or ordinal_key > best_ordinal_key:
            best_ordinal_key = ordinal_key
            best_ordinal_name = name

    best_pipeline = pipelines[best_name]
    best_pred = best_pipeline.predict(X_test).astype(int)
    proba_df = predict_proba_frame(best_pipeline, X_test).reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    metadata_cols = [col for col in METADATA_COLS if col in test_df.columns]
    predictions = test_df[metadata_cols].copy()
    predictions["actual_price_vnd"] = test_df["price_vnd"].astype(float)
    predictions["actual_price_class_id"] = test_df["price_class_id"].astype(int)
    predictions["actual_price_class_label"] = test_df["price_class_label"].astype(str)
    predictions["pred_price_class_id"] = best_pred
    predictions["pred_price_class_label"] = pd.Series(best_pred).map(PRICE_CLASS_LABELS).to_numpy()
    predictions = pd.concat([predictions.reset_index(drop=True), proba_df], axis=1)
    predictions["abs_class_error"] = np.abs(predictions["pred_price_class_id"] - predictions["actual_price_class_id"])
    predictions["is_adjacent_or_exact"] = (predictions["abs_class_error"] <= 1).astype(int)
    predictions["is_severe_misclassification"] = (predictions["abs_class_error"] >= 2).astype(int)
    predictions["suspicious_price_flag"] = test_df.get("suspicious_price_flag", pd.Series(0, index=test_df.index)).fillna(0).astype(int)
    predictions["suspicious_reason"] = test_df.get("suspicious_reason", pd.Series("", index=test_df.index)).fillna("").astype(str)
    predictions["experiment_name"] = best_name
    predictions.to_csv(args.predictions_path, index=False, encoding="utf-8-sig")

    joblib.dump(
        {
            "experiment_name": best_name,
            "pipeline": best_pipeline,
            "feature_config": feature_config,
            "feature_cols": feature_cols,
            "price_class_labels": PRICE_CLASS_LABELS,
        },
        args.model_path,
    )

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "price_vnd_band_classification",
        "target": "price_class_id",
        "source_data": str(args.source_path),
        "output_data": str(args.output_data_path),
        "predictions_path": str(args.predictions_path),
        "model_path": str(args.model_path),
        "class_definition": {
            "budget": "price_vnd < 500000",
            "economy": "500000 <= price_vnd < 1000000",
            "mid_range": "1000000 <= price_vnd < 2000000",
            "upscale": "2000000 <= price_vnd < 5000000",
            "premium_luxury": "price_vnd >= 5000000",
        },
        "class_distribution": class_distribution(df["price_class_id"]),
        "split": {
            "method": "GroupShuffleSplit",
            "group_column": "hotel_id",
            "test_size": 0.2,
            "random_state": RANDOM_STATE,
            "train_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
            "train_hotels": int(df.iloc[train_idx]["hotel_id"].nunique()),
            "test_hotels": int(df.iloc[test_idx]["hotel_id"].nunique()),
            "train_class_distribution": class_distribution(y_train),
            "test_class_distribution": class_distribution(y_test),
        },
        "feature_config": feature_config,
        "dropped_columns": {
            "leakage": [col for col in LEAKAGE_COLS if col in df.columns],
            "id": [col for col in ID_DROP_COLS if col in df.columns],
            "derived_from_target": [col for col in TARGET_DERIVED_COLS if col in df.columns],
        },
        "experiments": experiments,
        "unavailable_candidates": unavailable,
        "best_experiment_by_macro_f1": best_name,
        "best_experiment_by_ordinal_safety": best_ordinal_name,
        "severe_misclassification_review": {
            "best_experiment": best_name,
            "severe_count": int(predictions["is_severe_misclassification"].sum()),
            "high_to_low_count": experiments[best_name]["metrics"]["business_metrics"]["high_to_low_confusion_count"],
            "low_to_high_count": experiments[best_name]["metrics"]["business_metrics"]["low_to_high_confusion_count"],
        },
        "comparison_to_regression_versions": {
            "note": "Regression metrics are not directly comparable; compare operational usefulness and error inspection only."
        },
    }
    args.report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "best_experiment_by_macro_f1": best_name,
        "best_experiment_by_ordinal_safety": best_ordinal_name,
        "best_metrics": experiments[best_name]["metrics"],
        "all_model_summary": {
            name: {
                "accuracy": exp["metrics"]["accuracy"],
                "balanced_accuracy": exp["metrics"]["balanced_accuracy"],
                "macro_f1": exp["metrics"]["macro_f1"],
                "weighted_f1": exp["metrics"]["weighted_f1"],
                "premium_luxury_recall": exp["metrics"]["business_metrics"]["premium_luxury_recall"],
                "adjacent_band_accuracy": exp["metrics"]["ordinal_metrics"]["adjacent_band_accuracy"],
                "severe_misclassification_rate": exp["metrics"]["ordinal_metrics"]["severe_misclassification_rate"],
            }
            for name, exp in experiments.items()
        },
        "unavailable_candidates": unavailable,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Processed classification data saved to: {args.output_data_path}")
    print(f"Prediction-level output saved to: {args.predictions_path}")
    print(f"Evaluation JSON saved to: {args.report_path}")
    print(f"Best model artifact saved to: {args.model_path}")


if __name__ == "__main__":
    main()
