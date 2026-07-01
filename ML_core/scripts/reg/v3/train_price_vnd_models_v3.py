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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

ML_CORE_ROOT = Path(__file__).resolve().parents[3]
V2_SCRIPT_PATH = ML_CORE_ROOT / "scripts/reg/v2/train_price_vnd_models_v2.py"
spec = importlib.util.spec_from_file_location("price_regression_v2", V2_SCRIPT_PATH)
base = importlib.util.module_from_spec(spec)
sys.modules["price_regression_v2"] = base
spec.loader.exec_module(base)


RAW_PATH = Path("data/raw/vietnam_rooms_properties_merged.csv")
PROCESSED_PATH = Path("data/processed/reg/v3/vietnam_price_vnd_modeling_v3.csv")
REPORT_PATH = Path("reports/reg/v3/price_vnd_model_evaluation_v3.json")
PREDICTIONS_PATH = Path("reports/reg/v3/price_vnd_predictions_v3.csv")
MODEL_DIR = Path("models/reg/v3")
RANDOM_STATE = 42
SEGMENT_LABELS = ["<10M", "10M-30M", ">30M"]


def assign_v3_segment(values):
    values = pd.Series(values, dtype="float64")
    labels = pd.Series(index=values.index, dtype="string")
    labels[values < 10_000_000] = "<10M"
    labels[(values >= 10_000_000) & (values < 30_000_000)] = "10M-30M"
    labels[values >= 30_000_000] = ">30M"
    return labels.fillna("Unknown")


def make_classifier():
    try:
        from xgboost import XGBClassifier

        return "XGBoostClassifier", XGBClassifier(
            n_estimators=450,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
        )
    except Exception:
        return "RandomForestClassifier", RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            class_weight={"<10M": 1.0, "10M-30M": 3.0, ">30M": 8.0},
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )


def make_regressor():
    try:
        from xgboost import XGBRegressor

        return "XGBoostRegressor", XGBRegressor(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
        )
    except Exception:
        from sklearn.ensemble import RandomForestRegressor

        return "RandomForestRegressor", RandomForestRegressor(
            n_estimators=350,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )


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


def segment_metric_dict(y_true_vnd, y_pred_vnd, labels=None):
    labels = assign_v3_segment(y_true_vnd) if labels is None else pd.Series(labels)
    result = {}
    for segment in SEGMENT_LABELS:
        mask = labels.to_numpy() == segment
        result[segment] = {"row_count": 0} if mask.sum() == 0 else regression_metrics(
            np.asarray(y_true_vnd)[mask],
            np.asarray(y_pred_vnd)[mask],
        )
    return result


def segment_underprediction_dict(y_true_vnd, y_pred_vnd, labels=None):
    labels = assign_v3_segment(y_true_vnd) if labels is None else pd.Series(labels)
    result = {}
    for segment in SEGMENT_LABELS:
        mask = labels.to_numpy() == segment
        result[segment] = {"row_count": 0} if mask.sum() == 0 else underprediction_metrics(
            np.asarray(y_true_vnd)[mask],
            np.asarray(y_pred_vnd)[mask],
        )
    return result


def classifier_metrics(y_true, y_pred, class_names):
    report = classification_report(y_true, y_pred, labels=class_names, output_dict=True, zero_division=0)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=class_names,
        zero_division=0,
    )
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=class_names, average="macro")), 6),
        "weighted_f1": round(float(f1_score(y_true, y_pred, labels=class_names, average="weighted")), 6),
        "per_class": {
            label: {
                "precision": round(float(precision[i]), 6),
                "recall": round(float(recall[i]), 6),
                "f1": round(float(f1[i]), 6),
                "support": int(support[i]),
            }
            for i, label in enumerate(class_names)
        },
        "classification_report": report,
    }


def confusion_dict(y_true, y_pred, class_names):
    matrix = confusion_matrix(y_true, y_pred, labels=class_names)
    return {
        "labels": class_names,
        "matrix": matrix.astype(int).tolist(),
        "rows_are_actual_columns_are_predicted": True,
    }


def threshold_routes(probabilities, class_names, over_30m_threshold=0.25, mid_threshold=0.35):
    proba_df = pd.DataFrame(probabilities, columns=class_names)
    routes = np.where(
        proba_df[">30M"].to_numpy() >= over_30m_threshold,
        ">30M",
        np.where(proba_df["10M-30M"].to_numpy() >= mid_threshold, "10M-30M", "<10M"),
    )
    return routes, proba_df


def fit_regressor(modeling_df, config, train_idx, train_mask, weight_mode="none"):
    feature_cols = [col for col in modeling_df.columns if col not in {"price_vnd", "hotel_id", "true_segment_v3"}]
    train_positions = np.asarray(train_idx)[train_mask]
    X_train = modeling_df.iloc[train_positions][feature_cols]
    y_train_vnd = modeling_df.iloc[train_positions]["price_vnd"].astype(float)
    y_train = np.log1p(y_train_vnd)
    name, model = make_regressor()
    pipeline = Pipeline([("preprocessor", base.make_preprocessor(config)), ("model", model)])
    fit_kwargs = {}
    if weight_mode == "high_price":
        fit_kwargs["model__sample_weight"] = np.where(y_train_vnd >= 30_000_000, 4.0, 1.0)
    pipeline.fit(X_train, y_train, **fit_kwargs)
    return name, pipeline, feature_cols


def predict_with_regressors(under_pipeline, high_pipeline, under_features, high_features, under_df, high_df, routes):
    pred = np.zeros(len(routes), dtype=float)
    regressor_names = np.empty(len(routes), dtype=object)
    raw_pred = np.zeros(len(routes), dtype=float)
    for route_label, pipeline, features, df, regressor_name in [
        ("<10M", under_pipeline, under_features, under_df, "under_10m"),
        ("10M-30M", high_pipeline, high_features, high_df, "high_price"),
        (">30M", high_pipeline, high_features, high_df, "high_price"),
    ]:
        mask = routes == route_label
        if mask.sum() == 0:
            continue
        values = np.expm1(pipeline.predict(df.loc[mask, features]))
        raw_pred[mask] = np.clip(values, 0, None)
        pred[mask] = raw_pred[mask]
        regressor_names[mask] = regressor_name
    return raw_pred, pred, regressor_names


def compute_luxury_calibration(high_pipeline, high_df, high_features, train_idx):
    train = high_df.iloc[train_idx].copy()
    over_mask = train["price_vnd"].astype(float) >= 30_000_000
    if over_mask.sum() == 0:
        return {"multiplier": 1.0, "q25_actual_over_30m_train": None}
    pred = np.clip(np.expm1(high_pipeline.predict(train.loc[over_mask, high_features])), 1, None)
    actual = train.loc[over_mask, "price_vnd"].astype(float).to_numpy()
    multiplier = float(np.nanmedian(actual / pred))
    multiplier = float(np.clip(multiplier, 1.0, 3.0))
    q25_actual = float(np.quantile(actual, 0.25))
    return {
        "multiplier": round(multiplier, 6),
        "q25_actual_over_30m_train": round(q25_actual, 2),
    }


def apply_luxury_calibration(raw_pred, routes, calibration):
    pred = raw_pred.copy()
    mask = routes == ">30M"
    if mask.sum() == 0:
        return pred
    pred[mask] = pred[mask] * calibration["multiplier"]
    if calibration["q25_actual_over_30m_train"] is not None:
        pred[mask] = np.maximum(pred[mask], calibration["q25_actual_over_30m_train"])
    return pred


def evaluate_routing(y_true, y_pred, true_segments, route_segments=None):
    payload = {
        "overall_metrics": regression_metrics(y_true, y_pred),
        "segment_metrics_by_true_segment": segment_metric_dict(y_true, y_pred, true_segments),
        "underprediction_metrics": {
            "overall": underprediction_metrics(y_true, y_pred),
            "by_true_segment": segment_underprediction_dict(y_true, y_pred, true_segments),
        },
    }
    if route_segments is not None:
        payload["segment_metrics_by_predicted_route"] = segment_metric_dict(y_true, y_pred, route_segments)
        payload["underprediction_metrics"]["by_predicted_route"] = segment_underprediction_dict(
            y_true,
            y_pred,
            route_segments,
        )
    return payload


def run_v3_experiment(name, baseline_df, v2_df, baseline_config, v2_config, train_idx, test_idx, under_feature_version):
    start = time.perf_counter()
    classifier_feature_cols = [col for col in v2_df.columns if col not in {"price_vnd", "hotel_id", "true_segment_v3"}]
    y_segments = v2_df["true_segment_v3"].astype(str)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_segments)
    class_names = list(label_encoder.classes_)

    clf_name, classifier = make_classifier()
    classifier_pipeline = Pipeline([("preprocessor", base.make_preprocessor(v2_config)), ("model", classifier)])
    class_weight_map = {"<10M": 1.0, "10M-30M": 3.0, ">30M": 8.0}
    sample_weight = y_segments.iloc[train_idx].map(class_weight_map).astype(float).to_numpy()
    classifier_pipeline.fit(
        v2_df.iloc[train_idx][classifier_feature_cols],
        y_encoded[train_idx],
        model__sample_weight=sample_weight,
    )

    test_probabilities = classifier_pipeline.predict_proba(v2_df.iloc[test_idx][classifier_feature_cols])
    argmax_routes = label_encoder.inverse_transform(test_probabilities.argmax(axis=1))
    threshold_route_labels, proba_df = threshold_routes(test_probabilities, class_names)
    true_test_segments = y_segments.iloc[test_idx].to_numpy()

    under_df = baseline_df if under_feature_version == "baseline" else v2_df
    under_config = baseline_config if under_feature_version == "baseline" else v2_config
    train_segments = y_segments.iloc[train_idx].to_numpy()
    under_name, under_pipeline, under_features = fit_regressor(
        under_df,
        under_config,
        train_idx,
        train_segments == "<10M",
    )
    high_name, high_pipeline, high_features = fit_regressor(
        v2_df,
        v2_config,
        train_idx,
        train_segments != "<10M",
        weight_mode="high_price",
    )

    oracle_routes = np.where(true_test_segments == "<10M", "<10M", "10M-30M")
    raw_oracle_pred, oracle_pred, _ = predict_with_regressors(
        under_pipeline,
        high_pipeline,
        under_features,
        high_features,
        under_df.iloc[test_idx].reset_index(drop=True),
        v2_df.iloc[test_idx].reset_index(drop=True),
        oracle_routes,
    )

    raw_argmax_pred, argmax_pred, _ = predict_with_regressors(
        under_pipeline,
        high_pipeline,
        under_features,
        high_features,
        under_df.iloc[test_idx].reset_index(drop=True),
        v2_df.iloc[test_idx].reset_index(drop=True),
        argmax_routes,
    )

    raw_threshold_pred, threshold_pred_uncal, threshold_regressor_names = predict_with_regressors(
        under_pipeline,
        high_pipeline,
        under_features,
        high_features,
        under_df.iloc[test_idx].reset_index(drop=True),
        v2_df.iloc[test_idx].reset_index(drop=True),
        threshold_route_labels,
    )
    calibration = compute_luxury_calibration(high_pipeline, v2_df, high_features, train_idx)
    threshold_pred_calibrated = apply_luxury_calibration(raw_threshold_pred, threshold_route_labels, calibration)
    actual_test_vnd = v2_df.iloc[test_idx]["price_vnd"].astype(float).to_numpy()

    metadata_cols = [col for col in base.PREDICTION_COLUMNS if col in v2_df.columns]
    predictions = v2_df.iloc[test_idx][metadata_cols].reset_index(drop=True).copy()
    predictions["actual_price_vnd"] = actual_test_vnd
    predictions["true_segment_v3"] = true_test_segments
    predictions["classifier_pred_segment"] = threshold_route_labels
    predictions["classifier_p_under_10m"] = proba_df["<10M"].to_numpy()
    predictions["classifier_p_10m_30m"] = proba_df["10M-30M"].to_numpy()
    predictions["classifier_p_over_30m"] = proba_df[">30M"].to_numpy()
    predictions["routing_strategy"] = "threshold_over30m_0.25_mid_0.35_with_calibration"
    predictions["regressor_name"] = threshold_regressor_names
    predictions["raw_pred_price_vnd"] = raw_threshold_pred
    predictions["calibrated_pred_price_vnd"] = threshold_pred_calibrated
    predictions["absolute_error"] = np.abs(actual_test_vnd - threshold_pred_calibrated)
    predictions["absolute_percentage_error"] = np.where(
        actual_test_vnd > 0,
        predictions["absolute_error"] / actual_test_vnd * 100,
        np.nan,
    )
    predictions["pred_minus_actual"] = threshold_pred_calibrated - actual_test_vnd
    predictions["pred_to_actual_ratio"] = threshold_pred_calibrated / actual_test_vnd
    predictions["is_underprediction"] = threshold_pred_calibrated < actual_test_vnd
    predictions["experiment_name"] = name

    report = {
        "classifier": {
            "model": clf_name,
            "feature_config": "v2_expanded_features",
            "thresholds": {"p_over_30m": 0.25, "p_10m_30m": 0.35},
            "argmax_metrics": classifier_metrics(true_test_segments, argmax_routes, SEGMENT_LABELS),
            "threshold_metrics": classifier_metrics(true_test_segments, threshold_route_labels, SEGMENT_LABELS),
            "argmax_confusion_matrix": confusion_dict(true_test_segments, argmax_routes, SEGMENT_LABELS),
            "threshold_confusion_matrix": confusion_dict(true_test_segments, threshold_route_labels, SEGMENT_LABELS),
        },
        "regressors": {
            "under_10m": {
                "model": under_name,
                "feature_config": f"{under_feature_version}_features",
                "train_rows": int((train_segments == "<10M").sum()),
            },
            "high_price": {
                "model": high_name,
                "feature_config": "v2_expanded_features",
                "train_rows": int((train_segments != "<10M").sum()),
                "sample_weight": "4.0 for >30M rows, 1.0 for 10M-30M rows",
                "luxury_calibration": calibration,
            },
        },
        "oracle_routing": evaluate_routing(actual_test_vnd, oracle_pred, true_test_segments, oracle_routes),
        "predicted_routing_argmax_no_calibration": evaluate_routing(
            actual_test_vnd,
            argmax_pred,
            true_test_segments,
            argmax_routes,
        ),
        "predicted_routing_threshold_no_calibration": evaluate_routing(
            actual_test_vnd,
            threshold_pred_uncal,
            true_test_segments,
            threshold_route_labels,
        ),
        "predicted_routing_threshold_calibrated": evaluate_routing(
            actual_test_vnd,
            threshold_pred_calibrated,
            true_test_segments,
            threshold_route_labels,
        ),
        "routing_confusion_matrix": confusion_dict(true_test_segments, threshold_route_labels, SEGMENT_LABELS),
        "fit_predict_seconds": round(time.perf_counter() - start, 3),
    }
    artifacts = {
        "classifier_pipeline": classifier_pipeline,
        "under_10m_pipeline": under_pipeline,
        "high_price_pipeline": high_pipeline,
        "label_encoder": label_encoder,
        "under_features": under_features,
        "high_features": high_features,
        "classifier_features": classifier_feature_cols,
        "calibration": calibration,
    }
    return report, predictions, artifacts


def load_previous_comparisons():
    comparisons = {}
    for version, path in [
        ("v1", Path("reports/reg/v1/price_vnd_model_evaluation.json")),
        ("v2", Path("reports/reg/v2/price_vnd_model_evaluation_v2.json")),
        ("v1_legacy_path", Path("reports/price_vnd_model_evaluation.json")),
        ("v2_legacy_path", Path("reports/price_vnd_model_evaluation_v2.json")),
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
    baseline_config = base.make_feature_config(df, amenity_features, "baseline")
    v2_config = base.make_feature_config(df, amenity_features, "v2")
    baseline_df = base.make_modeling_df(df, baseline_config)
    v2_df = base.make_modeling_df(df, v2_config)
    v2_df["true_segment_v3"] = assign_v3_segment(v2_df["price_vnd"]).astype(str)
    baseline_df["true_segment_v3"] = v2_df["true_segment_v3"].to_numpy()

    v2_df.to_csv(args.processed_path, index=False, encoding="utf-8-sig")

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    y_split = np.log1p(v2_df["price_vnd"])
    train_idx, test_idx = next(splitter.split(v2_df, y_split, groups=v2_df["hotel_id"]))

    experiment_specs = [
        ("under_10m_baseline_features__high_price_v2_features", "baseline"),
        ("under_10m_v2_features__high_price_v2_features", "v2"),
    ]
    experiments = {}
    prediction_frames = []
    best_artifacts = None
    best_experiment_name = None
    best_mae = float("inf")
    for experiment_name, under_feature_version in experiment_specs:
        result, predictions, artifacts = run_v3_experiment(
            experiment_name,
            baseline_df,
            v2_df,
            baseline_config,
            v2_config,
            train_idx,
            test_idx,
            under_feature_version,
        )
        experiments[experiment_name] = result
        prediction_frames.append(predictions)
        mae = result["predicted_routing_threshold_calibrated"]["overall_metrics"]["MAE"]
        if mae < best_mae:
            best_mae = mae
            best_artifacts = artifacts
            best_experiment_name = experiment_name

    prediction_output = pd.concat(prediction_frames, ignore_index=True)
    prediction_output.to_csv(args.predictions_path, index=False, encoding="utf-8-sig")
    model_path = args.model_dir / "price_vnd_v3_routed_model.joblib"
    joblib.dump({"experiment_name": best_experiment_name, **best_artifacts}, model_path)

    split_info = {
        "method": "GroupShuffleSplit",
        "group_column": "hotel_id",
        "test_size": 0.2,
        "random_state": RANDOM_STATE,
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "train_hotels": int(v2_df.iloc[train_idx]["hotel_id"].nunique()),
        "test_hotels": int(v2_df.iloc[test_idx]["hotel_id"].nunique()),
    }
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "target": "price_vnd",
        "task": "Two-stage price prediction with segment classifier and routed regressors",
        "raw_path": str(args.raw_path),
        "processed_path": str(args.processed_path),
        "predictions_path": str(args.predictions_path),
        "model_path": str(model_path),
        "best_experiment_name": best_experiment_name,
        "split": split_info,
        "segment_definition": {
            "<10M": "price_vnd < 10000000",
            "10M-30M": "10000000 <= price_vnd < 30000000",
            ">30M": "price_vnd >= 30000000",
        },
        "data": {
            "rows_after_target_filter": int(len(v2_df)),
            "unique_hotels": int(v2_df["hotel_id"].nunique()),
            "v3_segment_counts": v2_df["true_segment_v3"].value_counts().to_dict(),
            "v3_segment_counts_train": v2_df.iloc[train_idx]["true_segment_v3"].value_counts().to_dict(),
            "v3_segment_counts_test": v2_df.iloc[test_idx]["true_segment_v3"].value_counts().to_dict(),
        },
        "dropped_columns": {
            "id": [col for col in base.ID_DROP_COLS if col in df.columns],
            "leakage": [col for col in base.LEAKAGE_COLS if col in df.columns],
            "constant": [col for col in base.CONSTANT_COLS if col in df.columns],
            "all_missing": [col for col in base.ALL_MISSING_COLS if col in df.columns],
            "long_text_or_unused": [col for col in base.TEXT_DROP_COLS if col in df.columns],
        },
        "experiments": experiments,
        "comparison_to_previous_versions": load_previous_comparisons(),
    }
    args.report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        name: {
            "classifier_over30m_recall": exp["classifier"]["threshold_metrics"]["per_class"][">30M"]["recall"],
            "overall_MAE": exp["predicted_routing_threshold_calibrated"]["overall_metrics"]["MAE"],
            "overall_MAPE_percent": exp["predicted_routing_threshold_calibrated"]["overall_metrics"]["MAPE_percent"],
            "over30m_pred_to_actual_median": exp["predicted_routing_threshold_calibrated"]["underprediction_metrics"][
                "by_true_segment"
            ][">30M"]["median_pred_to_actual_ratio"],
        }
        for name, exp in experiments.items()
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Processed data saved to: {args.processed_path}")
    print(f"Prediction errors saved to: {args.predictions_path}")
    print(f"Evaluation report saved to: {args.report_path}")
    print(f"Best model artifact saved to: {model_path}")


if __name__ == "__main__":
    main()
