import argparse
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight


ROOT = Path(__file__).resolve().parents[3]
V2_SCRIPT_PATH = ROOT / "scripts/classify/v2/train_price_classification_v2.py"
spec = importlib.util.spec_from_file_location("price_classification_v2", V2_SCRIPT_PATH)
v2 = importlib.util.module_from_spec(spec)
sys.modules["price_classification_v2"] = v2
spec.loader.exec_module(v2)

SOURCE_PATH = Path("data/processed/classify/v2/vietnam_price_classification_v2.csv")
FALLBACK_SOURCE_PATH = Path("data/processed/classify/v1/vietnam_price_classification_v1.csv")
OUTPUT_DATA_PATH = Path("data/processed/classify/v3/vietnam_price_classification_v3.csv")
REPORT_PATH = Path("reports/classify/v3/price_classification_evaluation_v3.json")
PREDICTIONS_PATH = Path("reports/classify/v3/price_classification_predictions_v3.csv")
ERROR_ANALYSIS_PATH = Path("reports/classify/v3/price_classification_error_analysis_v3.csv")
MODEL_PATH = Path("models/classify/v3/price_classification_v3_model.joblib")
V2_REPORT_PATH = Path("reports/classify/v2/price_classification_evaluation_v2.json")
RANDOM_STATE = 42


def to_stage_class(y):
    y = np.asarray(y, dtype=int)
    out = np.full(len(y), 1, dtype=int)
    out[y <= 1] = 0
    out[y >= 3] = 2
    return out


def make_lgbm_binary_or_multiclass():
    return v2.get_lgbm_classifier(class_weight=None)


def compute_stage_weights(y_stage, middle_multiplier):
    weights = compute_sample_weight(class_weight="balanced", y=y_stage).astype(float)
    weights[np.asarray(y_stage) == 1] *= float(middle_multiplier)
    return weights


def compute_specialist_weights(y_original, mid_boundary_multiplier=1.0, premium_multiplier=1.0):
    weights = compute_sample_weight(class_weight="balanced", y=y_original).astype(float)
    y_original = np.asarray(y_original)
    weights[y_original == 2] *= float(mid_boundary_multiplier)
    weights[y_original == 4] *= float(premium_multiplier)
    return weights


class HierarchicalPriceClassifier:
    def __init__(
        self,
        feature_config,
        middle_multiplier=1.5,
        low_model_factory=make_lgbm_binary_or_multiclass,
        stage_model_factory=make_lgbm_binary_or_multiclass,
        high_model_factory=make_lgbm_binary_or_multiclass,
    ):
        self.feature_config = feature_config
        self.middle_multiplier = middle_multiplier
        self.low_model_factory = low_model_factory
        self.stage_model_factory = stage_model_factory
        self.high_model_factory = high_model_factory
        self.stage_pipeline = None
        self.low_pipeline = None
        self.high_pipeline = None

    @staticmethod
    def _class_proba_for_classes(pipeline, X, target_classes):
        proba = pipeline.predict_proba(X)
        full = np.zeros((len(X), len(target_classes)), dtype=float)
        model_classes = list(pipeline.named_steps["model"].classes_)
        for source_idx, class_id in enumerate(model_classes):
            if int(class_id) in target_classes:
                full[:, target_classes.index(int(class_id))] = proba[:, source_idx]
        row_sums = full.sum(axis=1)
        missing = row_sums <= 0
        if missing.any():
            full[missing, :] = 1.0 / len(target_classes)
            row_sums = full.sum(axis=1)
        return full / row_sums[:, None]

    def fit(self, X, y):
        y = np.asarray(y, dtype=int)
        y_stage = to_stage_class(y)
        self.stage_pipeline = Pipeline(
            [
                ("preprocessor", v2.make_preprocessor(self.feature_config)),
                ("model", self.stage_model_factory()),
            ]
        )
        self.stage_pipeline.fit(
            X,
            y_stage,
            model__sample_weight=compute_stage_weights(y_stage, self.middle_multiplier),
        )

        low_mask = y <= 1
        self.low_pipeline = Pipeline(
            [
                ("preprocessor", v2.make_preprocessor(self.feature_config)),
                ("model", self.low_model_factory()),
            ]
        )
        self.low_pipeline.fit(
            X.iloc[low_mask],
            y[low_mask],
            model__sample_weight=compute_specialist_weights(y[low_mask]),
        )

        high_mask = y >= 3
        self.high_pipeline = Pipeline(
            [
                ("preprocessor", v2.make_preprocessor(self.feature_config)),
                ("model", self.high_model_factory()),
            ]
        )
        self.high_pipeline.fit(
            X.iloc[high_mask],
            y[high_mask],
            model__sample_weight=compute_specialist_weights(y[high_mask], premium_multiplier=1.25),
        )
        return self

    def predict_proba(self, X):
        stage_proba = self._class_proba_for_classes(self.stage_pipeline, X, [0, 1, 2])
        low_proba = self._class_proba_for_classes(self.low_pipeline, X, [0, 1])
        high_proba = self._class_proba_for_classes(self.high_pipeline, X, [3, 4])
        full = np.zeros((len(X), 5), dtype=float)
        full[:, 0] = stage_proba[:, 0] * low_proba[:, 0]
        full[:, 1] = stage_proba[:, 0] * low_proba[:, 1]
        full[:, 2] = stage_proba[:, 1]
        full[:, 3] = stage_proba[:, 2] * high_proba[:, 0]
        full[:, 4] = stage_proba[:, 2] * high_proba[:, 1]
        row_sums = full.sum(axis=1)
        return full / row_sums[:, None]

    def predict(self, X):
        stage_pred = self.stage_pipeline.predict(X).astype(int)
        pred = np.full(len(X), 2, dtype=int)
        low_idx = np.where(stage_pred == 0)[0]
        high_idx = np.where(stage_pred == 2)[0]
        if len(low_idx):
            pred[low_idx] = self.low_pipeline.predict(X.iloc[low_idx]).astype(int)
        if len(high_idx):
            pred[high_idx] = self.high_pipeline.predict(X.iloc[high_idx]).astype(int)
        return pred


def hierarchy_selection_key(metrics):
    ordinal = metrics["ordinal_metrics"]
    business = metrics["business_metrics"]
    v3_metrics = metrics["v2_boundary_metrics"]
    return (
        v3_metrics["mid_range_f1"],
        v3_metrics["mid_range_recall"],
        metrics["macro_f1"],
        metrics["balanced_accuracy"],
        ordinal["adjacent_band_accuracy"],
        -ordinal["severe_misclassification_rate"],
        business["premium_luxury_recall"],
    )


def macro_selection_key(metrics):
    return v2.selection_key(metrics)


def load_previous_baselines():
    out = {"v1": v2.load_v1_report()}
    if V2_REPORT_PATH.exists():
        report = json.loads(V2_REPORT_PATH.read_text(encoding="utf-8"))
        selected = report.get("selected_v2_model")
        metrics = report.get("experiments", {}).get(selected, {}).get("metrics", {})
        out["v2_selected"] = {
            "experiment": selected,
            "macro_f1": metrics.get("macro_f1"),
            "mid_range_f1": metrics.get("v2_boundary_metrics", {}).get("mid_range_f1"),
            "mid_range_recall": metrics.get("v2_boundary_metrics", {}).get("mid_range_recall"),
            "balanced_accuracy": metrics.get("balanced_accuracy"),
            "adjacent_band_accuracy": metrics.get("ordinal_metrics", {}).get("adjacent_band_accuracy"),
            "severe_misclassification_rate": metrics.get("ordinal_metrics", {}).get("severe_misclassification_rate"),
            "premium_luxury_recall": metrics.get("business_metrics", {}).get("premium_luxury_recall"),
        }
    return out


def compare_metrics(current, baseline):
    if not baseline:
        return {}
    return {
        "macro_f1_delta": round(float(current["macro_f1"] - baseline["macro_f1"]), 6),
        "mid_range_f1_delta": round(float(current["v2_boundary_metrics"]["mid_range_f1"] - baseline["mid_range_f1"]), 6),
        "mid_range_recall_delta": round(float(current["v2_boundary_metrics"]["mid_range_recall"] - baseline["mid_range_recall"]), 6),
        "balanced_accuracy_delta": round(float(current["balanced_accuracy"] - baseline["balanced_accuracy"]), 6),
        "adjacent_band_accuracy_delta": round(float(current["ordinal_metrics"]["adjacent_band_accuracy"] - baseline["adjacent_band_accuracy"]), 6),
        "severe_misclassification_rate_delta": round(float(current["ordinal_metrics"]["severe_misclassification_rate"] - baseline["severe_misclassification_rate"]), 6),
        "premium_luxury_recall_delta": round(float(current["business_metrics"]["premium_luxury_recall"] - baseline["premium_luxury_recall"]), 6),
    }


def fit_hierarchy_experiment(name, feature_config, X_train, y_train, X_test, y_test, test_prices, middle_multiplier):
    start = time.perf_counter()
    model = HierarchicalPriceClassifier(feature_config=feature_config, middle_multiplier=middle_multiplier)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    metrics = v2.evaluate_classifier(y_test, pred, prices=test_prices)
    return {
        "name": name,
        "model": model,
        "pred": pred,
        "proba": proba,
        "experiment": {
            "model": "HierarchicalPriceClassifier",
            "model_family": "two_stage_low_middle_high_with_low_high_specialists",
            "feature_set": feature_config["feature_set"],
            "stage_definition": {
                "low": "budget + economy",
                "middle": "mid_range",
                "high": "upscale + premium_luxury",
                "low_specialist": "budget vs economy",
                "high_specialist": "upscale vs premium_luxury",
            },
            "middle_stage_weight_multiplier": float(middle_multiplier),
            "sample_weight": "balanced at stage and specialist levels; high specialist premium_luxury multiplied by 1.25",
            "fit_predict_seconds": round(float(time.perf_counter() - start), 3),
            "metrics": metrics,
        },
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
    return parser.parse_args()


def main():
    args = parse_args()
    for path in [args.output_data_path, args.report_path, args.predictions_path, args.error_analysis_path, args.model_path]:
        path.parent.mkdir(parents=True, exist_ok=True)

    source_path = args.source_path if args.source_path.exists() else args.fallback_source_path
    df = pd.read_csv(source_path)
    df["price_vnd"] = pd.to_numeric(df["price_vnd"], errors="coerce")
    df = df[df["price_vnd"].notna() & (df["price_vnd"] > 0)].copy()
    df["price_class_id"] = df["price_vnd"].apply(v2.assign_price_class).astype(int)
    df["price_class_label"] = df["price_class_id"].map(v2.PRICE_CLASS_LABELS)
    df = v2.add_mid_market_features(df)
    df.to_csv(args.output_data_path, index=False, encoding="utf-8-sig")

    y = df["price_class_id"].astype(int).to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(df, y, groups=df["hotel_id"]))
    raw_train_df = df.iloc[train_idx].reset_index(drop=True)
    raw_test_df = df.iloc[test_idx].reset_index(drop=True)
    y_train = raw_train_df["price_class_id"].astype(int).to_numpy()
    y_test = raw_test_df["price_class_id"].astype(int).to_numpy()
    train_df, test_df, market_encoder = v2.add_market_context_features(raw_train_df, raw_test_df, y_train)

    feature_config = v2.build_feature_config(train_df, "market_context_features")
    feature_cols = feature_config["numeric_features"] + feature_config["categorical_features"] + feature_config["text_features"]
    X_train = v2.prepare_features(train_df, feature_cols)
    X_test = v2.prepare_features(test_df, feature_cols)
    test_prices = test_df["price_vnd"].astype(float).to_numpy()

    experiments = {}
    fitted = {}
    for multiplier in [1.0, 1.25, 1.5, 1.75, 2.0, 2.5]:
        name = f"hierarchy_lgbm_middle_weight_{str(multiplier).replace('.', '_')}"
        result = fit_hierarchy_experiment(name, feature_config, X_train, y_train, X_test, y_test, test_prices, multiplier)
        experiments[name] = result["experiment"]
        fitted[name] = result
        m = result["experiment"]["metrics"]
        print(
            json.dumps(
                {
                    "experiment": name,
                    "macro_f1": m["macro_f1"],
                    "mid_range_f1": m["v2_boundary_metrics"]["mid_range_f1"],
                    "mid_range_recall": m["v2_boundary_metrics"]["mid_range_recall"],
                    "adjacent_band_accuracy": m["ordinal_metrics"]["adjacent_band_accuracy"],
                    "severe_misclassification_rate": m["ordinal_metrics"]["severe_misclassification_rate"],
                    "premium_luxury_recall": m["business_metrics"]["premium_luxury_recall"],
                },
                ensure_ascii=False,
            )
        )

    best_hierarchy_name = max(experiments, key=lambda name: hierarchy_selection_key(experiments[name]["metrics"]))
    best_macro_name = max(experiments, key=lambda name: macro_selection_key(experiments[name]["metrics"]))
    best_ordinal_name = max(experiments, key=lambda name: v2.ordinal_safety_key(experiments[name]["metrics"]))
    best_business_name = max(experiments, key=lambda name: v2.business_safety_key(experiments[name]["metrics"]))
    selected_name = best_hierarchy_name
    selected = fitted[selected_name]
    selected_metrics = experiments[selected_name]["metrics"]

    predictions = v2.make_predictions(
        test_df,
        selected["pred"],
        selected["proba"],
        selected_name,
        experiments[selected_name]["model_family"],
        threshold_probas=None,
    )
    predictions.to_csv(args.predictions_path, index=False, encoding="utf-8-sig")

    error_analysis = v2.build_error_analysis(predictions, selected_name)
    error_analysis = v2.enrich_error_analysis_from_test(error_analysis, predictions, test_df)
    error_analysis.to_csv(args.error_analysis_path, index=False, encoding="utf-8-sig")

    baselines = load_previous_baselines()
    joblib.dump(
        {
            "experiment_name": selected_name,
            "model_family": experiments[selected_name]["model_family"],
            "model": selected["model"],
            "feature_config": feature_config,
            "feature_cols": feature_cols,
            "price_class_labels": v2.PRICE_CLASS_LABELS,
            "market_context_encoder": market_encoder,
            "stage_definition": experiments[selected_name]["stage_definition"],
            "leakage_controls": {
                "dropped_columns": {
                    "leakage": v2.LEAKAGE_COLS,
                    "id": v2.ID_DROP_COLS,
                    "derived_from_target": v2.TARGET_DERIVED_COLS,
                },
                "train_fold_aggregate_features": v2.MARKET_CONTEXT_SPECS,
            },
        },
        args.model_path,
    )

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "price_vnd_band_classification",
        "target": "price_class_id",
        "source_data": str(source_path),
        "output_data": str(args.output_data_path),
        "predictions_path": str(args.predictions_path),
        "error_analysis_path": str(args.error_analysis_path),
        "model_path": str(args.model_path),
        "v3_strategy": "two_stage_hierarchy_low_middle_high_with_low_high_specialists",
        "class_definition": {
            "budget": "price_vnd < 500000",
            "economy": "500000 <= price_vnd < 1000000",
            "mid_range": "1000000 <= price_vnd < 2000000",
            "upscale": "2000000 <= price_vnd < 5000000",
            "premium_luxury": "price_vnd >= 5000000",
        },
        "baselines": baselines,
        "class_distribution": v2.class_distribution(df["price_class_id"]),
        "split": {
            "method": "GroupShuffleSplit",
            "group_column": "hotel_id",
            "test_size": 0.2,
            "random_state": RANDOM_STATE,
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "train_hotels": int(train_df["hotel_id"].nunique()),
            "test_hotels": int(test_df["hotel_id"].nunique()),
            "train_class_distribution": v2.class_distribution(y_train),
            "test_class_distribution": v2.class_distribution(y_test),
        },
        "feature_sets": {
            "expanded_luxury_features": "Inherited from V2/V1-compatible luxury features.",
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
            "market_context_features": list(v2.MARKET_CONTEXT_SPECS.keys()),
            "selected_feature_config": feature_config,
        },
        "leakage_controls": {
            "dropped_columns": {
                "leakage": [col for col in v2.LEAKAGE_COLS if col in df.columns],
                "id": [col for col in v2.ID_DROP_COLS if col in df.columns],
                "derived_from_target": [col for col in v2.TARGET_DERIVED_COLS if col in df.columns],
            },
            "train_fold_aggregate_features": {
                "note": "Market context features are fitted on train fold only, with global train median fallback.",
                "specs": v2.MARKET_CONTEXT_SPECS,
                "global_train_median_class": market_encoder.global_median_,
            },
        },
        "experiments": experiments,
        "best_experiment_by_hierarchy_mid_range_priority": best_hierarchy_name,
        "best_experiment_by_macro_f1": best_macro_name,
        "best_experiment_by_ordinal_safety": best_ordinal_name,
        "best_experiment_by_business_safety": best_business_name,
        "selected_v3_model": selected_name,
        "comparison_to_v1": compare_metrics(selected_metrics, baselines.get("v1")),
        "comparison_to_v2_selected": compare_metrics(selected_metrics, baselines.get("v2_selected")),
        "recommended_v4_direction": {
            "boundary_specialists": "Add economy-vs-mid_range and mid_range-vs-upscale specialists as overrides when hierarchy stage probabilities are close.",
            "class_rate_market_context": "Replace median class encoders with class-rate encoders for economy, mid_range, upscale, and premium_luxury rates.",
            "threshold_tuning": "Tune stage low/middle/high thresholds on grouped inner validation to trade precision for mid_range recall explicitly.",
        },
    }
    args.report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "selected_v3_model": selected_name,
        "best_experiment_by_hierarchy_mid_range_priority": best_hierarchy_name,
        "best_experiment_by_macro_f1": best_macro_name,
        "best_experiment_by_ordinal_safety": best_ordinal_name,
        "best_experiment_by_business_safety": best_business_name,
        "selected_metrics": selected_metrics,
        "comparison_to_v1": report["comparison_to_v1"],
        "comparison_to_v2_selected": report["comparison_to_v2_selected"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Processed classification data saved to: {args.output_data_path}")
    print(f"Prediction-level output saved to: {args.predictions_path}")
    print(f"Error analysis output saved to: {args.error_analysis_path}")
    print(f"Evaluation JSON saved to: {args.report_path}")
    print(f"Best model artifact saved to: {args.model_path}")


if __name__ == "__main__":
    main()
