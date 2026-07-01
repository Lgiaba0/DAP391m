import argparse
import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RAW_PATH = Path("data/raw/vietnam_rooms_properties_merged.csv")
PROCESSED_PATH = Path("data/processed/reg/v1/vietnam_price_vnd_modeling.csv")
REPORT_PATH = Path("reports/reg/v1/price_vnd_model_evaluation.json")
RANDOM_STATE = 42


ID_DROP_COLS = [
    "hotel_id",
    "block_id",
    "source_url",
    "hotel_name",
    "street_address",
]

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

CONSTANT_COLS = [
    "checkin",
    "checkout",
    "country",
    "country_code",
    "best_rating",
    "is_genius",
    "price_data_available",
    "no_smoking",
    "pets_allowed",
    "free_cancellation",
]

ALL_MISSING_COLS = [
    "score_staff",
    "score_facilities",
    "score_cleanliness",
    "score_comfort",
    "score_location",
    "score_wifi",
    "score_value",
    "child_free_age",
]

TEXT_DROP_COLS = [
    "description",
    "room_names_sample",
]

BASE_NUMERIC_FEATURES = [
    "star_rating_clean",
    "star_unknown",
    "room_index",
    "max_persons",
    "num_rate_options",
    "review_score",
    "property_type_id",
    "latitude",
    "longitude",
    "dest_ufi",
    "review_count",
    "num_room_types",
    "desc_word_count",
    "amenities_count",
    "amenity_density",
    "popularity_proxy",
    "review_to_star",
    "room_quality_tier",
    "bedroom_count",
    "has_view",
    "has_sea_view",
    "has_ocean_view",
    "has_city_view",
    "has_mountain_view",
    "has_garden_view",
    "has_balcony_room",
    "has_terrace",
    "has_private_pool",
    "is_family_room",
    "is_shared_room",
]

BASE_BINARY_FEATURES = [
    "breakfast_included",
    "has_breakfast_option",
]

BASE_CATEGORICAL_FEATURES = [
    "source_city",
    "city",
    "district",
    "region",
    "property_type",
    "local_currency",
    "score_tier",
    "room_type_extracted",
    "bed_type",
]


QUALITY_ORDER = {
    "unknown": 0,
    "basic": 1,
    "standard": 2,
    "superior": 3,
    "deluxe": 4,
    "premium": 5,
    "executive": 6,
    "luxury": 7,
    "suite": 8,
    "villa": 9,
    "presidential": 10,
}

NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
}


def normalize_room_name(value):
    if pd.isna(value):
        return ""
    return str(value).lower()


def has_any(text, patterns):
    return int(any(re.search(pattern, text) for pattern in patterns))


def extract_quality_tier(text):
    tier = QUALITY_ORDER["unknown"]
    for keyword, value in QUALITY_ORDER.items():
        if keyword != "unknown" and re.search(rf"\b{re.escape(keyword)}\b", text):
            tier = max(tier, value)
    if re.search(r"\beconomy\b|\bbudget\b", text):
        tier = max(tier, QUALITY_ORDER["basic"])
    return tier


def extract_room_type(text):
    ordered_rules = [
        ("villa", [r"\bvilla\b"]),
        ("suite", [r"\bsuite\b"]),
        ("dormitory", [r"\bdormitory\b", r"\bbunk bed\b", r"\bmixed dorm\b"]),
        ("apartment", [r"\bapartment\b"]),
        ("studio", [r"\bstudio\b"]),
        ("bungalow", [r"\bbungalow\b"]),
        ("chalet", [r"\bchalet\b"]),
        ("family", [r"\bfamily\b"]),
        ("double", [r"\bdouble\b"]),
        ("twin", [r"\btwin\b"]),
        ("single", [r"\bsingle\b"]),
        ("bed", [r"\bbed\b"]),
    ]
    for label, patterns in ordered_rules:
        if has_any(text, patterns):
            return label
    return "room"


def extract_bed_type(text):
    ordered_rules = [
        ("king", [r"\bking\b"]),
        ("queen", [r"\bqueen\b"]),
        ("double", [r"\bdouble\b"]),
        ("twin", [r"\btwin\b"]),
        ("single", [r"\bsingle\b"]),
        ("bunk", [r"\bbunk\b"]),
        ("sofa", [r"\bsofa\b"]),
    ]
    for label, patterns in ordered_rules:
        if has_any(text, patterns):
            return label
    return "unknown"


def extract_bedroom_count(text):
    match = re.search(r"\b(\d+)[-\s]?bedroom\b", text)
    if match:
        return int(match.group(1))
    for word, value in NUMBER_WORDS.items():
        if re.search(rf"\b{word}[-\s]?bedroom\b", text):
            return value
    return 0


def add_room_name_features(df):
    room_text = df["room_name"].map(normalize_room_name)
    df = df.copy()
    df["room_quality_tier"] = room_text.map(extract_quality_tier)
    df["room_type_extracted"] = room_text.map(extract_room_type)
    df["bed_type"] = room_text.map(extract_bed_type)
    df["bedroom_count"] = room_text.map(extract_bedroom_count)
    df["has_view"] = room_text.map(lambda x: has_any(x, [r"\bview\b"]))
    df["has_sea_view"] = room_text.map(lambda x: has_any(x, [r"\bsea view\b"]))
    df["has_ocean_view"] = room_text.map(lambda x: has_any(x, [r"\bocean view\b"]))
    df["has_city_view"] = room_text.map(lambda x: has_any(x, [r"\bcity view\b"]))
    df["has_mountain_view"] = room_text.map(lambda x: has_any(x, [r"\bmountain view\b"]))
    df["has_garden_view"] = room_text.map(lambda x: has_any(x, [r"\bgarden view\b"]))
    df["has_balcony_room"] = room_text.map(lambda x: has_any(x, [r"\bbalcony\b"]))
    df["has_terrace"] = room_text.map(lambda x: has_any(x, [r"\bterrace\b"]))
    df["has_private_pool"] = room_text.map(
        lambda x: has_any(x, [r"\bprivate pool\b", r"\bpool villa\b"])
    )
    df["is_family_room"] = room_text.map(lambda x: has_any(x, [r"\bfamily\b"]))
    df["is_shared_room"] = room_text.map(
        lambda x: has_any(x, [r"\bdormitory\b", r"\bbunk bed\b", r"\bmixed dorm\b"])
    )
    return df


def load_and_preprocess(raw_path):
    df = pd.read_csv(raw_path)
    df = df[df["price_vnd"].notna() & (df["price_vnd"] > 0)].copy()

    if "district" in df.columns:
        df["district"] = df["district"].astype("string").replace("-1", "Unknown")

    df["star_unknown"] = (pd.to_numeric(df["star_rating"], errors="coerce") == 0).astype(int)
    df["star_rating_clean"] = pd.to_numeric(df["star_rating"], errors="coerce").replace(0, np.nan)
    df = add_room_name_features(df)
    df["room_name"] = df["room_name"].fillna("")

    amenity_features = []
    for col in [col for col in df.columns if col.startswith("amenity_")]:
        non_null_values = set(pd.to_numeric(df[col], errors="coerce").dropna().unique())
        if non_null_values.issubset({0, 1}):
            amenity_features.append(col)
    numeric_features = [col for col in BASE_NUMERIC_FEATURES if col in df.columns]
    binary_features = [col for col in BASE_BINARY_FEATURES + amenity_features if col in df.columns]
    categorical_features = [col for col in BASE_CATEGORICAL_FEATURES if col in df.columns]
    text_features = ["room_name"]

    selected_cols = (
        ["hotel_id", "price_vnd"]
        + numeric_features
        + binary_features
        + categorical_features
        + text_features
    )
    selected_cols = list(dict.fromkeys(selected_cols))
    modeling_df = df[selected_cols].copy()

    for col in numeric_features + binary_features:
        modeling_df[col] = pd.to_numeric(modeling_df[col], errors="coerce")

    for col in categorical_features:
        modeling_df[col] = modeling_df[col].astype("string").fillna("Unknown")

    return modeling_df, {
        "numeric_features": numeric_features,
        "binary_features": binary_features,
        "categorical_features": categorical_features,
        "text_features": text_features,
        "dropped_columns": {
            "id": [col for col in ID_DROP_COLS if col in df.columns],
            "leakage": [col for col in LEAKAGE_COLS if col in df.columns],
            "constant": [col for col in CONSTANT_COLS if col in df.columns],
            "all_missing": [col for col in ALL_MISSING_COLS if col in df.columns],
            "long_text": [col for col in TEXT_DROP_COLS if col in df.columns],
            "duplicates_or_unused": [
                col
                for col in ["property_type_label", "crawled_at", "star_rating"]
                if col in df.columns
            ],
        },
    }


def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def make_preprocessor(feature_config):
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )
    binary_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", make_one_hot_encoder()),
        ]
    )

    transformers = []
    if feature_config["numeric_features"]:
        transformers.append(("num", numeric_transformer, feature_config["numeric_features"]))
    if feature_config["binary_features"]:
        transformers.append(("bin", binary_transformer, feature_config["binary_features"]))
    if feature_config["categorical_features"]:
        transformers.append(("cat", categorical_transformer, feature_config["categorical_features"]))
    transformers.append(
        (
            "room_name_tfidf",
            TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=5, max_features=100),
            "room_name",
        )
    )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_models():
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor

    return {
        "LightGBM": LGBMRegressor(
            n_estimators=600,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def evaluate_predictions(y_true_log, y_pred_log):
    y_true_vnd = np.expm1(y_true_log)
    y_pred_vnd = np.expm1(y_pred_log)
    y_pred_vnd = np.clip(y_pred_vnd, a_min=0, a_max=None)

    mae = mean_absolute_error(y_true_vnd, y_pred_vnd)
    rmse = math.sqrt(mean_squared_error(y_true_vnd, y_pred_vnd))
    mape = np.mean(np.abs((y_true_vnd - y_pred_vnd) / y_true_vnd)) * 100
    rmsle = math.sqrt(mean_squared_error(y_true_log, y_pred_log))
    r2 = r2_score(y_true_vnd, y_pred_vnd)

    return {
        "MAE": round(float(mae), 2),
        "RMSE": round(float(rmse), 2),
        "MAPE_percent": round(float(mape), 4),
        "RMSLE": round(float(rmsle), 6),
        "R2": round(float(r2), 6),
    }


def train_and_evaluate(modeling_df, feature_config):
    y = np.log1p(modeling_df["price_vnd"])
    X = modeling_df.drop(columns=["price_vnd"])

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y, groups=modeling_df["hotel_id"]))

    X_train = X.iloc[train_idx].drop(columns=["hotel_id"])
    X_test = X.iloc[test_idx].drop(columns=["hotel_id"])
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    models = build_models()
    results = {}

    for model_name, model in models.items():
        start = time.perf_counter()
        pipeline = Pipeline(
            steps=[
                ("preprocessor", make_preprocessor(feature_config)),
                ("model", model),
            ]
        )
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        metrics = evaluate_predictions(y_test, y_pred)
        metrics["fit_predict_seconds"] = round(time.perf_counter() - start, 3)
        results[model_name] = metrics

    split_info = {
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "train_hotels": int(modeling_df.iloc[train_idx]["hotel_id"].nunique()),
        "test_hotels": int(modeling_df.iloc[test_idx]["hotel_id"].nunique()),
    }
    return results, split_info


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-path", type=Path, default=RAW_PATH)
    parser.add_argument("--processed-path", type=Path, default=PROCESSED_PATH)
    parser.add_argument("--report-path", type=Path, default=REPORT_PATH)
    return parser.parse_args()


def main():
    args = parse_args()
    args.processed_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)

    modeling_df, feature_config = load_and_preprocess(args.raw_path)
    processed_write_status = "written"
    try:
        modeling_df.to_csv(args.processed_path, index=False, encoding="utf-8-sig")
    except PermissionError:
        processed_write_status = "skipped_permission_denied"
        print(f"Could not overwrite processed data because it is locked: {args.processed_path}")

    model_results, split_info = train_and_evaluate(modeling_df, feature_config)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "Predict room price in VND using log1p(price_vnd)",
        "raw_path": str(args.raw_path),
        "processed_path": str(args.processed_path),
        "processed_write_status": processed_write_status,
        "target": "price_vnd",
        "target_transform": "log1p",
        "split": {
            "method": "GroupShuffleSplit",
            "group_column": "hotel_id",
            "test_size": 0.2,
            "random_state": RANDOM_STATE,
            **split_info,
        },
        "data": {
            "rows_after_target_filter": int(len(modeling_df)),
            "unique_hotels": int(modeling_df["hotel_id"].nunique()),
        },
        "features": feature_config,
        "models": model_results,
    }
    args.report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report["models"], ensure_ascii=False, indent=2))
    print(f"Processed data saved to: {args.processed_path}")
    print(f"Evaluation report saved to: {args.report_path}")


if __name__ == "__main__":
    main()
