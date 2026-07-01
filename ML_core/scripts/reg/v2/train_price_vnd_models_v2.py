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
PROCESSED_PATH = Path("data/processed/reg/v2/vietnam_price_vnd_modeling_v2.csv")
REPORT_PATH = Path("reports/reg/v2/price_vnd_model_evaluation_v2.json")
PREDICTIONS_PATH = Path("reports/reg/v2/price_vnd_predictions_v2.csv")
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

V2_NUMERIC_FEATURES = [
    "star_rating_clean",
    "star_unknown",
    "room_index",
    "max_persons",
    "num_rate_options",
    "review_score",
    "latitude",
    "longitude",
    "review_count",
    "num_room_types",
    "desc_word_count",
    "amenities_count",
    "amenity_density",
    "popularity_proxy",
    "review_to_star",
    "room_quality_tier",
    "bedroom_count",
    "luxury_amenity_score",
    "wellness_amenity_score",
    "parking_amenity_score",
    "family_amenity_score",
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
    "has_beachfront",
    "has_ocean_front",
    "has_bay_view",
    "has_lagoon",
    "has_penthouse",
    "has_residence",
    "has_royal",
    "has_presidential",
    "has_spa_package",
    "has_all_inclusive",
    "has_pool_access",
    "has_private_beach",
    "has_villa",
    "has_suite",
    "has_apartment",
    "has_multi_bedroom",
    "has_four_bedroom_plus",
    "name_has_resort",
    "name_has_villa",
    "name_has_spa",
    "name_has_luxury",
    "name_has_beach",
    "name_has_hotel",
    "name_has_homestay",
    "name_has_hostel",
    "name_has_apartment",
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

V2_CATEGORICAL_FEATURES = [
    "source_city",
    "city",
    "district",
    "region",
    "property_type",
    "property_type_id",
    "dest_ufi",
    "local_currency",
    "score_tier",
    "room_type_extracted",
    "bed_type",
]

PREDICTION_COLUMNS = [
    "hotel_id",
    "room_name",
    "city",
    "source_city",
    "district",
    "region",
    "property_type",
    "star_rating_clean",
]

PRICE_SEGMENTS = [
    (0, 1_000_000, "<1M"),
    (1_000_000, 3_000_000, "1M-3M"),
    (3_000_000, 10_000_000, "3M-10M"),
    (10_000_000, 30_000_000, "10M-30M"),
    (30_000_000, np.inf, ">30M"),
]

LUXURY_AMENITY_COLS = [
    "amenity_outdoor_pool",
    "amenity_indoor_pool",
    "amenity_spa",
    "amenity_fitness_center",
    "amenity_restaurant",
    "amenity_bar",
    "amenity_room_service",
    "amenity_concierge",
    "amenity_hot_tub",
    "amenity_sauna",
    "amenity_minibar",
    "amenity_balcony",
    "amenity_garden",
    "amenity_airport_shuttle",
]

WELLNESS_AMENITY_COLS = [
    "amenity_spa",
    "amenity_fitness_center",
    "amenity_hot_tub",
    "amenity_sauna",
]

PARKING_AMENITY_COLS = [
    "amenity_free_parking",
    "amenity_paid_parking",
]

FAMILY_AMENITY_COLS = [
    "amenity_family_rooms",
    "amenity_kids_club",
    "amenity_garden",
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


def normalize_text(value):
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
    match = re.search(r"\b(\d+)[-\s]?bedrooms?\b", text)
    if match:
        return int(match.group(1))
    for word, value in NUMBER_WORDS.items():
        if re.search(rf"\b{word}[-\s]?bedrooms?\b", text):
            return value
    return 0


def add_room_name_features(df):
    room_text = df["room_name"].map(normalize_text)
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
    df["has_beachfront"] = room_text.map(lambda x: has_any(x, [r"\bbeachfront\b", r"\bbeach front\b"]))
    df["has_ocean_front"] = room_text.map(lambda x: has_any(x, [r"\boceanfront\b", r"\bocean front\b"]))
    df["has_bay_view"] = room_text.map(lambda x: has_any(x, [r"\bbay view\b", r"\bbayfront\b"]))
    df["has_lagoon"] = room_text.map(lambda x: has_any(x, [r"\blagoon\b"]))
    df["has_penthouse"] = room_text.map(lambda x: has_any(x, [r"\bpenthouse\b"]))
    df["has_residence"] = room_text.map(lambda x: has_any(x, [r"\bresidences?\b"]))
    df["has_royal"] = room_text.map(lambda x: has_any(x, [r"\broyal\b"]))
    df["has_presidential"] = room_text.map(lambda x: has_any(x, [r"\bpresidential\b"]))
    df["has_spa_package"] = room_text.map(lambda x: has_any(x, [r"\bspa inclusive\b", r"\bspa package\b"]))
    df["has_all_inclusive"] = room_text.map(lambda x: has_any(x, [r"\ball inclusive\b", r"\ball-inclusive\b"]))
    df["has_pool_access"] = room_text.map(lambda x: has_any(x, [r"\bpool access\b"]))
    df["has_private_beach"] = room_text.map(lambda x: has_any(x, [r"\bprivate beach\b"]))
    df["has_villa"] = room_text.map(lambda x: has_any(x, [r"\bvilla\b"]))
    df["has_suite"] = room_text.map(lambda x: has_any(x, [r"\bsuite\b"]))
    df["has_apartment"] = room_text.map(lambda x: has_any(x, [r"\bapartment\b"]))
    df["has_multi_bedroom"] = (df["bedroom_count"] >= 2).astype(int)
    df["has_four_bedroom_plus"] = (df["bedroom_count"] >= 4).astype(int)
    return df


def add_hotel_name_features(df):
    name_text = df["hotel_name"].map(normalize_text)
    df = df.copy()
    df["name_has_resort"] = name_text.map(lambda x: has_any(x, [r"\bresort\b"]))
    df["name_has_villa"] = name_text.map(lambda x: has_any(x, [r"\bvilla\b"]))
    df["name_has_spa"] = name_text.map(lambda x: has_any(x, [r"\bspa\b"]))
    df["name_has_luxury"] = name_text.map(lambda x: has_any(x, [r"\bluxury\b"]))
    df["name_has_beach"] = name_text.map(lambda x: has_any(x, [r"\bbeach\b"]))
    df["name_has_hotel"] = name_text.map(lambda x: has_any(x, [r"\bhotel\b"]))
    df["name_has_homestay"] = name_text.map(lambda x: has_any(x, [r"\bhomestay\b"]))
    df["name_has_hostel"] = name_text.map(lambda x: has_any(x, [r"\bhostel\b"]))
    df["name_has_apartment"] = name_text.map(lambda x: has_any(x, [r"\bapartment\b", r"\bapartments\b"]))
    return df


def add_amenity_scores(df):
    df = df.copy()
    score_groups = {
        "luxury_amenity_score": LUXURY_AMENITY_COLS,
        "wellness_amenity_score": WELLNESS_AMENITY_COLS,
        "parking_amenity_score": PARKING_AMENITY_COLS,
        "family_amenity_score": FAMILY_AMENITY_COLS,
    }
    for score_col, source_cols in score_groups.items():
        existing_cols = [col for col in source_cols if col in df.columns]
        if existing_cols:
            df[score_col] = df[existing_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
        else:
            df[score_col] = 0
    return df


def assign_price_segment(values):
    values = pd.Series(values)
    labels = pd.Series(index=values.index, dtype="string")
    for lower, upper, label in PRICE_SEGMENTS:
        labels[(values >= lower) & (values < upper)] = label
    return labels.fillna("Unknown")


def load_and_preprocess(raw_path):
    df = pd.read_csv(raw_path)
    df = df[df["price_vnd"].notna() & (df["price_vnd"] > 0)].copy()

    if "district" in df.columns:
        df["district"] = df["district"].astype("string").replace("-1", "Unknown").fillna("Unknown")

    df["star_unknown"] = (pd.to_numeric(df["star_rating"], errors="coerce") == 0).astype(int)
    df["star_rating_clean"] = pd.to_numeric(df["star_rating"], errors="coerce").replace(0, np.nan)
    df = add_room_name_features(df)
    df = add_hotel_name_features(df)
    df = add_amenity_scores(df)
    df["room_name"] = df["room_name"].fillna("")
    df["description"] = df["description"].fillna("")

    amenity_features = []
    for col in [col for col in df.columns if col.startswith("amenity_")]:
        non_null_values = set(pd.to_numeric(df[col], errors="coerce").dropna().unique())
        if non_null_values.issubset({0, 1}):
            amenity_features.append(col)

    return df, amenity_features


def make_feature_config(df, amenity_features, version):
    if version == "baseline":
        numeric_features = [col for col in BASE_NUMERIC_FEATURES if col in df.columns]
        categorical_features = [col for col in BASE_CATEGORICAL_FEATURES if col in df.columns]
        text_features = ["room_name"]
    elif version == "v2":
        numeric_features = [col for col in V2_NUMERIC_FEATURES if col in df.columns]
        categorical_features = [col for col in V2_CATEGORICAL_FEATURES if col in df.columns]
        text_features = ["room_name", "description"]
    else:
        raise ValueError(f"Unknown feature version: {version}")

    binary_features = [col for col in BASE_BINARY_FEATURES + amenity_features if col in df.columns]
    return {
        "numeric_features": numeric_features,
        "binary_features": binary_features,
        "categorical_features": categorical_features,
        "text_features": text_features,
    }


def make_modeling_df(df, feature_config):
    selected_cols = (
        ["hotel_id", "price_vnd"]
        + [col for col in PREDICTION_COLUMNS if col in df.columns]
        + feature_config["numeric_features"]
        + feature_config["binary_features"]
        + feature_config["categorical_features"]
        + feature_config["text_features"]
    )
    selected_cols = list(dict.fromkeys(selected_cols))
    modeling_df = df[selected_cols].copy()

    for col in feature_config["numeric_features"] + feature_config["binary_features"]:
        modeling_df[col] = pd.to_numeric(modeling_df[col], errors="coerce")

    for col in feature_config["categorical_features"]:
        modeling_df[col] = modeling_df[col].astype("string").fillna("Unknown")

    for col in feature_config["text_features"]:
        modeling_df[col] = modeling_df[col].fillna("")

    return modeling_df


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
    if "room_name" in feature_config["text_features"]:
        transformers.append(
            (
                "room_name_tfidf",
                TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=3, max_features=200),
                "room_name",
            )
        )
    if "description" in feature_config["text_features"]:
        transformers.append(
            (
                "description_tfidf",
                TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=10, max_features=300),
                "description",
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_models():
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor

    return {
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
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def safe_r2(y_true, y_pred):
    if len(y_true) < 2:
        return None
    return round(float(r2_score(y_true, y_pred)), 6)


def evaluate_predictions(y_true_vnd, y_pred_vnd):
    y_true_vnd = np.asarray(y_true_vnd, dtype=float)
    y_pred_vnd = np.clip(np.asarray(y_pred_vnd, dtype=float), a_min=0, a_max=None)
    absolute_error = np.abs(y_true_vnd - y_pred_vnd)
    ape = np.where(y_true_vnd > 0, absolute_error / y_true_vnd, np.nan)

    return {
        "row_count": int(len(y_true_vnd)),
        "MAE": round(float(mean_absolute_error(y_true_vnd, y_pred_vnd)), 2),
        "RMSE": round(float(math.sqrt(mean_squared_error(y_true_vnd, y_pred_vnd))), 2),
        "MAPE_percent": round(float(np.nanmean(ape) * 100), 4),
        "RMSLE": round(float(math.sqrt(mean_squared_error(np.log1p(y_true_vnd), np.log1p(y_pred_vnd)))), 6),
        "R2": safe_r2(y_true_vnd, y_pred_vnd),
        "Median_Absolute_Error": round(float(np.median(absolute_error)), 2),
        "P90_Absolute_Error": round(float(np.percentile(absolute_error, 90)), 2),
    }


def evaluate_segments(y_true_vnd, y_pred_vnd):
    segments = assign_price_segment(y_true_vnd)
    metrics = {}
    for _, _, label in PRICE_SEGMENTS:
        mask = segments == label
        if mask.sum() == 0:
            metrics[label] = {"row_count": 0}
            continue
        metrics[label] = evaluate_predictions(np.asarray(y_true_vnd)[mask], np.asarray(y_pred_vnd)[mask])
    return metrics


def build_prediction_frame(metadata, actual_vnd, pred_vnd, experiment_name, model_name):
    pred_vnd = np.clip(np.asarray(pred_vnd, dtype=float), a_min=0, a_max=None)
    actual_vnd = np.asarray(actual_vnd, dtype=float)
    result = metadata.copy()
    result["actual_price_vnd"] = actual_vnd
    result["pred_price_vnd"] = pred_vnd
    result["absolute_error"] = np.abs(actual_vnd - pred_vnd)
    result["absolute_percentage_error"] = np.where(
        actual_vnd > 0,
        result["absolute_error"] / actual_vnd * 100,
        np.nan,
    )
    result["price_segment"] = assign_price_segment(actual_vnd).to_numpy()
    result["experiment_name"] = experiment_name
    result["model_name"] = model_name
    return result


def run_experiment(experiment_name, modeling_df, feature_config, train_idx, test_idx, capped_target=False):
    feature_cols_to_drop = ["price_vnd", "hotel_id"]
    X = modeling_df.drop(columns=list(dict.fromkeys(feature_cols_to_drop)))

    y_actual_vnd = modeling_df["price_vnd"].astype(float)
    target_vnd = y_actual_vnd.copy()
    cap_value = None
    if capped_target:
        cap_value = float(y_actual_vnd.iloc[train_idx].quantile(0.99))
        target_vnd = target_vnd.clip(upper=cap_value)
    y = np.log1p(target_vnd)

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    actual_test_vnd = y_actual_vnd.iloc[test_idx].to_numpy()
    metadata_cols = [col for col in PREDICTION_COLUMNS if col in modeling_df.columns]
    metadata = modeling_df.iloc[test_idx][metadata_cols].copy()

    models = build_models()
    results = {}
    prediction_frames = []

    for model_name, model in models.items():
        start = time.perf_counter()
        pipeline = Pipeline(
            steps=[
                ("preprocessor", make_preprocessor(feature_config)),
                ("model", model),
            ]
        )
        pipeline.fit(X_train, y_train)
        y_pred_log = pipeline.predict(X_test)
        y_pred_vnd = np.expm1(y_pred_log)
        fit_predict_seconds = round(time.perf_counter() - start, 3)

        results[model_name] = {
            "overall_metrics": evaluate_predictions(actual_test_vnd, y_pred_vnd),
            "segment_metrics": evaluate_segments(actual_test_vnd, y_pred_vnd),
            "fit_predict_seconds": fit_predict_seconds,
        }
        if cap_value is not None:
            results[model_name]["target_cap_p99_train_vnd"] = round(cap_value, 2)

        prediction_frames.append(
            build_prediction_frame(metadata, actual_test_vnd, y_pred_vnd, experiment_name, model_name)
        )

    return results, prediction_frames


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-path", type=Path, default=RAW_PATH)
    parser.add_argument("--processed-path", type=Path, default=PROCESSED_PATH)
    parser.add_argument("--report-path", type=Path, default=REPORT_PATH)
    parser.add_argument("--predictions-path", type=Path, default=PREDICTIONS_PATH)
    return parser.parse_args()


def main():
    args = parse_args()
    args.processed_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_path.parent.mkdir(parents=True, exist_ok=True)

    df, amenity_features = load_and_preprocess(args.raw_path)
    baseline_config = make_feature_config(df, amenity_features, "baseline")
    v2_config = make_feature_config(df, amenity_features, "v2")
    baseline_df = make_modeling_df(df, baseline_config)
    v2_df = make_modeling_df(df, v2_config)

    processed_write_status = "written"
    try:
        v2_df.to_csv(args.processed_path, index=False, encoding="utf-8-sig")
    except PermissionError:
        processed_write_status = "skipped_permission_denied"
        print(f"Could not overwrite processed data because it is locked: {args.processed_path}")

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    y_split = np.log1p(v2_df["price_vnd"])
    train_idx, test_idx = next(splitter.split(v2_df, y_split, groups=v2_df["hotel_id"]))

    experiments = {}
    all_prediction_frames = []
    experiment_specs = [
        ("baseline_v1_compatible", baseline_df, baseline_config, False),
        ("v2_expanded_features", v2_df, v2_config, False),
        ("v2_p99_capped_target", v2_df, v2_config, True),
    ]

    for experiment_name, modeling_df, feature_config, capped_target in experiment_specs:
        model_results, prediction_frames = run_experiment(
            experiment_name,
            modeling_df,
            feature_config,
            train_idx,
            test_idx,
            capped_target=capped_target,
        )
        experiments[experiment_name] = {
            "target_transform": "log1p(price_vnd_capped_p99)" if capped_target else "log1p(price_vnd)",
            "feature_config": feature_config,
            "models": model_results,
        }
        all_prediction_frames.extend(prediction_frames)

    predictions_write_status = "written"
    try:
        pd.concat(all_prediction_frames, ignore_index=True).to_csv(
            args.predictions_path,
            index=False,
            encoding="utf-8-sig",
        )
    except PermissionError:
        predictions_write_status = "skipped_permission_denied"
        print(f"Could not overwrite predictions because it is locked: {args.predictions_path}")

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
        "task": "Predict room price in VND using log1p(price_vnd)",
        "raw_path": str(args.raw_path),
        "processed_path": str(args.processed_path),
        "processed_write_status": processed_write_status,
        "predictions_path": str(args.predictions_path),
        "predictions_write_status": predictions_write_status,
        "split": split_info,
        "data": {
            "rows_after_target_filter": int(len(v2_df)),
            "unique_hotels": int(v2_df["hotel_id"].nunique()),
            "price_segment_counts": assign_price_segment(v2_df["price_vnd"]).value_counts().to_dict(),
        },
        "dropped_columns": {
            "id": [col for col in ID_DROP_COLS if col in df.columns],
            "leakage": [col for col in LEAKAGE_COLS if col in df.columns],
            "constant": [col for col in CONSTANT_COLS if col in df.columns],
            "all_missing": [col for col in ALL_MISSING_COLS if col in df.columns],
            "long_text_or_unused": [col for col in TEXT_DROP_COLS if col in df.columns],
            "duplicates_or_unused": [
                col
                for col in ["property_type_label", "crawled_at", "star_rating"]
                if col in df.columns
            ],
        },
        "experiments": experiments,
    }
    args.report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        experiment_name: {
            model_name: model_result["overall_metrics"]
            for model_name, model_result in experiment_result["models"].items()
        }
        for experiment_name, experiment_result in experiments.items()
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Processed data saved to: {args.processed_path}")
    print(f"Prediction errors saved to: {args.predictions_path}")
    print(f"Evaluation report saved to: {args.report_path}")


if __name__ == "__main__":
    main()
