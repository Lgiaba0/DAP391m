# Data Preprocessing Guide for ML Training - Target `price_vnd` - V2

This document extends `data_preprocessing_price_vnd.md` with stronger evaluation, richer luxury/location features, and additional experiments for the long-tailed hotel room price problem.

## 1. V2 Goal

Main task stays the same:

```text
Predict room price in VND
Target: price_vnd
Target transform: log1p(price_vnd)
```

V2 should keep the baseline leakage controls from V1, then improve the pipeline in four areas:

```text
1. Report metrics by price segment
2. Add stronger room/property luxury features
3. Treat ID/code columns as categorical where appropriate
4. Compare baseline versus robust/capped-target experiments
```

The main problem is not only model choice. The target has a very long right tail, and the grouped test set can contain new luxury hotels that are not represented well in training.

## 2. Keep V1 Leakage Rules

Continue to drop direct price leakage columns:

```python
leakage_cols = [
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
```

Continue to avoid using direct hotel identity as a memorization feature:

```python
id_drop_cols = [
    "hotel_id",
    "block_id",
    "source_url",
    "hotel_name",
    "street_address",
]
```

Important note:

- Keep `hotel_id` only for `GroupShuffleSplit`.
- Drop `hotel_id` from the feature matrix after the split.
- Do not use full one-hot encoding for `hotel_name`.

## 3. Train/Test Split

Continue using grouped splitting by `hotel_id`:

```python
from sklearn.model_selection import GroupShuffleSplit

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)
```

This split is harder than random row split, but it is more realistic because the model must predict prices for unseen hotels.

## 4. Primary and Secondary Metrics

Evaluate predictions on the real VND scale:

```python
y_pred_vnd = np.expm1(y_pred_log)
y_test_vnd = np.expm1(y_test_log)
```

Report these overall metrics:

```text
MAE
RMSE
MAPE
RMSLE
R2
Median Absolute Error
P90 Absolute Error
```

Metric notes:

- `RMSLE` and `MAPE` are more useful than raw `RMSE` for a long-tailed price target.
- `R2` on raw VND can look poor because a few very expensive rooms dominate squared error.
- Keep `R2`, but do not use it as the only success criterion.

## 5. Segment-Based Evaluation

V2 must report metrics by price segment. Use fixed VND bands:

```python
price_segments = [
    (0, 1_000_000, "<1M"),
    (1_000_000, 3_000_000, "1M-3M"),
    (3_000_000, 10_000_000, "3M-10M"),
    (10_000_000, 30_000_000, "10M-30M"),
    (30_000_000, np.inf, ">30M"),
]
```

For each segment, report:

```text
row_count
MAE
RMSE
MAPE
RMSLE
R2
Median Absolute Error
P90 Absolute Error
```

Reasons:

- The model may perform acceptably for common rooms but fail on luxury villas and presidential suites.
- Overall `RMSE` and `R2` can hide this behavior.
- Segment metrics make it clear whether the next improvement should target mainstream rooms or luxury-tail rooms.

## 6. Save Prediction Error Analysis

In addition to the JSON report, V2 should save a prediction-level CSV:

```text
reports/price_vnd_predictions_v2.csv
```

Recommended columns:

```text
hotel_id
room_name
city
source_city
district
region
property_type
star_rating_clean
actual_price_vnd
pred_price_vnd
absolute_error
absolute_percentage_error
price_segment
model_name
```

Use this file to inspect top-error cases:

```python
predictions.sort_values("absolute_error", ascending=False).head(50)
```

This is important because the largest errors are likely to be luxury resorts, beachfront villas, presidential suites, multi-bedroom villas, or unusual packages.

## 7. Numeric Features

Recommended numeric features for V2:

```python
numeric_features = [
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
]
```

V2 change:

- Remove `property_type_id` and `dest_ufi` from numeric features.
- Treat them as categorical/code features instead.

## 8. Categorical Features

Recommended categorical features:

```python
categorical_features = [
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
```

Use:

```python
OneHotEncoder(handle_unknown="ignore")
```

Reasons:

- `property_type_id` is an identifier/code, not a numeric quantity.
- `dest_ufi` is also a destination code, not a continuous numeric measurement.
- Tree models can sometimes handle numeric codes, but categorical encoding is cleaner and less misleading.

## 9. Binary Amenity Features

Keep binary `0/1` columns:

```python
binary_features = [
    "breakfast_included",
    "has_breakfast_option",
]
```

Also keep actual binary amenity columns that start with `amenity_`, but only if their non-null values are in `{0, 1}`.

Do not automatically classify every `amenity_` column as binary, because columns such as `amenity_density` are numeric continuous features.

Example:

```python
amenity_binary_features = []

for col in df.columns:
    if col.startswith("amenity_"):
        values = set(pd.to_numeric(df[col], errors="coerce").dropna().unique())
        if values.issubset({0, 1}):
            amenity_binary_features.append(col)
```

## 10. Luxury Amenity Scores

Create aggregate amenity scores to help the model identify high-positioning properties.

Recommended features:

```python
luxury_amenity_cols = [
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

wellness_amenity_cols = [
    "amenity_spa",
    "amenity_fitness_center",
    "amenity_hot_tub",
    "amenity_sauna",
]

parking_amenity_cols = [
    "amenity_free_parking",
    "amenity_paid_parking",
]

family_amenity_cols = [
    "amenity_family_rooms",
    "amenity_kids_club",
    "amenity_garden",
]
```

Create scores:

```python
df["luxury_amenity_score"] = df[luxury_amenity_cols].sum(axis=1)
df["wellness_amenity_score"] = df[wellness_amenity_cols].sum(axis=1)
df["parking_amenity_score"] = df[parking_amenity_cols].sum(axis=1)
df["family_amenity_score"] = df[family_amenity_cols].sum(axis=1)
```

Only include columns that exist in the dataset.

## 11. Expanded Rule-Based Features from `room_name`

Keep V1 rule-based room features:

```python
room_quality_tier
room_type_extracted
bed_type
bedroom_count
has_view
has_sea_view
has_ocean_view
has_city_view
has_mountain_view
has_garden_view
has_balcony_room
has_terrace
has_private_pool
is_family_room
is_shared_room
```

Add V2 luxury-tail features:

```python
room_name_keyword_features = [
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
]
```

Suggested keyword rules:

```text
"beachfront" / "beach front" -> has_beachfront
"oceanfront" / "ocean front" -> has_ocean_front
"bay view" / "bayfront" -> has_bay_view
"lagoon" -> has_lagoon
"penthouse" -> has_penthouse
"residence" / "residences" -> has_residence
"royal" -> has_royal
"presidential" -> has_presidential
"spa inclusive" / "spa package" -> has_spa_package
"all inclusive" / "all-inclusive" -> has_all_inclusive
"pool access" -> has_pool_access
"private beach" -> has_private_beach
"villa" -> has_villa
"suite" -> has_suite
"apartment" -> has_apartment
```

Improve bedroom extraction so it handles:

```text
four-bedroom
four bedrooms
4-bedroom
4 bedroom
4 bedrooms
```

Create:

```python
has_multi_bedroom = int(bedroom_count >= 2)
has_four_bedroom_plus = int(bedroom_count >= 4)
```

## 12. TF-IDF for `room_name`

Continue using TF-IDF for `room_name`.

Recommended V2 configuration:

```python
room_name_tfidf = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=3,
    max_features=200
)
```

V2 changes:

- Lower `min_df` from `5` to `3` so rare luxury phrases are less likely to be dropped.
- Increase `max_features` from `100` to `200`.

Important:

- Continue adding explicit rule-based luxury features.
- Do not rely only on TF-IDF for rare expensive-room keywords.

## 13. TF-IDF for `description`

V2 should add a separate TF-IDF vectorizer for `description`.

Recommended configuration:

```python
description_tfidf = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=10,
    max_features=300
)
```

Reasons:

- Many property-level luxury signals may appear in description, not in `room_name`.
- Examples: `private beach`, `beachfront`, `luxury resort`, `spa`, `ocean`, `bay`, `pool villa`, `airport transfer`, `inclusive package`.

If runtime or dimensionality becomes too high, reduce `max_features` to `150`.

## 14. Non-Memorizing Features from `hotel_name`

V1 drops `hotel_name` to avoid memorization. V2 should still avoid full hotel-name one-hot encoding.

However, it is acceptable to extract general keyword flags from `hotel_name`, because these are not direct hotel identity features.

Suggested features:

```python
hotel_name_keyword_features = [
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
```

Do not keep raw `hotel_name` as a text vectorizer in the baseline V2, because brand names may create memorization or unstable generalization.

## 15. Optional Robust Target Experiment

Keep the main experiment as:

```python
y = np.log1p(df["price_vnd"])
```

Add a second experiment with P99 capped target:

```python
upper = df["price_vnd"].quantile(0.99)
df["price_vnd_capped"] = df["price_vnd"].clip(upper=upper)
y = np.log1p(df["price_vnd_capped"])
```

Evaluate both:

```text
baseline_log_target
p99_capped_log_target
```

Use the capped experiment to understand outlier sensitivity. Do not replace the main target unless the product/business goal is explicitly to ignore extreme luxury prices.

## 16. Models to Try

Keep the three V1 models:

```text
LightGBM
XGBoost
Random Forest
```

Recommended V2 model priority:

```text
1. XGBoost
2. LightGBM
3. Random Forest
```

Reason:

- XGBoost currently performs best on baseline metrics.
- Random Forest is useful as a robust non-boosting baseline, but may struggle with sparse TF-IDF features and high-cardinality one-hot features.

## 17. Experiment Tracking

V2 should store results as experiments instead of a single flat model report.

Recommended output:

```text
reports/price_vnd_model_evaluation_v2.json
```

Suggested JSON structure:

```json
{
  "created_at_utc": "...",
  "target": "price_vnd",
  "split": {
    "method": "GroupShuffleSplit",
    "group_column": "hotel_id",
    "test_size": 0.2,
    "random_state": 42
  },
  "experiments": {
    "baseline_v1_compatible": {
      "models": {}
    },
    "v2_expanded_features": {
      "models": {}
    },
    "v2_p99_capped_target": {
      "models": {}
    }
  }
}
```

For each model, include:

```text
overall_metrics
segment_metrics
fit_predict_seconds
```

## 18. Recommended V2 Pipeline

Recommended preprocessing flow:

```text
Load vietnam_rooms_properties_merged.csv
-> filter rows with valid price_vnd
-> keep hotel_id for grouped split
-> create star_unknown and star_rating_clean
-> replace district = -1 with Unknown
-> extract room_name rule-based features
-> extract expanded luxury room_name keyword features
-> extract non-memorizing hotel_name keyword features
-> create amenity aggregate scores
-> keep room_name for TF-IDF
-> keep description for TF-IDF
-> drop leakage columns
-> move dest_ufi and property_type_id to categorical features
-> GroupShuffleSplit by hotel_id
-> fit preprocessors only on train
-> train LightGBM, XGBoost, Random Forest
-> predict log price
-> convert predictions with expm1
-> report overall metrics
-> report segment metrics
-> save prediction-level error CSV
-> save experiment JSON
```

## 19. Priority Checklist

Implement V2 in this order:

```text
1. Add segment metrics and prediction error CSV
2. Move dest_ufi and property_type_id to categorical features
3. Add expanded room_name luxury keyword features
4. Add amenity aggregate scores
5. Add description TF-IDF
6. Add hotel_name keyword flags without raw hotel_name encoding
7. Add P99 capped-target experiment
8. Compare experiments in one JSON report
```

This order is recommended because it first improves diagnosis, then improves generalizable signal, and only then tests robust target handling.

## 20. Interpretation Guideline

When reading V2 results:

- If `<1M`, `1M-3M`, and `3M-10M` segments improve but `>30M` remains poor, the model is learning mainstream prices but still lacks luxury-tail signal.
- If `>30M` improves after room-name and description features, luxury text features are important.
- If capped-target results improve dramatically while uncapped results stay weak, outlier sensitivity is the main issue.
- If grouped split performance remains much worse than random row split, hotel-level generalization is the main challenge.

In summary: V2 should not only chase a better overall score. It should make the model easier to diagnose across price tiers and add richer generalizable features for luxury, location, and package signals.
