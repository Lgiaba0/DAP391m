# Data Preprocessing and Modeling Guide - Target `price_vnd` - V5

This document extends `data_preprocessing_price_vnd_v4.md` with a luxury-tail and outlier-aware strategy.

V4 improved high-price routing and materially improved the `10M-30M` segment, but the `>30M` luxury tail remains the weakest range. V5 should keep the stable V4 routing foundation, then treat `>30M` not as generic noise, but as a small valid pricing regime that needs its own audit, routing signal, and constrained calibration.

## 1. V5 Goal

Main task stays the same:

```text
Predict room price in VND
Target: price_vnd
Main target transform for regressors: log1p(price_vnd)
```

V5 changes the modeling focus:

```text
1. Preserve the V4 low-price and binary high-price routing foundation
2. Add an explicit luxury-tail audit for large price outliers
3. Split high-price handling into 10M-30M and >30M-like behavior
4. Add a secondary luxury-tail classifier inside the >=10M route
5. Apply luxury lower-bound calibration only when confidence is high
6. Improve >30M pred_to_actual_ratio without breaking <10M
```

V5 should still report the same detailed evaluation segments:

```text
<10M
10M-30M
>30M
```

But the operational routing should become:

```text
low_price:       price_vnd < 10M
high_mid_price:  10M <= price_vnd < 30M
luxury_tail:     price_vnd >= 30M
```

## 2. Why V5 Is Needed

Best V4 high-price-focused experiment:

```text
threshold_0_25_classifier_weight_config_c_luxury_score_4_high_price_lightgbm_quantile_0_6
calibration_mode: uncalibrated
```

V4 best metrics:

```text
Overall:
MAE: 1,882,485.88 VND
MAPE: 38.6554%
R2: 0.285118

<10M:
MAE: 774,148.19 VND
MAPE: 37.8589%
R2: -0.350070

10M-30M:
MAE: 7,749,302.20 VND
MAPE: 49.0140%
median pred_to_actual_ratio: 1.116394
R2: -2.751131

>30M:
MAE: 58,903,086.06 VND
MAPE: 67.0508%
median pred_to_actual_ratio: 0.305239
underprediction_rate: 95.2381%
severe_underprediction_count_ratio_below_0_5: 18 out of 21
R2: -1.235530
```

V4 routing improved but did not solve the luxury tail:

```text
route_recall_over_10m: 0.71875
route_precision_over_10m: 0.528736
false_negative_over_10m_count: 18 out of 64
false_positive_over_10m_count: 41
```

Interpretation:

```text
V4 is useful for finding more high-price rows and improves 10M-30M.
The remaining bottleneck is not only binary high-price routing.
The bottleneck is detecting and pricing valid luxury-tail outliers.
```

V5 should therefore avoid treating all outliers as bad data. Expensive villas, presidential suites, beachfront residences, and multi-bedroom private pool rooms are valid target cases.

## 3. Preserve V1/V2/V3/V4 Leakage Rules

Continue to drop direct price leakage columns from all feature matrices:

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

Keep `hotel_id` only for grouped splitting, then remove it from model feature matrices.

Do not use raw `hotel_name` as a full text or one-hot feature. Continue using only general keyword flags:

```text
name_has_resort
name_has_villa
name_has_spa
name_has_luxury
name_has_beach
name_has_hotel
name_has_homestay
name_has_hostel
name_has_apartment
```

## 4. Train/Test Split

Use the same group-based split as previous versions:

```python
from sklearn.model_selection import GroupShuffleSplit

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42,
)
```

Split by `hotel_id`, not by row.

Use the same train/test indices for:

```text
binary high-price classifier
secondary luxury-tail classifier
low-price regressor
high-mid regressor
luxury-tail experiments
threshold experiments
calibration experiments
oracle routing evaluation
predicted routing evaluation
```

## 5. Outlier Audit Before Modeling

Before changing models, V5 must audit high outliers and classify them as likely valid or suspicious.

Create an audit table for rows:

```text
price_vnd >= 30M
price_vnd >= 50M
price_vnd >= 100M
top 50 rows by price_vnd
top 50 rows by V4 absolute error
```

Recommended output:

```text
reports/v5/price_vnd_luxury_outlier_audit_v5.csv
```

Recommended audit columns:

```text
hotel_id
room_name
city
source_city
district
region
property_type
star_rating_clean
max_persons
bedroom_count
room_type_extracted
bed_type
luxury_route_score
luxury_amenity_score
wellness_amenity_score
price_vnd
true_segment_v5
outlier_bucket
valid_luxury_signal_count
suspicious_price_flag
suspicious_reason
```

Example suspicious rules:

```python
suspicious_price_flag = (
    (price_vnd >= 100_000_000)
    & (star_rating_clean < 4)
    & (bedroom_count <= 1)
    & (luxury_route_score < 3)
    & (max_persons <= 2)
)
```

Do not automatically remove suspicious rows. Report them first.

Recommended handling:

```text
valid luxury outlier -> keep
suspicious outlier -> run a diagnostic experiment excluding it
unknown -> keep in main experiment and mark in prediction CSV
```

## 6. V5 Segment and Targets

Create detailed reporting segment labels:

```python
def assign_v5_segment(price_vnd):
    if price_vnd < 10_000_000:
        return "<10M"
    if price_vnd < 30_000_000:
        return "10M-30M"
    return ">30M"
```

Create the primary binary high-price target:

```python
def assign_high_price_target(price_vnd):
    return int(price_vnd >= 10_000_000)
```

Create a secondary luxury-tail target inside high-price rows:

```python
def assign_luxury_tail_target(price_vnd):
    return int(price_vnd >= 30_000_000)
```

Recommended labels:

```text
primary route:
0 -> <10M
1 -> >=10M

secondary high-price route:
0 -> 10M-30M-like
1 -> >30M-like
```

## 7. Preserve the V4 `<10M` Path

Keep the low-price path stable:

```text
under_10m_regressor: XGBoostRegressor
under_10m_feature_config: baseline_features
training rows: price_vnd < 10M
target: log1p(price_vnd)
```

Do not train the `<10M` regressor on high-price rows.

Do not apply high-price or luxury-tail lower bounds to rows routed as `<10M`.

Reason:

```text
V4 showed that aggressive high-price calibration can damage <10M through false positives.
V5 should protect <10M by making luxury calibration narrower, not broader.
```

## 8. Primary Binary High-Price Router

Keep the V4 primary router:

```text
classifier_high_price: predict price_vnd >= 10M
feature_config: v2_expanded_features
```

Recommended classifier candidates:

```text
1. XGBoostClassifier
2. LightGBMClassifier
3. RandomForestClassifier
```

Use V4-style thresholds:

```text
p_high_price >= 0.20
p_high_price >= 0.25
p_high_price >= 0.30
p_high_price >= 0.35
```

V5 should not chase recall blindly. Select primary high-price thresholds with a low-price guardrail:

```text
<10M MAPE should not exceed V3 <10M MAPE by more than 5 percentage points
```

Reference guardrail:

```text
V3 <10M MAPE: 33.3671%
V5 soft ceiling: about 38.5%
```

## 9. Secondary Luxury-Tail Classifier

Train a secondary classifier only on training rows where:

```text
price_vnd >= 10M
```

Target:

```text
0 -> 10M-30M
1 -> >30M
```

Recommended model candidates:

```text
1. XGBoostClassifier
2. LightGBMClassifier
3. RandomForestClassifier
4. LogisticRegression as a calibrated simple baseline
```

Use the V2 expanded feature set plus V5 luxury signal features.

Because `>30M` is very small, use both sample weighting and threshold tuning:

```text
10M-30M: weight = 1
>30M: weight = 6, 8, 10, 12
```

Test luxury-tail probability thresholds:

```text
p_luxury_tail >= 0.10
p_luxury_tail >= 0.15
p_luxury_tail >= 0.20
p_luxury_tail >= 0.25
p_luxury_tail >= 0.30
```

Secondary classifier metrics to report:

```text
accuracy_inside_high_price
precision_luxury_tail
recall_luxury_tail
f1_luxury_tail
macro_f1_inside_high_price
confusion_matrix_inside_high_price
false_negative_luxury_tail_count
false_positive_luxury_tail_count
```

Most important secondary metric:

```text
recall_luxury_tail with controlled false positives
```

## 10. Stronger Luxury Signal Features

Keep all V1/V2/V4 features, then add V5 luxury confidence features.

Recommended binary features:

```text
is_villa_or_residence
is_suite_or_penthouse
is_presidential_or_royal
has_private_pool_or_pool_access
has_beach_or_ocean_front
has_multi_bedroom_luxury
is_large_capacity_luxury
is_five_star_luxury
is_resort_luxury
is_luxury_tail_keyword_strong
```

Example:

```python
is_villa_or_residence = has_villa | has_residence
is_suite_or_penthouse = has_suite | has_penthouse
is_presidential_or_royal = has_presidential | has_royal
has_private_pool_or_pool_access = has_private_pool | has_pool_access
has_beach_or_ocean_front = has_beachfront | has_ocean_front | has_private_beach
has_multi_bedroom_luxury = (bedroom_count >= 2) & (has_villa | has_residence | has_suite)
is_large_capacity_luxury = (max_persons >= 4) & (has_villa | has_suite | has_private_pool)
is_five_star_luxury = (star_rating_clean >= 5) & (luxury_amenity_score >= 6)
is_resort_luxury = name_has_resort & (luxury_amenity_score >= 5)
```

Create a stricter luxury confidence score:

```python
luxury_tail_score = (
    3 * has_presidential
    + 3 * has_penthouse
    + 3 * has_four_bedroom_plus
    + 2 * has_private_pool
    + 2 * has_private_beach
    + 2 * has_beachfront
    + 2 * has_villa
    + 2 * has_residence
    + has_suite
    + has_pool_access
    + has_ocean_front
    + name_has_resort
    + name_has_villa
    + name_has_luxury
    + (star_rating_clean >= 5).astype(int)
    + (max_persons >= 4).astype(int)
    + (luxury_amenity_score >= 7).astype(int)
)
```

Test thresholds:

```text
luxury_tail_score >= 4
luxury_tail_score >= 5
luxury_tail_score >= 6
luxury_tail_score >= 7
```

## 11. Routing Logic

V5 deployable routing should be hierarchical:

```python
if p_high_price < high_price_threshold and luxury_route_score < high_price_rule_threshold:
    route = "<10M"
else:
    route = ">=10M"

if route == ">=10M":
    if p_luxury_tail >= luxury_tail_threshold:
        subroute = ">30M_like"
    elif luxury_tail_score >= luxury_tail_score_threshold:
        subroute = ">30M_like"
    else:
        subroute = "10M-30M_like"
else:
    subroute = "<10M"
```

Important:

```text
Do not apply >30M calibration only because primary route is >=10M.
Apply >30M calibration only when the secondary luxury-tail signal is strong.
```

This is the main V5 correction to V4. V4's broad calibration could harm `<10M` false positives. V5 should make luxury-tail treatment narrow and confidence-gated.

## 12. Regressor Design

Train three candidate pricing paths:

```text
under_10m_regressor:
  rows: price_vnd < 10M
  feature_config: baseline_features
  target: log1p(price_vnd)

high_mid_regressor:
  rows: 10M <= price_vnd < 30M
  feature_config: v2_expanded_features + V5 luxury features
  target: log1p(price_vnd)

high_price_shared_regressor:
  rows: price_vnd >= 10M
  feature_config: v2_expanded_features + V5 luxury features
  target: log1p(price_vnd)

luxury_tail_specialist:
  rows: price_vnd >= 30M
  feature_config: v2_expanded_features + V5 luxury features
  target: log1p(price_vnd)
  use only as diagnostic or blended specialist unless validation proves stable
```

Reason:

```text
10M-30M improved in V4, but >30M still underpredicts.
V5 should test whether a high-mid model plus luxury-tail calibration is better than one shared high-price regressor.
```

Recommended high-mid models:

```text
XGBoostRegressor log MSE
LightGBMRegressor log MSE
LightGBMRegressor quantile alpha 0.60
LightGBMRegressor quantile alpha 0.70
```

Recommended shared high-price models:

```text
XGBoostRegressor with >30M weight 6, 8, 10, 12
LightGBMRegressor quantile alpha 0.60, 0.70, 0.80
```

Recommended luxury-tail specialist experiments:

```text
RandomForestRegressor on >30M rows
XGBoostRegressor on >30M rows with shallow trees
KNN/nearest-neighbor baseline on transformed luxury features
median-by-luxury-score fallback baseline
```

Only deploy `luxury_tail_specialist` if oracle and predicted subroute evaluations show stable improvement.

## 13. Blended Luxury Prediction

Because `>30M` has very few rows, a standalone luxury-tail model may be unstable. Prefer blending when the row is `>30M_like`.

Recommended blending:

```python
shared_pred = high_price_shared_regressor.predict(row)
tail_pred = luxury_tail_specialist.predict(row)

if subroute == ">30M_like":
    pred = 0.65 * shared_pred + 0.35 * tail_pred
else:
    pred = high_mid_regressor.predict(row)
```

Test blend weights:

```text
tail_blend_0_20
tail_blend_0_35
tail_blend_0_50
```

If the luxury specialist is unstable, use the shared high-price regressor plus confidence-gated calibration instead.

## 14. Confidence-Gated Calibration

V5 should report uncalibrated and calibrated versions, but calibration must be gated.

Recommended calibration groups:

```text
10M-30M_like
>30M_like_by_classifier
>30M_like_by_rule
>30M_like_by_both_classifier_and_rule
```

Use lower bounds only for strong luxury-tail confidence:

```python
strong_luxury = (
    route == ">=10M"
    and p_luxury_tail >= luxury_tail_threshold
    and luxury_tail_score >= luxury_tail_score_threshold
)

if strong_luxury:
    pred = pred * luxury_tail_multiplier
    pred = max(pred, q25_actual_over_30m_train)
```

Test lower bounds:

```text
q10_actual_over_30m_train
q25_actual_over_30m_train
q40_actual_over_30m_train
```

Also test softer lower bounds based on predicted probability:

```python
lower_bound = (
    p_luxury_tail * q25_actual_over_30m_train
    + (1 - p_luxury_tail) * q10_actual_10m_30m_train
)
```

Cap multipliers:

```text
min multiplier: 1.0
max multiplier: 2.5
```

Do not use broad lower-bound calibration for every row routed `>=10M`.

## 15. Outlier Treatment Experiments

V5 must compare outlier handling strategies explicitly.

Main experiment:

```text
keep_all_valid_prices
target: log1p(price_vnd)
```

Diagnostic experiments:

```text
exclude_suspicious_outliers_only
winsorize_suspicious_outliers_only
train_luxury_specialist_on_valid_luxury_only
```

Do not run a primary experiment that caps all high-price rows at P99.

Reason:

```text
The >30M rows are exactly the rows V5 is trying to learn.
Global capping removes the signal needed for luxury-tail pricing.
```

If a suspicious row is excluded, report:

```text
excluded_row_count
excluded_price_min
excluded_price_max
excluded_reasons
metrics_with_all_rows
metrics_excluding_suspicious_rows
```

## 16. Oracle Evaluations

V5 should report three oracle evaluations.

### Oracle Primary High-Price Routing

Use the true binary group:

```text
true price_vnd < 10M -> low-price regressor
true price_vnd >= 10M -> high-price path
```

### Oracle High-Price Subroute

Use the true high-price sub-segment:

```text
10M-30M -> high_mid_regressor or high_mid calibration
>30M -> luxury-tail treatment
```

This answers:

```text
If the secondary luxury-tail route were perfect, can V5 improve >30M?
```

### Oracle Outlier Validity

Use the audit label:

```text
valid luxury outlier -> keep/apply luxury-tail handling
suspicious outlier -> evaluate separately
```

This answers:

```text
Are extreme errors caused by valid luxury rows or likely data issues?
```

## 17. Predicted Routing Evaluation

Predicted routing is the deployable V5 system:

```text
primary high-price probability
+ high-price rule override
+ secondary luxury-tail probability
+ strict luxury-tail score override
+ confidence-gated calibration
```

Report all major threshold configurations:

```text
high_threshold_0_20_luxury_threshold_0_15_score_5
high_threshold_0_25_luxury_threshold_0_20_score_5
high_threshold_0_30_luxury_threshold_0_20_score_6
high_threshold_0_35_luxury_threshold_0_25_score_6
```

For each threshold, report:

```text
overall_metrics
segment_metrics_by_true_segment
segment_metrics_by_predicted_route
segment_metrics_by_predicted_subroute
high_price_focus_metrics
luxury_tail_focus_metrics
underprediction_metrics
primary_classifier_metrics
secondary_classifier_metrics
routing_confusion_matrix
subrouting_confusion_matrix
```

## 18. Luxury-Tail Focus Metrics

V5 must add a dedicated metrics block for rows with true price `>30M`.

Recommended structure:

```json
{
  "luxury_tail_focus": {
    "true_over_30m_row_count": 21,
    "subroute_recall_over_30m": 0.0,
    "subroute_precision_over_30m": 0.0,
    "false_negative_over_30m_count": 0,
    "false_positive_over_30m_like_count": 0,
    "metrics_over_30m": {},
    "underprediction_over_30m": {},
    "metrics_over_50m": {},
    "underprediction_over_50m": {},
    "metrics_over_100m": {},
    "underprediction_over_100m": {}
  }
}
```

Important luxury-tail metrics:

```text
subroute_recall_over_30m
false_negative_over_30m_count
median_pred_to_actual_ratio_over_30m
p10_pred_to_actual_ratio_over_30m
underprediction_rate_over_30m
severe_underprediction_count_ratio_below_0_5_over_30m
p90_underprediction_error_over_30m
MAE_over_30m
MAPE_over_30m
```

Do not select V5 only by overall MAE.

Recommended selection priority:

```text
1. <10M MAPE stays within guardrail
2. route_recall_over_10m does not regress badly from V4
3. subroute_recall_over_30m improves
4. >30M median pred_to_actual_ratio improves
5. >30M severe underprediction count decreases
6. 10M-30M does not regress materially from V4
7. overall MAE remains acceptable
```

## 19. Prediction-Level CSV

Save prediction-level output:

```text
reports/v5/price_vnd_predictions_v5.csv
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
max_persons
bedroom_count
actual_price_vnd
true_segment_v5
true_is_high_price
true_is_luxury_tail
primary_p_high_price
primary_route_high_price
luxury_route_score
high_price_rule_override
secondary_p_luxury_tail
secondary_route_luxury_tail
luxury_tail_score
luxury_tail_rule_override
predicted_route
predicted_subroute
regressor_name
high_price_regressor_variant
luxury_tail_variant
raw_pred_price_vnd
blended_pred_price_vnd
calibrated_pred_price_vnd
calibration_mode
calibration_group
absolute_error
absolute_percentage_error
pred_minus_actual
pred_to_actual_ratio
is_underprediction
is_severe_underprediction_ratio_below_0_5
suspicious_price_flag
suspicious_reason
experiment_name
threshold_name
```

Use this file to inspect:

```python
predictions[predictions["true_segment_v5"] == ">30M"]
predictions[predictions["actual_price_vnd"] >= 50_000_000]
predictions[predictions["actual_price_vnd"] >= 100_000_000]
predictions.sort_values("absolute_error", ascending=False).head(50)
```

## 20. Evaluation JSON

Save experiment report:

```text
reports/v5/price_vnd_model_evaluation_v5.json
```

Recommended structure:

```json
{
  "created_at_utc": "...",
  "target": "price_vnd",
  "task": "Luxury-tail-aware routed price prediction",
  "split": {
    "method": "GroupShuffleSplit",
    "group_column": "hotel_id",
    "test_size": 0.2,
    "random_state": 42
  },
  "segment_definition": {
    "<10M": "price_vnd < 10000000",
    "10M-30M": "10000000 <= price_vnd < 30000000",
    ">30M": "price_vnd >= 30000000",
    "high_price": "price_vnd >= 10000000",
    "luxury_tail": "price_vnd >= 30000000"
  },
  "preserved_v4_foundation": {
    "low_price_path": {},
    "primary_high_price_router": {}
  },
  "outlier_audit": {
    "over_30m_count": 0,
    "over_50m_count": 0,
    "over_100m_count": 0,
    "suspicious_outlier_count": 0,
    "audit_path": "reports/v5/price_vnd_luxury_outlier_audit_v5.csv"
  },
  "experiments": {
    "high_0_25_luxury_0_20_score_5_shared_quantile_0_70": {
      "primary_classifier": {},
      "secondary_classifier": {},
      "regressors": {},
      "calibration": {},
      "oracle_primary_routing": {},
      "oracle_subroute": {},
      "predicted_routing": {},
      "high_price_focus": {},
      "luxury_tail_focus": {}
    }
  },
  "best_experiment_by_luxury_tail_focus": "...",
  "best_experiment_by_high_price_focus": "...",
  "best_experiment_by_overall_mae": "...",
  "comparison_to_previous_versions": {}
}
```

## 21. Output Paths

Use versioned folders:

```text
scripts/v5/train_price_vnd_models_v5.py
data/processed/v5/vietnam_price_vnd_modeling_v5.csv
reports/v5/price_vnd_luxury_outlier_audit_v5.csv
reports/v5/price_vnd_predictions_v5.csv
reports/v5/price_vnd_model_evaluation_v5.json
models/v5/price_vnd_v5_routed_model.joblib
```

Keep older versions unchanged:

```text
scripts/v1
scripts/v2
scripts/v3
scripts/v4
data/processed/v1
data/processed/v2
data/processed/v3
data/processed/v4
reports/v1
reports/v2
reports/v3
reports/v4
models/v3
models/v4
```

## 22. Recommended V5 Pipeline

Recommended preprocessing and modeling flow:

```text
Load vietnam_rooms_properties_merged.csv
-> filter rows with valid price_vnd
-> keep hotel_id for grouped split
-> create star_unknown and star_rating_clean
-> replace district = -1 with Unknown
-> extract V1 room_name features
-> extract V2 luxury room_name keyword features
-> extract hotel_name keyword flags
-> create amenity aggregate scores
-> create V4 luxury_route_score
-> create V5 luxury_tail_score and strong luxury composite flags
-> keep room_name for TF-IDF
-> keep description for TF-IDF
-> drop leakage columns
-> create v5 segment target: <10M, 10M-30M, >30M
-> create binary high-price target: price_vnd >=10M
-> create secondary luxury-tail target: price_vnd >=30M
-> create luxury outlier audit table
-> GroupShuffleSplit by hotel_id
-> train primary high-price classifier
-> train secondary luxury-tail classifier inside high-price train rows
-> tune primary high-price thresholds with <10M guardrail
-> tune secondary luxury-tail thresholds
-> train preserved low-price regressor on rows <10M
-> train high-mid regressor on rows 10M-30M
-> train shared high-price regressors on rows >=10M
-> optionally train diagnostic luxury-tail specialist on rows >30M
-> test blended luxury predictions
-> apply confidence-gated calibration
-> evaluate oracle primary routing
-> evaluate oracle high-price subroute
-> evaluate predicted routing
-> report high_price_focus metrics
-> report luxury_tail_focus metrics
-> report underprediction metrics
-> save outlier audit CSV
-> save prediction-level CSV
-> save evaluation JSON
-> save best V5 model artifact
```

## 23. Priority Checklist

Implement V5 in this order:

```text
1. Copy V4 preprocessing and feature engineering
2. Add V5 luxury_tail_score and composite luxury flags
3. Add outlier audit CSV for >30M, >50M, >100M, and top errors
4. Preserve V4 low-price path
5. Preserve V4 primary binary high-price classifier
6. Add secondary luxury-tail classifier inside >=10M rows
7. Add hierarchical routing and subrouting
8. Train high-mid regressor for 10M-30M
9. Keep shared high-price regressor as baseline
10. Add luxury-tail specialist and blended prediction experiments
11. Add confidence-gated calibration only for strong >30M-like rows
12. Add oracle primary and oracle subroute evaluations
13. Add luxury_tail_focus metrics
14. Save V5 prediction CSV, outlier audit CSV, and JSON report
15. Save best V5 model artifact under models/v5
16. Compare V5 against V4 and V3
```

## 24. Success Criteria

V5 should be considered successful if it improves valid luxury-tail behavior without breaking mainstream performance.

Primary success criteria:

```text
<10M MAPE remains <= 38.5%
route_recall_over_10m remains close to or above V4's 0.71875
subroute_recall_over_30m improves versus V4 implicit luxury behavior
>30M median_pred_to_actual_ratio improves from 0.305 to at least 0.45
>30M severe_underprediction_count_ratio_below_0_5 decreases from 18/21 to below 14/21
10M-30M median_pred_to_actual_ratio remains close to V4's 1.116
10M-30M MAPE does not regress materially from 49.014%
```

Stretch targets:

```text
route_recall_over_10m >= 0.75
subroute_recall_over_30m >= 0.50
>30M median_pred_to_actual_ratio >= 0.50
>30M severe_underprediction_count_ratio_below_0_5 <= 12/21
```

Do not require `>30M` R2 to become positive immediately. With only a small luxury-tail test set, R2 can remain unstable and negative even when underprediction behavior improves.

## 25. Interpretation Guideline

When reading V5 results:

- If `<10M` gets worse, raise the primary high-price threshold or make high-price rule override stricter.
- If `10M-30M` gets worse, inspect whether too many high-mid rows are being classified as `>30M_like`.
- If `>30M` is still underpredicted but secondary recall is low, lower the luxury-tail threshold or increase secondary classifier weights.
- If `>30M` is still underpredicted even with oracle subrouting, the bottleneck is the regressor, feature quality, or limited `>30M` data.
- If confidence-gated calibration improves `>30M` but hurts `10M-30M`, make the luxury gate stricter.
- If excluding suspicious outliers improves only the diagnostic metrics, do not remove those rows from the main benchmark unless the audit confirms data errors.
- If blended luxury prediction beats shared high-price prediction in oracle but not predicted routing, improve the secondary luxury classifier.

In summary: V5 should not treat large `price_vnd` values as generic outliers to remove. It should separate likely valid luxury-tail rows from suspicious records, add a secondary `>30M_like` signal, and apply stronger pricing correction only when luxury confidence is high. The main V5 objective is fewer severe underpredictions for true luxury rooms while keeping the stable `<10M` path protected.
