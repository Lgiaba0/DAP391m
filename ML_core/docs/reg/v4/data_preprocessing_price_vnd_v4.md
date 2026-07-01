# Data Preprocessing and Modeling Guide - Target `price_vnd` - V4

This document extends `data_preprocessing_price_vnd_v3.md` with a high-price-focused strategy.

V3 confirms that the `<10M` path is the most stable part of the routed model, while the `10M-30M` and `>30M` ranges still suffer from strong underprediction. V4 should therefore keep the best V3 low-price pipeline and only redesign the routing and modeling strategy for rows above `10M`.

## 1. V4 Goal

Main task stays the same:

```text
Predict room price in VND
Target: price_vnd
Main target transform for regressors: log1p(price_vnd)
```

V4 changes the modeling focus:

```text
1. Preserve the best V3 pipeline for rows below 10M
2. Replace the 3-class primary classifier with a binary high-price classifier
3. Optimize routing recall for rows >=10M
4. Train and calibrate a stronger high-price regressor
5. Report high-price-focused metrics separately from overall metrics
```

V4 should treat `price_vnd >= 10_000_000` as the main improvement area.

```text
low_price:  price_vnd < 10M
high_price: price_vnd >= 10M
```

Keep the original detailed evaluation segments:

```text
<10M
10M-30M
>30M
```

## 2. Why V4 Is Needed

Best V3 experiment:

```text
under_10m_baseline_features__high_price_v2_features
```

V3 best metrics by true segment:

```text
<10M:
MAE: 609,936.85 VND
MAPE: 33.3671%
median pred_to_actual_ratio: 1.049847

10M-30M:
MAE: 8,674,545.99 VND
MAPE: 53.2649%
median pred_to_actual_ratio: 0.596008

>30M:
MAE: 60,020,604.01 VND
MAPE: 67.9360%
median pred_to_actual_ratio: 0.286118
underprediction_rate: 95.2381%
severe_underprediction_count_ratio_below_0_5: 17 out of 21
```

V3 classifier recall for the `>30M` class was only:

```text
>30M recall: 0.238095
```

This means most expensive rooms were not routed strongly enough into the luxury/high-price path.

V3 oracle routing also performed better than predicted routing:

```text
oracle routing MAE: 1,610,807.11 VND
predicted routing calibrated MAE: 1,772,037.58 VND
```

Interpretation:

```text
If routing improves, the two-stage design can improve.
The main V4 target is therefore high-price routing recall and high-price underprediction reduction.
```

## 3. Preserve V1/V2/V3 Leakage Rules

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

Keep `hotel_id` only for `GroupShuffleSplit`, then remove it from model feature matrices.

Do not use raw `hotel_name` as a full text or one-hot feature. Continue using only non-memorizing keyword flags such as:

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

Use the same group-based split as V1/V2/V3:

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
low/high classifier
low-price regressor
high-price regressor
threshold experiments
calibration experiments
oracle routing evaluation
predicted routing evaluation
```

## 5. Keep the Best V3 `<10M` Path

V4 should keep the best V3 low-price regressor as the default low-price path:

```text
under_10m_regressor: XGBoostRegressor
under_10m_feature_config: baseline_features
training rows: price_vnd < 10M
target: log1p(price_vnd)
```

Reason:

- `<10M` is the largest and most stable segment.
- V3 baseline features performed better than V2 features for this segment.
- V4 should avoid damaging mainstream performance while optimizing high-price rows.

Do not use V2 expanded features for the low-price regressor unless a V4 diagnostic experiment clearly improves `<10M`.

## 6. V4 Segment and Binary Targets

Create detailed reporting segment labels:

```python
def assign_v4_segment(price_vnd):
    if price_vnd < 10_000_000:
        return "<10M"
    if price_vnd < 30_000_000:
        return "10M-30M"
    return ">30M"
```

Create the primary binary routing target:

```python
def assign_high_price_target(price_vnd):
    return int(price_vnd >= 10_000_000)
```

Recommended labels:

```text
0 -> <10M
1 -> >=10M
```

The binary target is easier and more useful than the V3 primary 3-class target because V4 mostly needs to avoid false negatives for high-price rows.

## 7. Binary High-Price Classifier

Train a binary classifier to predict whether a row should go to the high-price path:

```text
classifier_high_price: predict price_vnd >= 10M
```

Recommended classifier candidates:

```text
1. XGBoostClassifier
2. LightGBMClassifier
3. RandomForestClassifier
```

Use the V2 expanded feature set for the classifier:

```text
numeric V2 features
binary amenity features
categorical V2 features
room_name TF-IDF
description TF-IDF
hotel_name keyword flags
luxury room_name flags
amenity aggregate scores
```

Classifier metrics to report:

```text
accuracy
precision_high_price
recall_high_price
f1_high_price
macro_f1
weighted_f1
confusion_matrix
false_negative_high_price_count
false_positive_high_price_count
```

Most important classifier metric:

```text
recall_high_price
```

A false positive sends a cheap row to the high-price model. That may hurt some `<10M` predictions.

A false negative sends an expensive row to the low-price model. That usually creates severe underprediction.

For V4, false negatives are more expensive than false positives.

## 8. Classifier Imbalance Handling

V3 train set had roughly:

```text
<10M: 5174 rows
10M-30M: 194 rows
>30M: 39 rows
```

For the binary classifier, use sample weights:

```text
<10M: 1
10M-30M: 5
>30M: 10
```

Example:

```python
sample_weight = np.select(
    [
        y_train_vnd < 10_000_000,
        (y_train_vnd >= 10_000_000) & (y_train_vnd < 30_000_000),
        y_train_vnd >= 30_000_000,
    ],
    [1.0, 5.0, 10.0],
)
```

Also test stronger settings:

```text
classifier_weight_config_a:
<10M = 1, 10M-30M = 4, >30M = 8

classifier_weight_config_b:
<10M = 1, 10M-30M = 5, >30M = 10

classifier_weight_config_c:
<10M = 1, 10M-30M = 6, >30M = 12
```

Optionally oversample high-price rows in the training set only:

```text
10M-30M: repeat 3 to 5 times
>30M: repeat 8 to 12 times
```

Do not oversample validation or test rows.

## 9. Threshold-Based High-Price Routing

Do not rely only on `argmax` or the default `0.50` probability cutoff.

Test multiple high-price thresholds:

```text
p_high_price >= 0.10
p_high_price >= 0.15
p_high_price >= 0.20
p_high_price >= 0.25
p_high_price >= 0.30
```

Recommended starting logic:

```python
if p_high_price >= high_price_threshold:
    route = ">=10M"
else:
    route = "<10M"
```

Select threshold using high-price-focused metrics:

```text
maximize recall_high_price
minimize false_negative_high_price_count
reduce >=10M underprediction
keep <10M MAPE within an acceptable degradation band
```

Recommended threshold selection rule:

```text
Pick the lowest threshold that improves >=10M recall materially
without causing unacceptable <10M MAPE degradation.
```

## 10. Rule-Based High-Price Override

Add a rule-based score that can override the classifier and route likely luxury rows into the high-price path.

Recommended score inputs:

```text
has_villa
has_suite
has_penthouse
has_residence
has_royal
has_presidential
has_private_pool
has_private_beach
has_beachfront
has_ocean_front
has_bay_view
has_lagoon
has_pool_access
has_multi_bedroom
has_four_bedroom_plus
name_has_resort
name_has_villa
name_has_spa
name_has_luxury
name_has_beach
star_rating_clean >= 5
luxury_amenity_score
wellness_amenity_score
```

Example score:

```python
luxury_route_score = (
    2 * has_presidential
    + 2 * has_penthouse
    + 2 * has_private_pool
    + 2 * has_four_bedroom_plus
    + has_villa
    + has_suite
    + has_residence
    + has_private_beach
    + has_beachfront
    + has_ocean_front
    + name_has_resort
    + name_has_villa
    + name_has_luxury
    + (star_rating_clean >= 5).astype(int)
    + (luxury_amenity_score >= 6).astype(int)
)
```

Recommended routing:

```python
if p_high_price >= high_price_threshold:
    route = ">=10M"
elif luxury_route_score >= luxury_route_threshold:
    route = ">=10M"
else:
    route = "<10M"
```

Test thresholds:

```text
luxury_route_score >= 2
luxury_route_score >= 3
luxury_route_score >= 4
```

The rule-based override should be evaluated carefully because it can increase false positives from `<10M`.

## 11. Optional Secondary Luxury Classifier

After a row is routed to `>=10M`, optionally classify whether it is likely `>30M`:

```text
secondary target inside high-price path:
10M-30M
>30M
```

Use this secondary classifier only for calibration or lower-bound selection, not as the primary router.

Reason:

- There are too few `>30M` rows to rely on a standalone primary classifier.
- But a secondary signal can help decide whether stronger luxury calibration should be applied.

Recommended labels:

```text
high_mid: 10M <= price_vnd < 30M
luxury_tail: price_vnd >= 30M
```

Recommended usage:

```python
if route == ">=10M" and p_luxury_tail >= 0.20:
    calibration_group = ">30M_like"
else:
    calibration_group = "10M-30M_like"
```

## 12. High-Price Regressor Design

Train the high-price regressor only on rows with:

```text
price_vnd >= 10M
```

Use:

```text
target: log1p(price_vnd)
feature_config: v2_expanded_features
```

Recommended primary high-price model:

```text
XGBoostRegressor with sample weights
```

Recommended additional experiments:

```text
LightGBMRegressor with standard squared-error objective
LightGBMRegressor with quantile objective
RandomForestRegressor as a non-boosting baseline
```

Do not use P99 capped target for the high-price regressor.

Reason:

```text
High-price rows are exactly the rows V4 is trying to learn.
Target capping removes the signal needed for >30M rows.
```

## 13. High-Price Regressor Sample Weights

The high-price regressor should weight `>30M` rows more strongly:

```text
10M-30M: weight = 1
>30M: weight = 6 to 10
```

Test at least:

```text
over30_weight_4
over30_weight_6
over30_weight_8
over30_weight_10
```

Example:

```python
sample_weight = np.where(y_train_vnd >= 30_000_000, 8.0, 1.0)
```

Select the high-price regressor by:

```text
>=10M MAPE
10M-30M MAPE
>30M median pred_to_actual_ratio
>30M severe underprediction count
>30M p90_underprediction_error
```

Do not select only by overall MAE.

## 14. Quantile and Asymmetric High-Price Experiments

Because underprediction is the main failure mode, test an asymmetric model for the high-price path.

Recommended LightGBM quantile experiments:

```text
objective = "quantile"
alpha = 0.60
alpha = 0.70
alpha = 0.80
```

Experiment names:

```text
high_price_xgboost_log_mse_weight_6
high_price_xgboost_log_mse_weight_8
high_price_lightgbm_quantile_0_60
high_price_lightgbm_quantile_0_70
high_price_lightgbm_quantile_0_80
```

Purpose:

```text
Predict slightly higher for uncertain expensive rows
Reduce severe underprediction
Accept limited overprediction if high-price recall and business safety improve
```

When using quantile regression, still evaluate on the real VND scale.

## 15. High-Price Calibration

V3 luxury multiplier was nearly neutral:

```text
luxury_multiplier: 1.000738
```

V4 should use validation-based calibration instead of train residual calibration whenever possible.

Recommended calibration groups:

```text
route >=10M
predicted 10M-30M-like
predicted >30M-like
rule-based luxury override
```

Use both multiplier and lower-bound calibration:

```python
if route == ">=10M":
    pred_price_vnd = pred_price_vnd * high_price_multiplier
    pred_price_vnd = max(pred_price_vnd, q10_actual_10m_30m_train)

if route == ">=10M" and luxury_like:
    pred_price_vnd = pred_price_vnd * luxury_multiplier
    pred_price_vnd = max(pred_price_vnd, q25_actual_over_30m_train)
```

Recommended lower bounds:

```text
route >=10M:
lower_bound = q10_actual_10m_30m_train

luxury_like or predicted >30M:
lower_bound = q25_actual_over_30m_train
```

Cap multipliers to prevent unstable predictions:

```text
min multiplier: 1.0
max multiplier: 3.0
```

Report calibrated and uncalibrated versions.

## 16. Oracle Evaluation

V4 should report two oracle evaluations.

### Oracle High-Price Routing

Use the true binary group:

```text
true price_vnd < 10M -> low-price regressor
true price_vnd >=10M -> high-price regressor
```

This answers:

```text
If the binary high-price route were perfect, how good would the system be?
```

### Oracle Luxury Calibration

Use the true detailed segment only for calibration selection:

```text
10M-30M -> high-price standard calibration
>30M -> luxury-tail calibration
```

This answers:

```text
If high-price sub-segment detection were perfect, how much would calibration help?
```

## 17. Predicted Routing Evaluation

Predicted routing is the deployable V4 system:

```text
classifier probability + threshold + optional rule-based override
```

Report all threshold configurations:

```text
threshold_0_10
threshold_0_15
threshold_0_20
threshold_0_25
threshold_0_30
```

For each threshold, report:

```text
overall_metrics
segment_metrics_by_true_segment
segment_metrics_by_predicted_route
high_price_focus_metrics
underprediction_metrics
classifier_metrics
routing_confusion_matrix
```

The recommended best V4 model should be selected primarily from `high_price_focus_metrics`.

## 18. High-Price Focus Metrics

V4 must add a dedicated metrics block for rows with true price `>=10M`.

Recommended structure:

```json
{
  "high_price_focus": {
    "true_over_10m_row_count": 64,
    "route_recall_over_10m": 0.0,
    "route_precision_over_10m": 0.0,
    "false_negative_over_10m_count": 0,
    "false_positive_over_10m_count": 0,
    "metrics_over_10m": {},
    "metrics_10m_30m": {},
    "metrics_over_30m": {},
    "underprediction_over_10m": {},
    "underprediction_10m_30m": {},
    "underprediction_over_30m": {}
  }
}
```

Important underprediction metrics:

```text
underprediction_rate
mean_pred_minus_actual
median_pred_to_actual_ratio
p10_pred_to_actual_ratio
p90_underprediction_error
severe_underprediction_count_ratio_below_0_5
```

For V4, the most important high-price metrics are:

```text
route_recall_over_10m
false_negative_over_10m_count
10M-30M median_pred_to_actual_ratio
>30M median_pred_to_actual_ratio
>30M severe_underprediction_count_ratio_below_0_5
```

## 19. Prediction-Level CSV

Save prediction-level output:

```text
reports/v4/price_vnd_predictions_v4.csv
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
true_segment_v4
true_is_high_price
classifier_p_high_price
classifier_route_high_price
luxury_route_score
rule_override_high_price
routing_strategy
predicted_route
regressor_name
high_price_regressor_variant
raw_pred_price_vnd
calibrated_pred_price_vnd
calibration_group
absolute_error
absolute_percentage_error
pred_minus_actual
pred_to_actual_ratio
is_underprediction
is_severe_underprediction_ratio_below_0_5
experiment_name
threshold_name
```

Use this file to inspect:

```python
predictions[predictions["true_is_high_price"] == 1]
predictions[predictions["true_segment_v4"] == ">30M"]
predictions.sort_values("absolute_error", ascending=False).head(50)
```

## 20. Evaluation JSON

Save experiment report:

```text
reports/v4/price_vnd_model_evaluation_v4.json
```

Recommended structure:

```json
{
  "created_at_utc": "...",
  "target": "price_vnd",
  "task": "High-price-focused routed price prediction",
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
    "high_price": "price_vnd >= 10000000"
  },
  "preserved_v3_low_price_path": {
    "model": "XGBoostRegressor",
    "feature_config": "baseline_features",
    "training_rows": "price_vnd < 10000000"
  },
  "experiments": {
    "threshold_0_15_weight_8_quantile_none": {
      "classifier": {},
      "regressors": {},
      "calibration": {},
      "oracle_routing": {},
      "predicted_routing": {},
      "high_price_focus": {}
    }
  },
  "best_experiment_by_high_price_focus": "...",
  "best_experiment_by_overall_mae": "...",
  "comparison_to_previous_versions": {}
}
```

## 21. Output Paths

Use versioned folders:

```text
scripts/v4/train_price_vnd_models_v4.py
data/processed/v4/vietnam_price_vnd_modeling_v4.csv
reports/v4/price_vnd_predictions_v4.csv
reports/v4/price_vnd_model_evaluation_v4.json
models/v4/price_vnd_v4_routed_model.joblib
```

Keep older versions unchanged:

```text
scripts/v1
scripts/v2
scripts/v3
data/processed/v1
data/processed/v2
data/processed/v3
reports/v1
reports/v2
reports/v3
models/v3
```

## 22. Recommended V4 Pipeline

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
-> create luxury_route_score
-> keep room_name for TF-IDF
-> keep description for TF-IDF
-> drop leakage columns
-> create v4 segment target: <10M, 10M-30M, >30M
-> create binary high-price target: price_vnd >=10M
-> GroupShuffleSplit by hotel_id
-> train binary high-price classifier with imbalance handling
-> tune high-price thresholds
-> train preserved low-price regressor on rows <10M with baseline features
-> train high-price regressors on rows >=10M with V2 expanded features
-> apply high-price sample weights for >30M rows
-> run quantile/asymmetric high-price experiments
-> apply rule-based high-price override
-> apply high-price and luxury-tail calibration
-> evaluate oracle high-price routing
-> evaluate predicted high-price routing
-> report high-price focus metrics
-> report underprediction metrics
-> save prediction-level CSV
-> save evaluation JSON
-> save best model artifact
```

## 23. Priority Checklist

Implement V4 in this order:

```text
1. Copy V3 preprocessing and feature engineering
2. Preserve best V3 low-price regressor configuration
3. Add v4 segment labels: <10M, 10M-30M, >30M
4. Add binary high-price target: price_vnd >=10M
5. Train binary high-price classifier with sample weights
6. Add threshold grid for p_high_price
7. Add luxury_route_score and rule-based override
8. Train high-price XGBoost regressors with >30M sample weights
9. Add LightGBM quantile high-price experiments
10. Add high-price and luxury-tail calibration
11. Add oracle high-price routing evaluation
12. Add predicted routing evaluation for each threshold
13. Add high_price_focus metrics
14. Save v4 prediction CSV and JSON report
15. Save best V4 model artifact under models/v4
16. Compare V4 against V3 best results
```

## 24. Success Criteria

V4 should be considered successful if it improves high-price behavior, even if overall MAE does not improve dramatically.

Primary success criteria:

```text
Recall route >=10M improves materially
False negatives for >=10M decrease
10M-30M median pred_to_actual_ratio increases
>30M median pred_to_actual_ratio increases
>30M severe underprediction count decreases
<10M metrics stay close to V3
```

Suggested metric targets:

```text
route_recall_over_10m: >= 0.75
route_recall_over_30m: >= 0.50
10M-30M median_pred_to_actual_ratio: improve from 0.596 to at least 0.75
>30M median_pred_to_actual_ratio: improve from 0.286 to at least 0.50
>30M severe_underprediction_count_ratio_below_0_5: reduce from 17/21 to below 12/21
<10M MAPE: remain close to V3, ideally not worse by more than 3 to 5 percentage points
```

Do not require `>30M` MAPE to become excellent immediately. The dataset has very few luxury-tail rows, so the first realistic V4 goal is to route high-price rows correctly and reduce extreme underprediction.

## 25. Interpretation Guideline

When reading V4 results:

- If `<10M` remains stable but `>=10M` improves, V4 is successful even if overall MAE changes only slightly.
- If `>=10M` recall is low, lower the high-price threshold, increase classifier weights, or strengthen the rule-based override.
- If `>=10M` recall is high but high-price MAPE remains weak, focus on the high-price regressor, quantile objective, and calibration.
- If `>30M` predictions remain too low, increase `>30M` regressor weights or apply stronger luxury lower bounds.
- If `<10M` performance becomes much worse, raise the high-price threshold or make the rule-based override stricter.
- If oracle high-price routing is much better than predicted routing, the bottleneck is the classifier/router.
- If oracle high-price routing is also weak, the bottleneck is the high-price regressor, feature quality, or limited training data.

In summary: V4 should keep the stable `<10M` model from V3 and treat `>=10M` as a separate high-price specialist problem. The main V4 objective is not only better overall metrics, but fewer missed high-price rows and less severe underprediction for expensive rooms, villas, suites, and luxury properties.
