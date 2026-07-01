# Price Classification Plan - V2

This plan improves the V1 ordinal price-band classifier. V1 already converted
the former `price_vnd` regression problem into a 5-class classification task.
V2 should keep the same target definition, split discipline, leakage rules, and
V5 luxury feature foundation, but focus on the weaknesses found in the first
classification run.

V2 should use:

```text
docs/classify/v2/price_classification_plan_v2.md
scripts/classify/v2/train_price_classification_v2.py
data/processed/classify/v2/vietnam_price_classification_v2.csv
reports/classify/v2/price_classification_predictions_v2.csv
reports/classify/v2/price_classification_evaluation_v2.json
reports/classify/v2/price_classification_error_analysis_v2.csv
models/classify/v2/price_classification_v2_model.joblib
```

## 1. Objective

Improve V1 classification quality, especially around the middle price bands:

```text
0 budget          price_vnd < 500,000
1 economy         500,000 <= price_vnd < 1,000,000
2 mid_range       1,000,000 <= price_vnd < 2,000,000
3 upscale         2,000,000 <= price_vnd < 5,000,000
4 premium_luxury  price_vnd >= 5,000,000
```

V1 showed that the classifier is not failing randomly. Most mistakes are
adjacent-band errors caused by boundary ambiguity and feature overlap:

```text
best V1 model by macro F1: LightGBM balanced
accuracy:                 0.646501
balanced_accuracy:        0.650007
macro_f1:                 0.654782
adjacent_band_accuracy:   0.981050
severe_error_rate:        0.018950
premium_luxury_recall:    0.699301
```

The weakest V1 class was:

```text
mid_range precision: 0.549242
mid_range recall:    0.469256
mid_range f1:        0.506108
```

V2 should therefore prioritize:

```text
1. Better mid_range recall and F1.
2. Better economy/mid_range/upscale boundaries.
3. Maintaining low severe misclassification.
4. Maintaining acceptable premium_luxury recall.
5. Keeping grouped split and leakage safety unchanged.
```

## 2. V1 Error Findings To Address

Use the V1 outputs as diagnostics:

```text
reports/classify/v1/price_classification_evaluation_v1.json
reports/classify/v1/price_classification_predictions_v1.csv
models/classify/v1/price_classification_v1_model.joblib
```

Important V1 confusion cases from the best macro-F1 model:

```text
actual mid_range, predicted economy: 92 rows
actual mid_range, predicted upscale: 64 rows
actual budget, predicted economy: 73 rows
actual upscale, predicted mid_range: 57 rows
actual upscale, predicted premium_luxury: 33 rows
actual premium_luxury, predicted upscale: 39 rows
```

Key interpretation:

```text
mid_range has weak class identity.
budget/economy mistakes are often close to the 500k boundary.
economy/mid_range/upscale mistakes are mostly adjacent and feature-overlap driven.
premium_luxury mistakes mostly happen near the 5M lower boundary or when luxury signals are weak.
upscale predicted as premium_luxury usually has strong luxury signals and prices near 5M.
```

The V2 plan should not simply add more model capacity. It should test modeling
strategies that match the ordinal structure and add market-context features
that help separate ambiguous neighboring bands.

## 3. Keep V1 Target Definition

Do not change the default class definition in V2. Keep the 5-class setup so V2
is directly comparable to V1.

```python
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
```

Readable labels:

```python
PRICE_CLASS_LABELS = {
    0: "budget",
    1: "economy",
    2: "mid_range",
    3: "upscale",
    4: "premium_luxury",
}
```

Do not make the 6-class split the V2 default. It can be added as a diagnostic
only after the 5-class V2 experiments are complete.

## 4. Input Data

Preferred V2 starting point:

```text
data/processed/classify/v1/vietnam_price_classification_v1.csv
```

Fallback starting point:

```text
data/processed/reg/v5/vietnam_price_vnd_modeling_v5.csv
```

Reason:

```text
V1 classification data already includes price_class_id and price_class_label.
V5 processed data contains the strongest luxury/high-price feature set.
The V2 script should be able to rebuild V2 data from either file.
```

Save the V2 modeling data to:

```text
data/processed/classify/v2/vietnam_price_classification_v2.csv
```

## 5. Leakage Rules

Preserve the V1 leakage rules exactly.

Drop all direct price leakage columns from feature matrices:

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

Drop direct identity or memorization columns:

```python
id_drop_cols = [
    "hotel_id",
    "block_id",
    "source_url",
    "hotel_name",
    "street_address",
]
```

Drop target-derived columns from model training:

```python
target_derived_cols = [
    "true_segment_v5",
    "true_is_high_price",
    "true_is_luxury_tail",
    "price_class_id",
    "price_class_label",
    "suspicious_price_flag",
    "suspicious_reason",
]
```

Keep `hotel_id` only for grouped splitting and reporting. Keep
`suspicious_price_flag` and `suspicious_reason` only for diagnostics and
prediction-level output, not for model training.

## 6. Train/Test Split

Use the same grouped split discipline as V1:

```python
from sklearn.model_selection import GroupShuffleSplit

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42,
)
```

Report class distribution in train and test.

Do not use row-level stratified splitting. Rooms from the same hotel must not
be split across train and test.

Optional V2 diagnostic:

```text
Run 5 to 10 GroupShuffleSplit seeds.
Select the seed with class proportions closest to the full dataset.
Keep the selected seed fixed in the evaluation JSON.
```

This is allowed only if the V1 seed produces materially distorted class
proportions. It must still group by `hotel_id`.

## 7. Feature Sets

V2 should test three feature sets.

### 7.1 expanded_luxury_features

This is the V1 default:

```text
numeric hotel/room metadata
categorical city, region, property_type, room_type_extracted, bed_type
room_name TF-IDF
description TF-IDF
amenity flags
amenity aggregate scores
name_has_* flags
room luxury keyword flags
luxury_route_score
luxury_tail_score
V5 composite luxury flags
```

Use this as the baseline feature set for comparability.

### 7.2 market_context_features

Add train-fold aggregate features that provide market context without row-level
or test leakage.

Candidate features:

```text
city_median_class_train
city_mean_class_train
source_city_median_class_train
region_median_class_train
property_type_median_class_train
star_rating_clean_median_class_train
city_property_type_median_class_train
source_city_property_type_median_class_train
city_star_median_class_train
property_type_star_median_class_train
```

Implementation rule:

```text
Fit these encoders on the training fold only.
Map learned aggregates into train and test.
Use global train median class as fallback for unseen groups.
Do not compute aggregates on full data before splitting.
```

These features are target-derived, but they are allowed only when computed
inside the training fold and applied to validation/test like a supervised
encoder. The evaluation JSON must record this explicitly.

### 7.3 mid_market_features

Add engineered features to separate `mid_range` from adjacent classes.

Candidate features:

```text
is_standard_hotel_room
is_deluxe_without_luxury_tail
is_3_to_4_star_full_service
is_midscale_property_type
has_midscale_amenity_profile
midscale_amenity_score
luxury_signal_without_premium_capacity
budget_like_low_signal
economy_mid_boundary_signal
mid_upscale_boundary_signal
```

Suggested definitions:

```text
is_standard_hotel_room:
  property_type is Hotel and room_quality_tier in standard/superior/deluxe range

is_deluxe_without_luxury_tail:
  room_name has deluxe/superior/view but luxury_tail_score <= 1

is_3_to_4_star_full_service:
  star_rating_clean between 3 and 4 and amenity_count/luxury_amenity_score is moderate

is_midscale_property_type:
  property_type in Hotel, Apartment, Resort, Homestay with no strong luxury tail

has_midscale_amenity_profile:
  luxury_amenity_score between 3 and 5 and luxury_tail_score <= 2

midscale_amenity_score:
  moderate amenities minus strong luxury-tail indicators

luxury_signal_without_premium_capacity:
  luxury_route_score >= 2 and max_persons <= 2 and bedroom_count <= 1
```

These features should be simple numeric or binary features. Avoid complicated
rules that duplicate the target.

## 8. Model Candidates

V2 should compare direct multi-class classifiers against ordinal-aware
strategies.

### 8.1 V1-compatible direct classifiers

Train:

```text
DummyClassifier(strategy="most_frequent")
LightGBMClassifier(class_weight="balanced")
XGBoostClassifier with balanced sample weights
```

Also test class-weight variants focused on `mid_range`:

```text
weight_config_baseline:
  balanced weights only

weight_config_mid_1_5:
  balanced weights, mid_range multiplied by 1.5

weight_config_mid_2_0:
  balanced weights, mid_range multiplied by 2.0

weight_config_mid_1_5_premium_1_25:
  balanced weights, mid_range multiplied by 1.5, premium_luxury multiplied by 1.25
```

Do not choose a model solely because `mid_range` recall improves. It must not
break severe misclassification safety or premium_luxury recall.

### 8.2 Ordinal threshold classifiers

Train four binary threshold models:

```text
threshold_500k:  predict price_vnd >= 500,000
threshold_1m:    predict price_vnd >= 1,000,000
threshold_2m:    predict price_vnd >= 2,000,000
threshold_5m:    predict price_vnd >= 5,000,000
```

Convert threshold probabilities to a class prediction:

```python
pred_class_id = (
    int(p_ge_500k >= t_500k)
    + int(p_ge_1m >= t_1m)
    + int(p_ge_2m >= t_2m)
    + int(p_ge_5m >= t_5m)
)
```

Start with thresholds:

```text
t_500k = 0.50
t_1m   = 0.50
t_2m   = 0.50
t_5m   = 0.50
```

Then tune thresholds on the training fold using an inner validation split or
cross-validation grouped by `hotel_id`.

Priority for tuning:

```text
1. improve mid_range recall and F1
2. keep adjacent_band_accuracy high
3. keep severe_misclassification_rate low
4. keep premium_luxury_recall acceptable
```

If threshold outputs are inconsistent, enforce monotonicity:

```text
p_ge_500k >= p_ge_1m >= p_ge_2m >= p_ge_5m
```

Simple fallback:

```text
Use isotonic/monotone post-processing on the four probabilities.
Or clip cumulative probabilities so higher thresholds cannot exceed lower thresholds.
```

### 8.3 Boundary specialist classifiers

Optional if time allows.

Train binary specialists for ambiguous neighboring pairs:

```text
budget vs economy
economy vs mid_range
mid_range vs upscale
upscale vs premium_luxury
```

Use them only when the primary model is uncertain:

```text
top2_probability_gap <= 0.20
or predicted/second-best classes are adjacent
```

Specialist models should not override severe cross-band cases. They should
only choose between adjacent candidate classes.

## 9. Probability Calibration

V2 should test calibrated probabilities for the best direct classifier and the
ordinal threshold classifiers.

Candidate methods:

```text
CalibratedClassifierCV(method="sigmoid")
CalibratedClassifierCV(method="isotonic")
```

Use grouped train/validation splitting where possible. If using
CalibratedClassifierCV with standard CV is not group-aware enough, prefer:

```text
1. Fit model on grouped train subset.
2. Predict probabilities on grouped calibration subset.
3. Fit calibration mapping.
4. Evaluate on held-out grouped test.
```

Report both raw and calibrated metrics.

## 10. Error Analysis Output

Save a dedicated error analysis file:

```text
reports/classify/v2/price_classification_error_analysis_v2.csv
```

Recommended rows:

```text
one row per actual/predicted class pair
```

Recommended columns:

```text
actual_price_class_label
pred_price_class_label
rows
error_rate_within_actual_class
median_price_vnd
mean_price_vnd
near_lower_boundary_rate
near_upper_boundary_rate
near_any_boundary_rate
median_pred_top1_probability
median_pred_top2_gap
median_star_rating_clean
median_luxury_route_score
median_luxury_tail_score
median_luxury_amenity_score
median_max_persons
median_bedroom_count
top_source_city_values
top_property_type_values
experiment_name
```

Boundary definitions:

```text
budget upper boundary:          500,000
economy lower/upper boundaries: 500,000 / 1,000,000
mid_range boundaries:           1,000,000 / 2,000,000
upscale boundaries:             2,000,000 / 5,000,000
premium_luxury lower boundary:  5,000,000
```

Mark rows as near-boundary when the price is within 15% of the band width from
the nearest class boundary. For `premium_luxury`, use within 1,000,000 VND of
the 5M lower boundary.

## 11. Ordinal-Aware Evaluation

Keep all V1 metrics:

```text
accuracy
balanced_accuracy
macro_f1
weighted_f1
per_class_precision
per_class_recall
per_class_f1
confusion_matrix
mean_absolute_class_error
median_absolute_class_error
adjacent_band_accuracy
severe_misclassification_rate
premium_luxury_recall
premium_luxury_precision
budget_recall
high_to_low_confusion_count
low_to_high_confusion_count
```

Add V2 boundary metrics:

```text
mid_range_recall
mid_range_precision
mid_range_f1
economy_mid_boundary_error_count
mid_upscale_boundary_error_count
budget_economy_boundary_error_count
upscale_premium_boundary_error_count
adjacent_error_count
non_adjacent_error_count
near_boundary_error_rate
far_from_boundary_error_rate
```

Definitions:

```python
abs_class_error = abs(pred_class_id - actual_class_id)
adjacent_error = abs_class_error == 1
non_adjacent_error = abs_class_error >= 2
near_boundary_error_rate = mean(error among rows marked near a class boundary)
far_from_boundary_error_rate = mean(error among rows not near a class boundary)
```

Report per-class mean absolute class error too:

```text
mae_class_error_budget
mae_class_error_economy
mae_class_error_mid_range
mae_class_error_upscale
mae_class_error_premium_luxury
```

## 12. Model Selection Rule

V2 should not select only by macro F1. Use a rule that reflects the V1 weakness.

Primary V2 selection priority:

```text
1. macro_f1
2. mid_range_f1
3. mid_range_recall
4. balanced_accuracy
5. adjacent_band_accuracy
6. severe_misclassification_rate
7. premium_luxury_recall
8. weighted_f1
9. accuracy
```

V2 should also name:

```text
best_experiment_by_macro_f1
best_experiment_by_mid_range_f1
best_experiment_by_ordinal_safety
best_experiment_by_business_safety
```

Business safety tie-breaker:

```text
1. severe_misclassification_rate lower is better
2. high_to_low_confusion_count lower is better
3. low_to_high_confusion_count lower is better
4. premium_luxury_recall higher is better
```

## 13. Minimum Useful V2 Targets

V2 is useful only if it beats V1 on the main weakness without breaking ordinal
safety.

V1 baseline to beat:

```text
macro_f1:                    0.654782
mid_range_f1:                0.506108
mid_range_recall:            0.469256
balanced_accuracy:           0.650007
adjacent_band_accuracy:      0.981050
severe_misclassification:    0.018950
premium_luxury_recall:       0.699301
```

Minimum V2 target:

```text
mid_range_f1 >= 0.55
mid_range_recall >= 0.55
macro_f1 >= V1 macro_f1 - 0.01
adjacent_band_accuracy >= 0.975
severe_misclassification_rate <= 0.025
premium_luxury_recall >= 0.65
```

Strong V2 target:

```text
macro_f1 > 0.67
mid_range_f1 >= 0.58
mid_range_recall >= 0.58
adjacent_band_accuracy >= V1
severe_misclassification_rate <= V1
premium_luxury_recall >= 0.70
```

## 14. Prediction-Level Output

Save:

```text
reports/classify/v2/price_classification_predictions_v2.csv
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
actual_price_class_id
actual_price_class_label
pred_price_class_id
pred_price_class_label
pred_proba_budget
pred_proba_economy
pred_proba_mid_range
pred_proba_upscale
pred_proba_premium_luxury
threshold_p_ge_500k
threshold_p_ge_1m
threshold_p_ge_2m
threshold_p_ge_5m
abs_class_error
is_adjacent_or_exact
is_severe_misclassification
nearest_boundary_vnd
distance_to_nearest_boundary_vnd
is_near_boundary
top1_probability
top2_probability_gap
model_family
experiment_name
suspicious_price_flag
suspicious_reason
```

For direct multi-class models, threshold probability columns can be blank. For
ordinal threshold models, class probability columns should be derived from the
threshold probabilities when possible.

## 15. Evaluation JSON

Save:

```text
reports/classify/v2/price_classification_evaluation_v2.json
```

Recommended structure:

```json
{
  "created_at_utc": "...",
  "task": "price_vnd_band_classification",
  "target": "price_class_id",
  "source_data": "data/processed/classify/v1/vietnam_price_classification_v1.csv",
  "output_data": "data/processed/classify/v2/vietnam_price_classification_v2.csv",
  "class_definition": {},
  "v1_baseline": {},
  "class_distribution": {},
  "split": {
    "method": "GroupShuffleSplit",
    "group_column": "hotel_id",
    "test_size": 0.2,
    "random_state": 42
  },
  "feature_sets": {},
  "leakage_controls": {
    "dropped_columns": {},
    "train_fold_aggregate_features": {}
  },
  "experiments": {},
  "best_experiment_by_macro_f1": "...",
  "best_experiment_by_mid_range_f1": "...",
  "best_experiment_by_ordinal_safety": "...",
  "best_experiment_by_business_safety": "...",
  "selected_v2_model": "...",
  "comparison_to_v1": {},
  "recommended_v3_direction": {}
}
```

## 16. Implementation Checklist

Implement V2 in this order:

```text
1. Create scripts/classify/v2/train_price_classification_v2.py.
2. Load data/processed/classify/v1/vietnam_price_classification_v1.csv.
3. Rebuild price_class_id and price_class_label if missing.
4. Save data/processed/classify/v2/vietnam_price_classification_v2.csv.
5. Split with GroupShuffleSplit by hotel_id.
6. Build the V1-compatible expanded_luxury_features pipeline.
7. Add mid_market_features.
8. Add train-fold market_context_features through a leakage-safe encoder.
9. Train DummyClassifier baseline.
10. Train direct LightGBM/XGBoost V1-compatible models.
11. Train direct models with mid_range weight variants.
12. Train ordinal threshold classifiers.
13. Optionally train adjacent boundary specialist classifiers.
14. Optionally calibrate the best direct and ordinal models.
15. Evaluate all models with V1 metrics.
16. Add V2 boundary and mid_range-focused metrics.
17. Save prediction-level CSV for the selected V2 model.
18. Save error analysis CSV for the selected V2 model.
19. Save evaluation JSON with all experiment results.
20. Save selected V2 model artifact.
21. Compare V2 against V1 baselines.
22. Decide whether V3 should test 6-class split, richer encoders, or raw rebuild.
```

## 17. V2 Success Criteria

V2 is successful if it improves the weak middle class while preserving the main
strengths of V1:

```text
mid_range recall improves materially
mid_range F1 improves materially
macro F1 is stable or improves
severe cross-band mistakes stay rare
adjacent-band accuracy stays high
premium_luxury recall stays acceptable
confusion matrix remains concentrated on neighboring classes
```

The preferred V2 model should be more ordinal-aware than V1, not merely more
complex.

## 18. Recommended V3 Direction

After V2:

```text
1. If ordinal threshold classification wins, tune threshold calibration more carefully.
2. If market_context_features help, make the target encoder reusable and cross-validated.
3. If mid_range remains weak, test a two-stage hierarchy:
   low/economy vs middle vs high/luxury, then specialist classifiers inside each group.
4. Test 6-class split only if premium_luxury remains stable.
5. Compare V5-processed starting point with a full raw rebuild from data/raw.
6. Consider conformal/uncertainty output for rows near price boundaries.
```

Do not move to V3 until V2 has a clear comparison against the V1 best model and
the V1 ordinal-safety model.
