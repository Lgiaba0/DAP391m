# Data Preprocessing and Modeling Guide - Target `price_vnd` - V3

This document extends `data_preprocessing_price_vnd_v2.md` with a two-stage modeling strategy for the long-tailed room price problem.

V1 and V2 show that one global regressor predicts mainstream prices better than luxury-tail prices. The weakest range is `>30M`, where the dataset has very few rows and the model often underpredicts expensive villas, residences, and presidential suites.

## 1. V3 Goal

Main task stays the same:

```text
Predict room price in VND
Target: price_vnd
Main target transform for regressors: log1p(price_vnd)
```

V3 changes the modeling strategy:

```text
1. Train a classifier to predict the price segment first
2. Route each row to a segment-specific price regressor
3. Add special handling for high-price and luxury-tail rows
4. Evaluate both oracle routing and predicted routing
```

The intended price segments are:

```text
<10M
10M-30M
>30M
```

The reason for grouping all rows below `10M` together is that V1/V2 perform acceptably in the common ranges below `10M`, while the main modeling challenge starts in the high-price tail.

## 2. Why V3 Is Needed

V2 added richer luxury features, but the result shows a mixed effect:

- Performance improved slightly in `10M-30M`.
- Performance became worse in the larger mainstream ranges.
- Performance remained weak in `>30M`.

This means a single global regressor is being pulled toward the dense mainstream price distribution. Expensive rows are rare, so the model learns conservative predictions and often underpredicts luxury-tail prices.

V3 should treat this as a routing problem:

```text
Input row
-> classifier predicts price segment
-> selected regressor predicts price within that segment
-> optional calibration corrects luxury-tail underprediction
```

## 3. Keep V1/V2 Leakage Rules

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

Keep `hotel_id` only for `GroupShuffleSplit`, then remove it from the model feature matrix.

## 4. Train/Test Split

Use the same group-based split as V1/V2:

```python
from sklearn.model_selection import GroupShuffleSplit

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42,
)
```

Split by `hotel_id`, not by row. This keeps evaluation realistic because the model must predict prices for unseen hotels.

Use the same train/test indices for:

```text
classifier
segment regressors
oracle routing evaluation
predicted routing evaluation
```

## 5. Segment Target for Classifier

Create a 3-class target from `price_vnd`:

```python
def assign_v3_segment(price_vnd):
    if price_vnd < 10_000_000:
        return "<10M"
    if price_vnd < 30_000_000:
        return "10M-30M"
    return ">30M"
```

Expected class imbalance:

```text
<10M: majority class
10M-30M: small class
>30M: very small luxury-tail class
```

Because `>30M` is rare, do not optimize only for accuracy. A classifier with high accuracy can still be bad if it misses most `>30M` rows.

## 6. Classifier Design

Recommended classifier candidates:

```text
XGBoost Classifier
LightGBM Classifier
Random Forest Classifier
```

Recommended priority:

```text
1. XGBoost Classifier
2. LightGBM Classifier
3. Random Forest Classifier
```

Use the V2 expanded feature set for the classifier because the classifier needs luxury and location signals.

Classifier metrics to report:

```text
accuracy
macro_f1
weighted_f1
per_class_precision
per_class_recall
per_class_f1
confusion_matrix
```

Most important classifier metric:

```text
recall for >30M
```

Missing a true `>30M` row is expensive because the row will be routed to a cheaper regressor and will likely be heavily underpredicted.

## 7. Classifier Imbalance Handling

Use one or more of these strategies:

```text
class_weight
sample_weight
oversampling minority classes in train only
threshold tuning based on predicted probabilities
```

Recommended starting sample weights:

```text
<10M: 1
10M-30M: 3
>30M: 8
```

Adjust weights after reading the confusion matrix. If recall for `>30M` is still poor, increase the `>30M` weight or lower the probability threshold for routing into the luxury path.

## 8. Threshold-Based Routing

Do not rely only on `argmax(class_probability)` for routing. Use threshold-based routing to reduce false negatives in `>30M`.

Recommended starting logic:

```python
if p_over_30m >= 0.25:
    route = ">30M"
elif p_10m_30m >= 0.35:
    route = "10M-30M"
else:
    route = "<10M"
```

Tune thresholds on the training validation fold or cross-validation folds. The objective is to improve recall for `>30M` while keeping false positives manageable.

Report both:

```text
argmax routing metrics
threshold routing metrics
```

## 9. Regressor Design

Train segment-specific regressors on `log1p(price_vnd)`.

Recommended initial design:

```text
regressor_under_10m: train on rows with price_vnd < 10M
regressor_high_price: train on rows with price_vnd >= 10M
```

Do not immediately train a separate `>30M` regressor as the primary path. The `>30M` class has too few rows for a reliable standalone model.

Instead:

```text
Use one high-price regressor for all rows >=10M
Add luxury-tail calibration when classifier route is >30M
```

This gives the high-price model more training data while still allowing special correction for `>30M`.

Optional experiment:

```text
regressor_10m_30m: train on 10M-30M
regressor_over_30m: train on >30M
```

Only keep this option if oracle routing proves that a separate `>30M` model performs better than the shared high-price model.

## 10. Regressor Feature Sets

Recommended feature sets:

```text
regressor_under_10m: baseline_v1_compatible or v2-lite
regressor_high_price: v2_expanded_features
```

Reason:

- The baseline feature set performs better for mainstream prices.
- V2 expanded features show some improvement in `10M-30M`.
- Luxury-tail prediction needs room-name, hotel-name, amenity, location, and description signals.

V3 should compare at least these regressor configurations:

```text
under_10m_baseline_features + high_price_v2_features
under_10m_v2_features + high_price_v2_features
```

## 11. High-Price Regressor Sample Weights

For the high-price regressor, use higher weights for `>30M` rows:

```text
10M-30M: weight = 1
>30M: weight = 3 to 5
```

Start with:

```python
sample_weight = np.where(y_train_vnd >= 30_000_000, 4.0, 1.0)
```

This tells the model that underfitting luxury-tail rows is more costly than small errors in the `10M-30M` range.

## 12. Luxury-Tail Calibration

The current model often underpredicts rows in `>30M`. Add a calibration step after the high-price regressor.

Apply calibration only when the classifier route is `>30M`.

Recommended starting options:

### Option A: Multiplicative Correction

Compute a multiplier from train or validation residuals:

```python
luxury_multiplier = median(actual_price_vnd / predicted_price_vnd)
```

Then:

```python
if route == ">30M":
    pred_price_vnd = pred_price_vnd * luxury_multiplier
```

Cap the multiplier to avoid unstable corrections:

```text
min multiplier: 1.0
max multiplier: 3.0
```

### Option B: Lower-Bound Correction

If the classifier routes a row to `>30M`, prevent implausibly low predictions:

```python
if route == ">30M":
    pred_price_vnd = max(pred_price_vnd, q25_actual_over_30m_train)
```

This reduces cases where a luxury villa is predicted as a mainstream room.

### Option C: Combined Correction

Use both:

```python
if route == ">30M":
    pred_price_vnd = pred_price_vnd * luxury_multiplier
    pred_price_vnd = max(pred_price_vnd, q25_actual_over_30m_train)
```

Start with Option C, then compare against no calibration.

## 13. Avoid P99 Target Capping for High-Price Model

Do not use `price_vnd_capped_p99` for the high-price or luxury-tail model.

Reason:

```text
P99 cap removes the exact signal needed to learn >30M prices.
```

If target capping is used at all, restrict it to the `<10M` model or keep it as a diagnostic experiment only.

For high-price rows, use:

```python
y = np.log1p(price_vnd)
```

## 14. Oracle Routing Evaluation

Before evaluating the full two-stage system, evaluate oracle routing.

Oracle routing means:

```text
Use the true segment to choose the correct regressor.
```

This answers:

```text
If routing were perfect, would segment-specific regressors improve the result?
```

Report oracle metrics:

```text
overall_metrics
segment_metrics for <10M, 10M-30M, >30M
underprediction_metrics
```

If oracle routing does not improve the result, the problem is mainly regressor/data/feature quality.

If oracle routing improves but predicted routing does not, the problem is mainly the classifier.

## 15. Predicted Routing Evaluation

Predicted routing means:

```text
Use classifier output to choose the regressor.
```

This is the deployable system.

Report:

```text
overall_metrics
segment_metrics by true segment
segment_metrics by predicted route
classifier_metrics
routing_confusion_matrix
top_error_rows
```

Compare predicted routing against:

```text
V1 best model
V2 baseline_v1_compatible
V2 expanded_features
V2 p99_capped_target
```

## 16. Underprediction Metrics

Because the key failure mode is underpredicting expensive rows, add special metrics:

```text
underprediction_rate
mean_pred_minus_actual
median_pred_to_actual_ratio
p10_pred_to_actual_ratio
p90_underprediction_error
```

Definitions:

```python
underprediction_rate = mean(pred_price_vnd < actual_price_vnd)
pred_to_actual_ratio = pred_price_vnd / actual_price_vnd
underprediction_error = max(actual_price_vnd - pred_price_vnd, 0)
```

Report these metrics overall and per segment.

For `>30M`, the desired first improvement is:

```text
Increase pred_to_actual_ratio
Reduce severe underprediction
Keep MAPE from getting worse
```

## 17. Prediction-Level CSV

Save prediction-level output:

```text
reports/price_vnd_predictions_v3.csv
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
true_segment_v3
classifier_pred_segment
classifier_p_under_10m
classifier_p_10m_30m
classifier_p_over_30m
routing_strategy
regressor_name
raw_pred_price_vnd
calibrated_pred_price_vnd
absolute_error
absolute_percentage_error
pred_minus_actual
pred_to_actual_ratio
is_underprediction
```

Use this file to inspect:

```python
predictions.sort_values("absolute_error", ascending=False).head(50)
predictions[predictions["true_segment_v3"] == ">30M"]
```

## 18. Evaluation JSON

Save experiment report:

```text
reports/price_vnd_model_evaluation_v3.json
```

Recommended structure:

```json
{
  "created_at_utc": "...",
  "target": "price_vnd",
  "task": "Two-stage price prediction with segment classifier and routed regressors",
  "split": {
    "method": "GroupShuffleSplit",
    "group_column": "hotel_id",
    "test_size": 0.2,
    "random_state": 42
  },
  "segment_definition": {
    "<10M": "price_vnd < 10000000",
    "10M-30M": "10000000 <= price_vnd < 30000000",
    ">30M": "price_vnd >= 30000000"
  },
  "classifier": {
    "model": "XGBoostClassifier",
    "metrics": {},
    "confusion_matrix": {}
  },
  "regressors": {
    "under_10m": {},
    "high_price": {}
  },
  "oracle_routing": {
    "overall_metrics": {},
    "segment_metrics": {},
    "underprediction_metrics": {}
  },
  "predicted_routing": {
    "overall_metrics": {},
    "segment_metrics": {},
    "underprediction_metrics": {}
  },
  "comparison_to_previous_versions": {}
}
```

## 19. Recommended V3 Pipeline

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
-> keep room_name for TF-IDF
-> keep description for TF-IDF
-> drop leakage columns
-> create v3 segment target: <10M, 10M-30M, >30M
-> GroupShuffleSplit by hotel_id
-> train segment classifier with imbalance handling
-> tune routing thresholds for >30M recall
-> train under_10m regressor on rows <10M
-> train high_price regressor on rows >=10M
-> apply high-price sample weights for >30M rows
-> evaluate oracle routing
-> evaluate predicted routing
-> apply luxury-tail calibration for predicted >30M rows
-> report classifier metrics
-> report regression metrics
-> report underprediction metrics
-> save prediction-level CSV
-> save evaluation JSON
```

## 20. Priority Checklist

Implement V3 in this order:

```text
1. Add v3 segment labels: <10M, 10M-30M, >30M
2. Add classifier training and classifier metrics
3. Add threshold-based routing for >30M recall
4. Add under_10m and high_price regressors
5. Add oracle routing evaluation
6. Add predicted routing evaluation
7. Add high-price sample weights
8. Add luxury-tail calibration
9. Add underprediction metrics
10. Save v3 prediction CSV and JSON report
11. Compare v3 against V1/V2 best results
```

## 21. Success Criteria

V3 should be considered successful if it improves the business-relevant failure modes, not only the overall score.

Primary success criteria:

```text
Recall for >30M classifier class improves
Severe underprediction in >30M decreases
Predicted routing does not materially hurt <10M performance
Oracle routing shows that segment-specific regressors are useful
```

Suggested metric targets:

```text
<10M MAPE: stay close to or better than V1/V2 baseline
10M-30M MAPE: improve versus V2 expanded XGBoost if possible
>30M pred_to_actual_ratio: increase materially versus current baseline
>30M severe underprediction count: decrease
```

Do not require `>30M` MAE/MAPE to become excellent immediately. With only a small number of luxury-tail rows, the first realistic goal is to route these rows correctly and reduce extreme underprediction.

## 22. Interpretation Guideline

When reading V3 results:

- If oracle routing improves but predicted routing does not, improve the classifier or routing thresholds.
- If classifier recall for `>30M` is low, increase class/sample weights or lower the `>30M` routing threshold.
- If classifier recall is good but `>30M` predictions remain too low, improve high-price regressor weights or luxury calibration.
- If `<10M` becomes worse, keep baseline features for the `<10M` regressor and reserve V2 expanded features for high-price routing.
- If a standalone `>30M` regressor performs worse than the shared high-price regressor, keep the shared high-price regressor and use calibration.

In summary: V3 should behave like a simple mixture-of-experts system. The classifier decides which pricing regime a row belongs to, the selected regressor predicts within that regime, and luxury-tail calibration reduces the current tendency to underpredict very expensive rooms.
