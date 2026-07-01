# Price Classification Plan - V1

This plan converts the previous `price_vnd` regression work into an ordinal
price-band classification task.

The previous regression versions are now grouped under `reg`:

```text
docs/reg/v1..v5
scripts/reg/v1..v5
data/processed/reg/v1..v5
reports/reg/v1..v5
models/reg/v3..v5
```

The new classification version should use:

```text
docs/classify/v1/price_classification_plan_v1.md
scripts/classify/v1/train_price_classification_v1.py
data/processed/classify/v1/vietnam_price_classification_v1.csv
reports/classify/v1/price_classification_predictions_v1.csv
reports/classify/v1/price_classification_evaluation_v1.json
models/classify/v1/price_classification_v1_model.joblib
```

## 1. Objective

Change the task from:

```text
Regression target: price_vnd
Prediction: exact room price in VND
```

to:

```text
Classification target: price_class
Prediction: room price band
```

The classification target should remain ordinal: a prediction one band away is
less severe than a prediction three bands away. Evaluation should therefore
include both standard classification metrics and ordinal error metrics.

## 2. References From Regression Versions

Reuse the following decisions from the regression docs:

```text
docs/reg/v3/data_preprocessing_price_vnd_v3.md
- Keep GroupShuffleSplit by hotel_id.
- Avoid row-level leakage between rooms from the same hotel.
- Treat long-tailed price behavior as a routing/class imbalance issue.

docs/reg/v4/data_preprocessing_price_vnd_v4.md
- Preserve leakage rules.
- Keep hotel_id only for grouped splitting.
- Use high-price/luxury signals from room name, hotel name, amenities, and stars.

docs/reg/v5/data_preprocessing_price_vnd_v5.md
- Keep V5 luxury-tail feature engineering.
- Keep suspicious outlier audit logic as diagnostics.
- Do not remove high prices automatically.
```

The classification version should not inherit the routed regression models. It
should inherit preprocessing, feature engineering, split discipline, leakage
rules, and luxury/high-price feature signals.

## 3. Proposed Price Classes

Use 5 classes for V1.

```text
0 budget          price_vnd < 500,000
1 economy         500,000 <= price_vnd < 1,000,000
2 mid_range       1,000,000 <= price_vnd < 2,000,000
3 upscale         2,000,000 <= price_vnd < 5,000,000
4 premium_luxury  price_vnd >= 5,000,000
```

Class distribution from `data/processed/reg/v5/vietnam_price_vnd_modeling_v5.csv`:

```text
budget:          1,089 rows   16.06%
economy:         1,940 rows   28.62%
mid_range:       1,609 rows   23.74%
upscale:         1,448 rows   21.36%
premium_luxury:    693 rows   10.22%
total:           6,779 rows
```

This is the recommended first version because it avoids the extreme imbalance
of the old regression reporting bands:

```text
<10M:    6,482 rows
10M-30M:   237 rows
>30M:       60 rows
```

A 6-class version can be tested later as a diagnostic:

```text
0 <500k
1 500k-1M
2 1M-2M
3 2M-5M
4 5M-10M
5 >=10M
```

Do not use the 6-class version as the default unless V1 shows strong recall for
the `premium_luxury` class.

## 4. Input Data

Primary options:

```text
Preferred starting point:
data/processed/reg/v5/vietnam_price_vnd_modeling_v5.csv

Raw rebuild path:
data/raw/vietnam_rooms_properties_merged.csv
```

V1 can start from the V5 processed dataset to move faster because it already
contains the strongest feature set:

```text
basic hotel and room features
location features
room_name-derived features
hotel_name keyword flags
amenity binary flags
amenity aggregate scores
luxury_route_score
luxury_tail_score
V5 luxury composite flags
suspicious_price_flag
```

If the team wants a fully reproducible preprocessing script, rebuild from raw
after the first classification baseline is working.

## 5. Leakage Rules

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

Keep `hotel_id` only for grouped train/test splitting and reporting.

Do not train on `true_segment_v5`, `true_is_high_price`, or
`true_is_luxury_tail`, because these are derived from `price_vnd`.

## 6. Target Creation

Create the classification label:

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

Create a readable label too:

```python
PRICE_CLASS_LABELS = {
    0: "budget",
    1: "economy",
    2: "mid_range",
    3: "upscale",
    4: "premium_luxury",
}
```

Save both:

```text
price_class_id
price_class_label
```

## 7. Train/Test Split

Use group-based splitting by `hotel_id`.

Recommended V1 default:

```python
from sklearn.model_selection import GroupShuffleSplit

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42,
)
```

After splitting, report class distribution in train and test.

If the class mix is materially distorted, test multiple random seeds and select
the split with the closest class distribution while still grouping by hotel.
Do not use a plain row-level stratified split, because that can leak hotel-level
patterns across train and test.

## 8. Feature Sets

Start with two feature sets:

```text
baseline_features
- numeric hotel/room metadata
- categorical city, region, property_type, room_type_extracted, bed_type
- room_name TF-IDF
- description TF-IDF

expanded_luxury_features
- baseline_features
- amenity flags
- amenity aggregate scores
- name_has_* flags
- room luxury keyword flags
- luxury_route_score
- luxury_tail_score
- V5 composite luxury flags
```

Recommended V1 default:

```text
expanded_luxury_features
```

Reason: classification must distinguish the `premium_luxury` band, and V5 added
the strongest signals for this purpose.

## 9. Model Candidates

Train simple baselines first:

```text
DummyClassifier(strategy="most_frequent")
LogisticRegression(class_weight="balanced")
RandomForestClassifier(class_weight="balanced")
```

Then train stronger candidates if dependencies are available:

```text
XGBoostClassifier
LightGBMClassifier
```

Recommended first production candidate:

```text
XGBoostClassifier with class/sample weights
```

Class weighting should protect minority classes:

```text
budget:          balanced weight
economy:         balanced weight
mid_range:       balanced weight
upscale:         balanced weight
premium_luxury:  balanced weight, optionally multiplied by 1.25 to 1.75
```

Do not optimize only for accuracy. The majority classes are too large.

## 10. Ordinal-Aware Evaluation

Report standard classification metrics:

```text
accuracy
balanced_accuracy
macro_f1
weighted_f1
per_class_precision
per_class_recall
per_class_f1
confusion_matrix
```

Report ordinal metrics:

```text
mean_absolute_class_error
median_absolute_class_error
adjacent_band_accuracy
severe_misclassification_rate
```

Definitions:

```python
abs_class_error = abs(pred_class_id - actual_class_id)
adjacent_band_accuracy = mean(abs_class_error <= 1)
severe_misclassification_rate = mean(abs_class_error >= 2)
```

Report business-focused metrics:

```text
premium_luxury_recall
premium_luxury_precision
budget_recall
high_to_low_confusion_count
low_to_high_confusion_count
```

Important confusion cases:

```text
actual premium_luxury predicted as budget/economy
actual budget/economy predicted as premium_luxury
```

## 11. Model Selection Rule

Select the best V1 model by this priority:

```text
1. macro_f1
2. balanced_accuracy
3. premium_luxury_recall
4. adjacent_band_accuracy
5. severe_misclassification_rate
6. weighted_f1
7. accuracy
```

Minimum acceptable targets for a useful V1:

```text
macro_f1 materially above DummyClassifier
balanced_accuracy materially above 20%
adjacent_band_accuracy >= 80%
severe_misclassification_rate <= 10%
premium_luxury_recall >= 50%
```

These thresholds are starting points. Update them after the first baseline run.

## 12. Prediction-Level Output

Save:

```text
reports/classify/v1/price_classification_predictions_v1.csv
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
abs_class_error
is_adjacent_or_exact
is_severe_misclassification
suspicious_price_flag
suspicious_reason
experiment_name
```

Use this file to inspect:

```python
predictions[predictions["actual_price_class_label"] == "premium_luxury"]
predictions[predictions["is_severe_misclassification"] == 1]
predictions.sort_values("pred_proba_premium_luxury", ascending=False).head(50)
```

## 13. Evaluation JSON

Save:

```text
reports/classify/v1/price_classification_evaluation_v1.json
```

Recommended structure:

```json
{
  "created_at_utc": "...",
  "task": "price_vnd_band_classification",
  "target": "price_class_id",
  "source_data": "data/processed/reg/v5/vietnam_price_vnd_modeling_v5.csv",
  "output_data": "data/processed/classify/v1/vietnam_price_classification_v1.csv",
  "class_definition": {},
  "class_distribution": {},
  "split": {
    "method": "GroupShuffleSplit",
    "group_column": "hotel_id",
    "test_size": 0.2,
    "random_state": 42
  },
  "dropped_columns": {
    "leakage": [],
    "id": [],
    "derived_from_target": []
  },
  "experiments": {},
  "best_experiment_by_macro_f1": "...",
  "best_experiment_by_ordinal_safety": "...",
  "comparison_to_regression_versions": {
    "note": "Regression metrics are not directly comparable; compare operational usefulness and error inspection only."
  }
}
```

## 14. Implementation Checklist

Implement V1 in this order:

```text
1. Create scripts/classify/v1/train_price_classification_v1.py.
2. Load data/processed/reg/v5/vietnam_price_vnd_modeling_v5.csv.
3. Filter rows with valid price_vnd.
4. Create price_class_id and price_class_label.
5. Save data/processed/classify/v1/vietnam_price_classification_v1.csv.
6. Drop leakage and target-derived columns.
7. Build preprocessing pipeline for numeric, categorical, and text columns.
8. Split with GroupShuffleSplit by hotel_id.
9. Train DummyClassifier baseline.
10. Train LogisticRegression balanced baseline.
11. Train RandomForest balanced baseline.
12. Train XGBoost/LightGBM if available.
13. Evaluate standard classification metrics.
14. Evaluate ordinal metrics.
15. Save prediction-level CSV.
16. Save evaluation JSON.
17. Save best model artifact.
18. Review severe misclassifications manually.
19. Decide whether V2 should test the 6-class split.
```

## 15. V1 Success Criteria

V1 is successful if it provides a stable and interpretable price-band classifier:

```text
macro_f1 is clearly above baseline
balanced_accuracy is clearly above baseline
adjacent_band_accuracy is high
severe cross-band mistakes are rare
premium_luxury recall is acceptable
confusion matrix mostly concentrates on neighboring classes
```

The first version should prioritize reliable class boundaries and inspection
quality over aggressive luxury-tail separation.

## 16. Recommended V2 Direction

After V1:

```text
1. Tune class weights for premium_luxury.
2. Try a 6-class split only if premium_luxury recall is stable.
3. Test ordinal classification strategies.
4. Add probability calibration.
5. Add threshold tuning for premium_luxury recall.
6. Compare raw rebuild from data/raw against V5-processed starting point.
```

Potential ordinal strategies:

```text
one-vs-threshold classifiers
rank/ordinal loss if available
post-processing to penalize far-away class predictions
```

Keep V1 simple first. The main change is from exact price prediction to useful,
stable, and explainable price-band prediction.
