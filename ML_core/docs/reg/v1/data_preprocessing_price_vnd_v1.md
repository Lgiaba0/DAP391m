# Data Preprocessing Guide for ML Training - Target `price_vnd`

This document summarizes the recommended preprocessing steps for `data/raw/vietnam_rooms_properties_merged.csv` before ML training, using room price in VND as the target.

## 1. Problem Definition

Main task:

```text
Predict room price in VND
Target: price_vnd
```

Because `price_vnd` has a very wide range, from cheap dorm beds to expensive villas and resorts, training directly on the raw price is not recommended. Use a log transform:

```python
y = np.log1p(df["price_vnd"])
```

After prediction, convert the prediction back to VND:

```python
price_vnd_pred = np.expm1(y_pred)
```

## 2. Data Grain

The dataset is room-level. Each row represents one room or one room-rate option for a hotel.

A hotel can have multiple room rows:

```text
hotel_id + room_name + room_index + block_id
```

Therefore, train/test splitting should not be done randomly by row. Split by `hotel_id` to avoid leakage between train and test.

## 3. Drop Columns That Should Not Be Used Directly

Drop ID, URL, identifier-like text, or fields that can make the model memorize hotels:

```python
drop_cols = [
    "hotel_id",
    "block_id",
    "source_url",
    "hotel_name",
    "street_address",
]
```

Reasons:

- `hotel_id` and `block_id` are IDs and have no meaningful numeric order.
- `hotel_name` and `street_address` have high cardinality and can make the model memorize specific hotels.
- `source_url` has no direct predictive value.

Note: keep `hotel_id` temporarily for group-based train/test splitting. Drop it from the feature matrix after the split.

## 4. Drop Leakage Columns for Target `price_vnd`

Since the target is `price_vnd`, remove columns that directly or indirectly contain price information:

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

Reasons:

- `price_usd` is almost a converted version of `price_vnd`, so it must be dropped.
- `cheapest_room_vnd`, `cheapest_room_usd`, and `most_expensive_usd` are price summaries from the same hotel.
- `price_category` is derived from price; keeping it would almost directly reveal the answer.
- `value_index` and `price_to_star` include price-related information in their formulas.

## 5. Drop Constant or Low-Information Columns

These columns currently have only one useful value or do not help the model:

```python
constant_cols = [
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
```

Currently, `checkin` and `checkout` contain only one date each, so they do not provide predictive value. If future crawls include many dates, they can be kept and used to create seasonality features.

## 6. Drop 100% Missing Columns

The following columns are completely empty:

```python
all_missing_cols = [
    "score_staff",
    "score_facilities",
    "score_cleanliness",
    "score_comfort",
    "score_location",
    "score_wifi",
    "score_value",
    "child_free_age",
]
```

Drop them directly.

## 7. Handle Missing Values and Special Values

### `district = -1`

`district = -1` should be treated as missing/unknown, not as a real district:

```python
df["district"] = df["district"].replace("-1", "Unknown")
```

### `star_rating = 0`

`star_rating = 0` should be treated as unknown/unrated, not as a true 0-star hotel:

```python
df["star_unknown"] = (df["star_rating"] == 0).astype(int)
df["star_rating_clean"] = df["star_rating"].replace(0, np.nan)
```

Then impute `star_rating_clean`, for example with the median.

### Numeric Missing Values

Some numeric columns have missing values:

```python
numeric_missing_cols = [
    "review_score",
    "popularity_proxy",
    "review_to_star",
]
```

Use:

```python
SimpleImputer(strategy="median")
```

For categorical missing values, use:

```python
SimpleImputer(strategy="constant", fill_value="Unknown")
```

## 8. Handle Outliers in `price_vnd`

`price_vnd` has large outliers. The recommended approach is to log-transform the target:

```python
y = np.log1p(df["price_vnd"])
```

If outliers are too influential, you can cap the target at the 99th percentile:

```python
upper = df["price_vnd"].quantile(0.99)
df["price_vnd_capped"] = df["price_vnd"].clip(upper=upper)
```

Be careful when removing or capping outliers, because expensive villas, presidential suites, and 5-star resorts are valid cases in a hotel pricing problem. For the baseline version, prefer `log1p(price_vnd)`.

## 9. Numeric Features to Keep

Recommended numeric features:

```python
numeric_features = [
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
]
```

Notes:

- `review_to_star` has many missing values because `star_rating = 0`, but it can still be used if imputed.
- `dest_ufi` is a destination code, so it can also be treated as categorical rather than numeric.
- If using linear models, scale numeric features with `StandardScaler`.
- If using tree-based models, scaling is usually less important.

## 10. Binary Features to Keep

Keep binary `0/1` columns as-is:

```python
binary_features = [
    "breakfast_included",
    "has_breakfast_option",
]
```

Also keep all columns that start with:

```text
amenity_
```

Examples:

```text
amenity_free_wifi
amenity_airport_shuttle
amenity_spa
amenity_restaurant
amenity_bar
amenity_free_parking
amenity_room_service
amenity_family_rooms
amenity_balcony
amenity_garden
```

These features are useful because amenities directly influence room price.

## 11. Categorical Features and Encoding

Use one-hot encoding for categorical features:

```python
categorical_features = [
    "source_city",
    "city",
    "district",
    "region",
    "property_type",
    "property_type_label",
    "local_currency",
    "score_tier",
]
```

Use:

```python
OneHotEncoder(handle_unknown="ignore")
```

Notes:

- `property_type` and `property_type_label` are almost duplicates. You can keep only one, preferably `property_type`.
- `city` has more distinct values than `source_city`, but it is still usable.
- `district` has many `Unknown` values, so do not expect too much signal from it.

## 12. Rule-Based Feature Extraction from `room_name`

`room_name` is very important because it contains signals about room quality and room type.

Examples:

```text
Deluxe King Room
Superior Double Room
Three-Bedroom Pool Villa
Presidential Suite
Bed in Mixed Dormitory Room
```

Create rule-based features such as:

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

Suggested `room_quality_tier` mapping:

```python
quality_order = {
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
```

Example rules:

```text
"deluxe" -> room_quality_tier = 4
"superior" -> room_quality_tier = 3
"presidential" -> room_quality_tier = 10
"villa" -> room_type_extracted = villa
"suite" -> room_type_extracted = suite
"dormitory" / "bunk bed" -> is_shared_room = 1
"sea view" / "ocean view" -> has_sea_view = 1
"balcony" -> has_balcony_room = 1
"private pool" / "pool villa" -> has_private_pool = 1
```

Encoding:

- `room_quality_tier`: ordinal numeric.
- `room_type_extracted`: one-hot.
- `bed_type`: one-hot.
- `bedroom_count`: numeric.
- `has_*` and `is_*`: binary.

## 13. Use TF-IDF for `room_name`

After creating rule-based features, also use TF-IDF on `room_name`.

Reasons:

- Rule-based extraction only captures keywords we manually define.
- TF-IDF lets the model learn additional price-related words or phrases such as `beachfront`, `royal`, `ocean`, `family`, `pool villa`, `dormitory`, and `penthouse`.

Recommended configuration:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

room_name_tfidf = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=5,
    max_features=100
)
```

Meaning:

- `ngram_range=(1, 2)`: use unigrams and bigrams, for example `deluxe`, `sea view`, and `pool villa`.
- `min_df=5`: keep only terms that appear in at least 5 rows.
- `max_features=100`: limit the number of text features to avoid too many columns.

Do not one-hot encode the full `room_name`, because it has nearly 2,000 unique values.

## 14. Other Long Text Columns

For the baseline version, drop long text columns:

```python
text_drop_cols = [
    "description",
    "room_names_sample",
]
```

Possible future improvements:

- `description`: apply a separate TF-IDF vectorizer.
- `street_address`: extract location keywords.
- `room_names_sample`: use it to estimate how diverse a hotel's room offerings are.

For the first version, only apply TF-IDF to `room_name`.

## 15. Correct Train/Test Split

Do not randomly split rows:

```python
train_test_split(df, test_size=0.2)
```

The same `hotel_id` can have many rooms. If one hotel appears in both train and test, the model can learn hotel-specific patterns from train, making test performance unrealistically high.

Use `GroupShuffleSplit` by `hotel_id`:

```python
from sklearn.model_selection import GroupShuffleSplit

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(df, y, groups=df["hotel_id"])
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]
y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]
```

Drop `hotel_id` from the feature matrix after splitting.

## 16. Overall Pipeline

Recommended preprocessing flow:

```text
Load vietnam_rooms_properties_merged.csv
-> define target price_vnd
-> create y = log1p(price_vnd)
-> handle district = -1
-> handle star_rating = 0
-> create rule-based features from room_name
-> keep room_name for TF-IDF
-> drop ID/leakage/null/constant columns
-> numeric imputation
-> categorical one-hot encoding
-> binary passthrough
-> TF-IDF(room_name)
-> GroupShuffleSplit by hotel_id
-> train model
-> predict log price
-> use expm1 to convert predictions back to VND
-> evaluate MAE/RMSE/MAPE on true VND price
```

## 17. Evaluation Metrics

For price prediction, report multiple metrics:

```text
MAE
RMSE
MAPE
RMSLE
```

Meaning:

- `MAE`: average absolute error in VND.
- `RMSE`: penalizes large errors more strongly.
- `MAPE`: percentage error, easy to explain.
- `RMSLE`: suitable for log-transformed targets.

Evaluate on the real price scale:

```python
y_pred_vnd = np.expm1(y_pred)
y_test_vnd = np.expm1(y_test)
```

## 18. Models to Try

```text
LightGBM
XGBoost
Random Forest
```
## 19. Important Note

With target `price_vnd`, `price_usd` must be dropped:

```python
drop_cols += ["price_usd"]
```

Otherwise, the model will almost directly learn the USD-to-VND conversion. Test results may look very good, but they will not represent a meaningful ML model.

In summary: use `price_vnd` as a log-transformed target, remove price leakage, handle missing values and outliers, one-hot encode categorical features, keep amenities as binary features, extract rule-based features from `room_name`, and add TF-IDF to capture additional useful room-name keywords and phrases.
