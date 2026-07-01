# Phase Next: Intent To Recommendation Pipeline

## Goal

Build a reusable recommendation pipeline where user intent is transformed into a
structured search-and-ranking request:

```text
User intent
  -> intent parser
  -> ML core price-band classifier
  -> unified feature vector
  -> web-search agent
  -> recommendation ranker
  -> user-facing recommendation
```

The ML core should own local model inference, feature engineering, vector
construction, confidence metadata, and ranking signals. The web-search agent
should own external discovery and evidence collection.

## Target Flow

1. Intent parser receives the raw user request.

   Example:

   ```text
   "Tim phong khach san Da Nang cho 2 nguoi, gan bien, khoang 1-2 trieu, co ho boi"
   ```

2. Intent parser outputs structured intent.

   ```json
   {
     "destination": "Da Nang",
     "guest_count": 2,
     "budget_text": "1-2 trieu",
     "amenities": ["pool"],
     "location_preferences": ["near beach"],
     "room_preferences": [],
     "date_range": null,
     "raw_query": "..."
   }
   ```

3. ML core classifies the target price band.

   Preferred model:

   ```text
   models/classify/v3/price_classification_v3_model.joblib
   ```

   Output:

   ```json
   {
     "price_class_id": 2,
     "price_class_label": "mid_range",
     "price_class_proba": {
       "budget": 0.03,
       "economy": 0.18,
       "mid_range": 0.61,
       "upscale": 0.16,
       "premium_luxury": 0.02
     },
     "confidence": 0.61
   }
   ```

4. Feature builder creates one unified feature vector.

   The vector should combine:

   ```text
   intent features
   location features
   budget and price-band features
   room capacity features
   amenity features
   property-type preferences
   luxury and mid-market signals
   uncertainty/confidence fields
   search-control fields
   ```

5. Web-search agent receives the vector and searches external sources.

   The vector should guide search terms, filters, and ranking:

   ```json
   {
     "destination": "Da Nang",
     "price_class_label": "mid_range",
     "price_min_vnd": 1000000,
     "price_max_vnd": 2000000,
     "guest_count": 2,
     "amenity_pool": 1,
     "near_beach": 1,
     "property_type_hotel": 1,
     "confidence_price_class": 0.61,
     "expand_budget_if_low_confidence": true
   }
   ```

6. Recommendation ranker combines ML signals and web evidence.

   Ranking should consider:

   ```text
   price-band fit
   hard intent match
   amenity match
   location match
   capacity fit
   review/source evidence
   model confidence
   explanation quality
   ```

7. Final response returns recommended rooms/hotels with reasons.

   Each recommendation should include:

   ```text
   name
   estimated price/range
   source URL
   why it matches
   tradeoffs
   confidence
   ```

## Proposed Module Layout

Use this split so `ML_core` stays ML-only and the application orchestration
lives outside it:

```text
DAP391m/
  ML_core/
    core/
      __init__.py
      schemas.py
      config.py
      exceptions.py
      price_band_service.py
      feature_builder.py
    data/
    docs/
    models/
    reports/
    scripts/
  intent/
    __init__.py
    intent_parser.py
  agents/
    __init__.py
    web_search_agent.py
    search_query_builder.py
    source_normalizer.py
  recommendation/
    __init__.py
    schemas.py
    ranker.py
  pipelines/
    __init__.py
    recommend_pipeline.py
  tests/
    test_intent_parser.py
    test_feature_builder.py
    test_recommend_pipeline.py
```

## Data Contracts

### IntentRequest

```json
{
  "raw_query": "string",
  "destination": "string|null",
  "check_in": "date|null",
  "check_out": "date|null",
  "guest_count": "integer|null",
  "room_count": "integer|null",
  "budget_min_vnd": "number|null",
  "budget_max_vnd": "number|null",
  "amenities": ["string"],
  "location_preferences": ["string"],
  "property_types": ["string"],
  "room_preferences": ["string"]
}
```

### PriceBandPrediction

```json
{
  "price_class_id": "integer",
  "price_class_label": "string",
  "probabilities": {
    "budget": "number",
    "economy": "number",
    "mid_range": "number",
    "upscale": "number",
    "premium_luxury": "number"
  },
  "confidence": "number",
  "price_min_vnd": "number|null",
  "price_max_vnd": "number|null"
}
```

### SearchFeatureVector

```json
{
  "destination": "string|null",
  "guest_count": "number",
  "price_min_vnd": "number|null",
  "price_max_vnd": "number|null",
  "price_class_id": "number",
  "price_class_confidence": "number",
  "amenity_pool": "number",
  "amenity_beach": "number",
  "amenity_breakfast": "number",
  "near_beach": "number",
  "near_center": "number",
  "property_type_hotel": "number",
  "property_type_apartment": "number",
  "property_type_resort": "number",
  "expand_budget": "boolean",
  "raw_query": "string"
}
```

## Implementation Plan

### Phase 1: Stabilize ML Core Interface

Create `ML_core/core/schemas.py` with typed ML-facing data contracts. Keep it
small and stable so the agent layer does not import training scripts directly.

Create `ML_core/core/price_band_service.py` that loads:

```text
models/classify/v3/price_classification_v3_model.joblib
```

The service should expose:

```python
predict_price_band(intent: IntentRequest) -> PriceBandPrediction
```

If the intent already contains an explicit budget range, map that budget to the
known class boundaries first and use the classifier as supporting confidence.

### Phase 2: Intent Parsing

Create `intent/intent_parser.py`.

Start with deterministic parsing for:

```text
destination
guest_count
budget_min_vnd
budget_max_vnd
amenities
location preferences
property type
```

Use simple dictionaries and regex first. LLM parsing can be added later behind
the same interface.

### Phase 3: Feature Vector Builder

Create `ML_core/core/feature_builder.py`.

Inputs:

```text
IntentRequest
PriceBandPrediction
```

Output:

```text
SearchFeatureVector
```

Rules:

```text
If price confidence < 0.55, set expand_budget = true.
If user gave explicit min/max budget, preserve it as hard search guidance.
If only price class exists, use class boundary as soft search guidance.
Encode amenities and location signals as binary features.
Keep raw_query for web-search query generation.
```

### Phase 4: Web-Search Agent Adapter

Create `agents/web_search_agent.py`.

The agent should accept only the vector contract, not raw model objects.

Responsibilities:

```text
generate search queries
call web/search provider
deduplicate candidates
extract price, location, amenities, rating, URL
return normalized candidate list
```

### Phase 5: Recommendation Ranking

Create `recommendation/ranker.py`.

Start with transparent scoring:

```text
score =
  0.35 * price_fit
  0.25 * amenity_fit
  0.20 * location_fit
  0.10 * capacity_fit
  0.10 * source_quality
```

Penalize:

```text
missing price
outside hard budget
unclear source
low intent match
```

### Phase 6: End-To-End Pipeline

Create `pipelines/recommend_pipeline.py`.

Expose:

```python
recommend(raw_user_query: str) -> RecommendationResponse
```

Minimum response:

```json
{
  "parsed_intent": {},
  "price_band": {},
  "feature_vector": {},
  "recommendations": [],
  "debug": {
    "search_queries": [],
    "ranking_version": "v1"
  }
}
```

## Acceptance Criteria

The next phase is complete when:

```text
1. One function can run the full pipeline from raw user query to ranked output.
2. Price-band classification is isolated behind a service interface.
3. Feature vector output is deterministic and serializable to JSON.
4. Web-search agent receives the vector instead of loosely structured text.
5. Ranking produces explanations tied to intent and evidence.
6. Tests cover intent parsing, budget mapping, vector creation, and ranking.
```

## Recommended First Build Order

```text
1. Add schemas.
2. Add deterministic intent parser.
3. Add budget-to-price-class mapper.
4. Wrap the V3 classifier in price_band_service.
5. Build SearchFeatureVector.
6. Stub web_search_agent with mock candidates.
7. Implement transparent ranker.
8. Wire recommend_pipeline end to end.
9. Add tests.
10. Replace mock web search with real provider/tooling.
```
