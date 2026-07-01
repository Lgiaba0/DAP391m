# ML Core

This folder contains the current machine-learning assets for the Vietnam room
price and recommendation pipeline.

## Current Structure

```text
ML_core/
  data/
    raw/                  Source CSV files
    processed/            Versioned modeling datasets
  core/                   ML service contracts, price-band inference, vectors
  docs/                   Experiment notes and next-phase plans
  models/                 Saved model artifacts
  reports/                Evaluation JSON, prediction CSV, audits, charts
  scripts/                Training scripts by task and version
```

## Current Best Artifacts

```text
models/classify/v3/price_classification_v3_model.joblib
models/reg/v5/price_vnd_v5_routed_model.joblib
reports/classify/v3/price_classification_evaluation_v3.json
reports/reg/v5/price_vnd_model_evaluation_v5.json
```

## Run Convention

Run existing training scripts from this folder so their relative paths resolve
against `ML_core`:

```powershell
cd ML_core
python scripts/classify/v3/train_price_classification_v3.py
python scripts/reg/v5/train_price_vnd_models_v5.py
```

Run lightweight ML-core validation:

```powershell
cd ML_core
python scripts/validate_ml_core.py
```

## Next Phase

The next phase should turn this folder from experiment storage into a reusable
ML service core:

```text
user intent -> price band classification -> feature vector -> web-search agent -> recommendation
```

Overall pipeline architecture lives outside `ML_core`:

```text
../docs/phase_next_intent_to_recommendation_pipeline.md
```
