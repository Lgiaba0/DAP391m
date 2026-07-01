# DAP391m Recommendation Project

This repository is split into a small ML-only core and the application layer
that uses it.

## Structure

```text
DAP391m/
  ML_core/                Price models, ML training artifacts, ML services
  intent/                 User intent parsing
  agents/                 Web-search query and source adapters
  recommendation/         Candidate schemas and ranking
  pipelines/              End-to-end orchestration
  samples/                Mock inputs and search results
  reports/recommendation/ Recommendation pipeline outputs
  scripts/                App-level demos and utilities
  tests/                  App-level tests
  docs/                   Project-level architecture docs
```

## Run

```powershell
python scripts/run_recommend_demo.py
python -m unittest discover tests
python ML_core/scripts/validate_ml_core.py
```

The intended flow is:

```text
user intent -> ML_core price-band/vector service -> web-search agent -> recommendation
```
