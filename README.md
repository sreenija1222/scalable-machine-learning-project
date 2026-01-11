# Mood & Energy Prediction

This project builds an end-to-end ML pipeline for predicting **daily energy** and **daily mood stability** from a combination of wearable-derived data and self-reported symptoms.

**UI Deployed on Streamlit Cloud**: https://scalable-machine-learning-project-zw6shfb5qtixuk8mswpkro.streamlit.app/

The pipeline is implemented as four Jupyter notebooks and uses **Hopsworks** for:
- **Feature Groups** (offline & online storage)
- **Feature Views** (training datasets)
- **Model Registry** (versioned model artifacts & metrics)

---

## Notebooks Structure

- `1_mcphases_feature_backfill.ipynb`: Creates a **master daily table** `mcphases_daily_fg` (v1) and writes it to Hopsworks as a Feature Group (`mcphases_daily_fg`)

- `2_mcphases_feature_pipeline.ipynb`: Reads the master Feature Group, engineers features and labels, and writes **four engineered Feature Groups** (v2) and **four Feature Views** (v1) for training

- `3_mcphases_training.ipynb`: Loads datasets from Feature Views, performs a **subject-wise split** (to avoid leakage), trains **4 RandomForest models**, saves artifacts locally, and registers models in the Hopsworks Model Registry with metrics

- `4_mcphases_batch_inference.ipynb`: Loads latest model versions from the Model Registry and runs **online inference** from direct user inputs by building model-ready features from a user input dictionary (one-hot encodes phase, aligns feature order, fixes dtypes) and **producing predictions with class probabilities**
---

## Datasets

We train **four classifiers** (3-class classification) to support two modes per target:

### Targets
- **Energy**: `y_energy_cls3` (3 classes)
- **Mood stability**: `y_mood_stability_cls3` (3 classes)

### Modes
- **Mode A**: Uses wearable data and self-reported symptoms as features
- **Mode B**: `lag1_energy` or `lag1_mood` (previous day's ground truths) are included as features

---

## Hopsworks Entities

### Master Feature Group
- `mcphases_daily_fg` (v1)  
  Primary key: `(subject_id, day_in_study)`  
  Online enabled
  Event time: `event_time`

### Engineered Feature Groups (created in feature pipeline)
- `mcphases_energy_modea_fg` (v2)
- `mcphases_energy_modeb_fg` (v2)
- `mcphases_mood_modea_fg` (v2)
- `mcphases_mood_modeb_fg` (v2)

All keyed by `(subject_id, day_in_study)`

### Feature Views
- `mcphases_energy_modea_fv` (v1)
- `mcphases_energy_modeb_fv` (v1)
- `mcphases_mood_modea_fv` (v1)
- `mcphases_mood_modeb_fv` (v1)

Each Feature View selects:
- primary keys
- model features
- the label column (for training)

### Models in the Model Registry
Models are registered as:
- `mcphases_energy_modea_randomforest`
- `mcphases_energy_modeb_randomforest`
- `mcphases_mood_modea_randomforest`
- `mcphases_mood_modeb_randomforest`

Each registered model artifact folder includes:
- `model.joblib`
- `feature_columns.json` (feature order used during training)
- `metrics.json` (full evaluation metrics)

---

## Notes/Decisions

* **No leakage split**: Training uses a **subject-wise split**, so the same subject does not appear in both train and test
* **Feature name sanitization**: Hopsworks sanitizes feature names to lowercase, so the pipeline normalizes feature names accordingly
