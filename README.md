# ID2223-Scalable machine leanringand Deep leanring models project
# Wellbeing Explorer — Daily Mood & Energy Prediction (mcPHASES)

A small end-to-end ML system that predicts **daily mood** and **daily energy** from menstrual-cycle and wellbeing signals, and presents the results in an interactive dashboard.

> Exploration only. Not diagnosis or medical advice.

---

## 1) What problem this solves

### Prediction task
We treat **each day** for each participant as one sample (tabular, longitudinal data).

**Inputs (per day)** are signals that are either:
- automatically available from common wearables/phones, or
- quick self-reports (seconds to enter)

Input fields contain
- cycle phase (e.g., menstruation / follicular / ovulation / luteal)
- sleep duration and sleep-related metrics
- resting heart rate
- stress (wearable-derived and/or self-reported)
- symptoms like cramps/headaches/sleep issues

**Targets (per day)**
- **Mood** (we operationalize this as *mood stability* using the `moodswing` item)
- **Energy** (derived from `fatigue`, inverted so “higher = more energy”)

We frame the final targets as **3 classes**:
- `0 = Low`, `1 = Medium`, `2 = High`

### Two model “routes” (to match real user behavior)
To support both “minimal logging” users and “more engaged” users, we train two variants:

**Mode A (wearables + quick inputs only)**
- Predict using only the day’s observable signals (phase/sleep/RHR/stress/symptoms…)
- Works even if the user doesn’t enter yesterday’s mood/energy

**Mode B (Mode A + optional lagged self-reports)**
- Same features as Mode A **plus** “yesterday’s mood/energy”
- Higher accuracy when users choose to provide yesterday’s values

This is implemented as a simple **routing** decision in the app:
- if yesterday values are provided → Mode B
- otherwise → Mode A

---

## 2) Dataset

We use **mcPHASES (PhysioNet)**: data from **42 menstruating young adults** across **two ~3-month intervals**, including:
- hormone-based cycle phase labels
- wearable-derived sleep, stress, heart rate, temperature signals
- daily self-reports (mood, fatigue, symptoms, etc.)

We use the dataset in a **daily resolution** format: one row per `(subject_id, day_in_study)` after merging relevant sources.

---

## 3) System overview (pipeline)

This project follows a scalable ML workflow with four main stages:

### (1) Backfill
- Load raw mcPHASES CSVs
- Standardize keys (`subject_id`, `day_in_study`) and clean types
- Collapse duplicate daily records (e.g., multiple wearable samples per day) into one daily row
- Merge tables into a **master daily dataset**
- (Optional) write to a Feature Store table (Hopsworks) after enforcing feature-name constraints

**Output:** `master_daily.parquet`

### (2) Feature Pipeline
- Select “user-feasible” features (wearables/phone + quick self-report)
- Encode categorical inputs (e.g., phase) and keep numeric/ordinal signals consistent
- Build targets:
  - `y_energy_*` from inverted fatigue
  - `y_mood_*` from inverted moodswing (mood stability proxy)
- Create two datasets per target:
  - Mode A: same-day features
  - Mode B: Mode A + lagged labels (e.g., lag1)

**Outputs (examples):**
- `mcphases_energy_modeA.parquet`
- `mcphases_energy_modeB.parquet`
- `mcphases_mood_modeA.parquet`
- `mcphases_mood_modeB.parquet`

### (3) Training
- Train a tabular model for each dataset (energy/mood × mode A/B)
- Use **subject-wise splitting** so test subjects are never seen in training (prevents leakage)
- Save:
  - model artifact
  - feature column order (`feature_columns.json`)
  - evaluation metrics (`metrics.json`)

**Models used**
- `RandomForestClassifier` (robust baseline for tabular data)
- `SimpleImputer(strategy="median")` inside a scikit-learn pipeline

### (4) Inference
- Load the correct model + feature schema
- Accept inputs from the UI
- Route to Mode A or Mode B
- Predict:
  - class label (Low/Medium/High)
  - probabilities for transparency/reminders of uncertainty

**Output:** JSON (for app display and debugging)

---

## 4) The UI (how we show value)

The dashboard is designed for two things:
1) **Daily logging + instant prediction**
2) **Pattern discovery across a month**

### Key UI components

#### A) View controls
- Month + year selector
- Toggle whether calendar dots show **Mood** or **Energy**
- Optional “Evaluate” mode for showing raw prediction payloads

#### B) Calendar view (month at a glance)
- Dots appear only on days the user saved
- Dot color intensity corresponds to predicted level (darker = higher)
- Switch between Mood dots and Energy dots

#### C) Trends section (time series)
Charts that help users connect inputs to predictions over time:
- Sleep (hours)
- Stress (0–5)
- Resting heart rate (bpm)
- Mood level (Low/Med/High)
- Energy level (Low/Med/High)

#### D) “Log a day” panel (quick entry)
- Date picker
- Cycle phase selector
- Sleep duration input (hours/minutes OR minutes)
- Resting heart rate slider
- Stress slider (0–5)
- Symptom sliders: cramps, headaches, sleep issues
- **Yesterday (optional)** checkbox:
  - if checked, user can input yesterday’s mood & energy
  - enables Mode B routing automatically

#### E) Model evaluation panel (debug/traceability)
- Shows the raw inference payload, e.g.:
  - `mood_pred`, `energy_pred`
  - `route` (modeA / modeB)
  - `mood_proba[]`, `energy_proba[]`

#### F) History table
- Shows saved days and key inputs for review
- Supports “edit a saved day” behavior by selecting a date

---

## 5) Technologies used

### Core stack
- **Python** (pandas, numpy)
- **scikit-learn** for modeling:
  - RandomForestClassifier
  - Pipelines + SimpleImputer
- **Parquet** for cached datasets (fast reloads, reproducible pipeline outputs)

### Feature store / registry (optional but supported)
- **Hopsworks Feature Store** for storing backfilled/feature-engineered datasets
- **Hopsworks Model Registry** for registering trained models and metrics
- `.env` based configuration for secrets and dataset paths

### App / Visualization
- **Streamlit** for the interactive “Wellbeing Explorer” dashboard
- Plotly (or Streamlit charting) for calendar + trend visualizations

---

## 6) Evaluation approach

### Why subject-wise split matters
Daily rows are highly correlated within each person. If you randomly split rows, the model can “memorize” person-specific patterns and look unrealistically strong.

We therefore split by `subject_id`:
- Train: subset of participants
- Test: completely unseen participants

### Metrics reported
- **Accuracy**: overall fraction correct
- **Balanced accuracy**: mean recall across classes (important if classes are imbalanced)
- **Macro F1**: averages F1 across classes equally (penalizes doing well on only the majority class)
- **Majority baseline**: accuracy you’d get by always predicting the most common class

Example results (will vary by run/split):
- Mode B often improves performance vs Mode A because lagged self-reports carry strong predictive signal.

---

## 7) How to run

### 1) Environment
Create/activate your environment and install dependencies:
- `pip install -r requirements.txt` (or your project’s equivalent)

Create a `.env` file at repo root with:
- `HOPSWORKS_API_KEY=...` (if using Hopsworks)
- `MCPHASES_DATA_DIR=/path/to/mcphases/csvs`

### 2) Run notebooks (in order)
1. `1_mcphases_feature_backfill.ipynb`
2. `2_mcphases_feature_pipeline.ipynb`
3. `3_mcphases_training.ipynb`
4. `4_mcphases_inference.ipynb`

### 3) Launch the UI
Run:
- `streamlit run app_streamlit.py`

---

## 8) Notes & limitations
- This is a learning + exploration tool, not medical advice.
- Missing targets are common in real self-report datasets; training uses only labeled rows.
- Mode B improves accuracy but requires yesterday inputs; Mode A is designed to work with minimal effort.

---

## 9) References
PhysioNet mcPHASES dataset:
