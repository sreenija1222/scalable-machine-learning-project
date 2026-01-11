from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List
from functools import lru_cache
import os
import hopsworks
import joblib
import pandas as pd

HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
MODEL_VERSION = 1
#MODEL_VERSION = int(os.getenv("HOPSWORKS_MODEL_VERSION", "1"))

MODEL_NAMES = {
    "mood_modeA": "mcphases_mood_modeb_randomforest",
    "mood_modeB": "mcphases_mood_modeb_randomforest",
    "energy_modeA": "mcphases_energy_modea_randomforest",
    "energy_modeB": "mcphases_energy_modeb_randomforest",
}


def _find_joblib(path: Path) -> Path:
    direct = path / "model.joblib"
    if direct.exists():
        return direct

    hits = list(path.rglob("*.joblib"))
    if not hits:
        raise FileNotFoundError(f"No .joblib found in {path}")
    return hits[0]


@lru_cache(maxsize=8)
def _load_model_from_hopsworks(model_name: str):
    if HOPSWORKS_PROJECT and HOPSWORKS_API_KEY:
        project = hopsworks.login(project=HOPSWORKS_PROJECT, api_key_value=HOPSWORKS_API_KEY)
    else:
        project = hopsworks.login()

    mr = project.get_model_registry()
    model = mr.get_model(model_name, version=MODEL_VERSION)
    model_dir = Path(model.download())

    joblib_path = _find_joblib(model_dir)
    return joblib.load(joblib_path)


@lru_cache(maxsize=1)
def get_models() -> Dict[str, Any]:
    return {
        "MOOD_A": _load_model_from_hopsworks(MODEL_NAMES["mood_modeA"]),
        "MOOD_B": _load_model_from_hopsworks(MODEL_NAMES["mood_modeB"]),
        "ENERGY_A": _load_model_from_hopsworks(MODEL_NAMES["energy_modeA"]),
        "ENERGY_B": _load_model_from_hopsworks(MODEL_NAMES["energy_modeB"]),
    }


def _as_int(x: Any, name: str) -> int:
    try:
        return int(x)
    except Exception as e:
        raise ValueError(f"{name} must be int-like, got {x!r}") from e


def _as_str(x: Any, name: str) -> str:
    if x is None:
        raise ValueError(f"missing required field: {name}")
    return str(x).strip()


def _clamp(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def _expected_feature_names(model) -> List[str]:
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    if hasattr(model, "named_steps"):
        for step in reversed(list(model.named_steps.values())):
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)

    raise ValueError(
        "model does not expose feature_names_in_. "
        "save the feature col list during training and load it here."
    )


def _align_to_expected(X: pd.DataFrame, expected: List[str]) -> pd.DataFrame:
    for c in expected:
        if c not in X.columns:
            X[c] = 0
    return X[expected]


PHASE_TO_BUCKET = {
    "Menstruation": "Menstrual",
    "Menstrual": "Menstrual",
    "Menses": "Menstrual",
    "Late Follicular": "Follicular",
    "Follicular": "Follicular",
    "Ovulation": "Fertility",
    "Fertility": "Fertility",
    "Luteal": "Luteal",
}

PHASE_COLS = [
    "phase_Fertility",
    "phase_Follicular",
    "phase_Luteal",
    "phase_Menstrual",
    "phase_nan",
]


def get_input_schema() -> Dict[str, Any]:
    return {
        "phase_values": ["Menstruation", "Late Follicular", "Ovulation", "Luteal"],
        "sleep_minutes_range": [0, 900],
        "resting_hr_range": [35, 110],
        "ordinal_0_5": True,
        "cls3_values": [0, 1, 2],
    }


def _build_feature_row(payload: Dict[str, Any]) -> Dict[str, Any]:
    phase_raw = _as_str(payload.get("phase"), "phase")

    is_weekend = _clamp(_as_int(payload.get("is_weekend"), "is_weekend"), 0, 1)
    sleep_mins = _clamp(_as_int(payload.get("sleep_duration_minutes"), "sleep_duration_minutes"), 0, 900)
    rhr = _clamp(_as_int(payload.get("resting_heart_rate__value"), "resting_heart_rate__value"), 35, 110)

    cramps = _clamp(_as_int(payload.get("cramps_num", 0), "cramps_num"), 0, 5)
    headaches = _clamp(_as_int(payload.get("headaches_num", 0), "headaches_num"), 0, 5)
    sleepissue = _clamp(_as_int(payload.get("sleepissue_num", 0), "sleepissue_num"), 0, 5)
    stress = _clamp(_as_int(payload.get("stress_num", 0), "stress_num"), 0, 5)

    row: Dict[str, Any] = {
        "is_weekend": is_weekend,
        "sleep_duration_minutes": sleep_mins,
        "resting_heart_rate__value": rhr,
        "cramps_num": cramps,
        "headaches_num": headaches,
        "sleepissue_num": sleepissue,
        "stress_num": stress,
    }

    if "lag1_mood" in payload and payload.get("lag1_mood") is not None:
        row["lag1_mood"] = _clamp(_as_int(payload["lag1_mood"], "lag1_mood"), 0, 2)
    if "lag1_energy" in payload and payload.get("lag1_energy") is not None:
        row["lag1_energy"] = _clamp(_as_int(payload["lag1_energy"], "lag1_energy"), 0, 2)

    for c in PHASE_COLS:
        row[c] = 0

    bucket = PHASE_TO_BUCKET.get(phase_raw)
    if bucket is None:
        row["phase_nan"] = 1
    else:
        row[f"phase_{bucket}"] = 1

    return row


def predict_one(payload: Dict[str, Any]) -> Dict[str, Any]:
    use_lag = ("lag1_mood" in payload and payload.get("lag1_mood") is not None) and (
        "lag1_energy" in payload and payload.get("lag1_energy") is not None
    )
    route = "modeB" if use_lag else "modeA"

    models = get_models()
    mood_model = models["MOOD_B"] if use_lag else models["MOOD_A"]
    energy_model = models["ENERGY_B"] if use_lag else models["ENERGY_A"]

    row = _build_feature_row(payload)
    X_raw = pd.DataFrame([row])

    mood_expected = _expected_feature_names(mood_model)
    energy_expected = _expected_feature_names(energy_model)

    X_mood = _align_to_expected(X_raw.copy(), mood_expected)
    X_energy = _align_to_expected(X_raw.copy(), energy_expected)

    mood_pred = int(mood_model.predict(X_mood)[0])
    energy_pred = int(energy_model.predict(X_energy)[0])

    out: Dict[str, Any] = {"mood_pred": mood_pred, "energy_pred": energy_pred, "route": route}

    if hasattr(mood_model, "predict_proba"):
        try:
            out["mood_proba"] = mood_model.predict_proba(X_mood)[0].tolist()
        except Exception:
            pass

    if hasattr(energy_model, "predict_proba"):
        try:
            out["energy_proba"] = energy_model.predict_proba(X_energy)[0].tolist()
        except Exception:
            pass

    return out
