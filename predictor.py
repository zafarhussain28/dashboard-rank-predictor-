"""Inference helpers for the trained JEE prediction artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite
from numbers import Real
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import sklearn


PROJECT_DIR = Path(__file__).resolve().parent
ARTIFACT_SKLEARN_VERSION = "1.6.1"
FEATURE_COLUMNS = (
    "physics_score",
    "chemistry_score",
    "maths_score",
    "total_score",
    "correct_questions",
    "wrong_questions",
    "unattempted_questions",
    "attempted_questions",
    "negative_marks",
    "total_time_seconds",
    "avg_time_per_question_sec",
)
STUDENT_INPUTS = (
    "physics_score",
    "chemistry_score",
    "maths_score",
    "correct_questions",
    "wrong_questions",
    "unattempted_questions",
    "total_time_seconds",
)
ARTIFACT_FILES = {
    "scaler": "Model2_Scaler.pkl",
    "percentile_model": "Model2_Percentile.pkl",
    "rank_model": "Model2_Rank.pkl",
}


class PredictorError(ValueError):
    """Raised when student inputs do not match the model's valid input domain."""


class ArtifactError(RuntimeError):
    """Raised when trained model artifacts are missing or incompatible."""


@dataclass(frozen=True)
class ModelArtifacts:
    scaler: Any
    percentile_model: Any
    rank_model: Any


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise PredictorError(f"{label} must be a number.")
    result = float(value)
    if not isfinite(result):
        raise PredictorError(f"{label} must be finite.")
    return result


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    result = _number(value, label)
    if not result.is_integer():
        raise PredictorError(f"{label} must be a whole number.")
    if not minimum <= result <= maximum:
        raise PredictorError(f"{label} must be between {minimum} and {maximum}.")
    return int(result)


def build_model_features(student_inputs: dict[str, Any]) -> dict[str, float | int]:
    """Validate the student-entered values and derive notebook features."""
    missing = [key for key in STUDENT_INPUTS if key not in student_inputs]
    if missing:
        raise PredictorError(f"Missing required inputs: {', '.join(missing)}.")

    scores = {
        key: _integer(student_inputs[key], key.replace("_", " ").title(), 0, 100)
        for key in STUDENT_INPUTS[:3]
    }
    correct = _integer(student_inputs["correct_questions"], "Correct questions", 0, 100)
    wrong = _integer(student_inputs["wrong_questions"], "Wrong questions", 0, 100)
    unattempted = _integer(
        student_inputs["unattempted_questions"], "Unattempted questions", 0, 100
    )
    total_time = _integer(
        student_inputs["total_time_seconds"], "Total time", 1, 10_800
    )

    attempted = correct + wrong
    if attempted == 0:
        raise PredictorError("Enter at least one correct or wrong answer.")
    if attempted + unattempted > 100:
        raise PredictorError("The total question count cannot exceed 100.")

    return {
        "physics_score": scores["physics_score"],
        "chemistry_score": scores["chemistry_score"],
        "maths_score": scores["maths_score"],
        "total_score": sum(scores.values()),
        "correct_questions": correct,
        "wrong_questions": wrong,
        "unattempted_questions": unattempted,
        "attempted_questions": attempted,
        "negative_marks": wrong,
        "total_time_seconds": total_time,
        "avg_time_per_question_sec": total_time / attempted,
    }


def validate_model_features(features: dict[str, Any]) -> None:
    missing = [key for key in FEATURE_COLUMNS if key not in features]
    if missing:
        raise PredictorError(f"Missing model features: {', '.join(missing)}.")

    values = {key: _number(features[key], key.replace("_", " ")) for key in FEATURE_COLUMNS}
    for key in FEATURE_COLUMNS[:3]:
        if not 0 <= values[key] <= 100:
            raise PredictorError(f"{key.replace('_', ' ').title()} must be between 0 and 100.")
    if not isclose(
        values["total_score"],
        sum(values[key] for key in FEATURE_COLUMNS[:3]),
        abs_tol=1e-9,
    ):
        raise PredictorError("Total score must equal the three subject scores combined.")
    if values["attempted_questions"] != (
        values["correct_questions"] + values["wrong_questions"]
    ):
        raise PredictorError("Attempted questions must equal correct plus wrong answers.")
    if values["negative_marks"] != values["wrong_questions"]:
        raise PredictorError("Negative marks must match wrong questions for this model.")
    if values["total_time_seconds"] <= 0 or values["attempted_questions"] <= 0:
        raise PredictorError("Time and attempted questions must be greater than zero.")
    expected_average = values["total_time_seconds"] / values["attempted_questions"]
    if not isclose(
        values["avg_time_per_question_sec"], expected_average, rel_tol=1e-9
    ):
        raise PredictorError("Average time must equal total time divided by attempted questions.")


def load_artifacts(artifact_dir: Path = PROJECT_DIR) -> ModelArtifacts:
    paths = {name: artifact_dir / filename for name, filename in ARTIFACT_FILES.items()}
    missing = [path.name for path in paths.values() if not path.is_file()]
    if missing:
        raise ArtifactError(f"Required model files are missing: {', '.join(missing)}.")
    if sklearn.__version__ != ARTIFACT_SKLEARN_VERSION:
        raise ArtifactError(
            "These model files were saved with scikit-learn "
            f"{ARTIFACT_SKLEARN_VERSION}; the installed version is {sklearn.__version__}. "
            "Install the versions listed in requirements.txt."
        )

    try:
        artifacts = ModelArtifacts(
            scaler=joblib.load(paths["scaler"]),
            percentile_model=joblib.load(paths["percentile_model"]),
            rank_model=joblib.load(paths["rank_model"]),
        )
    except Exception as exc:
        raise ArtifactError(
            "The trained model files could not be loaded. Verify that the repository "
            "artifacts are intact and dependencies match requirements.txt."
        ) from exc

    scaler_features = getattr(artifacts.scaler, "n_features_in_", None)
    if scaler_features != len(FEATURE_COLUMNS):
        raise ArtifactError(
            f"The scaler expects {scaler_features} features; {len(FEATURE_COLUMNS)} are required."
        )
    scaler_names = getattr(artifacts.scaler, "feature_names_in_", None)
    if scaler_names is not None and tuple(scaler_names) != FEATURE_COLUMNS:
        raise ArtifactError("The scaler feature names do not match the notebook feature order.")
    for name, model in (
        ("percentile", artifacts.percentile_model),
        ("rank", artifacts.rank_model),
    ):
        if getattr(model, "n_features_in_", None) != len(FEATURE_COLUMNS):
            raise ArtifactError(f"The {name} model does not accept the expected 11 features.")

    return artifacts


def predict_features(
    features: dict[str, Any], artifacts: ModelArtifacts
) -> dict[str, float]:
    """Scale an ordered feature row once and query both saved regressors."""
    validate_model_features(features)
    input_frame = pd.DataFrame(
        [[features[key] for key in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS
    )
    try:
        scaled_features = artifacts.scaler.transform(input_frame)
        percentile = float(artifacts.percentile_model.predict(scaled_features)[0])
        rank = float(artifacts.rank_model.predict(scaled_features)[0])
    except Exception as exc:
        raise ArtifactError("Prediction failed while applying the saved model pipeline.") from exc
    if not isfinite(percentile) or not isfinite(rank):
        raise ArtifactError("The model returned a non-finite prediction.")
    return {"percentile": percentile, "rank": rank}


def predict_student(
    student_inputs: dict[str, Any], artifacts: ModelArtifacts
) -> tuple[dict[str, float | int], dict[str, float]]:
    features = build_model_features(student_inputs)
    predictions = predict_features(features, artifacts)
    return features, predictions