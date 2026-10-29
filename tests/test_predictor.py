from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from predictor import (
    FEATURE_COLUMNS,
    ArtifactError,
    ModelArtifacts,
    PredictorError,
    build_model_features,
    load_artifacts,
    predict_features,
)


SAMPLE_INPUTS = {
    "physics_score": 52,
    "chemistry_score": 63,
    "maths_score": 48,
    "correct_questions": 51,
    "wrong_questions": 19,
    "unattempted_questions": 30,
    "total_time_seconds": 10_150,
}


def test_real_artifacts_load_and_predict_sample():
    artifacts = load_artifacts()
    features = build_model_features(SAMPLE_INPUTS)
    predictions = predict_features(features, artifacts)

    assert tuple(features) == FEATURE_COLUMNS
    assert artifacts.scaler.n_features_in_ == len(FEATURE_COLUMNS)
    assert artifacts.percentile_model.n_features_in_ == len(FEATURE_COLUMNS)
    assert artifacts.rank_model.n_features_in_ == len(FEATURE_COLUMNS)
    assert np.isfinite(predictions["percentile"])
    assert np.isfinite(predictions["rank"])


def test_feature_values_match_notebook_order_and_derivations():
    features = build_model_features(SAMPLE_INPUTS)

    assert tuple(features) == FEATURE_COLUMNS
    assert features["total_score"] == 163
    assert features["attempted_questions"] == 70
    assert features["negative_marks"] == 19
    assert features["avg_time_per_question_sec"] == 145


def test_scaler_is_called_once_and_both_models_receive_its_output():
    class Scaler:
        calls = 0

        def transform(self, frame):
            self.calls += 1
            assert list(frame.columns) == list(FEATURE_COLUMNS)
            return np.array([[float(index) for index in range(len(FEATURE_COLUMNS))]])

    class Model:
        def __init__(self, value):
            self.value = value
            self.received = None

        def predict(self, values):
            self.received = values.copy()
            return [self.value]

    scaler = Scaler()
    percentile_model = Model(72.5)
    rank_model = Model(123_456.4)
    artifacts = ModelArtifacts(scaler, percentile_model, rank_model)

    predictions = predict_features(build_model_features(SAMPLE_INPUTS), artifacts)

    assert scaler.calls == 1
    np.testing.assert_array_equal(percentile_model.received, rank_model.received)
    assert predictions == {"percentile": 72.5, "rank": 123_456.4}


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("physics_score", 101, "between 0 and 100"),
        ("wrong_questions", -1, "between 0 and 100"),
        ("correct_questions", 1.5, "whole number"),
        ("total_time_seconds", 10_801, "between 1 and 10800"),
        ("unattempted_questions", 31, "cannot exceed 100"),
    ],
)
def test_rejects_invalid_student_values(field, value, message):
    values = {**SAMPLE_INPUTS, field: value}

    with pytest.raises(PredictorError, match=message):
        build_model_features(values)


def test_rejects_missing_student_input():
    values = {key: value for key, value in SAMPLE_INPUTS.items() if key != "maths_score"}

    with pytest.raises(PredictorError, match="Missing required inputs"):
        build_model_features(values)


def test_rejects_zero_attempts():
    values = {
        **SAMPLE_INPUTS,
        "correct_questions": 0,
        "wrong_questions": 0,
        "unattempted_questions": 90,
    }

    with pytest.raises(PredictorError, match="at least one"):
        build_model_features(values)


def test_missing_artifacts_report_actionable_error(tmp_path):
    with pytest.raises(ArtifactError, match="Required model files are missing"):
        load_artifacts(tmp_path)


def test_model_failure_is_reported_as_inference_error():
    class Scaler:
        def transform(self, frame):
            return frame.to_numpy()

    class FailingModel:
        def predict(self, values):
            raise RuntimeError("estimator failure")

    artifacts = ModelArtifacts(Scaler(), FailingModel(), FailingModel())
    features = build_model_features(SAMPLE_INPUTS)

    with pytest.raises(ArtifactError, match="Prediction failed"):
        predict_features(features, artifacts)


def test_streamlit_app_renders_without_errors():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()

    assert not app.exception
    assert any("Dashboard Rank Predictor" in element.value for element in app.markdown)

    app.button[0].click().run()

    assert not app.exception
    assert any("Prediction estimates" in element.value for element in app.subheader)