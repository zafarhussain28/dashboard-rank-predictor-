# Dashboard Rank Predictor

Dashboard Rank Predictor is a browser-based JEE mock-test performance analyzer. Enter a student's subject scores, answer counts, and test time to get estimated percentile and rank from the project's saved machine-learning models. It also shows the entered subject scores and the values derived for the models. The estimates are based on synthetic training data, so they are for practice and exploration only, not official results.

## Run locally

Use Python 3.12. The model artifacts were serialized with scikit-learn 1.6.1, so that exact version is pinned in `requirements.txt`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

The app loads `Model2_Scaler.pkl`, `Model2_Percentile.pkl`, and `Model2_Rank.pkl` from the project directory. Keep these files beside `app.py`. No model training occurs at startup. `Synthetic_JEE_Mock_Dataset.csv` is used for training/evaluation only and is not required at runtime.

## Using the dashboard

1. Start the app with `streamlit run app.py` and open the local address printed in the terminal, usually `http://localhost:8501`.
2. Enter Physics, Chemistry, and Mathematics scores from 0 to 100.
3. Enter the numbers of correct, wrong, and unattempted questions, plus the total test time in seconds. Counts must be whole numbers, their sum cannot exceed 100, and at least one answer must have been attempted.
4. Select **Analyze performance**. The app displays the predicted percentile and rank, a chart of the subject scores you entered, and a table of all 11 feature values sent to the models. Values marked as derived are calculated by the app from your entries.
5. Change the inputs and analyze again to compare another mock-test result. The displayed rank is rounded to a whole number.

The form starts with sample values based on the example in the training notebook. Replace them with the mock-test results you want to analyze. Stop the local server with `Ctrl+C` in the terminal when you are finished.

## Inputs and inference

The form accepts section scores (Physics, Chemistry, Mathematics, each 0–100), correct answers, wrong answers, unattempted questions, and total test time (1–10,800 seconds). Answer counts must be whole numbers and add up to no more than 100; at least one answer must have been attempted.

The notebook's 11 model inputs, in order, are:

1. `physics_score`
2. `chemistry_score`
3. `maths_score`
4. `total_score`
5. `correct_questions`
6. `wrong_questions`
7. `unattempted_questions`
8. `attempted_questions`
9. `negative_marks`
10. `total_time_seconds`
11. `avg_time_per_question_sec`

The app derives total score as the sum of section scores, attempted questions as correct plus wrong, negative marks as the wrong-answer count, and average time as total time divided by attempted questions. These relationships match the notebook's example and the synthetic dataset. The derived 11-value row is ordered exactly as above, transformed once by the saved `StandardScaler`, and passed to the separate percentile and rank `GradientBoostingRegressor` models. Percentile is displayed to two decimal places; rank is rounded to a whole-number estimate.

## Tests

```bash
pytest -q
```

Tests cover loading and using the real artifacts, feature order and derivation, single-pass scaling, invalid and missing inputs, missing artifacts, inference errors, and Streamlit rendering.

## Deployment

For Streamlit Community Cloud, push the app and all three model artifacts to the repository, select `app.py` as the entry point, and select Python 3.12 in the app's advanced settings. Cloud installs the pinned compatible scikit-learn version from `requirements.txt`. The model artifacts total about 279 KB and are below typical repository hosting limits. The CSV is not needed to run the app. No secrets are required.

Only load these pickle files from a repository source you trust: deserializing joblib/pickle artifacts can execute code. The checked-in files are the artifacts associated with this repository's current commit.

## Model and limitations

The original notebook trains two Gradient Boosting regressors on a `StandardScaler`-transformed feature matrix. Its targets are `approx_percentile` and `approx_rank`. The dataset is explicitly synthetic; its patterns may not represent real exam scores, candidate populations, year-specific formats, or outcomes. The app does not claim model accuracy, confidence intervals, official ranks, or cutoffs. Actual results can differ substantially.
