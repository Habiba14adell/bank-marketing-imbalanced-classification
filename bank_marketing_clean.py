"""Clean workflow for the Bank Marketing imbalanced classification project.

This script keeps the evaluation honest by splitting the data before any
resampling, then using pipelines for preprocessing, sampling, and modeling.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline
from scipy.io.arff import loadarff
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path("phpkIxskf.arff")
RANDOM_STATE = 42
TEST_SIZE = 0.25
DROP_DURATION = False


COLUMN_NAMES = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "balance",
    "housing",
    "loan",
    "contact",
    "day",
    "month",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
    "y",
]


def load_bank_marketing(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the ARFF dataset and decode byte-string categorical values."""
    raw_data, _metadata = loadarff(path)
    df = pd.DataFrame(raw_data)
    df.columns = COLUMN_NAMES

    for column in df.select_dtypes(include=["object"]).columns:
        df[column] = df[column].str.decode("utf-8")

    target_map = {"1": 0, "2": 1, "no": 0, "yes": 1}
    df["y"] = df["y"].map(target_map).astype(int)
    return df


def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    numeric_features = x.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = x.select_dtypes(exclude=["number"]).columns.tolist()

    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ]
    )


def get_experiments(preprocessor: ColumnTransformer) -> dict[str, Pipeline]:
    logistic = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
    )
    balanced_logistic = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=RANDOM_STATE,
    )
    forest = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", logistic),
            ]
        ),
        "Balanced Logistic Regression": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", balanced_logistic),
            ]
        ),
        "SMOTE + Logistic Regression": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("sampler", SMOTE(random_state=RANDOM_STATE, k_neighbors=3)),
                ("model", logistic),
            ]
        ),
        "Random Oversampling + Logistic Regression": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("sampler", RandomOverSampler(random_state=RANDOM_STATE)),
                ("model", logistic),
            ]
        ),
        "Balanced Random Forest": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", forest),
            ]
        ),
    }


def evaluate_model(name: str, model: Pipeline, x_train, x_test, y_train, y_test) -> dict:
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_score = model.predict_proba(x_test)[:, 1]

    precision, recall, f1, _support = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="binary",
        zero_division=0,
    )

    print(f"\n{name}")
    print("-" * len(name))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["no", "yes"], zero_division=0))

    return {
        "model": name,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc_score(y_test, y_score),
        "average_precision": average_precision_score(y_test, y_score),
    }


def main() -> None:
    df = load_bank_marketing()

    if DROP_DURATION:
        df = df.drop(columns=["duration"])

    x = df.drop(columns=["y"])
    y = df["y"]

    print("Class distribution:")
    print(y.value_counts().rename(index={0: "no", 1: "yes"}))

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessor = build_preprocessor(x_train)
    experiments = get_experiments(preprocessor)

    results = [
        evaluate_model(name, model, x_train, x_test, y_train, y_test)
        for name, model in experiments.items()
    ]

    results_df = pd.DataFrame(results).sort_values(
        by=["average_precision", "f1"],
        ascending=False,
    )

    print("\nModel comparison")
    print("----------------")
    print(results_df.to_string(index=False, float_format="{:.3f}".format))


if __name__ == "__main__":
    main()
