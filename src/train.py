"""Train and evaluate the phishing email detection model.

Pipeline overview
-----------------
The model is a single scikit-learn ``Pipeline`` that consumes raw email text
and combines two feature families with a ``FeatureUnion``:

1. ``TfidfVectorizer`` - word + bigram TF-IDF over the email body.
2. ``FunctionTransformer(numeric_feature_matrix)`` - hand-engineered URL and
   heuristic features (see ``src/feature_extraction.py``), standardized.

A ``LogisticRegression`` classifier is trained on top. We also train a
``RandomForestClassifier`` for comparison and keep whichever scores higher on
the held-out test set.

Outputs (written to ``outputs/`` and ``models/``):
- Printed accuracy, precision/recall/F1 and a full classification report.
- ``outputs/confusion_matrix.png`` - confusion matrix heatmap.
- ``outputs/metrics.json`` - machine-readable metrics.
- ``models/phishing_model.joblib`` - the fitted pipeline for inference.

Usage:
    python -m src.train --data data/emails.csv
"""
from __future__ import annotations

import argparse
import json
import os

import joblib
import matplotlib

matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.model_selection import train_test_split

from src.feature_extraction import FEATURE_NAMES, numeric_feature_matrix

LABELS = ["safe", "phishing"]  # 0 = safe, 1 = phishing


def build_pipeline(classifier) -> Pipeline:
    """Assemble the full text + engineered-feature pipeline."""
    tfidf = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_features=5000,
        sublinear_tf=True,
    )
    numeric = Pipeline(
        steps=[
            ("extract", FunctionTransformer(numeric_feature_matrix, validate=False)),
            ("scale", StandardScaler()),
        ]
    )
    features = FeatureUnion(
        transformer_list=[("tfidf", tfidf), ("numeric", numeric)]
    )
    return Pipeline(steps=[("features", features), ("clf", classifier)])


def evaluate(name, pipeline, X_test, y_test) -> dict:
    preds = pipeline.predict(X_test)
    metrics = {
        "model": name,
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds)),
        "recall": float(recall_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds)),
    }
    return metrics, preds


def main() -> None:
    parser = argparse.ArgumentParser(description="Train phishing detector")
    parser.add_argument("--data", default="data/emails.csv")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-out", default="models/phishing_model.joblib")
    parser.add_argument("--outputs", default="outputs")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        raise SystemExit(
            f"Dataset not found at {args.data}. "
            "Run: python -m src.generate_dataset --out data/emails.csv"
        )

    df = pd.read_csv(args.data)
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    X = df["text"].astype(str).values
    y = (df["label"].str.lower() == "phishing").astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    print(f"Loaded {len(df)} emails "
          f"({int(y.sum())} phishing / {int((1 - y).sum())} safe)")
    print(f"Train: {len(X_train)}  |  Test: {len(X_test)}\n")

    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=1000, C=4.0),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, random_state=args.seed, n_jobs=-1
        ),
    }

    results = []
    fitted = {}
    for name, clf in candidates.items():
        pipe = build_pipeline(clf)
        pipe.fit(X_train, y_train)
        metrics, _ = evaluate(name, pipe, X_test, y_test)
        results.append(metrics)
        fitted[name] = pipe
        print(f"{name:>20}  acc={metrics['accuracy']:.4f}  "
              f"precision={metrics['precision']:.4f}  "
              f"recall={metrics['recall']:.4f}  f1={metrics['f1']:.4f}")

    best = max(results, key=lambda m: m["f1"])
    best_name = best["model"]
    best_pipe = fitted[best_name]
    print(f"\nBest model: {best_name} (F1={best['f1']:.4f})\n")

    # Detailed report + confusion matrix for the best model.
    preds = best_pipe.predict(X_test)
    print("Classification report (best model):")
    print(classification_report(y_test, preds, target_names=LABELS))

    cm = confusion_matrix(y_test, preds)
    os.makedirs(args.outputs, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d"
    )
    ax.set_title(f"Confusion Matrix - {best_name}")
    plt.tight_layout()
    cm_path = os.path.join(args.outputs, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix -> {cm_path}")

    # Engineered-feature importances (from the RandomForest model), for insight.
    metrics_payload = {"results": results, "best_model": best_name}
    rf_pipe = fitted.get("RandomForest")
    if rf_pipe is not None:
        importances = rf_pipe.named_steps["clf"].feature_importances_
        # The numeric engineered features are the final len(FEATURE_NAMES)
        # columns of the FeatureUnion (the TF-IDF block comes first).
        eng = importances[-len(FEATURE_NAMES):]
        order = list(np.argsort(eng)[::-1])
        print("Top engineered features (RandomForest importance):")
        for i in order[:10]:
            print(f"  {FEATURE_NAMES[i]:>26}: {eng[i]:.4f}")
        print()
        metrics_payload["engineered_feature_importance"] = {
            FEATURE_NAMES[i]: float(eng[i]) for i in order
        }
    with open(os.path.join(args.outputs, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"Saved metrics -> {os.path.join(args.outputs, 'metrics.json')}")

    os.makedirs(os.path.dirname(args.model_out) or ".", exist_ok=True)
    joblib.dump(best_pipe, args.model_out)
    print(f"Saved model  -> {args.model_out}")


if __name__ == "__main__":
    main()
