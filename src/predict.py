"""Classify new emails as Phishing or Safe using the trained model.

Usage
-----
Classify text passed on the command line:
    python -m src.predict --text "Your account is suspended! Verify at http://1.2.3.4/login"

Classify the contents of a file:
    python -m src.predict --file path/to/email.txt

Run the built-in demo on a few example emails:
    python -m src.predict --demo
"""
from __future__ import annotations

import argparse
import os
import sys

import joblib

# Importing keeps the FunctionTransformer reference resolvable when unpickling.
from src.feature_extraction import numeric_feature_matrix  # noqa: F401

DEFAULT_MODEL = "models/phishing_model.joblib"
LABELS = {0: "Safe", 1: "Phishing"}


def load_model(path: str):
    if not os.path.exists(path):
        raise SystemExit(
            f"Model not found at {path}. Train it first with: python -m src.train"
        )
    return joblib.load(path)


def classify(model, text: str) -> dict:
    pred = int(model.predict([text])[0])
    result = {"label": LABELS[pred], "prediction": pred}
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([text])[0]
        result["confidence"] = float(proba[pred])
        result["phishing_probability"] = float(proba[1])
    return result


def _print(text: str, result: dict) -> None:
    preview = text.replace("\n", " ")[:80]
    conf = result.get("confidence")
    conf_str = f"  (confidence {conf:.1%})" if conf is not None else ""
    print(f"[{result['label']:>8}]{conf_str}  {preview}")


DEMO_EMAILS = [
    ("Subject: URGENT account suspended\n\nDear customer, your PayPal account "
     "has been suspended. Verify immediately at http://198.51.100.23/verify.php"),
    ("Subject: Lunch tomorrow?\n\nHey, are you free for lunch around noon? "
     "Let me know! See the notes at https://docs.google.com/document/123"),
    ("Subject: You won a prize!!!\n\nCongratulations! Claim your FREE $1000 gift "
     "card now at http://bit.ly/winPrize42 before it expires."),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify emails as Phishing/Safe")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--text", help="Raw email text to classify")
    parser.add_argument("--file", help="Path to a file containing the email text")
    parser.add_argument("--demo", action="store_true", help="Run on example emails")
    args = parser.parse_args()

    model = load_model(args.model)

    if args.demo:
        for email in DEMO_EMAILS:
            _print(email, classify(model, email))
        return

    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        text = sys.stdin.read()
        if not text.strip():
            parser.error("Provide --text, --file, --demo, or pipe text via stdin.")

    _print(text, classify(model, text))


if __name__ == "__main__":
    main()
