"""
predict.py — Classify a single SMS message as spam or ham.

Usage:
    python predict.py "Your message here"

Requires trained artifacts in the models/ directory.
Run SMS_Spam_Detection.ipynb first to generate them.
"""

import sys
import os
import joblib


MODEL_PATH    = os.path.join("models", "best_model.pkl")
TFIDF_PATH    = os.path.join("models", "tfidf_vectorizer.pkl")


def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(TFIDF_PATH):
        print(
            "Error: model artifacts not found.\n"
            "Run the notebook first to train and save the model."
        )
        sys.exit(1)
    model     = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(TFIDF_PATH)
    return model, vectorizer


def predict(message: str, model, vectorizer) -> str:
    vec  = vectorizer.transform([message])
    pred = model.predict(vec)[0]
    return "SPAM 🚨" if pred == 1 else "HAM ✅"


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py \"<message>\"")
        sys.exit(1)

    message          = " ".join(sys.argv[1:])
    model, vectorizer = load_artifacts()
    result           = predict(message, model, vectorizer)

    print(f"\nMessage : {message}")
    print(f"Result  : {result}\n")


if __name__ == "__main__":
    main()
