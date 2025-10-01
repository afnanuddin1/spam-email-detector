# train_spam.py
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, f1_score
import joblib

def load_dataset():

    """Load dataset spam.csv or fallback to sms.csv."""
    here = Path(".")
    csv_path = (here / "spam.csv") if (here / "spam.csv").exists() else (here / "sms.csv")
    if not csv_path.exists():
        raise FileNotFoundError("Place 'spam.csv' or 'sms.csv' next to this script.")
    print(f"Loading: {csv_path}")
    # Kaggle file often needs latin-1
    df = pd.read_csv(csv_path, encoding="latin-1")
    # Handle both schemas:
    if {"v1", "v2"}.issubset(df.columns):
        df = df[["v1", "v2"]]
        df.columns = ["label", "text"]
    elif {"label", "text"}.issubset(df.columns):
        df = df[["label", "text"]]
    else:
        raise ValueError(f"Unexpected columns: {list(df.columns)}")

    # Clean
    df["label"] = df["label"].astype(str).str.strip().str.lower().map({"ham": 0, "spam": 1})
    df = df.dropna(subset=["text", "label"])
    return df

def main():
    df = load_dataset()
    print("Rows:", len(df), "\nLabel counts:\n", df["label"].value_counts())

    X = df["text"].astype(str)
    y = df["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Baselines: NB and Log Reg
    nb = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1,1), min_df=2)),
        ("clf", MultinomialNB())
    ])
    logreg = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=2)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])

    print("\nTraining baselines...")
    models = [("Naive Bayes", nb), ("Log Reg", logreg)]
    best_name, best_model, best_f1 = None, None, -1.0
    for name, m in models:
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
        f1 = f1_score(y_test, preds, zero_division=0)
        acc = accuracy_score(y_test, preds)
        print(f"{name}:  F1={f1:.3f}  Acc={acc:.3f}")
        if f1 > best_f1:
            best_name, best_model, best_f1 = name, m, f1

    print(f"\nSelected baseline: {best_name} (F1={best_f1:.3f})")

    # Light tuning around the Log Reg
    print("\nGrid-search tuning (Logistic Regression)...")
    tune_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])
    param_grid = {
        "tfidf__ngram_range": [(1,1), (1,2)],
        "tfidf__min_df": [1,2,3],
        "clf__C": [0.5, 1.0, 2.0, 4.0]
    }
    gs = GridSearchCV(tune_pipe, param_grid, cv=5, n_jobs=-1, scoring="f1")
    gs.fit(X_train, y_train)

    tuned = gs.best_estimator_
    print("Best params:", gs.best_params_)

    # Final evaluation on held-out test
    preds = tuned.predict(X_test)
    print("\nClassification report (tuned):")
    print(classification_report(y_test, preds, digits=3, zero_division=0))
    if hasattr(tuned, "predict_proba"):
        proba = tuned.predict_proba(X_test)[:, 1]
        print("ROC-AUC:", round(roc_auc_score(y_test, proba), 3))

    # Save tuned model
    out = Path("spam_detector_tuned.joblib")
    joblib.dump(tuned, out)
    print(f"\nSaved tuned model to {out.resolve()}")

if __name__ == "__main__":
    main()
