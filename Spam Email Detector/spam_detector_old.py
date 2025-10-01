from pathlib import Path
import pandas as pd

# Try spam.csv first, else fall back to sms.csv
csv_path = Path("spam.csv")
if not csv_path.exists():
    alt = Path("sms.csv")
    if alt.exists():
        csv_path = alt
    else:
        raise FileNotFoundError(
            "Place 'spam.csv' (Kaggle) or 'sms.csv' (tiny sample) next to spam_detector.py"
        )

print(f"Loading: {csv_path}")
df = pd.read_csv(csv_path, encoding="latin-1")

# Handle both schemas:
# - Kaggle: columns 'v1' (label), 'v2' (text)
# - Our tiny file: columns 'label', 'text'
if {"v1", "v2"}.issubset(df.columns):
    df = df[["v1", "v2"]]
    df.columns = ["label", "text"]
elif {"label", "text"}.issubset(df.columns):
    df = df[["label", "text"]]
else:
    raise ValueError(
        f"Unexpected columns: {list(df.columns)}. Need either ['v1','v2'] or ['label','text']."
    )

# Clean labels
df["label"] = df["label"].astype(str).str.strip().str.lower().map({"ham": 0, "spam": 1})
df = df.dropna(subset=["text", "label"])
print("Rows:", len(df), " Label counts:\n", df["label"].value_counts())



# Step 4: split into train and test sets
from sklearn.model_selection import train_test_split

X = df["text"].astype(str)
y = df["label"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

print("\nTraining set size:", len(X_train), "Test size:", len(X_test))
print("Example training message:", X_train.iloc[0])


# Step 5

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(
    lowercase = True,
    stop_words = 'english',
    ngram_range = (1, 1),
    min_df = 1 
)),
    ("clf", MultinomialNB())
])

pipe.fit(X_train, y_train)

preds = pipe.predict(X_test)

print("\n=== Step 5: Baseline Naive Bayes ===")
print("Accuracy:", round(accuracy_score(y_test, preds), 3))
print("Confusion matrix:\n", confusion_matrix(y_test, preds))
print(classification_report(y_test, preds, digits=3))


# Step 6: Logistic Regression

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

nb_pipe = pipe
logit_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase = True,
        stop_words = 'english',
        ngram_range = (1, 2),
        min_df = 1
    )),
    ("clf", LogisticRegression(max_iter=500, class_weight="balanced"))
])

models = [("Naive Bayes", nb_pipe), ("Log Reg", logit_pipe)]

print("\n=== Step 6: Compare Models ===")
best_model, best_name, best_f1 = None, None, -1.0
for name, mdl in models:
    mdl.fit(X_train, y_train)
    preds = mdl.predict(X_test)
    f1 = f1_score(y_test, preds, zero_division=0)
    acc = accuracy_score(y_test, preds)
    print(f"{name}: F1={f1:.3f}  Acc={acc:.3f}")
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds, digits=3, zero_division=0))
    if f1 > best_f1:
        best_name, best_pipe, best_f1 = name, mdl, f1

print(f"Selected best model: {best_name} (F1={best_f1:.3f})")


# Step 7: Save the best model

import joblib
from pathlib import Path

model_path = Path("spam_detector.joblib")
joblib.dump(best_pipe, model_path)
print(f"\nSaved best model ({best_name}) {model_path.resolve()}")

loaded = joblib.load(model_path)

examples = [
    "WIN a brand new iPhone!!! Click here now",
    "Hey, are we still meeting at 3 pm?",
    "URGENT: your account will be closed unless you verify at http://phish.me"
]

print("\nDemo predictions:")
for msg in examples:
    pred = loaded.predict([msg])[0]
    label = "SPAM" if pred == 1 else "HAM"
    print(f"{label}: {msg}")


# cross-validated tuning (Log Reg)
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score


tune_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words='english')),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

param_grid = {
    "tfidf__ngram_range": [(1,1), (1,2)],       # unigrams or bigrams
    "tfidf__min_df": [1, 2],                    # ignore super rare words
    "clf__C": [0.5, 1.0, 2.0, 4.0]              # regularization strength
}

gs = GridSearchCV(
    estimator = tune_pipe,
    param_grid = param_grid,
    cv = 5,
    n_jobs = -1,
    scoring = 'f1',
)

gs.fit(X_train, y_train)
print("\n=== Step 8: Grid Search results ===")
print("Best params:", gs.best_params_)


# evaluate tune model on test set

best = gs.best_estimator_
preds = best.predict(X_test)
print("Classification report (tuned):")
print(classification_report(y_test, preds, digits=3, zero_division=0))


# if model supports probabilities, print AUC too

if hasattr(best, "predict_proba"):
    proba = best.predict_proba(X_test)[:, 1]
    print("ROC-AUC:", round(roc_auc_score(y_test, proba), 3))


# replace saved model with tuned one

import joblib

joblib.dump(best_model, "spam_detector.joblib")
print(f"\nSaved best model ({best_name}) to spam_detector.joblib")

loaded = joblib.load("spam_detector.joblib")