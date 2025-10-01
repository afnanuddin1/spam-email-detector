# predict.py
import sys
import joblib

MODEL_PATH = "spam_detector_tuned.joblib"
model = joblib.load(MODEL_PATH)

def classify(msg: str, threshold: float = 0.7):
    if hasattr(model, "predict_proba"):
        p_spam = model.predict_proba([msg])[0][1]
        label = "SPAM" if p_spam >= threshold else "HAM"
        return label, p_spam
    # fallback if model has no proba
    pred = model.predict([msg])[0]
    return ("SPAM" if pred == 1 else "HAM", None)

def main():
    if len(sys.argv) > 1:
        msg = " ".join(sys.argv[1:])
        label, p = classify(msg)
        if p is None:
            print(f"[{label}] {msg}")
        else:
            print(f"[{label}] (P(spam)={p:.2f})  {msg}")
    else:
        print("Type messages to classify (blank line to quit).")
        while True:
            msg = input("> ").strip()
            if msg == "":
                break
            label, p = classify(msg)
            if p is None:
                print(f"[{label}] {msg}")
            else:
                print(f"[{label}] (P(spam)={p:.2f})  {msg}")

if __name__ == "__main__":
    main()
