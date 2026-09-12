from pathlib import Path
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

BASE = Path(__file__).parent
df = pd.read_csv(BASE / "data" / "qr_dataset.csv")

features = ["length","digits","special_chars","has_url","https","num_dots",
            "num_slashes","num_at","num_hyphen","num_equals","num_question",
            "suspicious_words","shortener","ip_like"]

X, y = df[features], df["label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, max_depth=6,
                               random_state=42, class_weight="balanced")
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("ML Accuracy:", round(accuracy_score(y_test, pred), 4))
print(classification_report(y_test, pred, target_names=["Safe","Suspicious"]))

out = BASE / "models" / "qr_suspicious_model.joblib"
out.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, out)
print("ML model saved:", out)
