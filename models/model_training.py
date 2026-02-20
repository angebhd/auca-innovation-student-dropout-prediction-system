import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os


df = pd.read_csv("data/cleaned/students_cleaned.csv")
df = df.dropna(subset=["dropout_risk"])



X = df.drop(columns=["student_id", "actual_dropout", "dropout_risk"])
y = df["dropout_risk"]


X = pd.get_dummies(X, drop_first=True)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(random_state=42,class_weight="balanced",)
                            
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

#
if not os.path.exists("models"):
    os.makedirs("models")

joblib.dump(model, "models/risk_model.pkl")
joblib.dump(X.columns.tolist(), "models/feature_columns.pkl")
print("Model and feature columns saved in 'models/'")
