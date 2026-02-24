# # import pandas as pd
# # from sklearn.model_selection import train_test_split
# # from sklearn.ensemble import RandomForestClassifier
# # from sklearn.metrics import classification_report, accuracy_score
# # import joblib
# # import os


# # df = pd.read_csv("data/cleaned/students_cleaned.csv")
# # df = df.dropna(subset=["dropout_risk"])



# # X = df.drop(columns=["student_id", "actual_dropout", "dropout_risk"])
# # y = df["dropout_risk"]


# # X = pd.get_dummies(X, drop_first=True)


# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# # model = RandomForestClassifier(random_state=42,class_weight="balanced",)
                            
# # model.fit(X_train, y_train)


# # y_pred = model.predict(X_test)
# # print("Accuracy:", accuracy_score(y_test, y_pred))
# # print(classification_report(y_test, y_pred))

# # #
# # if not os.path.exists("models"):
# #     os.makedirs("models")

# # joblib.dump(model, "models/risk_model.pkl")
# # joblib.dump(X.columns.tolist(), "models/feature_columns.pkl")
# # print("Model and feature columns saved in 'models/'")
# """
# Student Dropout Prediction — Model Training Script
# AUCA Innovation Lab | Umoja Team

# Run this once from the project root after cleaning your data:

#     python src/train_model.py

# Output:
#     models/risk_model.pkl       — trained Random Forest model
#     models/feature_columns.pkl  — list of features used
# """

# import sys
# import os
# import joblib
# import pandas as pd
# import numpy as np
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.metrics import (
#     classification_report,
#     accuracy_score,
#     confusion_matrix,
#     ConfusionMatrixDisplay,
# )

# # ── Path setup ─────────────────────────────────────────────────────────────────
# PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# if PROJECT_ROOT not in sys.path:
#     sys.path.insert(0, PROJECT_ROOT)

# CLEANED_DIR   = os.path.join(PROJECT_ROOT, "data", "cleaned")
# MODEL_DIR     = os.path.join(PROJECT_ROOT, "models")
# MODEL_PATH    = os.path.join(MODEL_DIR, "risk_model.pkl")
# FEATURES_PATH = os.path.join(MODEL_DIR, "feature_columns.pkl")
# PLOTS_DIR     = os.path.join(MODEL_DIR, "plots")

# CANDIDATE_FEATURES = [
#     "age", "admission_grade",
#     "semester_1_gpa", "semester_2_gpa", "semester_3_gpa",
#     "current_gpa", "failed_courses", "attendance_rate",
#     "absences_count", "late_submissions", "online_portal_logins",
#     "library_visits", "participation_score", "distance_from_campus",
# ]

# TARGET = "dropout_risk"

# FRIENDLY_NAMES = {
#     "current_gpa":          "Current Overall Grade",
#     "attendance_rate":      "Class Attendance (%)",
#     "failed_courses":       "Courses Failed",
#     "semester_1_gpa":       "Semester 1 Grade",
#     "semester_2_gpa":       "Semester 2 Grade",
#     "semester_3_gpa":       "Semester 3 Grade",
#     "absences_count":       "Total Absences",
#     "late_submissions":     "Late Assignments",
#     "admission_grade":      "Admission Score",
#     "participation_score":  "Class Participation",
#     "online_portal_logins": "Online Platform Usage",
#     "library_visits":       "Library Visits",
#     "distance_from_campus": "Distance from Campus",
#     "age":                  "Student Age",
# }


# def print_section(title):
#     print(f"\n{'='*55}\n  {title}\n{'='*55}")


# def load_data():
#     print_section("STEP 1 — Loading Cleaned Data")
#     if not os.path.exists(CLEANED_DIR):
#         raise FileNotFoundError(
#             f"Cleaned data folder not found: {CLEANED_DIR}\n"
#             "Please clean a dataset first using the dashboard Data Suite."
#         )
#     csv_files = [f for f in os.listdir(CLEANED_DIR) if f.endswith(".csv")]
#     if not csv_files:
#         raise FileNotFoundError(
#             "No cleaned CSV files found in data/cleaned/.\n"
#             "Please clean a dataset first using the dashboard Data Suite."
#         )
#     latest = max([os.path.join(CLEANED_DIR, f) for f in csv_files], key=os.path.getctime)
#     df = pd.read_csv(latest)
#     print(f"  File    : {os.path.basename(latest)}")
#     print(f"  Rows    : {len(df)}")
#     print(f"  Columns : {len(df.columns)}")
#     return df


# def prepare_data(df):
#     print_section("STEP 2 — Preparing Features & Target")
#     if TARGET not in df.columns:
#         raise ValueError(f"Target column '{TARGET}' not found. Available: {list(df.columns)}")

#     features = [f for f in CANDIDATE_FEATURES if f in df.columns]
#     if len(features) < 3:
#         raise ValueError(f"Not enough features found. Got: {features}")

#     print(f"  Target   : {TARGET}")
#     print(f"  Features : {len(features)} found")
#     for f in features:
#         print(f"    ✔  {FRIENDLY_NAMES.get(f, f)}")

#     df_model = df[features + [TARGET]].dropna()
#     print(f"\n  Usable rows (after removing blanks): {len(df_model)}")

#     X = df_model[features]
#     y = df_model[TARGET]

#     print("\n  Class distribution:")
#     for label, count in y.value_counts().items():
#         print(f"    {label:8s} → {count} students ({count/len(y)*100:.1f}%)")

#     return X, y, features


# def train(X, y):
#     print_section("STEP 3 — Training Random Forest Model")
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42, stratify=y
#     )
#     print(f"  Training set : {len(X_train)} students")
#     print(f"  Test set     : {len(X_test)} students")

#     model = RandomForestClassifier(
#         n_estimators=200, max_depth=10, min_samples_split=5,
#         class_weight="balanced", random_state=42, n_jobs=-1,
#     )
#     model.fit(X_train, y_train)
#     print("  Model trained ")
#     return model, X_train, X_test, y_train, y_test


# def evaluate(model, X, y, X_test, y_test, features):
#     print_section("STEP 4 — Evaluating Model Performance")
#     y_pred   = model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     print(f"  Overall Accuracy : {accuracy * 100:.1f}%")

#     cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
#     print(f"  Cross-Validation : {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%")

#     print("\n  Performance per risk level:")
#     report = classification_report(y_test, y_pred, output_dict=True)
#     for label in ["Low", "Medium", "High"]:
#         if label in report:
#             r = report[label]
#             print(f"    {label:8s} → Accuracy: {r['precision']*100:.0f}%  |  Recall: {r['recall']*100:.0f}%  |  Students tested: {int(r['support'])}")

#     # Save plots
#     os.makedirs(PLOTS_DIR, exist_ok=True)

#     fig, ax = plt.subplots(figsize=(6, 5))
#     cm = confusion_matrix(y_test, y_pred, labels=["Low", "Medium", "High"])
#     ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Low", "Medium", "High"]).plot(ax=ax, colorbar=False, cmap="Blues")
#     ax.set_title("Predicted vs Actual Risk Level", fontsize=12, fontweight="bold")
#     plt.tight_layout()
#     plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"), dpi=150)
#     plt.close()

#     importance_df = pd.DataFrame({
#         "Feature":    [FRIENDLY_NAMES.get(f, f) for f in features],
#         "Importance": model.feature_importances_,
#     }).sort_values("Importance", ascending=True)

#     fig, ax = plt.subplots(figsize=(8, 5))
#     colors = plt.cm.RdYlGn_r(importance_df["Importance"] / importance_df["Importance"].max())
#     ax.barh(importance_df["Feature"], importance_df["Importance"], color=colors, edgecolor="black", linewidth=0.6)
#     ax.set_xlabel("Influence on Prediction", fontweight="bold")
#     ax.set_title("What Drives Student Dropout Risk?", fontsize=13, fontweight="bold")
#     ax.grid(True, axis="x", alpha=0.2, linestyle="--")
#     plt.tight_layout()
#     plt.savefig(os.path.join(PLOTS_DIR, "feature_importance.png"), dpi=150)
#     plt.close()

#     print(f"\n  Charts saved → {PLOTS_DIR}")
#     print("\n  Top 3 factors driving dropout risk:")
#     for _, row in importance_df.sort_values("Importance", ascending=False).head(3).iterrows():
#         print(f"    → {row['Feature']} ({row['Importance']*100:.1f}% influence)")


# def save_model(model, features):
#     print_section("STEP 5 — Saving Model")
#     os.makedirs(MODEL_DIR, exist_ok=True)
#     joblib.dump(model,    MODEL_PATH)
#     joblib.dump(features, FEATURES_PATH)
#     print(f"  Model saved    → {MODEL_PATH}")
#     print(f"  Features saved → {FEATURES_PATH}")


# def main():
#     print("\n AUCA Student Dropout Prediction — Model Training")
#     print("Umoja Team\n")
#     try:
#         df = load_data()
#         X, y, features = prepare_data(df)
#         model, X_train, X_test, y_train, y_test = train(X, y)
#         evaluate(model, X, y, X_test, y_test, features)
#         save_model(model, features)
#         print("\n" + "="*55)
#         print("   Training complete!")
#         print("  Open the dashboard → Risk Prediction page")
#         print("  to start assessing students.")
#         print("="*55 + "\n")
#     except (FileNotFoundError, ValueError) as e:
#         print(f"\n Error:\n  {e}\n")
#         sys.exit(1)
#     except Exception as e:
#         print(f"\n Unexpected Error:\n  {e}\n")
#         raise


# if __name__ == "__main__":
#     main()
