import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# STEP 1: Dataset (heart.csv)
# ============================================================
df = pd.read_csv("dataset/heart.csv")

print("===== STEP 1: DATASET =====")
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nAll columns:", list(df.columns))
print("\nMissing values:")
print(df.isnull().sum())

# ============================================================
# STEP 2: Feature engineering
# Original features are kept. New useful features are added.
# ============================================================

# Age groups: 0 = young, 1 = middle, 2 = senior, 3 = elderly
df["age_group"] = pd.cut(
    df["age"],
    bins=[0, 40, 55, 65, 120],
    labels=[0, 1, 2, 3],
).astype("float")

# Cholesterol level: 0 = normal, 1 = borderline, 2 = high
df["chol_level"] = pd.cut(
    df["chol"],
    bins=[0, 200, 239, 1000],
    labels=[0, 1, 2],
).astype("float")

# Resting blood pressure level: 0 = normal, 1 = elevated, 2 = high
df["bp_level"] = pd.cut(
    df["trestbps"],
    bins=[0, 120, 139, 300],
    labels=[0, 1, 2],
).astype("float")

# How close the max heart rate is to the age-predicted max
df["max_hr_ratio"] = df["thalach"] / (220 - df["age"])

# All original columns except target, plus the new engineered features
X = df.drop("target", axis=1)
y = df["target"]

print("\n===== STEP 2: FEATURE ENGINEERING =====")
print("New features added: age_group, chol_level, bp_level, max_hr_ratio")
print("Total features used:", X.shape[1])
print("Feature names:")
print(list(X.columns))
print("\nSample after feature engineering:")
print(X[["age", "age_group", "chol", "chol_level", "trestbps", "bp_level", "thalach", "max_hr_ratio"]].head())

# ============================================================
# STEP 3: Data preprocessing
# Missing values are filled, then features are scaled.
# ============================================================
imputer = SimpleImputer(strategy="median")
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns)

print("\n===== STEP 3: DATA PREPROCESSING =====")
print("Missing values after imputation:")
print(X_imputed.isnull().sum().sum())
print("Features scaled with StandardScaler.")
print("\nSample after scaling:")
print(X_scaled.head())

# ============================================================
# STEP 4: Train / Test split
# Split is done after dataset + feature engineering + preprocessing.
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\n===== STEP 4: TRAIN / TEST SPLIT =====")
print("Train size:", X_train.shape[0])
print("Test size :", X_test.shape[0])

# ============================================================
# STEP 5: ML Model
# ============================================================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("\n===== STEP 5: ML MODEL =====")
print("Model: Logistic Regression")

# ============================================================
# STEP 6: Prediction + Accuracy
# ============================================================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n===== STEP 6: PREDICTION (ACCURACY) =====")
print(f"Accuracy : {accuracy:.4f}")

# ============================================================
# STEP 7: F1-score / Precision
# ============================================================
precision = precision_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n===== STEP 7: F1-SCORE / PRECISION =====")
print(f"Precision: {precision:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=["Predicted 0", "Predicted 1"],
    yticklabels=["Actual 0", "Actual 1"],
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png", bbox_inches="tight")
plt.close()
print("\nConfusion matrix image saved as confusion_matrix.png")
