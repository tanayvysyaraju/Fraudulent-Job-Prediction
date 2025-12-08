import os
import joblib
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

print("\n=== STARTING VALIDATION ON FAKEPOSTINGS DATASET ===")

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # RF_Model_On_RealFake/
DATA_PATH = os.path.join("data", "FakePostings_cleaned.csv")

# Model artifacts saved directly inside RF_Model_On_RealFake/
model_path = os.path.join(BASE_DIR, "rf_real_fake_model.pkl")
tfidf_path = os.path.join(BASE_DIR, "real_fake_tfidf.pkl")
cols_path = os.path.join(BASE_DIR, "real_fake_structured_cols.pkl")

# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print("Loaded FakePostings dataset:", df.shape)

# ---------------------------------------------------------
# Load model + preprocessors
# ---------------------------------------------------------
rf = joblib.load(model_path)
tfidf = joblib.load(tfidf_path)
structured_cols = joblib.load(cols_path)

print("\nLoaded model + TF-IDF + structured columns")

# ---------------------------------------------------------
# 1. TEXT PROCESSING
# ---------------------------------------------------------
df["text"] = df["text"].fillna("unknown").astype(str)
X_text = tfidf.transform(df["text"])
print("TF-IDF shape:", X_text.shape)

# ---------------------------------------------------------
# 2. STRUCTURED FEATURES
# ---------------------------------------------------------
df_struct = df.reindex(columns=structured_cols, fill_value="unknown").copy()

# Convert objects → category codes
obj_cols = df_struct.select_dtypes(include="object").columns
for col in obj_cols:
    df_struct[col] = df_struct[col].astype("category").cat.codes

df_struct = df_struct.fillna(0)
X_struct = sparse.csr_matrix(df_struct.values.astype(float))

print("Structured shape:", X_struct.shape)

# ---------------------------------------------------------
# 3. Combine features
# ---------------------------------------------------------
X_final = sparse.hstack([X_text, X_struct], format="csr")
print("\nCombined feature matrix:", X_final.shape)

# ---------------------------------------------------------
# TRUE LABELS (all 1s)
# ---------------------------------------------------------
true_labels = df["fraudulent"].astype(int)
print("\nTrue label distribution (should be all 1s):")
print(true_labels.value_counts())

# ---------------------------------------------------------
# 4. Predict
# ---------------------------------------------------------
preds = rf.predict(X_final)
df["predicted_fraud"] = preds

print("\nPrediction Distribution (Real = 0, Fraud = 1):")
print(df["predicted_fraud"].value_counts())

# ---------------------------------------------------------
# 5. Confusion Matrix
# ---------------------------------------------------------
cm = confusion_matrix(true_labels, preds)
print("\nConfusion Matrix (rows = TRUE, cols = PREDICTED):")
print(cm)

# Extract TP / FN correctly for "all fraud" dataset
# True labels: always 1
# → row 1 contains counts for true fraud
if cm.shape == (2, 2):
    tn, fp, fn, tp = cm.ravel()
else:
    # Only one class appears
    tp = cm[0][0] if preds[0] == 1 else 0
    fn = 0

# ---------------------------------------------------------
# 6. Metrics
# ---------------------------------------------------------
accuracy = accuracy_score(true_labels, preds)
print("\nValidation Accuracy:", accuracy)

fraud_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
print("Fraud Recall (important):", fraud_recall)

print("\nClassification Report:")
print(classification_report(true_labels, preds, zero_division=0))

# ---------------------------------------------------------
# 7. Save results
# ---------------------------------------------------------
output_path = os.path.join(BASE_DIR, "model_real_fake_validation_results_fakepostings.csv")
df.to_csv(output_path, index=False)

print("\nSaved predictions to:", output_path)
print("\n=== VALIDATION COMPLETE ===\n")
