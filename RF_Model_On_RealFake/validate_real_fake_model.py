import os
import joblib
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    auc,
    recall_score
)

print("\n=== STARTING VALIDATION ON FAKEPOSTINGS DATASET ===")

# ---------------------------------------------------------
# paths
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # RF_Model_On_RealFake folder
MODEL_DIR = BASE_DIR                                   # pkls are saved directly here

DATA_PATH = os.path.join("data", "FakePostings_cleaned.csv")   # must be cleaned version
df = pd.read_csv(DATA_PATH)

print("Loaded FakePostings dataset:", df.shape)

# ---------------------------------------------------------
# expected artifacts
# ---------------------------------------------------------
model_path = os.path.join(MODEL_DIR, "rf_real_fake_model.pkl")
tfidf_path = os.path.join(MODEL_DIR, "real_fake_tfidf.pkl")
cols_path = os.path.join(MODEL_DIR, "real_fake_structured_cols.pkl")
svd_path = os.path.join(MODEL_DIR, "real_fake_svd.pkl")   # added: svd from training

rf = joblib.load(model_path)
tfidf = joblib.load(tfidf_path)
structured_cols = joblib.load(cols_path)
svd = joblib.load(svd_path)

print("\nLoaded model + TF-IDF + structured columns + SVD")

# ---------------------------------------------------------
# 1. text processing
# ---------------------------------------------------------
df["text"] = df["text"].fillna("unknown").astype(str)

# tf-idf transform using the same vectorizer
X_text_tfidf = tfidf.transform(df["text"])

# reduce tf-idf with svd to match training features
X_text_svd = svd.transform(X_text_tfidf)

print("TF-IDF shape:", X_text_tfidf.shape)
print("SVD-reduced TF-IDF shape:", X_text_svd.shape)

# ---------------------------------------------------------
# 2. structured features
# ---------------------------------------------------------
df_struct = df.reindex(columns=structured_cols, fill_value="unknown").copy()

# convert objects → category codes (same as training)
obj_cols = df_struct.select_dtypes(include="object").columns
for col in obj_cols:
    df_struct[col] = df_struct[col].astype("category").cat.codes

df_struct = df_struct.fillna(0)
X_struct = sparse.csr_matrix(df_struct.values.astype(float))

print("Structured shape:", X_struct.shape)

# ---------------------------------------------------------
# 3. combine features
# ---------------------------------------------------------
X_final = sparse.hstack([X_text_svd, X_struct], format="csr")
print("\nCombined feature matrix:", X_final.shape)

# ---------------------------------------------------------
# 4. true labels (all 1s in fakepostings)
# ---------------------------------------------------------
true_labels = df["fraudulent"]
print("\nTrue label distribution (should be all 1):")
print(true_labels.value_counts())

# ---------------------------------------------------------
# 5. predict
# ---------------------------------------------------------
preds = rf.predict(X_final)
proba = rf.predict_proba(X_final)[:, 1]
df["predicted_fraud"] = preds

print("\nPrediction Distribution (Real = 0, Fraud = 1):")
print(df["predicted_fraud"].value_counts())

# ---------------------------------------------------------
# 6. normal accuracy (will likely be low — expected)
# ---------------------------------------------------------
accuracy = accuracy_score(true_labels, preds)
print("\nValidation Accuracy (normal accuracy metric):", accuracy)

# ---------------------------------------------------------
# 7. fraud detection recall + pr-auc
# ---------------------------------------------------------
cm = confusion_matrix(true_labels, preds)

fraud_recall = recall_score(true_labels, preds, zero_division=0)

precision, recall, thresholds = precision_recall_curve(true_labels, proba)
pr_auc = auc(recall, precision)

print("\nFRAUD Recall (how many frauds were caught):", fraud_recall)
print("PR-AUC Score:", pr_auc)

# ---------------------------------------------------------
# 8. full classification report
# ---------------------------------------------------------
print("\nClassification Report:")
print(classification_report(true_labels, preds, zero_division=0))

# ---------------------------------------------------------
# 9. confusion matrix
# ---------------------------------------------------------
print("\nConfusion Matrix (rows = true, cols = predicted):")
print(cm)

# ---------------------------------------------------------
# 10. save predictions
# ---------------------------------------------------------
output_path = os.path.join(MODEL_DIR, "model_real_fake_validation_results_fakepostings.csv")
df.to_csv(output_path, index=False)

print("\nSaved predictions to:", output_path)
print("\n=== VALIDATION COMPLETE ===\n")
