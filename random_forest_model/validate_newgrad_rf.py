import os
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

print("\n=== VALIDATING NEW GRAD JOBS WITH RANDOM FOREST ===")

# ------------------------------------------------------------------------------
# 1. SET WORKING DIRECTORY
# ------------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)   # project root

FEATURE_DIR = os.path.join(ROOT_DIR, "feature_engineering")
DATA_DIR = os.path.join(ROOT_DIR, "data")

# ------------------------------------------------------------------------------
# 2. LOAD TRAINING-TIME ARTIFACTS
# ------------------------------------------------------------------------------

print("Loading TF-IDF, model, and structured columns...")

tfidf = joblib.load(os.path.join(FEATURE_DIR, "tfidf_vectorizer.pkl"))
rf_model = joblib.load(os.path.join(BASE_DIR, "random_forest_all_features.pkl"))

# We need X_train to know which structured features were used during training
X_train = joblib.load(os.path.join(FEATURE_DIR, "X_train.pkl"))
structured_cols = X_train.columns.to_list()

print(f"Structured feature count expected: {len(structured_cols)}")

# ------------------------------------------------------------------------------
# 3. LOAD NEW GRAD DATA
# ------------------------------------------------------------------------------
newgrad_path = os.path.join(DATA_DIR, "combined_newgrad_data.csv")
df = pd.read_csv(newgrad_path)

print("\nLoaded combined_newgrad_data.csv")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# ------------------------------------------------------------------------------
# 4. TEXT PROCESSING
# ------------------------------------------------------------------------------
df["text"] = (
    df["Position Title"].astype(str)
    + " "
    + df["Qualifications"].astype(str)
    + " "
    + df["Company Industry"].astype(str)
)

# fill missing text
df["text"] = df["text"].fillna("")

print("\nSample combined text:")
print(df["text"].head(3))

# ------------------------------------------------------------------------------
# 5. APPLY TF-IDF
# ------------------------------------------------------------------------------
print("\nTransforming text with TF-IDF...")
X_text = tfidf.transform(df["text"])

print("TF-IDF transformed text shape:", X_text.shape)

# ------------------------------------------------------------------------------
# 6. STRUCTURED FEATURES 
# ------------------------------------------------------------------------------

df_struct = df.reindex(columns=structured_cols, fill_value=0)

# Convert booleans to numeric (same as training)
bool_cols = df_struct.select_dtypes(include="bool").columns
if len(bool_cols) > 0:
    df_struct[bool_cols] = df_struct[bool_cols].astype(np.int8)

df_struct = df_struct.fillna(0)

X_struct_sparse = sparse.csr_matrix(df_struct.values.astype(float))

# ------------------------------------------------------------------------------
# 7. COMBINE FEATURES: [TF-IDF | STRUCTURED]
# ------------------------------------------------------------------------------

X_final = sparse.hstack([X_text, X_struct_sparse], format="csr")

print("\nFINAL feature matrix shape:", X_final.shape)

# ------------------------------------------------------------------------------
# 8. PREDICT
# ------------------------------------------------------------------------------

preds = rf_model.predict(X_final)

df["rf_pred_fraud"] = preds

pred_counts = df["rf_pred_fraud"].value_counts().reindex([0,1], fill_value=0)

summary_df = pd.DataFrame({
    "Meaning": ["Real job posting", "Fraud job posting"],
    "Count": [pred_counts[0], pred_counts[1]]
}, index=["0", "1"])

print("\n=== Prediction Summary ===")
print(summary_df)

# ------------------------------------------------------------------------------
# 9. SUMMARY BY CATEGORY
# ------------------------------------------------------------------------------

if "category" in df.columns:
    print("\nDistribution by NewGrad category:")
    print(df.groupby("category")["rf_pred_fraud"].value_counts())

# ------------------------------------------------------------------------------
# 10. SAVE RESULTS
# ------------------------------------------------------------------------------

output_path = os.path.join(BASE_DIR, "newgrad_rf_predictions.csv")
df.to_csv(output_path, index=False)

print(f"\nSaved predictions to: {output_path}")
print("\n== DONE! ==")
