import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack

# ----------------------------------------------------
# 1. Load trained model, vectorizer, and feature list
# ----------------------------------------------------
rf_clf = joblib.load("random_forest_model/random_forest_all_features.pkl")
tfidf = joblib.load("feature_engineering/tfidf_vectorizer.pkl")
structured_cols = joblib.load("feature_engineering/structured_feature_columns.pkl")


print("Loaded RF model + TF-IDF + structured columns.")
print("Structured features expected:", len(structured_cols))

# ----------------------------------------------------
# 2. Load the combined NEW GRAD dataset
# ----------------------------------------------------
df = pd.read_csv("combined_newgrad_data.csv")

print("\nLoaded combined_newgrad_data.csv")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# ----------------------------------------------------
# 3. Build the same TEXT column used in training
# ----------------------------------------------------
def make_text(row):
    parts = [
        row.get("Position Title", ""),
        row.get("Qualifications", ""),
        row.get("Work Model", ""),
        row.get("Location", "")
    ]
    return " ".join(str(p) for p in parts if pd.notna(p) and str(p).strip() != "")

df["text"] = df.apply(make_text, axis=1)
df["location"] = df["Location"].fillna("")

print("\nSample combined text:")
print(df["text"].head(3))

# ----------------------------------------------------
# 4. Transform TEXT using saved TF-IDF
# ----------------------------------------------------
X_text = tfidf.transform(df["text"])
print("\nTF-IDF transformed text shape:", X_text.shape)

# ----------------------------------------------------
# 5. Build STRUCTURED FEATURES with correct columns
# ----------------------------------------------------
X_struct = pd.DataFrame(
    0, index=df.index, columns=structured_cols, dtype=float
)

# Map your newgrad columns → model columns if names overlap
column_map = {
    "category": "category",
    "h1b_sponsored": "H1b Sponsored",
    "is_new_grad": "Is New Grad",
}

for model_col, newgrad_col in column_map.items():
    if model_col in X_struct.columns and newgrad_col in df.columns:
        tmp = df[newgrad_col].copy()
        tmp = tmp.replace({"Yes": 1, "No": 0, "Y": 1, "N": 0})
        X_struct[model_col] = pd.to_numeric(tmp, errors="coerce").fillna(0)
        print(f"Filled structured feature: {model_col} from column {newgrad_col}")

X_struct_sparse = csr_matrix(X_struct.to_numpy(dtype=float))

# ----------------------------------------------------
# 6. Combine TF-IDF + structured features
# ----------------------------------------------------
X_final = hstack([X_text, X_struct_sparse], format="csr")
print("\nFINAL feature matrix shape:", X_final.shape)

# ----------------------------------------------------
# 7. Predict with Random Forest
# ----------------------------------------------------
df["rf_pred_fraud"] = rf_clf.predict(X_final)
df["rf_fraud_probability"] = rf_clf.predict_proba(X_final)[:, 1]

print("\nPrediction distribution (0=real, 1=fraud):")
print(df["rf_pred_fraud"].value_counts())

if "category" in df.columns:
    print("\nDistribution BY CATEGORY:")
    print(df.groupby(["category", "rf_pred_fraud"]).size().unstack(fill_value=0))

# ----------------------------------------------------
# 8. SAVE RESULTS
# ----------------------------------------------------
output_path = "newgrad_rf_predictions.csv"
df.to_csv(output_path, index=False)

print(f"\nSaved results to: {output_path}")
