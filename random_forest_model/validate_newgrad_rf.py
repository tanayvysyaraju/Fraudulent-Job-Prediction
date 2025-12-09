import os
import joblib
import numpy as np
import pandas as pd

print("\n=== VALIDATING NEW GRAD JOBS WITH RANDOM FOREST  ===")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))    
ROOT_DIR = os.path.dirname(BASE_DIR)                      

FEATURE_DIR = os.path.join(ROOT_DIR, "feature_engineering")
DATA_DIR = os.path.join(ROOT_DIR, "data")


# load training-time artifacts
print("loading tf-idf, svd, model, and structured columns...")

tfidf = joblib.load(os.path.join(FEATURE_DIR, "tfidf_vectorizer.pkl"))
rf_model = joblib.load(os.path.join(BASE_DIR, "random_forest_all_features.pkl"))
svd = joblib.load(os.path.join(BASE_DIR, "rf_tfidf_svd_300.pkl"))

# use X_train just to recover which structured columns were used
X_train = joblib.load(os.path.join(FEATURE_DIR, "X_train.pkl"))
structured_cols = X_train.columns.to_list()
print(f"structured feature count expected: {len(structured_cols)}")


# 3. load new grad data
newgrad_path = os.path.join(DATA_DIR, "combined_newgrad_data.csv")
df = pd.read_csv(newgrad_path)

print("\nloaded combined_newgrad_data.csv")
print("shape:", df.shape)
print("columns:", df.columns.tolist())

# text processing
df["text"] = (
    df["Position Title"].astype(str)
    + " "
    + df["Qualifications"].astype(str)
    + " "
    + df["Company Industry"].astype(str)
)

df["text"] = df["text"].fillna("")

print("\nsample combined text:")
print(df["text"].head(3))


# tf-idf -> svd 
print("\ntransforming text with tf-idf...")
X_text_tfidf = tfidf.transform(df["text"])
print("tf-idf transformed text shape:", X_text_tfidf.shape)

print("applying truncated svd (same 300 components as training)...")
X_text_svd = svd.transform(X_text_tfidf)
print("svd-reduced text shape:", X_text_svd.shape)


# structured features 
df_struct = df.reindex(columns=structured_cols, fill_value=0).copy()

bool_cols = df_struct.select_dtypes(include="bool").columns
if len(bool_cols) > 0:
    df_struct[bool_cols] = df_struct[bool_cols].astype(np.int8)

df_struct = df_struct.fillna(0)

# dense numpy array, because training used dense stacking
X_struct_dense = df_struct.values.astype(float)
print("structured dense shape:", X_struct_dense.shape)

# combine features: [svd(tf-idf) | structured]
X_final = np.hstack([X_text_svd, X_struct_dense])
print("\nFINAL feature matrix shape:", X_final.shape)

# predict
preds = rf_model.predict(X_final)
df["rf_pred_fraud"] = preds

pred_counts = df["rf_pred_fraud"].value_counts().reindex([0, 1], fill_value=0)

summary_df = pd.DataFrame(
    {
        "Meaning": ["Real job posting", "Fraud job posting"],
        "Count": [pred_counts[0], pred_counts[1]],
    },
    index=["0", "1"],
)

print("\n=== Prediction Summary ===")
print(summary_df)

# summary by category

if "category" in df.columns:
    print("\ndistribution by NewGrad category:")
    print(df.groupby("category")["rf_pred_fraud"].value_counts())

# save results
output_path = os.path.join(BASE_DIR, "newgrad_rf_predictions.csv")
df.to_csv(output_path, index=False)

print(f"\nsaved predictions to: {output_path}")
print("\n DONE!\n")
