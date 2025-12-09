import os
import joblib
import numpy as np
import pandas as pd
from scipy import sparse

print("\n=== VALIDATING NEW GRAD JOBS WITH RANDOM FOREST  ===")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))    
ROOT_DIR = os.path.dirname(BASE_DIR)                      

FEATURE_DIR = os.path.join(ROOT_DIR, "feature_engineering")
DATA_DIR = os.path.join(ROOT_DIR, "data")


# load training-time artifacts
print("loading tf-idf, svd, model, and structured columns...")

rf_model = joblib.load(os.path.join(BASE_DIR, "random_forest_all_features.pkl"))

X_scraped_tfidf = joblib.load(os.path.join(BASE_DIR,"../feature_engineering/scraped_tfidf_matrix.pkl"))  # sparse matrix
X_scraped_struct = joblib.load(os.path.join(BASE_DIR,"../feature_engineering/scraped_feature_list.pkl")) # DataFrame 

tfidf = joblib.load(os.path.join(BASE_DIR, "../feature_engineering/tfidf_vectorizer.pkl"))
svd = joblib.load(os.path.join(BASE_DIR, "../XGboost_model_folder/tfidf_svd.pkl"))

#need to transform with same svd as tested on
X_scraped_tfidf_svd = svd.transform(X_scraped_tfidf)

#dont think we need this method but keeping in case it breaks
def to_numeric_sparse(df):
    df = df.copy()
    bool_cols = df.select_dtypes(include="bool").columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(np.int8)
    return sparse.csr_matrix(df.values)

X_scraped_struct_sparse = to_numeric_sparse(X_scraped_struct)

# Convert SVD output to sparse (optional)
X_scraped_tfidf_sparse = sparse.csr_matrix(X_scraped_tfidf_svd)

# Combine SVD + structured features
X_scraped_combined = sparse.hstack([X_scraped_tfidf_sparse, X_scraped_struct_sparse])

#end rksiha
# predict
preds = rf_model.predict(X_scraped_combined)
X_scraped_struct["rf_pred_fraud"] = preds

pred_counts = X_scraped_struct["rf_pred_fraud"].value_counts().reindex([0, 1], fill_value=0)

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

if "category" in X_scraped_struct.columns:
    print("\ndistribution by NewGrad category:")
    print(X_scraped_struct.groupby("category")["rf_pred_fraud"].value_counts())

# save results
output_path = os.path.join(BASE_DIR, "newgrad_rf_predictions.csv")
X_scraped_struct.to_csv(output_path, index=False)

print(f"\nsaved predictions to: {output_path}")
print("\n DONE!\n")
