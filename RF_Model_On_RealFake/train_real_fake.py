import os
import joblib
import pandas as pd
import numpy as np

from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# =============================================================
#                TRAINING ON CLEANED REAL/FAKE DATA
# =============================================================

print("\n=== STARTING RANDOM FOREST TRAINING ===")

DATA_PATH = "data/RealAndFake_cleaned.csv"
df = pd.read_csv(DATA_PATH)
print("Loaded cleaned training dataset:", df.shape)

# -------------------------------------------------------------
# 1. Split into text + structured features
# -------------------------------------------------------------
text_col = "text"
target_col = "fraudulent"

y = df[target_col]

# Structured = all columns EXCEPT text + target
structured_cols = [
    col for col in df.columns
    if col not in [text_col, target_col]
]

df_struct = df[structured_cols].copy()

# Convert object columns → categorical numeric
obj_cols = df_struct.select_dtypes(include="object").columns
for col in obj_cols:
    df_struct[col] = df_struct[col].astype("category").cat.codes

df_struct = df_struct.fillna(0)
X_struct = sparse.csr_matrix(df_struct.values.astype(float))

# -------------------------------------------------------------
# 2. TF-IDF vectorization
# -------------------------------------------------------------
print("\nVectorizing text with TF-IDF...")

tfidf = TfidfVectorizer(
    max_features=5000,     # avoids huge feature space
    ngram_range=(1,2),     # captures phrases → better performance
    stop_words="english"
)

X_text = tfidf.fit_transform(df[text_col])

print("TF-IDF shape:", X_text.shape)
print("Structured shape:", X_struct.shape)

# -------------------------------------------------------------
# 3. Combine features
# -------------------------------------------------------------
X_full = sparse.hstack([X_text, X_struct], format="csr")
print("\nCombined feature matrix:", X_full.shape)

# -------------------------------------------------------------
# 4. Train-test split
# -------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_full, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# -------------------------------------------------------------
# 5. Random Forest (anti-overfitting settings)
# -------------------------------------------------------------
print("\nTraining Random Forest...")

rf = RandomForestClassifier(
    n_estimators=300,          # strong but not huge
    max_depth=25,              # keeps trees from memorizing
    min_samples_split=20,
    min_samples_leaf=10,
    max_features="sqrt",       # recommended for high-dimensional TF-IDF
    n_jobs=-1,
    random_state=42
)

rf.fit(X_train, y_train)

print("\n=== MODEL TRAINED ===")

# -------------------------------------------------------------
# 6. Evaluate on Test Set
# -------------------------------------------------------------
print("\nPredicting on test set...")
y_pred = rf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("\nTEST ACCURACY:", acc)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# -------------------------------------------------------------
# 7. SAVE ARTIFACTS
# -------------------------------------------------------------
SAVE_DIR = "model_real_fake"
os.makedirs(SAVE_DIR, exist_ok=True)

joblib.dump(rf, f"{SAVE_DIR}/rf_real_fake_model.pkl")
joblib.dump(tfidf, f"{SAVE_DIR}/real_fake_tfidf.pkl")
joblib.dump(structured_cols, f"{SAVE_DIR}/real_fake_structured_cols.pkl")

print("\nSaved model and vectorizer to /model_real_fake/")
print(" - rf_real_fake_model.pkl")
print(" - real_fake_tfidf.pkl")
print(" - real_fake_structured_cols.pkl")

print("\n=== TRAINING COMPLETE ===\n")

