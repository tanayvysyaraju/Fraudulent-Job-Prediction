import os
import joblib
import pandas as pd
import numpy as np

from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from imblearn.over_sampling import SMOTE


print("\n=== STARTING CLEAN RANDOM FOREST TRAINING (NO LEAKAGE) ===")

DATA_PATH = "data/RealAndFake_cleaned.csv"
df = pd.read_csv(DATA_PATH)
print("Loaded cleaned training dataset:", df.shape)


text_col = "text"
target_col = "fraudulent"

y = df[target_col]

structured_cols = [
    col for col in df.columns
    if col not in [text_col, target_col]
]

df_struct = df[structured_cols].copy()

# Convert object columns to categorical numeric
obj_cols = df_struct.select_dtypes(include="object").columns
for col in obj_cols:
    df_struct[col] = df_struct[col].astype("category").cat.codes

df_struct = df_struct.fillna(0)


# 2. Train/Test Split before SMOTE
print("\nSplitting dataset before SMOTE...")

X_text_raw = df[text_col]
X_struct_raw = df_struct

X_train_text, X_test_text, X_train_struct, X_test_struct, y_train, y_test = train_test_split(
    X_text_raw, X_struct_raw, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Train size:", y_train.shape[0])
print("Test size:", y_test.shape[0])


# TF-IDF vectorizer (fit ONLY on training text)
print("\nVectorizing text (TF-IDF)...")

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words="english"
)

X_train_text_vec = tfidf.fit_transform(X_train_text)
X_test_text_vec = tfidf.transform(X_test_text)

print("TF-IDF train shape:", X_train_text_vec.shape)
print("TF-IDF test shape:", X_test_text_vec.shape)


X_train_struct_sparse = sparse.csr_matrix(X_train_struct.values.astype(float))
X_test_struct_sparse = sparse.csr_matrix(X_test_struct.values.astype(float))


# Combine TF-IDF + Structured
X_train_full = sparse.hstack([X_train_text_vec, X_train_struct_sparse], format="csr")
X_test_full = sparse.hstack([X_test_text_vec, X_test_struct_sparse], format="csr")

print("\nCombined Train Matrix:", X_train_full.shape)
print("Combined Test Matrix:", X_test_full.shape)


# Apply SMOTE ONLY to training set 
print("\nApplying SMOTE to training set only...")

sm = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = sm.fit_resample(X_train_full, y_train)

print("Before SMOTE:", y_train.value_counts().to_dict())
print("After SMOTE:", pd.Series(y_train_balanced).value_counts().to_dict())


# Train Random Forest (clean + strong settings)
print("\nTraining Random Forest model...")

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=30,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features="sqrt",
    n_jobs=-1,
    random_state=42
)

rf.fit(X_train_balanced, y_train_balanced)

print("\n=== MODEL TRAINED SUCCESSFULLY ===")

# Evaluate performance

print("\nPredicting on test set (unseen, untouched)...")

y_pred = rf.predict(X_test_full)

print("\nTEST ACCURACY:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


SAVE_DIR = os.path.dirname(os.path.abspath(__file__)) 
os.makedirs(SAVE_DIR, exist_ok=True)

joblib.dump(rf, os.path.join(SAVE_DIR, "rf_real_fake_model.pkl"))
joblib.dump(tfidf, os.path.join(SAVE_DIR, "real_fake_tfidf.pkl"))
joblib.dump(structured_cols, os.path.join(SAVE_DIR, "real_fake_structured_cols.pkl"))

print("\nSaved model + vectorizer directly into RF_Model_On_RealFake/")
print(" - rf_real_fake_model.pkl")
print(" - real_fake_tfidf.pkl")
print(" - real_fake_structured_cols.pkl")

print("\n=== CLEAN TRAINING COMPLETE ===\n")