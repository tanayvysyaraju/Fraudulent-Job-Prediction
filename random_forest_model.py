import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score


# ----------------------------------------------------
# 1. Load engineered train/test data
# ----------------------------------------------------
print("Loading engineered train/test data...")

train_full = pd.read_csv("data/train_engineered.csv")
test_full  = pd.read_csv("data/test_engineered.csv")

print("Train shape (with label):", train_full.shape)
print("Test shape  (with label):", test_full.shape)

# ----------------------------------------------------
# 2. Separate X and y
# ----------------------------------------------------
y_train = train_full["fraudulent"].values
y_test  = test_full["fraudulent"].values

X_train_df = train_full.drop(columns=["fraudulent"])
X_test_df  = test_full.drop(columns=["fraudulent"])

print("\nFeature columns:", X_train_df.columns.tolist())
print("Number of features (including text):", X_train_df.shape[1])

# ----------------------------------------------------
# 3. TF-IDF on the 'text' column
# ----------------------------------------------------
print("\nVectorizing text with TF-IDF on training set only...")

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words="english",
    min_df=5
)

tfidf.fit(X_train_df["text"])

X_train_text = tfidf.transform(X_train_df["text"])
X_test_text  = tfidf.transform(X_test_df["text"])

print("TF-IDF train shape:", X_train_text.shape)
print("TF-IDF test shape :", X_test_text.shape)

# ----------------------------------------------------
# 4. Structured features (everything EXCEPT 'text' and raw 'location')
# ----------------------------------------------------
# Drop raw text + raw location; keep all engineered features
X_train_struct = X_train_df.drop(columns=["text", "location"])
X_test_struct  = X_test_df.drop(columns=["text", "location"])

# Ensure columns align between train and test
X_test_struct = X_test_struct.reindex(columns=X_train_struct.columns, fill_value=0)

print("\nStructured feature shape (train):", X_train_struct.shape)
print("Structured feature shape (test) :", X_test_struct.shape)

# Coerce any non-numeric columns to numeric and fill NaNs with 0
X_train_struct = X_train_struct.apply(pd.to_numeric, errors="coerce").fillna(0)
X_test_struct  = X_test_struct.apply(pd.to_numeric, errors="coerce").fillna(0)

# Sanity check: make sure no object columns remain
obj_cols = X_train_struct.select_dtypes(include="object").columns.tolist()
if obj_cols:
    print("\n[WARNING] The following structured columns are still object dtype and will be dropped:", obj_cols)
    X_train_struct = X_train_struct.drop(columns=obj_cols)
    X_test_struct  = X_test_struct.drop(columns=obj_cols)

print("\nDtypes of structured features (train):")
print(X_train_struct.dtypes)

# Convert structured features to sparse so we can hstack with TF-IDF
X_train_struct_sparse = csr_matrix(X_train_struct.to_numpy(dtype=float))
X_test_struct_sparse  = csr_matrix(X_test_struct.to_numpy(dtype=float))

# ----------------------------------------------------
# 5. Combine: [TF-IDF | structured]
# ----------------------------------------------------
X_train_all = hstack([X_train_text, X_train_struct_sparse], format="csr")
X_test_all  = hstack([X_test_text,  X_test_struct_sparse],  format="csr")

print("\nCombined train shape:", X_train_all.shape)
print("Combined test shape :", X_test_all.shape)

# ----------------------------------------------------
# 6. Define Random Forest model
# ----------------------------------------------------
rf_clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    n_jobs=-1,
    class_weight="balanced",
    random_state=42
)

# ----------------------------------------------------
# 7. K-Fold Cross-Validation (on ALL features)
# ----------------------------------------------------
print("\nRunning 5-fold stratified cross-validation on training set (all features)...")

k = 5
cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

cv_f1_scores = cross_val_score(
    rf_clf,
    X_train_all,
    y_train,
    cv=cv,
    scoring="f1"
)

cv_acc_scores = cross_val_score(
    rf_clf,
    X_train_all,
    y_train,
    cv=cv,
    scoring="accuracy"
)

print(f"Cross-validated F1 scores:       {cv_f1_scores}")
print(f"Mean F1: {cv_f1_scores.mean():.4f} | Std: {cv_f1_scores.std():.4f}")

print(f"\nCross-validated Accuracy scores: {cv_acc_scores}")
print(f"Mean Acc: {cv_acc_scores.mean():.4f} | Std: {cv_acc_scores.std():.4f}\n")

# ----------------------------------------------------
# 8. Train final model on full training data (ALL features)
# ----------------------------------------------------
print("Training final Random Forest model on full training data (all features)...")

rf_clf.fit(X_train_all, y_train)

print("Final Random Forest model trained.\n")

# ----------------------------------------------------
# 9. Evaluate on held-out test set
# ----------------------------------------------------
print("Evaluating Random Forest on HELD-OUT test set (all features)...")

y_pred = rf_clf.predict(X_test_all)

print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
print("\nTest Classification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Random Forest – Confusion Matrix (Test Set, All Features)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("rf_confusion_matrix_all_features.png")
plt.close()

print("Confusion matrix saved as rf_confusion_matrix_all_features.png\n")

# ----------------------------------------------------
# 10. Feature importance analysis
# ----------------------------------------------------
print("Analyzing feature importances...")

importances = rf_clf.feature_importances_
n_text_features = X_train_text.shape[1]

# Split importance vector into text vs structured parts
text_importances = importances[:n_text_features]
struct_importances = importances[n_text_features:]

# --- Top TF-IDF keywords ---
feature_names_text = tfidf.get_feature_names_out()
top_text_idx = np.argsort(text_importances)[-30:][::-1]

print("\nTop 30 Most Important TF-IDF Keywords:")
for i in top_text_idx:
    print(f"- {feature_names_text[i]} (importance={text_importances[i]:.6f})")

# --- Top structured features ---
struct_feature_names = X_train_struct.columns.to_numpy()
top_struct_idx = np.argsort(struct_importances)[-20:][::-1]

print("\nTop 20 Most Important Structured Features:")
for i in top_struct_idx:
    print(f"- {struct_feature_names[i]} (importance={struct_importances[i]:.6f})")

# ----------------------------------------------------
# 11. Save trained model + vectorizer + structured columns
# ----------------------------------------------------
joblib.dump(rf_clf, "random_forest_all_features.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")  # this will overwrite the old one
joblib.dump(struct_feature_names.tolist(), "structured_feature_columns.pkl")

print("\nSaved:")
print("- random_forest_all_features.pkl")
print("- tfidf_vectorizer.pkl")
print("- structured_feature_columns.pkl")