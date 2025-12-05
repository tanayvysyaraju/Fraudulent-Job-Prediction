import joblib
import numpy as np
import os
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

print("Loading saved engineered data...")

# Ensure working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# === LOAD SAVED MATRICES ===
tfidf = joblib.load("../feature_engineering/tfidf_vectorizer.pkl")

X_train = joblib.load("../feature_engineering/X_train.pkl")
X_test  = joblib.load("../feature_engineering/X_test.pkl")

X_train_tfidf = joblib.load("../feature_engineering/X_train_tfidf.pkl")
X_test_tfidf  = joblib.load("../feature_engineering/X_test_tfidf.pkl")

y_train = joblib.load("../feature_engineering/y_train.pkl")
y_test  = joblib.load("../feature_engineering/y_test.pkl")

print("Data loaded successfully.\n")

# === Convert booleans to numeric for sparse matrix ===
def to_numeric_sparse(df):
    df = df.copy()
    bool_cols = df.select_dtypes(include="bool").columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(np.int8)
    return sparse.csr_matrix(df.values), df.columns.to_numpy()

X_train_struct_sparse, struct_columns = to_numeric_sparse(X_train)
X_test_struct_sparse, _ = to_numeric_sparse(X_test)

# === Combine matrices ===
X_train_combined = sparse.hstack([X_train_tfidf, X_train_struct_sparse])
X_test_combined  = sparse.hstack([X_test_tfidf,  X_test_struct_sparse])

print("Final train shape:", X_train_combined.shape)
print("Final test shape :", X_test_combined.shape, "\n")

# === Define Random Forest (regularized) ===
rf = RandomForestClassifier(
    n_estimators=350,
    max_depth=40,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced_subsample",
    n_jobs=-1,
    oob_score=True,
    random_state=42
)

# === Cross-validation ===
print("Running 5-fold stratified cross-validation (F1 and Accuracy)...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_f1 = cross_val_score(rf, X_train_combined, y_train, cv=cv, scoring="f1")
cv_acc = cross_val_score(rf, X_train_combined, y_train, cv=cv, scoring="accuracy")

print("F1 scores:", cv_f1)
print(f"Mean F1: {cv_f1.mean():.4f} | Std: {cv_f1.std():.4f}\n")

print("Accuracy scores:", cv_acc)
print(f"Mean Acc: {cv_acc.mean():.4f} | Std: {cv_acc.std():.4f}\n")

# === Train final model ===
print("Training final Random Forest model...")
rf.fit(X_train_combined, y_train)
print("Model trained.\n")

print("OOB Score:", rf.oob_score_)

# === Evaluate on test set ===
print("Evaluating on held-out test set...")
y_pred = rf.predict(X_test_combined)

print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Random Forest – Confusion Matrix (Test Set)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("rf_confusion_matrix.png")
plt.close()
print("Confusion matrix saved as rf_confusion_matrix.png")

# === Feature importances ===
print("\nExtracting feature importances...")

importances = rf.feature_importances_
n_tfidf = X_train_tfidf.shape[1]

text_importances = importances[:n_tfidf]
struct_importances = importances[n_tfidf:]

feature_names_text = tfidf.get_feature_names_out()
top_text_idx = np.argsort(text_importances)[-25:][::-1]

print("\nTop TF-IDF Keywords Indicating FRAUD:")
for idx in top_text_idx:
    print(f"- {feature_names_text[idx]} ({text_importances[idx]:.6f})")

top_struct_idx = np.argsort(struct_importances)[-15:][::-1]

print("\nTop Structured Features:")
for idx in top_struct_idx:
    print(f"- {struct_columns[idx]} ({struct_importances[idx]:.6f})")

joblib.dump(rf, "random_forest_all_features.pkl")
print("\nModel saved as random_forest_all_features.pkl")