import joblib
import numpy as np
import os
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    average_precision_score,
    precision_recall_curve,
    recall_score
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

# adding truncated svd
from sklearn.decomposition import TruncatedSVD

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
# before combining, apply svd reduction on tf-idf (xgboost-style dimensionality reduction)
print("applying truncated svd (n_components=300) to tf-idf features...")

svd = TruncatedSVD(n_components=300, random_state=42)
X_train_svd = svd.fit_transform(X_train_tfidf)
X_test_svd = svd.transform(X_test_tfidf)

# convert the structured features back to dense
X_train_struct_dense = X_train_struct_sparse.toarray()
X_test_struct_dense = X_test_struct_sparse.toarray()

# combine reduced tf-idf + structured features
X_train_combined = np.hstack([X_train_svd, X_train_struct_dense])
X_test_combined  = np.hstack([X_test_svd,  X_test_struct_dense])

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
print("Running 5-fold stratified cross-validation (F1, Accuracy, PR-AUC)...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_f1 = cross_val_score(rf, X_train_combined, y_train, cv=cv, scoring="f1")
cv_acc = cross_val_score(rf, X_train_combined, y_train, cv=cv, scoring="accuracy")
cv_pr_auc = cross_val_score(rf, X_train_combined, y_train, cv=cv, scoring="average_precision")

print("F1 scores:", cv_f1)
print(f"Mean F1: {cv_f1.mean():.4f} | Std: {cv_f1.std():.4f}\n")

print("Accuracy scores:", cv_acc)
print(f"Mean Acc: {cv_acc.mean():.4f} | Std: {cv_acc.std():.4f}\n")

print("PR-AUC scores:", cv_pr_auc)
print(f"Mean PR-AUC: {cv_pr_auc.mean():.4f} | Std: {cv_pr_auc.std():.4f}\n")

# === Train final model ===
print("Training final Random Forest model...")
rf.fit(X_train_combined, y_train)
print("Model trained.\n")

print("OOB Score:", rf.oob_score_)

# === Evaluate on test set ===
print("Evaluating on held-out test set...")
y_pred = rf.predict(X_test_combined)
y_proba = rf.predict_proba(X_test_combined)[:, 1]

print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# test pr-auc
test_pr_auc = average_precision_score(y_test, y_proba)
print("Test PR-AUC (Average Precision):", test_pr_auc)

# recall on fraudulent class
fraud_recall = recall_score(y_test, y_pred, pos_label=1)
print("Fraud Recall (class 1 recall):", fraud_recall)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Random Forest (SVD) – Confusion Matrix (Test Set)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("rf_svd_confusion_matrix.png")
plt.close()
print("Confusion matrix saved as rf_svd_confusion_matrix.png")

# === Precision-Recall Curve ===
precision, recall, _ = precision_recall_curve(y_test, y_proba)

plt.figure(figsize=(6,4))
plt.plot(recall, precision)
plt.title("Random Forest (SVD) – Precision–Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.tight_layout()
plt.savefig("rf_svd_pr_curve.png")
plt.close()
print("Precision-Recall curve saved as rf_svd_pr_curve.png")

# === Feature importances ===
print("\nExtracting feature importances...")

importances = rf.feature_importances_

# model now uses svd instead of raw tf-idf so text importance is based on svd components
n_svd = X_train_svd.shape[1]
text_importances = importances[:n_svd]
struct_importances = importances[n_svd:]

top_svd_idx = np.argsort(text_importances)[-20:][::-1]

print("\nTop SVD Components Contributing to Fraud Predictions:")
for idx in top_svd_idx:
    print(f"- SVD component {idx} ({text_importances[idx]:.6f})")

top_struct_idx = np.argsort(struct_importances)[-15:][::-1]

print("\nTop Structured Features:")
for idx in top_struct_idx:
    print(f"- {struct_columns[idx]} ({struct_importances[idx]:.6f})")

# save model + svd
joblib.dump(rf, "random_forest_all_features.pkl")
joblib.dump(svd, "rf_tfidf_svd_300.pkl")

print("\nModel saved as random_forest_all_features.pkl")
print("SVD saved as rf_tfidf_svd_300.pkl")
