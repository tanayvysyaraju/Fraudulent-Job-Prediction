import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from scipy import sparse

from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    average_precision_score,
    precision_recall_curve,
    recall_score,
)

print("Loading TF-IDF and structured feature data...")


#Load saved data 
os.chdir(os.path.dirname(os.path.abspath(__file__)))

tfidf = joblib.load("../feature_engineering/tfidf_vectorizer.pkl")
X_train = joblib.load("../feature_engineering/X_train.pkl")
X_test = joblib.load("../feature_engineering/X_test.pkl")
X_train_tfidf = joblib.load("../feature_engineering/X_train_tfidf.pkl")
X_test_tfidf = joblib.load("../feature_engineering/X_test_tfidf.pkl")
y_train = joblib.load("../feature_engineering/y_train.pkl")
y_test = joblib.load("../feature_engineering/y_test.pkl")

print("Data loaded successfully.\n")


# Structured features → numeric 
X_train_numeric = X_train.copy()
bool_cols_train = X_train_numeric.select_dtypes(include="bool").columns
X_train_numeric[bool_cols_train] = X_train_numeric[bool_cols_train].astype(np.int8)

X_test_numeric = X_test.copy()
bool_cols_test = X_test_numeric.select_dtypes(include="bool").columns
X_test_numeric[bool_cols_test] = X_test_numeric[bool_cols_test].astype(np.int8)

# Sparse matrices for structured features
X_train_struct_sparse = sparse.csr_matrix(X_train_numeric.values)
X_test_struct_sparse = sparse.csr_matrix(X_test_numeric.values)


# Apply SVD (300 components) on TF-IDF
print("Applying TruncatedSVD (n_components=300) on TF-IDF features...")

svd = TruncatedSVD(n_components=300, random_state=42)
X_train_svd = svd.fit_transform(X_train_tfidf)
X_test_svd = svd.transform(X_test_tfidf)

print("SVD-transformed TF-IDF shapes:")
print("  X_train_svd:", X_train_svd.shape)
print("  X_test_svd :", X_test_svd.shape)


# Combine SVD components + structured features
X_train_struct_dense = X_train_struct_sparse.toarray()
X_test_struct_dense = X_test_struct_sparse.toarray()

X_train_combined = np.hstack([X_train_svd, X_train_struct_dense])
X_test_combined = np.hstack([X_test_svd, X_test_struct_dense])

print("\nFinal combined feature shapes:")
print("  X_train_combined:", X_train_combined.shape)
print("  X_test_combined :", X_test_combined.shape)


# Logistic Regression model
print("\nRunning 5-fold stratified cross-validation on training set with SVD features...")

base_log_reg = LogisticRegression(
    max_iter=2000,
    n_jobs=-1,
    class_weight="balanced",
    solver="lbfgs",
)

k = 5
cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

# F1 scores
cv_f1_scores = cross_val_score(
    base_log_reg,
    X_train_combined,
    y_train,
    cv=cv,
    scoring="f1"
)

# Accuracy scores
cv_acc_scores = cross_val_score(
    base_log_reg,
    X_train_combined,
    y_train,
    cv=cv,
    scoring="accuracy"
)

# PR-AUC (average precision) scores
cv_pr_auc_scores = cross_val_score(
    base_log_reg,
    X_train_combined,
    y_train,
    cv=cv,
    scoring="average_precision"
)

print(f"Cross-validated F1 scores:       {cv_f1_scores}")
print(f"Mean F1: {cv_f1_scores.mean():.4f} | Std: {cv_f1_scores.std():.4f}")

print(f"\nCross-validated Accuracy scores: {cv_acc_scores}")
print(f"Mean Acc: {cv_acc_scores.mean():.4f} | Std: {cv_acc_scores.std():.4f}")

print(f"\nCross-validated PR-AUC scores:   {cv_pr_auc_scores}")
print(f"Mean PR-AUC: {cv_pr_auc_scores.mean():.4f} | Std: {cv_pr_auc_scores.std():.4f}\n")

# Cross-validated predictions on train for diagnostics
print("Generating cross-validated predictions for training set (for diagnostics)...")
y_train_cv_pred = cross_val_predict(
    base_log_reg,
    X_train_combined,
    y_train,
    cv=cv
)

print("\nCross-validated TRAIN classification report:")
print(classification_report(y_train, y_train_cv_pred))


# Train final model on full training set
print("\nTraining final Logistic Regression model on full training data (with SVD)...")

log_reg = base_log_reg
log_reg.fit(X_train_combined, y_train)

print("Final model trained.\n")


# Evaluation on held-out test set
print("Evaluating model on HELD-OUT test set...")

y_pred = log_reg.predict(X_test_combined)
y_proba = log_reg.predict_proba(X_test_combined)[:, 1]

test_acc = accuracy_score(y_test, y_pred)
print("\nTest Accuracy:", test_acc)

print("\nTest Classification Report:\n", classification_report(y_test, y_pred))

# PR-AUC (average precision) on test set
test_pr_auc = average_precision_score(y_test, y_proba)
print("Test PR-AUC (Average Precision):", test_pr_auc)

# Fraud recall (recall on class 1)
fraud_recall = recall_score(y_test, y_pred, pos_label=1)
print("Fraud Recall (Recall on class 1):", fraud_recall)


# Confusion Matrix (on test set)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Logistic Regression (SVD) – Confusion Matrix (Test Set)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("logreg_svd_confusion_matrix.png")
plt.close()

print("\nConfusion Matrix:")
print(cm)


#Precision–Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_proba)

plt.figure(figsize=(6, 5))
plt.plot(recall, precision)
plt.title("Precision–Recall Curve – Logistic Regression (SVD)")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.tight_layout()
plt.savefig("logreg_svd_pr_curve.png")
plt.close()

print("\nSaved Precision–Recall curve as logreg_svd_pr_curve.png")


#Save model + SVD + structured column names
joblib.dump(log_reg, "logistic_model.pkl")            # SVD-based logistic model
joblib.dump(svd, "logreg_tfidf_svd_300.pkl")          # SVD transformer
joblib.dump(X_train_numeric.columns.to_list(),
            "logreg_structured_feature_columns.pkl")

print("\nSaved:")
print(" - logistic_model.pkl")
print(" - logreg_tfidf_svd_300.pkl")
print(" - logreg_structured_feature_columns.pkl")

print("\n=== LOGISTIC REGRESSION (SVD) TRAINING COMPLETE ===\n")
