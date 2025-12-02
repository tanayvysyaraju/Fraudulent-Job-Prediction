import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict

print("Loading TF-IDF data...")

# 1. Load saved data
tfidf = joblib.load("tfidf_vectorizer.pkl")
X_train_tfidf = joblib.load("X_train_tfidf.pkl")
X_test_tfidf = joblib.load("X_test_tfidf.pkl")
y_train = joblib.load("y_train.pkl")
y_test = joblib.load("y_test.pkl")

print("Data loaded successfully.\n")

# 2. K-fold cross-validation (Stratified)
print("Running 5-fold stratified cross-validation on training set...")

base_log_reg = LogisticRegression(
    max_iter=1000,
    n_jobs=-1,
    class_weight="balanced"
)

k = 5
cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

# F1 scores
cv_f1_scores = cross_val_score(
    base_log_reg,
    X_train_tfidf,
    y_train,
    cv=cv,
    scoring="f1"
)

# Accuracy scores
cv_acc_scores = cross_val_score(
    base_log_reg,
    X_train_tfidf,
    y_train,
    cv=cv,
    scoring="accuracy"
)

print(f"Cross-validated F1 scores:       {cv_f1_scores}")
print(f"Mean F1: {cv_f1_scores.mean():.4f} | Std: {cv_f1_scores.std():.4f}")

print(f"\nCross-validated Accuracy scores: {cv_acc_scores}")
print(f"Mean Acc: {cv_acc_scores.mean():.4f} | Std: {cv_acc_scores.std():.4f}\n")

# Optional: get CV-based predictions on train for a more realistic view
print("Generating cross-validated predictions for training set (for diagnostics)...")
y_train_cv_pred = cross_val_predict(
    base_log_reg,
    X_train_tfidf,
    y_train,
    cv=cv
)

print("\nCross-validated TRAIN classification report:")
print(classification_report(y_train, y_train_cv_pred))


# 3. Train final model on full training set
print("\nTraining final Logistic Regression model on full training data...")

log_reg = base_log_reg
log_reg.fit(X_train_tfidf, y_train)

print("Final model trained.\n")

# 4. Evaluation on test set
print("Evaluating model on HELD-OUT test set...")

y_pred = log_reg.predict(X_test_tfidf)

print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
print("\nTest Classification Report:\n", classification_report(y_test, y_pred))

# 5. Confusion Matrix (on test set)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Logistic Regression – Confusion Matrix (Test Set)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

# 6. Feature Importance (top words)
feature_names = tfidf.get_feature_names_out()
coeffs = log_reg.coef_[0]

top_fraud_idx = coeffs.argsort()[-20:]   # largest positive coefficients
top_real_idx = coeffs.argsort()[:20]     # largest negative coefficients

print("\nTop Keywords Indicating FRAUD:")
for word in feature_names[top_fraud_idx]:
    print("-", word)

print("\nTop Keywords Indicating REAL Jobs:")
for word in feature_names[top_real_idx]:
    print("-", word)

# 7. Save final trained model
joblib.dump(log_reg, "logistic_model.pkl")
print("\nModel saved as logistic_model.pkl")