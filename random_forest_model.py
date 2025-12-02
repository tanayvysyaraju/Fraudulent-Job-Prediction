import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

# ----------------------------------------------------
# 1. Load data
# ----------------------------------------------------
print("Loading TF-IDF data...")

tfidf = joblib.load("tfidf_vectorizer.pkl")
X_train_tfidf = joblib.load("X_train_tfidf.pkl")
X_test_tfidf = joblib.load("X_test_tfidf.pkl")
y_train = joblib.load("y_train.pkl")
y_test = joblib.load("y_test.pkl")

print("Data loaded successfully.\n")

# ----------------------------------------------------
# 2. Define Random Forest model
# ----------------------------------------------------
rf_clf = RandomForestClassifier(
    n_estimators=200,        # number of trees
    max_depth=None,         # let trees grow fully (you can tune this)
    min_samples_split=2,
    min_samples_leaf=1,
    n_jobs=-1,              # use all cores
    class_weight="balanced",# handle slight imbalance
    random_state=42
)

# ----------------------------------------------------
# 3. K-Fold Cross-Validation (on training set)
# ----------------------------------------------------
print("Running 5-fold stratified cross-validation on training set...")

k = 5
cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

cv_f1_scores = cross_val_score(
    rf_clf,
    X_train_tfidf,
    y_train,
    cv=cv,
    scoring="f1"
)

cv_acc_scores = cross_val_score(
    rf_clf,
    X_train_tfidf,
    y_train,
    cv=cv,
    scoring="accuracy"
)

print(f"Cross-validated F1 scores:       {cv_f1_scores}")
print(f"Mean F1: {cv_f1_scores.mean():.4f} | Std: {cv_f1_scores.std():.4f}")

print(f"\nCross-validated Accuracy scores: {cv_acc_scores}")
print(f"Mean Acc: {cv_acc_scores.mean():.4f} | Std: {cv_acc_scores.std():.4f}\n")

# ----------------------------------------------------
# 4. Train final model on full training data
# ----------------------------------------------------
print("Training final Random Forest model on full training data...")

rf_clf.fit(X_train_tfidf, y_train)

print("Final Random Forest model trained.\n")

# ----------------------------------------------------
# 5. Evaluate on held-out test set
# ----------------------------------------------------
print("Evaluating Random Forest on HELD-OUT test set...")

y_pred = rf_clf.predict(X_test_tfidf)

print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
print("\nTest Classification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Random Forest – Confusion Matrix (Test Set)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("rf_confusion_matrix.png")
plt.close()

print("Confusion matrix saved as rf_confusion_matrix.png\n")

# ----------------------------------------------------
# 6. Feature importance: top TF-IDF features
# ----------------------------------------------------
print("Analyzing feature importances...")

feature_names = tfidf.get_feature_names_out()
importances = rf_clf.feature_importances_

# Get indices of top 30 most important features
top_idx = np.argsort(importances)[-30:][::-1]

print("\nTop 30 Most Important Keywords (by RF feature importance):")
for i in top_idx:
    print(f"- {feature_names[i]} (importance={importances[i]:.6f})")

# ----------------------------------------------------
# 7. Save trained model
# ----------------------------------------------------
joblib.dump(rf_clf, "random_forest_model.pkl")
print("\nRandom Forest model saved as random_forest_model.pkl")