# User Manual
## Fraudulent Job Posting Detection System

This manual provides step-by-step instructions to set up and run the fraudulent job posting detection system.

---

## Table of Contents

1. [Prerequisites & Installation](#prerequisites--installation)
2. [Data Preparation](#data-preparation)
3. [Running the Pipeline](#running-the-pipeline)
4. [Understanding Outputs](#understanding-outputs)
5. [Troubleshooting](#troubleshooting)

---

## Prerequisites & Installation

### System Requirements
- **Python**: Version 3.8 or higher
- **Memory**: At least 4GB RAM (8GB recommended)
- **Storage**: ~500MB for data and models

### Installation Steps

```bash
# 1. Clone/download project and navigate to directory
cd FraudulentJobPosting

# 2. Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python -c "import pandas, sklearn, scipy; print('Installation successful!')"
```

**Installs**: pandas, numpy, scikit-learn, scipy, matplotlib, seaborn, plotly, joblib, jupyter

---

## Data Preparation

### Required Data Files
Place CSV files in the `data/` directory:
- `FakePostings.csv` - Primary training dataset
- `RealAndFake.csv` - Additional training dataset
- `*NewGrad.csv` files (optional, for validation)

### Data Structure
CSV files must contain: `title`, `company_profile`, `description`, `requirements`, `benefits`, `location`, `salary_range`, `employment_type`, `industry`, and `fraudulent` (0=legitimate, 1=fraudulent).

---

## Running the Pipeline

### Phase 1: Data Processing
```bash
python dataprocessing.py
```
**Output**: `data/CleanedJobPostings.csv`  
**What it does**: Cleans text, combines datasets, removes duplicates and low-quality entries.

---

### Phase 2: Exploratory Data Analysis (Optional)
```bash
python eda.py
```
**Output**: Visualizations and statistical summaries  
**Purpose**: Understand data patterns and fraud indicators.

---

### Phase 3: Feature Engineering
```bash
jupyter notebook feature_engineering/feature_engineering.ipynb
```
Run all cells in the notebook.

**Output Files** (in `feature_engineering/`):
- `tfidf_vectorizer.pkl`, `X_train_tfidf.pkl`, `X_test_tfidf.pkl`
- `X_train.pkl`, `X_test.pkl`, `y_train.pkl`, `y_test.pkl`
- `structured_feature_columns.pkl`

**What it does**: Creates TF-IDF text features (5000 features, n-grams) and structured features (industry, location, salary), then combines them.

**Time**: ~5-15 minutes

---

### Phase 4: Model Training

#### Logistic Regression
```bash
cd logit_model && python logReg_model.py && cd ..
```
**Output**: `logit_model/logistic_model.pkl`, `logit_model/confusion_matrix.png`  
**What it does**: 5-fold cross-validation, trains model, evaluates on test set.  
**Time**: ~2-5 minutes

#### Random Forest
```bash
cd random_forest_model && python random_forest_model.py && cd ..
```
**Output**: `random_forest_model/random_forest_all_features.pkl`, `rf_confusion_matrix.png`  
**What it does**: 5-fold cross-validation, hyperparameter-tuned training, feature importance analysis.  
**Time**: ~10-30 minutes

---

### Phase 5: Validation on New Data (Optional)
```bash
python newgradcleaning.py
python random_forest_model/validate_newgrad_rf.py
```
**Output**: `newgrad_rf_predictions.csv` with predictions and probability scores  
**What it does**: Applies trained model to new graduate job postings with feature alignment.

---

## Understanding Outputs

### Performance Metrics
- **Accuracy**: Overall correct predictions (0-1, higher is better)
- **F1-Score**: Balance of precision/recall (critical for imbalanced data)
- **Precision**: Of predicted frauds, how many are actually fraud
- **Recall**: Of actual frauds, how many are correctly identified
- **Confusion Matrix**: Visual breakdown (TN, FP, FN, TP)

### Prediction File
`newgrad_rf_predictions.csv` contains:
- Original columns
- `rf_pred_fraud`: Binary prediction (0=legitimate, 1=fraudulent)
- `rf_fraud_probability`: Confidence score (0.0-1.0, closer to 1.0 = more likely fraud)

### Feature Importance
Random Forest outputs top fraud-indicating features (TF-IDF keywords and structured features) to understand model decisions.

---

## Troubleshooting

### FileNotFoundError
**Solution**: Verify files in `data/` directory, check exact file names (case-sensitive), ensure running from project root.

### ModuleNotFoundError
**Solution**: Activate virtual environment, reinstall: `pip install -r requirements.txt`

### Memory Error
**Solution**: Close other applications, reduce data size, use smaller `max_features` in TF-IDF, process in batches.

### Shape Mismatch Error
**Solution**: Ensure feature engineering completed fully, verify all `.pkl` files exist, re-run feature engineering if needed.

### Poor Model Performance
**Causes**: Class imbalance, data quality issues, insufficient features  
**Solution**: Review EDA outputs, check confusion matrix, experiment with hyperparameters.

### Jupyter Notebook Issues
**Solution**: `pip install jupyter` or `pip install notebook`

### Path Errors (Windows)
**Solution**: Code uses `os.path.join()` automatically. Use forward slashes or ensure Python 3.8+.

---

## Quick Reference

```bash
# Complete Pipeline
pip install -r requirements.txt
python cleaning/dataprocessing.py
jupyter notebook feature_engineering/feature_engineering.ipynb  # Run all cells
python logit_model/logReg_model.py
python random_forest_model/random_forest_model.py
python newgradcleaning.py  # Optional
python random_forest_model/validate_newgrad_rf.py  # Optional
```

**Total Time**: ~30-60 minutes for complete pipeline

---

## Getting Help

- **Documentation**: `CODE_FLOW.md`, `README.md`, `PROJECT_REPORT.md`
- **Review error messages** for specific guidance
- **Check file paths** and data format requirements

---

## Next Steps

1. Review confusion matrices and performance metrics
2. Analyze predictions in output CSV
3. Examine feature importances
4. Iterate on parameters based on results

---

**Last Updated**: [Date]  
**Version**: 1.0
