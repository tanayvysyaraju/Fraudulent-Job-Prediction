# Fraudulent Job Postings Prediction

This project builds a machine learning system to detect fraudulent job postings by combining text analysis (TF-IDF + SVD) with structured metadata features (salary, industry, location legitimacy, etc.). The model is trained on labeled Kaggle datasets and validated on newly scraped real-world job postings to test generalization beyond the training distribution.

## Problem Overview

Online job boards are seeing a rapid rise in fake job advertisements that mimic legitimate postings through similar language, buzzwords, and formatting. At the same time, real job descriptions have changed significantly between 2019–2025, creating additional challenges for detection.  
Our objective is to classify postings as real or fraudulent while identifying the linguistic and structural patterns that distinguish the two.

## Novelty

Our approach differs from standard methods by:

- Combining text + structured data using 10,000 TF-IDF features, SVD components, salary normalization, industry grouping, and location legitimacy scoring.  
- Evaluating multiple model families—Logistic Regression, Random Forest, Linear SVM, and XGBoost—revealing distinct failure modes.  
- Validating on a completely separate scraped dataset of 8,000+ recent new-graduate job postings across six industries to measure real-world robustness.  

This hybrid approach captures both manipulated language patterns and suspicious metadata signals.

## Pipeline Overview

1. **Data Collection**  
   - Kaggle datasets (real + fraudulent postings)  
   - Scraped new-graduate postings for out-of-domain validation  

2. **Preprocessing & Feature Engineering**  
   - Combined text fields  
   - TF-IDF vectorization (10,000 features)  
   - SVD reduction (300 components)  
   - Structured features: salary, industry, employment type, location legitimacy  

3. **Models Implemented**
   - Logistic Regression  
   - Random Forest  
   - Linear SVM  
   - XGBoost  

4. **Evaluation**  
   - Stratified 5-fold CV  
   - Metrics: Recall, F1, Accuracy, PR-AUC  
   - Real-world validation on scraped 2024–2025 postings  

## Key Findings

### Logistic Regression  
High test accuracy but poor generalization, predicting >99% of new jobs as fraud.

### Random Forest  
Good test metrics but predicted 0 frauds in real-world data.

### Linear SVM  
High recall and stable behavior; predicted 50 fraud cases.

### XGBoost (Final Model)  
- Accuracy: 0.984  
- Recall: 0.963  
- PR-AUC: 0.9977  
- Predicted 17.7% of scraped postings as fraud  
- Captured classic scam indicators such as “flexible hours,” “earn 5000,” and “immediate hiring.”

**Chosen as final model** for best balance of performance and realistic behavior.

## Team Contributions

- **Tanay:** Data collection, Random Forest, CV framework, evaluation, repo structure, documentation  
- **Krisha:** TF-IDF/SVD engineering, XGBoost + SVM modeling, tuning, feature analysis  
- **Renuka:** Cleaning workflows, visualizations, Logistic Regression, validation pipeline, real-world testing  

## What We Learned

- Data quality and consistency strongly affect performance  
- True validation requires out-of-domain testing  
- Feature engineering is crucial for fraud detection  
- Collaborative workflows and version control are essential  
