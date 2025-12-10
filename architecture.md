┌─────────────────────┐
│   Raw Datasets      │
│                     │
│ • FakePostings.csv  │
│ • RealAndFake.csv   │
│ • *NewGrad.csv      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Data Processing    │
│  Module             │
│                     │
│ • Text Normalization│
│ • Duplicate Removal │
│ • Quality Filtering │
│ • Data Combination  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Cleaned Dataset    │
│  (CleanedJobPostings│
│       .csv)         │
└──────────┬──────────┘
           │
    ┌──────┴──────┐
    │             │
    ▼             ▼
┌─────────┐  ┌─────────────────────┐
│   EDA   │  │ Feature Engineering │
│ Module  │  │      Module         │
│         │  │                     │
│ • Viz   │  │ • TF-IDF (Text)     │
│ • Stats │  │ • Structured        │
│ • Insights│ │ • Feature Comb.    │
└─────────┘  └──────────┬──────────┘
                        │
                        ▼
            ┌───────────────────────┐
            │  Engineered Features  │
            │                       │
            │ • X_train_tfidf.pkl   │
            │ • X_train.pkl         │
            │ • y_train.pkl         │
            │ • Vectorizers         │
            └───────────┬───────────┘
                        │
            ┌───────────┴───────────┐
            │                       │
            ▼                       ▼
    ┌───────────────┐      ┌───────────────┐
    │   Logistic    │      │   Random      │
    │  Regression   │      │   Forest      │
    │               │      │               │
    │ • Training    │      │ • Training    │
    │ • CV          │      │ • CV          │
    │ • Evaluation  │      │ • Evaluation  │
    └───────┬───────┘      └───────┬───────┘
            │                       │
            └───────────┬───────────┘
                        │
                        ▼
            ┌───────────────────────┐
            │  Model Evaluation     │
            │                       │
            │ • Accuracy            │
            │ • F1-Score            │
            │ • Confusion Matrices  │
            └───────────┬───────────┘
                        │
                        ▼
            ┌───────────────────────┐
            │  Validation Pipeline  │
            │                       │
            │ • New Data Processing │
            │ • Feature Alignment   │
            │ • Predictions         │
            └───────────────────────┘
