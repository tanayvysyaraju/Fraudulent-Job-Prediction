import pandas as pd
import numpy as np

# File paths
path1 = "/Users/renukaparhad/Desktop/FradulentJob/Fraudulent-Job-Prediction/data/FakePostings.csv"
path2 = "/Users/renukaparhad/Desktop/FradulentJob/Fraudulent-Job-Prediction/data/RealAndFake.csv"

# Load datasets
df_fake = pd.read_csv(path1)
df_rf = pd.read_csv(path2)


# Clean text-like columns
def clean_text_columns(df):
    text_cols = df.select_dtypes(include="object").columns
    df[text_cols] = df[text_cols].apply(lambda col: col.fillna("").str.lower().str.strip())
    return df

df_fake = clean_text_columns(df_fake).drop_duplicates()
df_rf = clean_text_columns(df_rf).drop_duplicates()


# Create unified text field using consistent columns
df_fake["text"] = (
    df_fake["title"] + " " +
    df_fake["company_profile"] + " " +
    df_fake["description"] + " " +
    df_fake["requirements"] + " " +
    df_fake["benefits"]
)

df_rf["text"] = (
    df_rf["title"] + " " +
    df_rf["company_profile"] + " " +
    df_rf["description"] + " " +
    df_rf["requirements"] + " " +
    df_rf["benefits"]
)

# Select relevant columns 
needed_cols = [
    "title", "company_profile",
    "location", "salary_range", "employment_type",
    "industry", "benefits",
    "requirements", "description",
    "fraudulent", "text"
]

df_fake_clean = df_fake[needed_cols]
df_rf_clean = df_rf[needed_cols]


# Combine datasets
df_all = pd.concat([df_fake_clean, df_rf_clean], ignore_index=True)


# REMOVE ROWS WITH > 5 EMPTY COLUMNS
empty_counts = (df_all == "").sum(axis=1)
df_all = df_all[empty_counts <= 5]


# Remove duplicate text fields
df_all = df_all.drop_duplicates(subset=["text"])

# Remove rows where text is extremely short
df_all = df_all[df_all["text"].str.len() > 30]

#fill any remaining empty or missing values with "unknown"
df_all = df_all.replace("", "unknown")
df_all = df_all.fillna("unknown")

# Save cleaned dataset
df_all.to_csv("data/CleanedJobPostings.csv", index=False)


print(df_all.head())
print(df_all.shape)
