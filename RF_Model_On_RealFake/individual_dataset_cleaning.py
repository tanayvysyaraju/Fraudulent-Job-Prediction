import pandas as pd
import numpy as np

real_path = "data/RealAndFake.csv"
fake_path = "data/FakePostings.csv"

df_real = pd.read_csv(real_path)
df_fake = pd.read_csv(fake_path)

print("Loaded files:")
print("Real:", df_real.shape)
print("Fake:", df_fake.shape)


#text cleaning
def clean_text_columns(df):
    text_cols = df.select_dtypes(include="object").columns
    df[text_cols] = (
        df[text_cols]
        .fillna("")
        .apply(lambda col: col.str.lower().str.strip())
    )
    return df


df_real = clean_text_columns(df_real).drop_duplicates()
df_fake = clean_text_columns(df_fake).drop_duplicates()


def create_text_field(df):
    df["text"] = (
        df["title"].astype(str) + " " +
        df["company_profile"].astype(str) + " " +
        df["description"].astype(str) + " " +
        df["requirements"].astype(str) + " " +
        df["benefits"].astype(str)
    )
    return df


df_real = create_text_field(df_real)
df_fake = create_text_field(df_fake)



needed_cols = [
    "title", "company_profile",
    "location", "salary_range", "employment_type",
    "industry", "benefits",
    "requirements", "description",
    "fraudulent", "text"
]

for col in needed_cols:
    if col not in df_real.columns:
        df_real[col] = "unknown"
    if col not in df_fake.columns:
        df_fake[col] = "unknown"

df_real_clean = df_real[needed_cols]
df_fake_clean = df_fake[needed_cols]



# removing rows with >5 empty columns
def drop_rows_with_too_many_empty(df):
    empty_counts = (df == "").sum(axis=1)
    return df[empty_counts <= 5]


df_real_clean = drop_rows_with_too_many_empty(df_real_clean)
df_fake_clean = drop_rows_with_too_many_empty(df_fake_clean)


df_real_clean = df_real_clean.drop_duplicates(subset=["text"])
df_fake_clean = df_fake_clean.drop_duplicates(subset=["text"])

# removing short text
df_real_clean = df_real_clean[df_real_clean["text"].str.len() > 30]
df_fake_clean = df_fake_clean[df_fake_clean["text"].str.len() > 30]


df_real_clean = df_real_clean.replace("", "unknown").fillna("unknown")
df_fake_clean = df_fake_clean.replace("", "unknown").fillna("unknown")

df_real_clean.to_csv("data/RealAndFake_cleaned.csv", index=False)
df_fake_clean.to_csv("data/FakePostings_cleaned.csv", index=False)

print("\n=== CLEANING COMPLETE ===")
print("Real cleaned shape:", df_real_clean.shape)
print("Fake cleaned shape:", df_fake_clean.shape)
