import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

#Global visual style
sns.set_theme(style="whitegrid", palette="Set2", font_scale=1.2)
plt.rcParams["figure.dpi"] = 110


df = pd.read_csv("data/CleanedJobPostings.csv")



#********Feature engineering for EDA*********
df["description_length"] = df["description"].apply(lambda x: len(str(x).split()))
df["requirements_length"] = df["requirements"].apply(lambda x: len(str(x).split()))
df["benefits_length"] = df["benefits"].apply(lambda x: len(str(x).split()))
df["profile_length"] = df["company_profile"].apply(lambda x: len(str(x).split()))
df["text_length"] = df["text"].apply(lambda x: len(str(x).split()))



# Description Length Histogram
plt.figure(figsize=(8,5))
sns.histplot(df["description_length"], bins=40, kde=False, color="#4C72B0")
plt.xlim(0, 600)
plt.title("Description Length Distribution", fontsize=16)
plt.xlabel("Words", fontsize=13)
plt.ylabel("Frequency", fontsize=13)
sns.despine()
plt.show()


# Density Plot (Real vs Fraud)
plt.figure(figsize=(8,5))
sns.kdeplot(
    data=df,
    x="description_length",
    hue="fraudulent",
    fill=True,
    alpha=0.6,
    common_norm=False,
    palette=["#4C72B0", "#DD8452"]
)
plt.xlim(0, 600)
plt.title("Density of Description Length (Real vs Fraud)", fontsize=16)
plt.xlabel("Description Length (Words)")
plt.ylabel("Density")
sns.despine()
plt.show()


# Employment Type by Fraud Label
plt.figure(figsize=(10,5))
sns.countplot(
    data=df,
    x="employment_type",
    hue="fraudulent",
    palette=["#4C72B0","#DD8452"]
)
plt.xticks(rotation=45)
plt.title("Employment Type by Fraud Label", fontsize=16)
plt.xlabel("Employment Type")
plt.ylabel("Count")
sns.despine()
plt.tight_layout()
plt.show()


# Top Fraudulent Industries
top_industries = df[df["fraudulent"]==1]["industry"].value_counts().head(10)

plt.figure(figsize=(10,5))
sns.barplot(
    x=top_industries.index,
    y=top_industries.values,
    color="#4C72B0"
)
plt.xticks(rotation=45, ha="right")
plt.title("Top Industries with Fraudulent Job Postings", fontsize=16)
plt.xlabel("Industry")
plt.ylabel("Fraud Count")
sns.despine()
plt.tight_layout()
plt.show()


# Correlation Heatmap
num_features = df[[
    "description_length",
    "requirements_length",
    "benefits_length",
    "profile_length",
    "fraudulent"
]]

plt.figure(figsize=(8,5))
sns.heatmap(
    num_features.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5,
    square=True,
    cbar_kws={"shrink":0.8}
)
plt.title("Correlation Heatmap", fontsize=16)
plt.show()



# Scatter Plot: Description vs Requirements Length (avoids overplotting)
plt.figure(figsize=(8,6))
sns.scatterplot(
    data=df.sample(3000),
    x="description_length",
    y="requirements_length",
    hue="fraudulent",
    alpha=0.5,
    palette=["#4C72B0","#DD8452"]
)
plt.xlim(0, 600)
plt.ylim(0, 600)
plt.title("Description vs Requirements Length (Sample of 3000)", fontsize=16)
sns.despine()
plt.show()



#Fraud Count Chart
fraud_counts = df["fraudulent"].value_counts().sort_index()

plt.figure(figsize=(6,4))
sns.barplot(
    x=["Real","Fraud"],
    y=fraud_counts.values,
    palette=["#4C72B0","#DD8452"]
)
plt.title("Distribution of Real vs Fraudulent Job Postings", fontsize=16)
plt.ylabel("Count")
sns.despine()
plt.show()



#Interactive Plotly: Fraudulent Industries
fraud_industries = df[df["fraudulent"]==1]["industry"].value_counts().head(15)

fig = px.bar(
    fraud_industries,
    title="Industries With Highest Fraud (Interactive)",
    labels={"value":"Count", "index":"industry"}
)
fig.show()


#Interactive Plotly: Scatter Plot
fig = px.scatter(
    df.sample(3000),
    x="description_length",
    y="requirements_length",
    color="fraudulent",
    opacity=0.6,
    title="Description vs Requirements Length (Interactive)"
)
fig.show()


#Interactive Plotly: Text Length Distribution
fig = px.histogram(
    df,
    x="text_length",
    color="fraudulent",
    nbins=50,
    title="Text Length Distribution (Interactive)"
)
fig.show()
