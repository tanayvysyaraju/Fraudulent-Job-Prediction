import os
import pandas as pd

# Path to your data folder
DATA_DIR = "data"
def load_and_combine_newgrad_data(data_dir=DATA_DIR):
    all_dfs = []
    for filename in os.listdir(data_dir):
        if filename.endswith("NewGrad.csv"):
            file_path = os.path.join(data_dir, filename)
            category = filename.replace("NewGrad.csv", "")
            df = pd.read_csv(file_path)
            df["category"] = category
            all_dfs.append(df)
    combined_df = pd.concat(all_dfs, ignore_index=True)
    return combined_df


if __name__ == "__main__":
    df = load_and_combine_newgrad_data()

    # Random sample of 25 rows
    print(df.sample(25))

    # Optional: save combined CSV
    df.to_csv("combined_newgrad_data.csv", index=False)