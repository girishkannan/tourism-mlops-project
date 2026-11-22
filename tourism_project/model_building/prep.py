"""Data preparation: clean the dataset and create train/test splits."""
import os
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = os.path.join("tourism_project", "data", "tourism.csv")
TARGET = "ProdTaken"

df = pd.read_csv(DATA_PATH)

# Drop index-like / identifier columns (unnamed index column and CustomerID)
drop_cols = [c for c in df.columns if c.startswith("Unnamed") or c.strip() == ""]
drop_cols.append("CustomerID")
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# Fix known inconsistent labels
if "Gender" in df.columns:
    df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

X = df.drop(columns=[TARGET])
y = df[TARGET]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Saved in the repo root so the workflow can upload them as artifacts
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data preparation complete.")
print(f"Train: {Xtrain.shape}, Test: {Xtest.shape}")
print(f"Columns used: {list(X.columns)}")
