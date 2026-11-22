"""Data registration: validate the raw dataset that lives inside the repo."""
import os
import sys
import pandas as pd

DATA_PATH = os.path.join("tourism_project", "data", "tourism.csv")

EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]

if not os.path.exists(DATA_PATH):
    sys.exit(f"Dataset not found at {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
if missing:
    sys.exit(f"Dataset is missing expected columns: {missing}")

print("Dataset registered successfully.")
print(f"Path   : {DATA_PATH}")
print(f"Shape  : {df.shape}")
print(f"Target distribution (ProdTaken):\n{df['ProdTaken'].value_counts()}")
print(f"Missing values per column:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
