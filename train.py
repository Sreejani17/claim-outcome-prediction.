
import numpy as np
import pandas as pd

# Generate synthetic insurance claims
np.random.seed(42)

n = 400

payers = ["Aetna", "Cigna", "UnitedHC", "Medicare"]
providers = ["Dr. Rao", "Dr. Shah", "Dr. Lee", "Dr. Kumar"]
cpt_codes = ["99213", "99214", "99215"]
diagnosis_codes = ["M54.5", "J06.9", "E11.9"]

statuses = np.random.choice(
    ["Paid", "Partially_Paid", "Denied"],
    size=n,
    p=[0.70, 0.20, 0.10]
)

df = pd.DataFrame({
    "payer": np.random.choice(payers, n),
    "provider": np.random.choice(providers, n),
    "cpt_code": np.random.choice(cpt_codes, n),
    "diagnosis_code": np.random.choice(diagnosis_codes, n),
    "billed_amount": np.round(np.random.uniform(100, 1500, n), 2),
    "patient_age": np.random.randint(18, 90, n),
    "status": statuses
})

print("Dataset created successfully!")
print(df.head())
print("\nStatus distribution:")
print(df["status"].value_counts(normalize=True).round(2))
print("\nDataset shape:", df.shape)

df.to_csv("claims.csv", index=False)
print("\nSaved dataset to claims.csv")

