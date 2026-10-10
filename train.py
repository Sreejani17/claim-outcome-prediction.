
import numpy as np
import pandas as pd

# Reproducible synthetic insurance claims
rng = np.random.default_rng(42)
n = 400

df = pd.DataFrame({
    "payer": rng.choice(
        ["Aetna", "Cigna", "UnitedHC", "Medicare"], n
    ),
    "provider": rng.choice(
        ["Dr. Rao", "Dr. Shah", "Dr. Lee", "Dr. Kumar"], n
    ),
    "cpt_code": rng.choice(
        ["99213", "99214", "99215"], n
    ),
    "diagnosis_code": rng.choice(
        ["M54.5", "J06.9", "E11.9"], n
    ),
    "billed_amount": np.round(
        rng.uniform(100, 1500, n), 2
    ),
    "patient_age": rng.integers(18, 90, n)
})

# Demonstration rules only; these are NOT real insurance rules.
status = []

for row in df.itertuples(index=False):
    amount = row.billed_amount
    cpt = row.cpt_code
    diagnosis = row.diagnosis_code

    if (
        amount >= 1250
        or (amount >= 1000 and cpt == "99215")
    ):
        outcome = "Denied"

    elif (
        amount >= 800
        or cpt == "99215"
        or diagnosis == "E11.9"
    ):
        outcome = "Partially_Paid"

    else:
        outcome = "Paid"

    # Add 5% label noise to avoid a perfectly clean toy dataset.
    if rng.random() < 0.05:
        outcome = rng.choice(
            ["Paid", "Partially_Paid", "Denied"]
        )

    status.append(outcome)

df["status"] = status

print("Synthetic dataset created successfully.")
print("\nFirst five records:")
print(df.head())

print("\nStatus counts:")
print(df["status"].value_counts())

print("\nStatus proportions:")
print(df["status"].value_counts(normalize=True).round(3))

print("\nDataset shape:", df.shape)

df.to_csv("claims.csv", index=False)
print("\nSaved dataset to claims.csv")

