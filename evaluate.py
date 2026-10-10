
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

# 1. Load dataset
df = pd.read_csv("claims.csv")

print("Dataset shape:", df.shape)

# 2. Feature engineering
df["high_value_claim"] = (
    df["billed_amount"] >= 750
).astype(int)

df["senior_patient"] = (
    df["patient_age"] >= 60
).astype(int)

# 3. Separate features and target
X = df.drop(columns=["status"])
y = df["status"]

categorical_features = [
    "payer",
    "provider",
    "cpt_code",
    "diagnosis_code"
]

numerical_features = [
    "billed_amount",
    "patient_age",
    "high_value_claim",
    "senior_patient"
]

# 4. Stratified train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

# 5. Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent")
                ),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore")
                )
            ]),
            categorical_features
        ),
        (
            "numerical",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(strategy="median")
                ),
                ("scaler", StandardScaler())
            ]),
            numerical_features
        )
    ]
)

# 6. Build model pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "classifier",
        RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42
        )
    )
])

# 7. Train model
model.fit(X_train, y_train)

# 8. Evaluate model
y_pred = model.predict(X_test)

print("\nAccuracy:", round(accuracy_score(y_test, y_pred), 4))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=["Paid", "Partially_Paid", "Denied"],
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred,
        labels=["Paid", "Partially_Paid", "Denied"]
    )
)

# 9. Save complete pipeline
joblib.dump(model, "claim_model.joblib")

print("\nModel saved successfully as claim_model.joblib")

