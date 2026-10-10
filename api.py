from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# Load the trained model
MODEL_PATH = Path(__file__).parent / "models" / "claim_model.joblib"
model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Insurance Claim Outcome Prediction API",
    description="Predicts insurance claim outcomes using machine learning.",
    version="1.0.0"
)


class ClaimInput(BaseModel):
    payer: str = Field(min_length=1, max_length=100)
    provider: str = Field(min_length=1, max_length=100)
    cpt_code: str = Field(min_length=1, max_length=30)
    diagnosis_code: str = Field(min_length=1, max_length=30)
    billed_amount: float = Field(gt=0, le=10_000_000)
    patient_age: int = Field(ge=0, le=120)


@app.get("/")
def home():
    return {
        "message": "Insurance Claim Outcome Prediction API is running",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict")
def predict_claim(claim: ClaimInput):
    try:
        # Convert request into a DataFrame
        data = pd.DataFrame([claim.model_dump()])

        # Apply the same feature engineering used during training
        data["high_value_claim"] = (
            data["billed_amount"] >= 750
        ).astype(int)

        data["senior_patient"] = (
            data["patient_age"] >= 60
        ).astype(int)

        # Predict outcome
        prediction = model.predict(data)[0]

        # Get probability for each possible outcome
        probabilities = model.predict_proba(data)[0]
        classes = model.named_steps["classifier"].classes_

        probability_dict = {
            str(label): round(float(probability), 4)
            for label, probability in zip(classes, probabilities)
        }

        return {
            "predicted_status": str(prediction),
            "probabilities": probability_dict
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}"
        )