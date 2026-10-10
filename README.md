		# Claim Outcome Prediction API

## Overview

A machine learning project that predicts synthetic insurance claim outcomes using a Random Forest classifier. The API is built with FastAPI and provides interactive documentation through Swagger UI.

## Features

* Synthetic insurance claim dataset generation
* Data preprocessing and categorical encoding
* Feature engineering
* Random Forest classification
* Model evaluation using accuracy, precision, recall, F1-score, and confusion matrix
* REST API for claim outcome predictions
* Prediction class probabilities
* Interactive API documentation

## Technologies

* Python
* Pandas and NumPy
* Scikit-learn
* FastAPI
* Uvicorn
* Joblib

## Dataset

The project uses 400 synthetically generated insurance claims. Features include payer, provider, CPT code, diagnosis code, billed amount, and patient age.

The target classes are Paid, Partially_Paid, and Denied. The dataset is synthetic and is not representative of real insurance claims.

## Model Evaluation

The current test-set accuracy is 98.75% on an 80-record test set. Since the outcomes are generated using predefined synthetic rules, this result should not be interpreted as real-world predictive performance.

## Setup

1. Create and activate a virtual environment.

2. Install dependencies:

   `pip install -r requirements.txt`

3. Generate the dataset:

   `python train.py`

4. Train and evaluate the model:

   `python evaluate.py`

5. Start the API:

   `python -m uvicorn api:app --reload`

6. Open `http://127.0.0.1:8000/docs`.

## API Endpoints

* `GET /` — API welcome message
* `GET /health` — health check
* `POST /predict` — predicts the outcome for a claim

## Example Request

```json
{
  "payer": "Aetna",
  "provider": "Dr. Kumar",
  "cpt_code": "99213",
  "diagnosis_code": "E11.9",
  "billed_amount": 1200.0,
  "patient_age": 45
}
```

## Disclaimer

This is an educational demonstration using synthetic data. It must not be used to make actual insurance claim decisions.
