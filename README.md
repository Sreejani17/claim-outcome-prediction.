# Insurance Claim Outcome Prediction

## Overview

This project predicts insurance claim outcomes using machine learning. It classifies claims into three categories: Paid, Partially Paid, and Denied.

## Features

* Synthetic insurance claim dataset generation
* Data preprocessing and feature engineering
* Random Forest classification
* Class imbalance handling using balanced class weights
* Model evaluation using accuracy, precision, recall, F1-score, and confusion matrix
* REST API built with FastAPI
* Interactive API documentation using Swagger UI

## Technologies

* Python
* Pandas and NumPy
* Scikit-learn
* FastAPI
* Uvicorn
* Joblib

## Project Structure

```text
claim-outcome-prediction/
├── train.py
├── evaluate.py
├── api.py
├── claims.csv
├── requirements.txt
├── README.md
└── models/
    └── claim_model.joblib
```

## Model Evaluation

The current model was evaluated on a held-out test set of 80 records.

| Metric                | Result |
| --------------------- | -----: |
| Accuracy              |    60% |
| Paid recall           |    79% |
| Partially Paid recall |    25% |
| Denied recall         |     0% |

The results show that the model performs better at identifying Paid claims than the other classes. The low recall for Denied claims is a limitation that needs further investigation and improvement.

## Run Locally

### 1. Create a virtual environment

```bash
python -m venv venv
```

### 2. Activate the environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Generate the dataset

```bash
python train.py
```

### 5. Train and evaluate the model

```bash
python evaluate.py
```

### 6. Start the API

```bash
python -m uvicorn api:app --reload
```

Open http://127.0.0.1:8000/docs to test the prediction endpoint.

## API Endpoints

* `GET /` — API welcome message
* `GET /health` — API health check
* `POST /predict` — Predict the outcome of an insurance claim

## Limitations

The dataset is synthetic and contains only 400 records. The evaluation results may not generalize to real insurance claims. The model currently has zero recall for the Denied class, so it should not be used for actual claim decisions without further development, representative data, and validation.

## Future Improvements

* Use a larger, representative dataset
* Improve minority-class detection
* Compare multiple classification algorithms
* Perform cross-validation and hyperparameter tuning
* Add automated tests and input validation
* Monitor model performance on new data
