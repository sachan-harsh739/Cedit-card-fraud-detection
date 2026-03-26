# Ultimate Production-Ready Credit Card Fraud Detection System

This project is an end-to-end Machine Learning solution designed to detect fraudulent credit card transactions accurately, featuring hyperparameter tuning, batch scoring, explainable AI (SHAP), scalable REST API, and Docker containerization.

## Overview

1. **Setup & Data Validation**: Uses robust preprocessing, checking for missing values, and eliminating duplicates.
2. **Exploratory Data Analysis**: Available in `notebooks/fraud_detection_analysis.ipynb`.
3. **Feature Engineering**: Features a time-based aggregation (`Hour_of_Day`) and scales data appropriately for model ingestion.
4. **Resampling Techniques (Zero-Leakage)**: Employs `SMOTETomek` nested effectively within `imblearn.pipeline.Pipeline` ensuring resampling ONLY occurs on training folds during Cross-Validation.
5. **Model Tuning & Evaluation**: Uses `RandomizedSearchCV` for XGBoost and evaluates Logistic Regression, Random Forest, and XGBoost using ROC-AUC, PR-AUC, Precision-Recall Curves, and SHAP Global Feature Importance.
6. **Explainability**: SHAP (SHapley Additive exPlanations) powers model interpretations on both global (summary plots) and local (waterfall plots) levels.
7. **Streamlit Dashboard**: A production-ready streamlit interface allowing fast-scoring via CSV batch uploads and single-transaction forms with dynamic SHAP explanation plots.
8. **Scalable REST API**: A high-performance asynchronous API served via FastAPI (`api_app.py`).
9. **Docker Support**: Containerized environment for unified deployment.

## Usage

### 1. Requirements Setup
```bash
pip install -r requirements.txt
```

### 2. Testing the Pipeline
Execute unit tests ensuring preprocessing algorithms are functioning:
```bash
pytest tests/
```

### 3. Model Training
```bash
python src/train_models.py
```
This generates the pipeline and best model dictionary in `models/fraud_model.pkl`. It also generates evaluation plots (inc. SHAP) in `reports/figures/`.

### 4. Interactive Dashboard
Launch the dynamic Streamlit UI:
```bash
streamlit run app.py
```

### 5. Serving the API (FastAPI)
Run the RESTful API via `uvicorn`:
```bash
uvicorn api_app:app --host 0.0.0.0 --port 8000
```
- Interactive API Docs available at `http://localhost:8000/docs`
- Send a `POST` request to `/predict` to score real-time JSON transactions.

### 6. Docker Containerization
```bash
# Build the image
docker build -t fraud_detector:latest .

# Run the container (Serving FastAPI on 8000)
docker run -p 8000:8000 fraud_detector:latest

# Or run the Streamlit app instead
docker run -p 8501:8501 -it fraud_detector:latest streamlit run app.py
```

## Results
- Evaluates Accuracy, Precision, Recall, F1-Score, ROC-AUC, and **PR-AUC**.
- Optimizes primarily for **Recall** to minimize False Negatives.

> *[Note: Insert Screenshots of Streamlit Dashboard & API Docs here]*
