 Credit Card Fraud Detection System

 Overview
This project is an enterprise-grade machine learning system designed to detect fraudulent credit card transactions using advanced techniques like SMOTETomek, PR-AUC optimization, and real-time API deployment.



 Features
 Zero Data Leakage Pipeline
 Imbalanced Data Handling (SMOTETomek)
 High Performance Model (Random Forest / XGBoost)
 FastAPI Backend for real-time predictions
 Streamlit Dashboard for UI
 SHAP Explainability
 Docker Deployment



️ Tech Stack
 Python
 Scikit-learn, XGBoost
 FastAPI
 Streamlit
 SHAP
 Docker


 Model Performance
 PR-AUC: 0.79
 ROC-AUC: 0.94
 Recall: 0.75+


 Project Structure
src/
models/
app.py
api_app.py
Dockerfile

How to Run

 1. Install dependencies
pip install -r requirements.txt
 2. Train model
python src/train_models.py
 3. Run API
uvicorn api_app:app --port 8000
 4. Run UI
streamlit run app.py

Dataset
Dataset not included due to size limitations.  
Download from Kaggle:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud


This project is an end-to-end Machine Learning solution designed to detect fraudulent credit card transactions accurately, featuring hyperparameter tuning, batch scoring, explainable AI (SHAP), scalable REST API, and Docker containerization.
 Overview

1.Setup & Data Validation**: Uses robust preprocessing, checking for missing values, and eliminating duplicates.
2.Exploratory Data Analysis**: Available in `notebooks/fraud_detection_analysis.ipynb`.
3.Feature Engineering**: Features a time-based aggregation (`Hour_of_Day`) and scales data appropriately for model ingestion.
4.Resampling Techniques (Zero-Leakage)**: Employs `SMOTETomek` nested effectively within `imblearn.pipeline.Pipeline` ensuring resampling ONLY occurs on training folds during Cross-Validation.
5.Model Tuning & Evaluation**: Uses `RandomizedSearchCV` for XGBoost and evaluates Logistic Regression, Random Forest, and XGBoost using ROC-AUC, PR-AUC, Precision-Recall Curves, and SHAP Global Feature Importance.
6.Explainability**: SHAP (SHapley Additive exPlanations) powers model interpretations on both global (summary plots) and local (waterfall plots) levels.
7.Streamlit Dashboard**: A production-ready streamlit interface allowing fast-scoring via CSV batch uploads and single-transaction forms with dynamic SHAP explanation plots.
8.Scalable REST API**: A high-performance asynchronous API served via FastAPI (`api_app.py`).
9.Docker Support**: Containerized environment for unified deployment.
 Usage
 1.Requirements Setup
```bash
pip install -r requirements.txt
```
 2.Testing the Pipeline
Execute unit tests ensuring preprocessing algorithms are functioning:
```bash
pytest tests/
```
3.Model Training
```bash
python src/train_models.py
```
This generates the pipeline and best model dictionary in `models/fraud_model.pkl`. It also generates evaluation plots (inc. SHAP) in `reports/figures/`.
 4.Interactive Dashboard
Launch the dynamic Streamlit UI:
```bash
streamlit run app.py
```
 5.Serving the API (FastAPI)
Run the RESTful API via `uvicorn`:
```bash
uvicorn api_app:app --host 0.0.0.0 --port 8000
```
- Interactive API Docs available at `http://localhost:8000/docs`
- Send a `POST` request to `/predict` to score real-time JSON transactions.
 6.Docker Containerization
```bash
 Build the image
docker build -t fraud_detector:latest .

 Run the container (Serving FastAPI on 8000)
docker run -p 8000:8000 fraud_detector:latest

Or run the Streamlit app instead
docker run -p 8501:8501 -it fraud_detector:latest streamlit run app.py
```

 Results
- Evaluates Accuracy, Precision, Recall, F1-Score, ROC-AUC, and PR-AUC.
- Optimizes primarily for Recall to minimize False Negatives.
