import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

@st.cache_resource
def load_model_data():
    try:
        data = joblib.load("models/fraud_model.pkl")
        return data
    except Exception as e:
        return None

st.title("Credit Card Fraud Detection System")
st.write("Detect fraudulent credit card transactions in real-time or via batch processing.")

model_data = load_model_data()

if model_data is None:
    st.warning("Model not found! Please run the training script (`python src/train_models.py`) first to generate `models/fraud_model.pkl`.")
elif not isinstance(model_data, dict):
    st.error("The saved model is outdated. Please re-run `python src/train_models.py` to get the latest model format.")
else:
    model = model_data['model']
    scaler = model_data['scaler']
    expected_features = model_data['features']
    
    st.sidebar.header("Model Info")
    st.sidebar.write(f"**Loaded Model**: {type(model).__name__}")
    
    tab1, tab2 = st.tabs(["Manual Input", "Batch CSV Upload"])
    
    with tab1:
        st.subheader("Manual Transaction Input")
        st.write("Provide raw features below. Time and Amount will be automatically scaled.")
        
        col1, col2, col3 = st.columns(3)
        input_data = {}
        input_data['Time'] = col1.number_input("Time (seconds)", value=0.0)
        input_data['Amount'] = col1.number_input("Amount ($)", value=0.0)
        
        for i in range(1, 29):
            col = col2 if i <= 14 else col3
            input_data[f'V{i}'] = col.number_input(f"V{i}", value=0.0)
            
        if st.button("Predict Single Transaction"):
            df = pd.DataFrame([input_data])
            df['Hour_of_Day'] = (df['Time'] // 3600) % 24
            
            # Scale features
            cols_to_scale = ['Amount', 'Time', 'Hour_of_Day']
            scaled_features = scaler.transform(df[cols_to_scale])
            
            df['Scaled_Amount'] = scaled_features[:, 0]
            df['Scaled_Time'] = scaled_features[:, 1]
            df['Scaled_Hour'] = scaled_features[:, 2]
            df.drop(cols_to_scale, axis=1, inplace=True)
            
            # Reorder to match training
            df = df[expected_features]
            
            prediction = model.predict(df)
            prob = model.predict_proba(df)[0][1] if hasattr(model, 'predict_proba') else None
            
            if prediction[0] == 1:
                st.error("🚨 FRAUDULENT TRANSACTION DETECTED 🚨")
            else:
                st.success("✅ Legit Transaction")
                
            if prob is not None:
                st.info(f"Fraud Probability (Confidence): {prob:.2%}")
                
            if hasattr(model, 'feature_importances_'):
                import shap
                import matplotlib.pyplot as plt
                
                st.subheader("Transaction Explanation (SHAP)")
                try:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer(df)
                    
                    fig, ax = plt.subplots(figsize=(8, 5))
                    shap.plots.waterfall(shap_values[0], show=False)
                    st.pyplot(fig)
                except Exception as e:
                    st.warning(f"Could not generate SHAP plot: {e}")
                
    with tab2:
        st.subheader("Batch Fast-Scoring")
        st.write("Upload a CSV file with the same basic schema (Time, Amount, V1-V28).")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(batch_df)} transactions.")
            
            if st.button("Process Batch"):
                try:
                    # Clean up if Class column exists
                    if 'Class' in batch_df.columns:
                        batch_df = batch_df.drop('Class', axis=1)
                        
                    batch_df['Hour_of_Day'] = (batch_df['Time'] // 3600) % 24
                    cols_to_scale = ['Amount', 'Time', 'Hour_of_Day']
                    scaled_features = scaler.transform(batch_df[cols_to_scale])
                    
                    batch_df['Scaled_Amount'] = scaled_features[:, 0]
                    batch_df['Scaled_Time'] = scaled_features[:, 1]
                    batch_df['Scaled_Hour'] = scaled_features[:, 2]
                    batch_df.drop(cols_to_scale, axis=1, inplace=True)
                    
                    batch_df = batch_df[expected_features]
                    
                    preds = model.predict(batch_df)
                    if hasattr(model, 'predict_proba'):
                        probs = model.predict_proba(batch_df)[:, 1]
                    else:
                        probs = [None] * len(preds)
                        
                    results_df = batch_df.copy()
                    results_df['Prediction'] = ["Fraud" if p == 1 else "Legit" for p in preds]
                    if probs[0] is not None:
                        results_df['Fraud_Probability'] = probs
                        
                    st.write("### Results Preview")
                    st.dataframe(results_df.head(50))
                    
                    num_frauds = sum(preds)
                    st.warning(f"Found {num_frauds} potentially fraudulent transactions out of {len(preds)}.")
                    
                    csv = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button("Download Full Prediction Results", data=csv, file_name="predictions.csv", mime="text/csv")
                except Exception as e:
                    st.error(f"Error processing batch: {e}")
