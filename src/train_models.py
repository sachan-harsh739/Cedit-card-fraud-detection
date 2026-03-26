import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (classification_report, precision_score, recall_score,
                             f1_score, accuracy_score, roc_auc_score, average_precision_score,
                             precision_recall_curve, confusion_matrix)
import shap
import logging

from data_preprocessing import load_and_validate_data, preprocess_data, get_train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def plot_confusion_matrix(y_true, y_pred, name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(f'reports/figures/cm_{name.replace(" ", "_")}.png')
    plt.close()

def plot_pr_curve(y_true, y_probs, name):
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    plt.figure()
    plt.plot(recall, precision, marker='.')
    plt.title(f'Precision-Recall Curve - {name}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig(f'reports/figures/pr_{name.replace(" ", "_")}.png')
    plt.close()
    
    # Calculate optimal threshold maximizing F1-Score
    fscore = (2 * precision * recall) / (precision + recall + 1e-10)
    ix = np.argmax(fscore)
    best_thresh = thresholds[ix] if ix < len(thresholds) else 0.5
    logger.info(f"Optimal Threshold for {name}: {best_thresh:.4f} (F1: {fscore[ix]:.4f})")
    
    return best_thresh

def plot_feature_importance(model, feature_names, name):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        plt.figure(figsize=(10,6))
        plt.title(f"Top 10 Feature Importances - {name}")
        plt.bar(range(10), importances[indices], align="center")
        plt.xticks(range(10), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.savefig(f'reports/figures/fi_{name.replace(" ", "_")}.png')
        plt.close()

def evaluate_model(model, X_test, y_test, name, optimal_thresh):
    y_probs = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_test)
    y_pred = (y_probs >= optimal_thresh).astype(int)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_probs)
    pr_auc = average_precision_score(y_test, y_probs)
    
    logger.info(f"--- Evaluation for {name} (Threshold: {optimal_thresh:.4f}) ---")
    logger.info(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")
    logger.info("\n" + classification_report(y_test, y_pred))
    
    plot_confusion_matrix(y_test, y_pred, name)
    
    return rec, roc_auc, pr_auc, {
        'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'roc_auc': roc_auc, 'pr_auc': pr_auc
    }

def generate_shap_plots(model, X_train, feature_names, name):
    logger.info(f"Generating SHAP values for {name}...")
    try:
        X_sample = shap.sample(X_train, 100)
        
        if isinstance(model, XGBClassifier) or isinstance(model, RandomForestClassifier):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
        else:
            explainer = shap.LinearExplainer(model, X_sample)
            shap_values = explainer.shap_values(X_sample)
            
        plt.figure()
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
        plt.savefig(f'reports/figures/shap_summary_{name.replace(" ", "_")}.png', bbox_inches='tight')
        plt.close()
    except Exception as e:
        logger.error(f"SHAP generation failed for {name}: {e}")

def train_and_evaluate(data_path='data/creditcard.csv'):
    df = load_and_validate_data(data_path)
    if df is None:
        logger.error(f"Failed to load data from {data_path}.")
        return
    
    logger.info("Preprocessing data...")
    df, scaler = preprocess_data(df)
    
    logger.info("Splitting data...")
    X_train, X_test, y_train, y_test = get_train_test_split(df)
    feature_names = X_train.columns.tolist()

    os.makedirs('models', exist_ok=True)
    os.makedirs('reports/figures', exist_ok=True)
    
    cv_strategy = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    smt = SMOTETomek(random_state=42, n_jobs=-1)
    
    pipelines = {
        'Logistic Regression': Pipeline([
            ('resample', smt),
            ('model', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
        ]),
        'Random Forest': Pipeline([
            ('resample', smt),
            ('model', RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1))
        ]),
        'XGBoost': Pipeline([
            ('resample', smt),
            ('model', XGBClassifier(eval_metric='logloss', random_state=42))
        ])
    }
    
    xgb_params = {
        'model__max_depth': [3, 5],
        'model__learning_rate': [0.1, 0.2],
        'model__n_estimators': [50]
    }
    
    best_model_pipeline = None
    best_pr_auc = 0
    best_model_name = ""
    best_threshold = 0.5
    best_metrics = {}
    best_params = {}

    for name, pipeline in pipelines.items():
        logger.info(f"Training Pipeline for {name}...")
        
        if name == 'XGBoost':
            search = RandomizedSearchCV(pipeline, xgb_params, n_iter=2, scoring='average_precision', 
                                        cv=cv_strategy, random_state=42, n_jobs=-1)
            search.fit(X_train, y_train)
            fitted_pipeline = search.best_estimator_
            params = search.best_params_
            logger.info(f"Best XGBoost params: {params}")
        else:
            fitted_pipeline = pipeline.fit(X_train, y_train)
            params = pipeline.get_params()
            
        final_model = fitted_pipeline.named_steps['model']
        plot_feature_importance(final_model, feature_names, name)
        generate_shap_plots(final_model, X_train, feature_names, name)
        
        y_probs_val = fitted_pipeline.predict_proba(X_test)[:, 1] if hasattr(fitted_pipeline, 'predict_proba') else fitted_pipeline.predict(X_test)
        opt_thresh = plot_pr_curve(y_test, y_probs_val, name)
        
        recall, roc_auc, pr_auc, metrics = evaluate_model(fitted_pipeline, X_test, y_test, name, opt_thresh)
        
        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            best_model_pipeline = fitted_pipeline
            best_model_name = name
            best_threshold = opt_thresh
            best_metrics = metrics
            best_params = params

    logger.info(f"Best model based on PR-AUC: {best_model_name} with PR-AUC: {best_pr_auc:.4f}")
    
    model_path = 'models/fraud_model.pkl'
    joblib.dump({
        'version': '1.0',
        'model_name': best_model_name,
        'model': best_model_pipeline.named_steps['model'],
        'scaler': scaler, 
        'features': feature_names,
        'optimal_threshold': float(best_threshold),
        'metrics': best_metrics,
        'hyperparameters': str(best_params)
    }, model_path)
    logger.info(f"Best model dictionary saved to {model_path}")

if __name__ == "__main__":
    train_and_evaluate(data_path='data/creditcard.csv')
