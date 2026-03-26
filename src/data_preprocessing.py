import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sys

def load_and_validate_data(filepath='data/creditcard.csv'):
    """Load the dataset and perform basic schema validation."""
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {filepath}")
        return None
    
    # Validation checks
    required_cols = ['Time', 'Amount', 'Class'] + [f'V{i}' for i in range(1, 29)]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Validation Error: Missing required columns: {missing_cols}")
        return None
    
    return df

def preprocess_data(df):
    """
    Handle duplicates, scale Time, Amount, and Hour_of_Day features.
    """
    # 1. Handle Duplicates
    df = df.drop_duplicates()
    
    # 2. Feature Engineering: Time-based aggregations
    if 'Time' in df.columns:
        df['Hour_of_Day'] = (df['Time'] // 3600) % 24
        
    # 3. Scale numerical features together
    if 'Amount' in df.columns and 'Time' in df.columns:
        scaler = StandardScaler()
        cols_to_scale = ['Amount', 'Time', 'Hour_of_Day']
        
        scaled_features = scaler.fit_transform(df[cols_to_scale])
        
        df.drop(cols_to_scale, axis=1, inplace=True)
        
        # Move scaled features to front
        df.insert(0, 'Scaled_Amount', scaled_features[:, 0])
        df.insert(1, 'Scaled_Time', scaled_features[:, 1])
        df.insert(2, 'Scaled_Hour', scaled_features[:, 2])
        return df, scaler
    return df, None

def get_train_test_split(df, test_size=0.2, random_state=42):
    X = df.drop('Class', axis=1)
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    return X_train, X_test, y_train, y_test
