import pytest
import pandas as pd
import numpy as np
import sys
import os

# Ensure src module is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from data_preprocessing import preprocess_data, get_train_test_split

@pytest.fixture
def sample_data():
    # Make a dummy dataframe mimicking the structure
    data = {
        'Time': [0, 3600, 7200, 10800, 14400, 18000],
        'Amount': [10.0, 20.0, 50.0, 100.0, 200.0, 500.0],
        'Class': [0, 1, 0, 0, 1, 0]
    }
    for i in range(1, 29):
        data[f'V{i}'] = np.random.randn(6)
    
    df = pd.DataFrame(data)
    # Add a duplicate for testing duplicate removal
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    return df

def test_preprocess_data(sample_data):
    assert len(sample_data) == 7
    processed_df, scaler = preprocess_data(sample_data.copy())
    
    # Should drop 1 duplicate
    assert len(processed_df) == 6
    
    # Check new columns exist
    assert 'Scaled_Amount' in processed_df.columns
    assert 'Scaled_Time' in processed_df.columns
    assert 'Scaled_Hour' in processed_df.columns
    assert 'Time' not in processed_df.columns
    assert 'Amount' not in processed_df.columns
    
def test_train_test_split(sample_data):
    processed_df, _ = preprocess_data(sample_data)
    # 6 rows total. If test_size 0.5, 3 train 3 test
    X_train, X_test, y_train, y_test = get_train_test_split(processed_df, test_size=0.5, random_state=42)
    assert len(X_train) == 3
    assert len(X_test) == 3
