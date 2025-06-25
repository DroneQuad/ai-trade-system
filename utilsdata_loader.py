import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_data(ticker, start_date, end_date):
    """Download data historis dari Yahoo Finance"""
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            raise ValueError("Data tidak ditemukan")
        return data
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return pd.DataFrame()

def preprocess_data(data, lookback=60):
    """Preprocessing data dan feature engineering"""
    if len(data) < lookback + 10:
        raise ValueError(f"Data tidak cukup. Dapat {len(data)} butuh minimal {lookback+10} baris.")
    
    # Normalisasi
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Close']])
    
    # Membuat dataset dengan lookback
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y, scaler
