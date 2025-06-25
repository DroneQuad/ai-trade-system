import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import io
import os
import warnings
warnings.filterwarnings('ignore')

# Coba impor Keras (tanpa TensorFlow)
try:
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout
    from keras.optimizers import Adam
    from keras.callbacks import Callback
    print("Using Standalone Keras")
except ImportError:
    try:
        # Coba impor dari TensorFlow jika Keras standalone tidak ada
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import Callback
        print("Using TensorFlow Keras")
    except ImportError:
        print("Error: Neither keras nor tensorflow is installed")

# Fungsi untuk download data
def load_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            return pd.DataFrame()
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

# Preprocessing data
def preprocess_data(data, lookback=60):
    if len(data) < lookback + 10:
        raise ValueError(f"Data tidak cukup. Dapat {len(data)} butuh minimal {lookback+10} baris.")
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Close']])
    
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y, scaler

# Membangun model LSTM
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(32, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(16, return_sequences=False),
        Dropout(0.2),
        Dense(8, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

# Callback untuk menampilkan progres training
class TrainingProgressCallback(Callback):
    def __init__(self, progress_bar, status_text, total_epochs):
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.total_epochs = total_epochs
        
    def on_epoch_end(self, epoch, logs=None):
        current_progress = (epoch + 1) / self.total_epochs
        self.progress_bar.progress(current_progress)
        self.status_text.text(f"Epoch {epoch+1}/{self.total_epochs} - Loss: {logs['loss']:.4f}, Val Loss: {logs['val_loss']:.4f}")

# Melatih model dengan progres
def train_model_with_progress(model, X_train, y_train, X_test, y_test, progress_bar, status_text, epochs=15, batch_size=32):
    callback = TrainingProgressCallback(progress_bar, status_text, epochs)
    history = model.fit(
        X_train, y_train, 
        validation_data=(X_test, y_test),
        epochs=epochs, 
        batch_size=batch_size,
        verbose=0,
        callbacks=[callback]
    )
    return history

# Strategi trading sederhana untuk backtest
def simple_trading_strategy(data, signals, initial_cash=10000):
    cash = initial_cash
    position = 0
    portfolio_value = [cash]
    trades = []
    
    for i in range(len(data)):
        price = data['Close'].iloc[i]
        signal = signals[i]
        
        # Jual jika ada posisi dan sinyal jual
        if position > 0 and signal == -1:
            cash += position * price
            position = 0
            trades.append(('SELL', price, i))
        
        # Beli jika tidak ada posisi dan sinyal beli
        elif position == 0 and signal == 1:
            position = cash * 0.02 / price  # 2% dari modal
            cash -= position * price
            trades.append(('BUY', price, i))
        
        # Hitung nilai portofolio
        portfolio_value.append(cash + position * price)
    
    return portfolio_value, trades

# Menjalankan backtest
def run_backtest(data, signals, initial_cash=10000):
    try:
        # Pastikan panjang data dan sinyal sama
        if len(data) != len(signals):
            min_length = min(len(data), len(signals))
            data = data.iloc[-min_length:]
            signals = signals[-min_length:]
        
        # Jalankan backtest sederhana
        portfolio_value, trades = simple_trading_strategy(data, signals, initial_cash)
        
        # Hitung metrik performa
        returns = (portfolio_value[-1] - initial_cash) / initial_cash * 100
        sharpe = 0  # Implementasi Sharpe Ratio yang sesungguhnya lebih kompleks
        drawdown = 0
        
        # Simulasi hasil untuk kompatibilitas
        class CerebroSim:
            pass
        
        cerebro = CerebroSim()
        cerebro.portfolio_value = portfolio_value
        
        return cerebro, sharpe, drawdown, returns, portfolio_value[-1]
    except Exception as e:
        print(f"Error in backtesting: {e}")
        return None

# Plot equity curve
def plot_equity_curve(cerebro):
    # Buat plot sederhana
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(cerebro.portfolio_value, label='Equity Curve')
    ax.set_title('Equity Curve')
    ax.set_xlabel('Hari')
    ax.set_ylabel('Nilai Portofolio ($)')
    ax.grid(True)
    ax.legend()
    
    return fig
