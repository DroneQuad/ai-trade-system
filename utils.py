import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import backtrader as bt
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
    print("Using Standalone Keras")
except ImportError:
    try:
        # Coba impor dari TensorFlow jika Keras standalone tidak ada
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.optimizers import Adam
        print("Using TensorFlow Keras")
    except ImportError:
        print("Error: Neither keras nor tensorflow is installed")

# Fungsi untuk download data
def load_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
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
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

# Melatih model
def train_model(model, X_train, y_train, X_test, y_test, epochs=30, batch_size=32):
    history = model.fit(X_train, y_train, 
                        validation_data=(X_test, y_test),
                        epochs=epochs, 
                        batch_size=batch_size,
                        verbose=0)
    return history

# Strategi backtrader
class AIStrategy(bt.Strategy):
    params = (('printlog', False),)

    def __init__(self):
        self.signal = 0  # Sinyal dari luar

    def next(self):
        # Ambil sinyal untuk hari ini
        self.signal = self.datas[0].signal[0]
        
        if self.signal == 1 and not self.position:
            size = min(self.broker.getvalue() * 0.02 / self.data.close[0], 
                      self.broker.getvalue() / self.data.close[0])
            self.buy(size=size)
        elif self.signal == -1 and self.position:
            self.close()

# Menjalankan backtest
def run_backtest(data, signals, initial_cash=10000):
    try:
        # Pastikan panjang data dan sinyal sama
        if len(data) != len(signals):
            min_length = min(len(data), len(signals))
            data = data.iloc[-min_length:]
            signals = signals[-min_length:]
        
        # Tambahkan sinyal ke dataframe
        data = data.copy()
        data['signal'] = signals
        
        cerebro = bt.Cerebro()
        cerebro.broker.set_cash(initial_cash)
        
        # Buat data feed dengan kolom sinyal
        data_feed = bt.feeds.PandasData(
            dataname=data,
            datetime=None,
            open='Open',
            high='High',
            low='Low',
            close='Close',
            volume='Volume',
            openinterest=-1,
            signal='signal'  # Kolom sinyal
        )
        cerebro.adddata(data_feed)
        
        cerebro.addstrategy(AIStrategy)
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        
        # Konfigurasi broker
        cerebro.broker.set_commission(commission=0.001)  # 0.1% komisi
        
        results = cerebro.run()
        strat = results[0]
        
        sharpe = strat.analyzers.sharpe.get_analysis()['sharperatio']
        drawdown = strat.analyzers.drawdown.get_analysis()['max']['drawdown']
        final_value = cerebro.broker.getvalue()
        returns = (final_value - initial_cash) / initial_cash * 100
        
        return cerebro, sharpe, drawdown, returns, final_value
    except Exception as e:
        print(f"Error in backtesting: {e}")
        return None

# Plot equity curve
def plot_equity_curve(cerebro):
    # Ekstrak nilai ekuitas dari broker
    equity = cerebro.broker.get_value()
    
    # Buat plot sederhana
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(equity, label='Equity Curve')
    ax.set_title('Equity Curve')
    ax.set_xlabel('Hari')
    ax.set_ylabel('Nilai Portofolio ($)')
    ax.grid(True)
    ax.legend()
    
    return fig
