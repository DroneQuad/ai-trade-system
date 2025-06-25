import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
import utils
import time
import gc

# Konfigurasi halaman
st.set_page_config(
    page_title="🚀 AI Trading System", 
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("🚀 AI Trading System")
st.markdown("""
**Sistem trading otomatis berbasis AI** | [GitHub](https://github.com/your-repo)
""")

# Inisialisasi session state
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'signals' not in st.session_state:
    st.session_state.signals = None
if 'test_data' not in st.session_state:
    st.session_state.test_data = None
if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = None

# Fungsi untuk menghasilkan sinyal trading
def generate_trading_signals(predictions, threshold):
    signal_diff = np.diff(predictions.flatten())
    signals = np.zeros_like(signal_diff)
    signals[signal_diff > threshold] = 1    # Buy signal
    signals[signal_diff < -threshold] = -1   # Sell signal
    return np.concatenate(([0], signals))    # Pad dengan nilai awal 0

# Tampilan utama
def main():
    # Sidebar konfigurasi
    st.sidebar.header("⚙️ Konfigurasi Trading")
    ticker = st.sidebar.text_input("Simbol Saham", "AAPL")
    start_date = st.sidebar.date_input("Tanggal Mulai", datetime(2020, 1, 1))
    end_date = st.sidebar.date_input("Tanggal Akhir", datetime.today())
    lookback = st.sidebar.slider("Lookback Period", 30, 200, 60)
    threshold = st.sidebar.slider("Signal Threshold", 0.01, 0.1, 0.02)
    initial_cash = st.sidebar.number_input("Modal Awal ($)", 1000, 1000000, 10000)
    
    # Reset session state
    if st.sidebar.button("🔄 Reset Session"):
        st.session_state.clear()
        gc.collect()
        st.experimental_rerun()

    # Download data
    with st.spinner("Mengunduh data..."):
        data = utils.load_data(ticker, start_date, end_date)
    
    if data.empty:
        st.error("⚠️ Data tidak ditemukan. Periksa simbol saham dan tanggal.")
        return

    # Tampilkan data
    st.subheader(f"📊 Data Historis {ticker}")
    st.dataframe(data.tail(), height=200)

    # Tab untuk analisis
    tab1, tab2, tab3 = st.tabs(["📈 Chart Saham", "🤖 Model AI", "🧪 Backtest"])

    with tab1:
        st.subheader("Performa Saham")
        fig, ax = plt.subplots(figsize=(12, 6))
        data['Close'].plot(ax=ax, title=f"Harga {ticker}", grid=True)
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Indikator Teknikal")
        col1, col2 = st.columns(2)
        with col1:
            window = st.slider("Moving Average Window", 5, 200, 50, key="ma_window")
            data['MA'] = data['Close'].rolling(window=window).mean()
            fig_ma, ax_ma = plt.subplots(figsize=(10, 4))
            data[['Close', 'MA']].plot(ax=ax_ma)
            st.pyplot(fig_ma)
            plt.close(fig_ma)

        with col2:
            rsi_window = st.slider("RSI Window", 5, 30, 14, key="rsi_window")
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).fillna(0)
            loss = (-delta.where(delta < 0, 0)).fillna(0)
            avg_gain = gain.rolling(rsi_window, min_periods=1).mean()
            avg_loss = loss.rolling(rsi_window, min_periods=1).mean()
            rs = avg_gain / (avg_loss + 1e-10)  # Hindari pembagian nol
            data['RSI'] = 100 - (100 / (1 + rs))
            fig_rsi, ax_rsi = plt.subplots(figsize=(10, 4))
            ax_rsi.plot(data['RSI'])
            ax_rsi.axhline(30, color='r', linestyle='--')
            ax_rsi.axhline(70, color='r', linestyle='--')
            st.pyplot(fig_rsi)
            plt.close(fig_rsi)

    with tab2:
        st.subheader("Pelatihan Model LSTM")
        try:
            # Batasi jumlah data untuk mengurangi beban
            if len(data) > 1000:
                data = data.iloc[-1000:]
                
            X, y, scaler = utils.preprocess_data(data, lookback)
        except Exception as e:
            st.error(f"❌ Error preprocessing: {str(e)}")
            return

        # Split data
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        if st.button("🚀 Latih Model", key="train_model") or st.session_state.trained_model:
            if not st.session_state.trained_model:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Bangun model
                model = utils.build_lstm_model((X_train.shape[1], 1))
                
                # Latih model dengan pembaruan progres
                history = utils.train_model_with_progress(
                    model, 
                    X_train, y_train, 
                    X_test, y_test,
                    progress_bar,
                    status_text
                )
                st.session_state.trained_model = (model, history, scaler)
            else:
                model, history, scaler = st.session_state.trained_model

            # Plot loss
            fig_loss, ax_loss = plt.subplots(figsize=(10, 4))
            ax_loss.plot(history.history['loss'], label='Train Loss')
            ax_loss.plot(history.history['val_loss'], label='Validation Loss')
            ax_loss.set_title('Model Loss')
            ax_loss.set_ylabel('Loss')
            ax_loss.set_xlabel('Epoch')
            ax_loss.legend()
            st.pyplot(fig_loss)
            plt.close(fig_loss)

            # Prediksi
            predictions = model.predict(X_test, verbose=0)
            predictions = scaler.inverse_transform(predictions)
            actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

            # Plot prediksi vs aktual (hanya 100 data terakhir)
            plot_size = min(100, len(actual_prices))
            fig_pred, ax_pred = plt.subplots(figsize=(12, 6))
            ax_pred.plot(actual_prices[-plot_size:], label='Harga Aktual')
            ax_pred.plot(predictions[-plot_size:], label='Prediksi LSTM', alpha=0.7)
            ax_pred.set_title('Prediksi vs Aktual')
            ax_pred.set_ylabel('Harga')
            ax_pred.legend()
            st.pyplot(fig_pred)
            plt.close(fig_pred)

            # Generate sinyal
            signals = generate_trading_signals(predictions, threshold)
            st.session_state.signals = signals
            st.session_state.test_data = data.iloc[-len(signals):].copy()

            st.success(f"✅ Model terlatih | Sinyal terdeteksi: {sum(signals != 0)}")
            st.session_state.backtest_results = None  # Reset hasil backtest sebelumnya

    with tab3:
        st.subheader("Backtesting Strategi")
        if st.session_state.signals is None:
            st.warning("⚠️ Latih model terlebih dahulu di tab Model AI")
            return

        signals = st.session_state.signals
        test_data = st.session_state.test_data

        if st.button("🚀 Jalankan Backtest", key="run_backtest"):
            with st.spinner("Menjalankan backtest..."):
                # Jalankan backtest di fungsi terpisah
                results = utils.run_backtest(test_data, signals, initial_cash)
                
            if results is None:
                st.error("❌ Error dalam backtesting")
                return
                
            cerebro, sharpe, drawdown, returns, final_value = results
            st.session_state.backtest_results = (cerebro, sharpe, drawdown, returns, final_value)
            
        if st.session_state.backtest_results:
            cerebro, sharpe, drawdown, returns, final_value = st.session_state.backtest_results
            
            # Tampilkan hasil
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Nilai Akhir", f"${final_value:,.2f}", f"{returns:.2f}%")
            col2.metric("Sharpe Ratio", f"{sharpe:.2f}")
            col3.metric("Max Drawdown", f"{drawdown:.2f}%")
            col4.metric("Jumlah Trade", f"{sum(signals != 0)}")
            
            # Plot hasil backtest (sederhana)
            st.subheader("Equity Curve")
            fig_equity = utils.plot_equity_curve(cerebro)
            st.pyplot(fig_equity)
            plt.close(fig_equity)
            
            # Risk Management
            st.subheader("🔒 Manajemen Risiko")
            st.info("""
            - **Position Sizing**: 2% modal per trade
            - **Stop Loss**: Auto trigger pada -5% per posisi
            - **Take Profit**: Auto trigger pada +8% per posisi
            """)

if __name__ == "__main__":
    main()
