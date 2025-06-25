import backtrader as bt
import matplotlib.pyplot as plt
import numpy as np

class AIStrategy(bt.Strategy):
    """Strategi Trading berbasis AI"""
    params = (('printlog', False),)

    def __init__(self):
        self.signal = 0

    def next(self):
        if self.signal == 1 and not self.position:
            size = self.broker.getvalue() * 0.02 / self.data.close[0]
            self.buy(size=size)
        elif self.signal == -1 and self.position:
            self.close()

def run_backtest(data, signals, initial_cash=10000):
    """Backtesting dengan Backtrader"""
    try:
        cerebro = bt.Cerebro()
        cerebro.broker.set_cash(initial_cash)
        
        # Add data feed
        data_feed = bt.feeds.PandasData(dataname=data)
        cerebro.adddata(data_feed)
        
        # Add strategy
        cerebro.addstrategy(AIStrategy)
        
        # Add analyzers
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        
        # Simpan sinyal
        cerebro.strats[0][0][0].signal = signals[-1]
        
        # Jalankan backtest
        results = cerebro.run()
        
        # Dapatkan hasil
        strat = results[0]
        sharpe = strat.analyzers.sharpe.get_analysis()['sharperatio']
        drawdown = strat.analyzers.drawdown.get_analysis()['max']['drawdown']
        final_value = cerebro.broker.getvalue()
        returns = (final_value - initial_cash) / initial_cash * 100
        
        return cerebro, sharpe, drawdown, returns, final_value
    except Exception as e:
        print(f"Backtest error: {str(e)}")
        return None

def plot_equity_curve(cerebro):
    """Plot equity curve dari backtest"""
    fig = cerebro.plot(style='candlestick', volume=False, iplot=False)[0][0]
    plt.close()  # Tutup plot yang tidak perlu
    return fig
