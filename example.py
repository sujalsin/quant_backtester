import os
import sys

# Add the lib directory to Python path
lib_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib")
sys.path.append(lib_dir)

import backtester_core as bt
from datetime import datetime, timedelta
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class MovingAverageCrossStrategy(bt.Strategy):
    def __init__(self, short_window=20, long_window=50):
        super().__init__()
        self.short_window = short_window
        self.long_window = long_window
        self.prices = {}
        
    def should_enter(self, asset):
        if asset.symbol not in self.prices:
            self.prices[asset.symbol] = []
        self.prices[asset.symbol].append(asset.price)
        
        if len(self.prices[asset.symbol]) < self.long_window:
            return False
            
        short_ma = np.mean(self.prices[asset.symbol][-self.short_window:])
        long_ma = np.mean(self.prices[asset.symbol][-self.long_window:])
        
        return short_ma > long_ma
        
    def should_exit(self, asset, position):
        short_ma = np.mean(self.prices[asset.symbol][-self.short_window:])
        long_ma = np.mean(self.prices[asset.symbol][-self.long_window:])
        
        return short_ma < long_ma

def generate_sample_data(symbol, days=365, volatility=0.02):
    data = []
    base_price = 100.0
    current_price = base_price
    start_date = datetime.now() - timedelta(days=days)
    
    for i in range(days):
        current_date = start_date + timedelta(days=i)
        return_multiplier = 1.0 + np.random.normal(0, volatility)
        current_price *= return_multiplier
        
        asset = bt.Asset()
        asset.symbol = symbol
        asset.asset_class = "equity"
        asset.price = current_price
        asset.volume = np.random.uniform(100000, 1000000)
        asset.timestamp = current_date
        
        data.append(asset)
    
    return data

def plot_results(equity_curve, symbol, prices):
    fig = make_subplots(rows=2, cols=1, 
                        subplot_titles=('Portfolio Value', f'{symbol} Price'),
                        vertical_spacing=0.1)
    
    # Plot equity curve
    fig.add_trace(
        go.Scatter(y=equity_curve, name="Portfolio Value"),
        row=1, col=1
    )
    
    # Plot asset price
    fig.add_trace(
        go.Scatter(y=prices, name=f"{symbol} Price"),
        row=2, col=1
    )
    
    fig.update_layout(height=800, title_text="Backtest Results")
    fig.show()

def main():
    # Initialize engine
    engine = bt.BacktestEngine()
    engine.set_initial_capital(100000.0)
    engine.set_commission_rate(0.001)
    
    # Generate sample data
    symbol = "AAPL"
    data = generate_sample_data(symbol)
    engine.add_data(symbol, data)
    
    # Create and set strategy
    strategy = MovingAverageCrossStrategy(short_window=20, long_window=50)
    engine.set_strategy(strategy)
    
    # Run backtest
    engine.run()
    
    # Get results
    total_return = engine.get_total_return()
    sharpe = engine.get_sharpe_ratio()
    max_dd = engine.get_max_drawdown()
    equity_curve = engine.get_equity_curve()
    
    print(f"Total Return: {total_return:.2%}")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Max Drawdown: {max_dd:.2%}")
    
    # Plot results
    prices = [asset.price for asset in data]
    plot_results(equity_curve, symbol, prices)

if __name__ == "__main__":
    main()
