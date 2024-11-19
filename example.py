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
    def __init__(self):
        super().__init__()
        self.short_window = 20
        self.long_window = 50
        self.prices = {}
        
    def should_enter(self, asset):
        try:
            if asset.symbol not in self.prices:
                self.prices[asset.symbol] = []
            self.prices[asset.symbol].append(asset.price)
            
            if len(self.prices[asset.symbol]) < self.long_window:
                return False
                
            short_ma = np.mean(self.prices[asset.symbol][-self.short_window:])
            long_ma = np.mean(self.prices[asset.symbol][-self.long_window:])
            
            return bool(short_ma > long_ma)  # Explicitly convert to bool
        except Exception as e:
            print(f"Error in should_enter: {e}")
            return False
        
    def should_exit(self, asset, position):
        try:
            if asset.symbol not in self.prices:
                return False
                
            short_ma = np.mean(self.prices[asset.symbol][-self.short_window:])
            long_ma = np.mean(self.prices[asset.symbol][-self.long_window:])
            
            return bool(short_ma < long_ma)  # Explicitly convert to bool
        except Exception as e:
            print(f"Error in should_exit: {e}")
            return False

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
        asset.timestamp = bt.TimePoint()  # Using default timestamp for now
        data.append(asset)
    
    return data

def plot_results(equity_curve, symbol, prices):
    fig = make_subplots(rows=2, cols=1, 
                       subplot_titles=('Equity Curve', 'Asset Price'),
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
    # Create backtest engine
    engine = bt.BacktestEngine()
    
    # Set parameters
    engine.set_initial_capital(100000.0)  # $100,000 initial capital
    engine.set_commission_rate(0.001)     # 0.1% commission rate
    engine.set_max_position_size(1000)    # Maximum position size
    engine.set_stop_loss(0.02)           # 2% stop loss
    
    # Generate and add sample data
    print("Generating sample data...")
    symbol = "AAPL"
    data = generate_sample_data(symbol)
    print(f"Generated {len(data)} data points")
    engine.add_data(symbol, data)
    
    # Create and set strategy
    print("Creating strategy...")
    strategy = MovingAverageCrossStrategy()
    engine.set_strategy(strategy)
    
    # Run backtest
    print("Starting backtest...")
    engine.run()
    print("Backtest completed")
    
    # Get results
    print("Getting results...")
    total_return = engine.get_total_return()
    sharpe_ratio = engine.get_sharpe_ratio()
    max_drawdown = engine.get_max_drawdown()
    win_rate = engine.get_win_rate()
    profit_factor = engine.get_profit_factor()
    avg_trade = engine.get_avg_trade()
    
    print(f"\nBacktest Results:")
    print(f"Total Return: {total_return:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2%}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Average Trade: ${avg_trade:.2f}")
    
    # Plot results
    print("Getting equity curve...")
    equity_curve = engine.get_equity_curve()
    prices = [asset.price for asset in data]
    print(f"Equity curve length: {len(equity_curve)}")
    print(f"Prices length: {len(prices)}")
    print("Plotting results...")
    plot_results(equity_curve, symbol, prices)

if __name__ == "__main__":
    main()
