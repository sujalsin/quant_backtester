import logging
import os
import sys
from datetime import datetime, timedelta

import backtester_core as bt

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Log Python path for debugging
logger.debug(f"Python path: {sys.path}")
lib_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib")
logger.debug(f"Looking for module in: {lib_dir}")

class BuyAndHoldStrategy(bt.Strategy):
    def should_enter(self, asset):
        # Simple buy and hold - buy at the start
        return True

    def should_exit(self, asset, position):
        # Never exit
        return False

def main():
    # Create synthetic price data
    data = []
    base_price = 100.0
    current_time = datetime.now()
    
    for i in range(5):  # 5 days of data
        asset = bt.Asset()
        asset.symbol = "TEST"
        asset.asset_class = "EQUITY"
        asset.price = base_price * (1 + 0.001 * i)  # Simple upward trend
        asset.volume = 1000000
        # Create a TimePoint object for timestamp
        time_point = bt.TimePoint()
        asset.timestamp = time_point
        data.append(asset)

    # Initialize backtester
    engine = bt.BacktestEngine()
    engine.set_initial_capital(10000.0)
    engine.set_commission_rate(0.001)  # 0.1% commission
    engine.set_max_position_size(1.0)  # Full position size
    engine.add_data("TEST", data)
    
    # Set strategy
    strategy = BuyAndHoldStrategy()
    engine.set_strategy(strategy)
    
    # Run backtest
    engine.run()
    
    # Print results
    print(f"Total Return: {engine.get_total_return():.2%}")
    print(f"Sharpe Ratio: {engine.get_sharpe_ratio():.2f}")
    print(f"Max Drawdown: {engine.get_max_drawdown():.2%}")
    print(f"Win Rate: {engine.get_win_rate():.2%}")
    print(f"Profit Factor: {engine.get_profit_factor():.2f}")
    print(f"Average Trade: {engine.get_avg_trade():.2f}")

if __name__ == "__main__":
    main()
