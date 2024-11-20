import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add the lib directory to Python path
lib_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib")
logger.debug(f"Python path before: {sys.path}")
sys.path.append(lib_dir)
logger.debug(f"Python path after: {sys.path}")
logger.debug(f"Looking for module in: {lib_dir}")

try:
    import backtester_core as bt
    logger.debug("Successfully imported backtester_core")
except Exception as e:
    logger.error(f"Failed to import backtester_core: {e}")
    raise

from datetime import datetime, timedelta
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class MovingAverageCrossStrategy(bt.Strategy):
    def __init__(self):
        super().__init__()
        self.short_window = 5   # More sensitive short window
        self.long_window = 15   # More sensitive long window
        self.prices = {}
        self.volumes = {}
        self.high_prices = {}
        self.low_prices = {}
        self.last_signal = {}  # To avoid repeated signals
        self.min_price_history = 20  # Shorter history requirement
        self.trend_threshold = 0.01  # More sensitive trend threshold (1%)
        self.volume_ma_window = 10  # Window for volume moving average
        self.atr_period = 14    # ATR calculation period
        self.risk_per_trade = 0.01  # Risk 1% of capital per trade
        self.atr_position_size = True  # Use ATR for position sizing
        self.rsi_period = 14    # RSI calculation period
        self.trailing_stop_atr = 2.0  # Trailing stop distance in ATR units
        self.trailing_stops = {}  # Track trailing stops for each position
        
        # Portfolio risk management parameters
        self.max_portfolio_risk = 0.02  # Maximum 2% portfolio risk
        self.max_correlation = 0.7      # Maximum correlation between assets
        self.max_sector_exposure = 0.4  # Maximum 40% exposure per sector
        self.sector_map = {
            'AAPL': 'TECH',
            'MSFT': 'TECH',
            'GOOGL': 'TECH'
        }
        self.position_correlations = {}  # Store correlations between assets
        self.sector_exposure = {}       # Track sector exposure
        self.volatility_scaling = 1.0   # Dynamic volatility scaling
        self.max_positions = 3          # Maximum concurrent positions
        
    def calculate_ema(self, prices, period):
        """Calculate exponential moving average"""
        if len(prices) < period:
            return None
        prices = np.array(prices)
        alpha = 2.0 / (period + 1)
        weights = (1 - alpha) ** np.arange(period)[::-1]
        weights /= weights.sum()
        ema = np.convolve(prices, weights, mode='valid')[-1]
        return ema
        
    def calculate_rsi(self, prices):
        """Calculate Relative Strength Index"""
        if len(prices) < self.rsi_period + 1:
            return 50  # Neutral if not enough data
            
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-self.rsi_period:])
        avg_loss = np.mean(losses[-self.rsi_period:])
        
        if avg_loss == 0:
            return 100
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
        
    def update_trailing_stop(self, symbol, current_price):
        """Update trailing stop for a position"""
        if symbol not in self.trailing_stops:
            atr = self.calculate_atr(symbol)
            if atr is None:
                return None
            self.trailing_stops[symbol] = current_price - (self.trailing_stop_atr * atr)
        else:
            atr = self.calculate_atr(symbol)
            if atr is None:
                return self.trailing_stops[symbol]
            new_stop = current_price - (self.trailing_stop_atr * atr)
            self.trailing_stops[symbol] = max(self.trailing_stops[symbol], new_stop)
        return self.trailing_stops[symbol]
        
    def calculate_atr(self, symbol):
        """Calculate Average True Range"""
        if len(self.prices[symbol]) < 2:
            return None
            
        true_ranges = []
        for i in range(1, len(self.prices[symbol])):
            high = self.high_prices[symbol][i]
            low = self.low_prices[symbol][i]
            prev_close = self.prices[symbol][i-1]
            
            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)
            
            true_range = max(tr1, tr2, tr3)
            true_ranges.append(true_range)
            
            if len(true_ranges) > self.atr_period:
                true_ranges.pop(0)
                
        if len(true_ranges) < self.atr_period:
            return None
            
        return np.mean(true_ranges)
        
    def calculate_portfolio_risk(self, positions):
        """Calculate total portfolio risk considering correlations"""
        if not positions:
            return 0.0
            
        total_risk = 0.0
        position_values = []
        
        # Calculate individual position risks
        for pos in positions:
            if pos.symbol not in self.prices:
                continue
            
            atr = self.calculate_atr(pos.symbol)
            if atr is None:
                continue
                
            position_risk = (atr * pos.quantity * self.risk_per_trade)
            position_values.append(position_risk)
            
        # Add correlation impact
        for i, pos1 in enumerate(positions):
            for j, pos2 in enumerate(positions):
                if i >= j:
                    continue
                    
                correlation = self.get_correlation(pos1.symbol, pos2.symbol)
                total_risk += (2 * position_values[i] * position_values[j] * correlation)
                
        # Add individual risks
        for risk in position_values:
            total_risk += risk * risk
            
        return np.sqrt(total_risk)
        
    def get_correlation(self, symbol1, symbol2):
        """Calculate correlation between two assets"""
        key = tuple(sorted([symbol1, symbol2]))
        if key not in self.position_correlations:
            if (symbol1 not in self.prices or 
                symbol2 not in self.prices or 
                len(self.prices[symbol1]) < self.min_price_history or 
                len(self.prices[symbol2]) < self.min_price_history):
                return 0.0
                
            # Calculate returns
            returns1 = np.diff(self.prices[symbol1][-self.min_price_history:])
            returns2 = np.diff(self.prices[symbol2][-self.min_price_history:])
            
            # Calculate correlation
            if len(returns1) == len(returns2) and len(returns1) > 0:
                correlation = np.corrcoef(returns1, returns2)[0, 1]
                self.position_correlations[key] = correlation if not np.isnan(correlation) else 0.0
            else:
                self.position_correlations[key] = 0.0
                
        return self.position_correlations[key]
        
    def update_sector_exposure(self, positions):
        """Update sector exposure tracking"""
        self.sector_exposure = {}
        total_value = sum(pos.quantity * self.prices[pos.symbol][-1] for pos in positions 
                         if pos.symbol in self.prices and len(self.prices[pos.symbol]) > 0)
        
        if total_value == 0:
            return
            
        for pos in positions:
            if pos.symbol not in self.prices or len(self.prices[pos.symbol]) == 0:
                continue
                
            sector = self.sector_map.get(pos.symbol, 'OTHER')
            position_value = pos.quantity * self.prices[pos.symbol][-1]
            self.sector_exposure[sector] = self.sector_exposure.get(sector, 0) + position_value / total_value
            
    def update_volatility_scaling(self):
        """Update volatility scaling based on market conditions"""
        total_atr = 0
        count = 0
        
        for symbol in self.prices:
            atr = self.calculate_atr(symbol)
            if atr is not None:
                total_atr += atr
                count += 1
                
        if count > 0:
            avg_atr = total_atr / count
            # Scale between 0.5 and 1.5 based on average ATR
            self.volatility_scaling = 1.0 + (0.5 * (1.0 - avg_atr / self.atr_period))
            self.volatility_scaling = max(0.5, min(1.5, self.volatility_scaling))
            
    def calculate_position_size(self, asset, capital):
        """Calculate position size with portfolio constraints"""
        # Get current positions from backtester
        positions = self.get_positions()
        
        # Update portfolio metrics
        self.update_sector_exposure(positions)
        self.update_volatility_scaling()
        
        # Calculate base position size
        atr = self.calculate_atr(asset.symbol)
        if atr is None:
            return 0
            
        # Calculate stop loss price (2 ATR below entry for long positions)
        stop_price = asset.price - (2 * atr)
        risk_amount = capital * self.risk_per_trade * self.volatility_scaling
        
        # Calculate position size based on risk amount and stop distance
        price_risk = asset.price - stop_price
        if price_risk <= 0:
            return 0
            
        position_size = risk_amount / price_risk
        
        # Apply portfolio constraints
        
        # 1. Check number of positions
        if len(positions) >= self.max_positions:
            return 0
            
        # 2. Check sector exposure
        sector = self.sector_map.get(asset.symbol, 'OTHER')
        if self.sector_exposure.get(sector, 0) >= self.max_sector_exposure:
            return 0
            
        # 3. Check correlation with existing positions
        for pos in positions:
            if self.get_correlation(asset.symbol, pos.symbol) > self.max_correlation:
                position_size *= 0.5  # Reduce position size if highly correlated
                
        # 4. Check portfolio risk
        portfolio_risk = self.calculate_portfolio_risk(positions)
        if portfolio_risk > self.max_portfolio_risk * capital:
            risk_ratio = (self.max_portfolio_risk * capital) / portfolio_risk
            position_size *= risk_ratio
            
        # 5. Apply maximum position size limit
        max_position = capital * 0.2  # 20% max position size
        position_value = position_size * asset.price
        if position_value > max_position:
            position_size = max_position / asset.price
            
        return position_size
        
    def calculate_trend_strength(self, prices):
        """Calculate trend strength using exponential moving averages"""
        if len(prices) < self.min_price_history:
            return 0
            
        short_ema = self.calculate_ema(prices[-self.min_price_history:], 5)
        med_ema = self.calculate_ema(prices[-self.min_price_history:], 10)
        long_ema = self.calculate_ema(prices[-self.min_price_history:], 20)
        
        if short_ema is None or med_ema is None or long_ema is None:
            return 0
            
        # Calculate trend alignment
        trend_strength = ((short_ema - long_ema) / long_ema +
                         (med_ema - long_ema) / long_ema) / 2
        
        return trend_strength
        
    def check_volume_confirmation(self, symbol, volume):
        """Check if current volume is above average"""
        if symbol not in self.volumes:
            self.volumes[symbol] = []
            
        self.volumes[symbol].append(volume)
        
        if len(self.volumes[symbol]) < self.volume_ma_window:
            return False
            
        volume_ma = np.mean(self.volumes[symbol][-self.volume_ma_window:])
        return volume > volume_ma * 1.2  # Require 20% above average volume
        
    def should_enter(self, asset):
        try:
            if asset.symbol not in self.prices:
                self.prices[asset.symbol] = []
                self.high_prices[asset.symbol] = []
                self.low_prices[asset.symbol] = []
                self.last_signal[asset.symbol] = False
                
            self.prices[asset.symbol].append(asset.price)
            self.high_prices[asset.symbol].append(asset.high)
            self.low_prices[asset.symbol].append(asset.low)
            
            if len(self.prices[asset.symbol]) < self.min_price_history:
                return False
                
            # Calculate EMAs
            short_ema = self.calculate_ema(self.prices[asset.symbol][-self.short_window:], self.short_window)
            long_ema = self.calculate_ema(self.prices[asset.symbol][-self.long_window:], self.long_window)
            
            if short_ema is None or long_ema is None:
                return False
                
            # Calculate trend strength
            trend_strength = self.calculate_trend_strength(self.prices[asset.symbol])
            
            # Calculate RSI
            rsi = self.calculate_rsi(self.prices[asset.symbol])
            
            # Check volume confirmation
            volume_confirmed = self.check_volume_confirmation(asset.symbol, asset.volume)
            
            # Calculate ATR
            atr = self.calculate_atr(asset.symbol)
            atr_ok = atr is not None and atr > 0
            
            # Get current positions
            positions = self.get_positions()
            
            # Check portfolio constraints
            if len(positions) >= self.max_positions:
                return False
                
            # Check sector exposure
            sector = self.sector_map.get(asset.symbol, 'OTHER')
            self.update_sector_exposure(positions)
            if self.sector_exposure.get(sector, 0) >= self.max_sector_exposure:
                return False
                
            # Check correlations with existing positions
            for pos in positions:
                if self.get_correlation(asset.symbol, pos.symbol) > self.max_correlation:
                    return False
                    
            # Generate signal with all confirmations
            should_enter = (short_ema > long_ema and 
                          trend_strength > self.trend_threshold and 
                          volume_confirmed and
                          atr_ok and
                          30 <= rsi <= 70 and  # Not overbought/oversold
                          not self.last_signal[asset.symbol])
            
            if should_enter:
                print(f"\nENTRY SIGNAL for {asset.symbol} at {asset.price:.2f}")
                print(f"Short EMA: {short_ema:.2f}, Long EMA: {long_ema:.2f}")
                print(f"Trend Strength: {trend_strength:.2%}")
                print(f"RSI: {rsi:.1f}")
                print(f"Volume: {asset.volume:,} (Above average)")
                print(f"ATR: {atr:.3f}")
                self.last_signal[asset.symbol] = True
                
            return should_enter
            
        except Exception as e:
            print(f"Error in should_enter: {e}")
            return False
        
    def should_exit(self, asset, position):
        try:
            if asset.symbol not in self.prices:
                return False
                
            # Update trailing stop
            trailing_stop = self.update_trailing_stop(asset.symbol, asset.price)
            if trailing_stop is None:
                trailing_stop = position.entry_price * 0.95  # Fallback to 5% fixed stop
                
            # Calculate EMAs
            short_ema = self.calculate_ema(self.prices[asset.symbol][-self.short_window:], self.short_window)
            long_ema = self.calculate_ema(self.prices[asset.symbol][-self.long_window:], self.long_window)
            
            if short_ema is None or long_ema is None:
                return False
                
            # Calculate trend strength
            trend_strength = self.calculate_trend_strength(self.prices[asset.symbol])
            
            # Calculate RSI
            rsi = self.calculate_rsi(self.prices[asset.symbol])
            
            # Check volume confirmation (more aggressive exit)
            volume_confirmed = self.check_volume_confirmation(asset.symbol, asset.volume)
            
            # Generate exit signal
            should_exit = (
                asset.price < trailing_stop or  # Trailing stop hit
                (short_ema < long_ema and volume_confirmed) or  # EMA cross with volume
                trend_strength < -self.trend_threshold or  # Trend reversal
                rsi > 70 or  # Overbought
                (volume_confirmed and short_ema < long_ema * 0.99)  # Fast exit on high volume
            )
            
            if should_exit:
                print(f"\nEXIT SIGNAL for {asset.symbol} at {asset.price:.2f}")
                print(f"Short EMA: {short_ema:.2f}, Long EMA: {long_ema:.2f}")
                print(f"Trend Strength: {trend_strength:.2%}")
                print(f"RSI: {rsi:.1f}")
                print(f"Volume: {asset.volume:,}")
                print(f"Trailing Stop: {trailing_stop:.2f}")
                print(f"P&L: ${(asset.price - position.entry_price) * position.quantity:.2f}")
                self.last_signal[asset.symbol] = False
                self.trailing_stops.pop(asset.symbol, None)
                
            return should_exit
            
        except Exception as e:
            print(f"Error in should_exit: {e}")
            return False
            
    def get_positions(self):
        """Get current positions from the backtester"""
        try:
            return self.backtester.get_positions()
        except AttributeError:
            return []  # Return empty list if backtester not initialized
            
def generate_sample_data(symbol, days=365, volatility=0.02, trend=0.0001):
    data = []
    base_price = 100.0
    current_price = base_price
    start_date = datetime.now() - timedelta(days=days)
    
    # Generate more realistic price movements
    prices = []
    for i in range(days):
        # Add trend component
        trend_component = 1.0 + trend
        # Add random walk component
        random_component = np.random.normal(0, volatility)
        # Add seasonal component (30-day cycle)
        seasonal_component = 0.005 * np.sin(2 * np.pi * i / 30)
        
        return_multiplier = trend_component * (1.0 + random_component + seasonal_component)
        current_price *= return_multiplier
        prices.append(current_price)
    
    # Smooth prices slightly to make them more realistic
    prices = np.array(prices)
    smoothed_prices = np.convolve(prices, np.ones(3)/3, mode='valid')
    
    # Create asset objects with the smoothed prices
    for i in range(len(smoothed_prices)):
        current_date = start_date + timedelta(days=i)
        
        asset = bt.Asset()
        asset.symbol = symbol
        asset.asset_class = "equity"
        asset.price = smoothed_prices[i]
        # Generate more realistic volume with some clustering
        base_volume = np.random.uniform(100000, 1000000)
        volume_multiplier = 1.0 + 0.5 * np.sin(2 * np.pi * i / 5)  # 5-day volume cycle
        asset.volume = int(base_volume * volume_multiplier)
        asset.high = smoothed_prices[i] * 1.01  # Add high price
        asset.low = smoothed_prices[i] * 0.99   # Add low price
        
        # Convert datetime to system_clock::time_point
        timestamp = bt.TimePoint()
        timestamp.from_datetime(current_date)
        asset.timestamp = timestamp
        
        data.append(asset)
    
    return data

def plot_results(equity_curve, symbol, prices):
    # Calculate daily returns
    daily_returns = np.diff(equity_curve) / equity_curve[:-1]
    
    # Calculate metrics
    total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
    volatility = np.std(daily_returns) * np.sqrt(252)  # Annualized
    sharpe = np.mean(daily_returns) * np.sqrt(252) / volatility if volatility != 0 else 0
    
    # Calculate drawdown
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / peak
    max_drawdown = np.max(drawdown)
    
    # Create figure with secondary y-axis
    fig = make_subplots(rows=3, cols=1,
                       subplot_titles=('Portfolio Performance', 'Asset Price', 'Drawdown'),
                       vertical_spacing=0.1,
                       row_heights=[0.5, 0.3, 0.2])
    
    # Plot equity curve
    fig.add_trace(
        go.Scatter(y=equity_curve, name="Portfolio Value",
                  hovertemplate="Value: $%{y:,.2f}<extra></extra>"),
        row=1, col=1
    )
    
    # Add horizontal line at initial capital
    fig.add_hline(y=equity_curve[0], line_dash="dash", line_color="gray",
                 annotation_text="Initial Capital", row=1, col=1)
    
    # Plot asset price
    fig.add_trace(
        go.Scatter(y=prices, name=f"{symbol} Price",
                  hovertemplate="Price: $%{y:,.2f}<extra></extra>"),
        row=2, col=1
    )
    
    # Plot drawdown
    fig.add_trace(
        go.Scatter(y=drawdown*100, name="Drawdown",
                  fill='tozeroy', fillcolor='rgba(255,0,0,0.2)',
                  hovertemplate="Drawdown: %{y:.1f}%<extra></extra>"),
        row=3, col=1
    )
    
    # Update layout with metrics
    metrics_text = (
        f"Total Return: {total_return:.1%}<br>"
        f"Annualized Volatility: {volatility:.1%}<br>"
        f"Sharpe Ratio: {sharpe:.2f}<br>"
        f"Max Drawdown: {max_drawdown:.1%}"
    )
    
    fig.update_layout(
        height=1000,
        title=dict(
            text="Backtest Results<br><sup>" + metrics_text + "</sup>",
            x=0.5,
            xanchor='center'
        ),
        showlegend=True,
        hovermode='x unified'
    )
    
    # Update y-axes
    fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
    fig.update_yaxes(title_text="Price ($)", row=2, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=3, col=1)
    
    fig.show()

def main():
    # Create backtest engine
    engine = bt.BacktestEngine()
    
    # Set parameters
    initial_capital = 100000.0
    engine.set_initial_capital(initial_capital)  # $100,000 initial capital
    engine.set_commission_rate(0.001)     # 0.1% commission rate
    engine.set_max_position_size(0.2)    # 20% maximum position size
    engine.set_stop_loss(0.02)           # 2% stop loss
    
    # Generate and add sample data for multiple symbols
    print("Generating sample data...")
    symbols = ["AAPL", "GOOGL", "MSFT"]  # Test with multiple symbols
    all_prices = {}
    
    for symbol in symbols:
        # Generate data with different characteristics
        if symbol == "AAPL":
            data = generate_sample_data(symbol, days=365, volatility=0.02, trend=0.0002)  # Upward trend
        elif symbol == "GOOGL":
            data = generate_sample_data(symbol, days=365, volatility=0.025, trend=-0.0001)  # Downward trend
        else:
            data = generate_sample_data(symbol, days=365, volatility=0.015, trend=0.0)  # Neutral trend
            
        print(f"Generated {len(data)} data points for {symbol}")
        print(f"First price: {data[0].price:.2f}, Last price: {data[-1].price:.2f}")
        engine.add_data(symbol, data)
        all_prices[symbol] = [asset.price for asset in data]
    
    # Create and set strategy
    print("\nCreating strategy...")
    strategy = MovingAverageCrossStrategy()
    engine.set_strategy(strategy)
    
    # Run backtest
    print("Starting backtest...")
    engine.run()
    print("Backtest completed")
    
    # Get results
    print("\nGetting results...")
    total_return = engine.get_total_return()
    sharpe_ratio = engine.get_sharpe_ratio()
    max_drawdown = engine.get_max_drawdown()
    win_rate = engine.get_win_rate()
    profit_factor = engine.get_profit_factor()
    avg_trade = engine.get_avg_trade()
    
    print(f"\nBacktest Results:")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Total Return: {total_return:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2%}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Average Trade: ${avg_trade:.2f}")
    
    # Plot results with debugging information
    print("\nGetting equity curve...")
    equity_curve = engine.get_equity_curve()
    print(f"Equity curve length: {len(equity_curve)}")
    print("Equity curve values:")
    print(f"First 5 values: {equity_curve[:5]}")
    print(f"Last 5 values: {equity_curve[-5:]}")
    print(f"Min value: ${min(equity_curve):,.2f}")
    print(f"Max value: ${max(equity_curve):,.2f}")
    
    # Get trade history
    trade_history = engine.get_trade_history()
    print(f"\nTrade History ({len(trade_history)} trades):")
    for i, trade in enumerate(trade_history):
        print(f"\nTrade {i+1}:")
        print(f"Symbol: {trade.symbol}")
        print(f"Entry Price: ${trade.entry_price:.2f}")
        print(f"Exit Price: ${trade.exit_price:.2f}")
        print(f"Quantity: {trade.quantity:.2f}")
        print(f"P&L: ${trade.pnl:.2f}")
        print(f"Commission: ${trade.commission:.2f}")
    
    # Enhanced plotting
    fig = make_subplots(rows=2, cols=1,
                       subplot_titles=('Portfolio Equity Curve', 'Asset Prices'),
                       vertical_spacing=0.15,
                       specs=[[{"secondary_y": True}],
                             [{"secondary_y": False}]])
    
    # Plot equity curve with better formatting
    fig.add_trace(
        go.Scatter(y=equity_curve, name="Portfolio Value",
                  line=dict(color='blue', width=2)),
        row=1, col=1
    )
    
    # Add drawdown on secondary y-axis
    drawdown = [100 * (max(equity_curve[:i+1]) - val) / max(equity_curve[:i+1]) if i > 0 else 0 
               for i, val in enumerate(equity_curve)]
    fig.add_trace(
        go.Scatter(y=drawdown, name="Drawdown %",
                  line=dict(color='red', width=1, dash='dash')),
        row=1, col=1, secondary_y=True
    )
    
    # Plot asset prices
    for symbol, prices in all_prices.items():
        fig.add_trace(
            go.Scatter(y=prices, name=f"{symbol} Price",
                      line=dict(width=1)),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        height=1000,
        title_text="Backtest Results",
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    # Update axes labels
    fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Price ($)", row=2, col=1)
    
    fig.show()

if __name__ == "__main__":
    main()
