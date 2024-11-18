#include "backtester.hpp"
#include <stdexcept>
#include <algorithm>
#include <numeric>

namespace quant {

void BacktestEngine::add_data(const std::string& symbol, const std::vector<Asset>& historical_data) {
    market_data_[symbol] = historical_data;
}

void BacktestEngine::set_strategy(std::unique_ptr<Strategy> strategy) {
    strategy_ = std::move(strategy);
}

void BacktestEngine::set_initial_capital(double capital) {
    initial_capital_ = capital;
    current_capital_ = capital;
}

void BacktestEngine::set_commission_rate(double rate) {
    commission_rate_ = rate;
}

double BacktestEngine::calculate_position_size(const Asset& asset, double current_capital) const {
    return current_capital * max_position_size_;
}

bool BacktestEngine::check_risk_limits(const Position& pos, const Asset& asset, double current_capital) const {
    // Check stop loss
    double unrealized_pnl = (asset.price - pos.entry_price) * pos.quantity;
    double pnl_pct = unrealized_pnl / current_capital;
    if (pnl_pct < -stop_loss_pct_) {
        return true;
    }

    // Check max drawdown
    double total_pnl = 0.0;
    for (const auto& result : trade_history_) {
        total_pnl += result.pnl;
    }
    double drawdown = (total_pnl + unrealized_pnl) / initial_capital_;
    if (drawdown < -max_drawdown_pct_) {
        return true;
    }

    return false;
}

void BacktestEngine::run() {
    if (!strategy_) {
        throw std::runtime_error("Strategy not set");
    }

    // Reset state
    current_capital_ = initial_capital_;
    std::unordered_map<std::string, Position> current_positions;
    equity_curve_.clear();
    equity_curve_.push_back(current_capital_);
    trade_history_.clear();

    // Process each asset's data
    for (const auto& [symbol, data] : market_data_) {
        for (const auto& asset : data) {
            // Check for exit signals on existing positions
            auto pos_it = current_positions.find(symbol);
            if (pos_it != current_positions.end()) {
                bool should_exit = strategy_->should_exit(asset, pos_it->second) ||
                                 !check_risk_limits(pos_it->second, asset, current_capital_);
                
                if (should_exit) {
                    // Calculate P&L
                    double pnl = (asset.price - pos_it->second.entry_price) * pos_it->second.quantity;
                    double commission = std::abs(asset.price * pos_it->second.quantity * commission_rate_);
                    pnl -= commission;
                    
                    // Update capital
                    current_capital_ += pnl;
                    equity_curve_.push_back(current_capital_);
                    
                    // Record trade
                    TradeResult trade{
                        symbol,
                        pos_it->second.entry_price,
                        asset.price,
                        pos_it->second.quantity,
                        pnl,
                        commission,
                        pos_it->second.entry_time,
                        asset.timestamp
                    };
                    trade_history_.push_back(trade);
                    
                    current_positions.erase(pos_it);
                }
            }
            // Check for entry signals when we don't have a position
            else if (strategy_->should_enter(asset)) {
                double position_size = calculate_position_size(asset, current_capital_);
                double quantity = position_size / asset.price;
                double commission = std::abs(asset.price * quantity * commission_rate_);
                
                // Only enter if we can afford commission
                if (position_size + commission <= current_capital_) {
                    current_capital_ -= commission;
                    equity_curve_.push_back(current_capital_);  // Update equity curve after commission
                    Position pos{
                        symbol,
                        quantity,
                        asset.price,
                        asset.timestamp
                    };
                    current_positions[symbol] = pos;
                }
            }
        }
    }
}

double BacktestEngine::get_total_return() const {
    return (current_capital_ - initial_capital_) / initial_capital_;
}

double BacktestEngine::get_sharpe_ratio() const {
    if (trade_history_.empty()) {
        return 0.0;
    }

    // Calculate returns
    std::vector<double> returns;
    returns.reserve(trade_history_.size());
    for (const auto& trade : trade_history_) {
        returns.push_back(trade.pnl / initial_capital_);
    }

    // Calculate mean return
    double mean = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();

    // Calculate standard deviation
    double sq_sum = std::inner_product(returns.begin(), returns.end(), returns.begin(), 0.0);
    double std_dev = std::sqrt(sq_sum / returns.size() - mean * mean);

    // Annualized Sharpe Ratio (assuming daily data)
    return std_dev == 0 ? 0 : (mean / std_dev) * std::sqrt(252);
}

double BacktestEngine::get_max_drawdown() const {
    if (equity_curve_.empty()) {
        return 0.0;
    }

    double max_drawdown = 0.0;
    double peak = equity_curve_[0];

    for (double value : equity_curve_) {
        if (value > peak) {
            peak = value;
        }
        double drawdown = (peak - value) / peak;
        max_drawdown = std::max(max_drawdown, drawdown);
    }

    return max_drawdown;
}

double BacktestEngine::get_win_rate() const {
    if (trade_history_.empty()) {
        return 0.0;
    }

    int winning_trades = std::count_if(trade_history_.begin(), trade_history_.end(),
                                     [](const TradeResult& trade) { return trade.pnl > 0; });
    return static_cast<double>(winning_trades) / trade_history_.size();
}

double BacktestEngine::get_profit_factor() const {
    if (trade_history_.empty()) {
        return 0.0;
    }

    double gross_profit = 0.0;
    double gross_loss = 0.0;

    for (const auto& trade : trade_history_) {
        if (trade.pnl > 0) {
            gross_profit += trade.pnl;
        } else {
            gross_loss -= trade.pnl;
        }
    }

    return gross_loss == 0 ? 0 : gross_profit / gross_loss;
}

double BacktestEngine::get_avg_trade() const {
    if (trade_history_.empty()) {
        return 0.0;
    }

    double total_pnl = std::accumulate(trade_history_.begin(), trade_history_.end(), 0.0,
                                     [](double sum, const TradeResult& trade) { return sum + trade.pnl; });
    return total_pnl / trade_history_.size();
}

std::vector<double> BacktestEngine::get_equity_curve() const {
    return equity_curve_;
}

} // namespace quant
