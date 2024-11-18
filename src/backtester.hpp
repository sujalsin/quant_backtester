#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <chrono>
#include <memory>

namespace quant {

struct Asset {
    std::string symbol;
    std::string asset_class;
    double price;
    double volume;
    std::chrono::system_clock::time_point timestamp;
};

struct Position {
    std::string symbol;
    double quantity;
    double entry_price;
    std::chrono::system_clock::time_point entry_time;
};

struct TradeResult {
    std::string symbol;
    double entry_price;
    double exit_price;
    double quantity;
    double pnl;
    double commission;
    std::chrono::system_clock::time_point entry_time;
    std::chrono::system_clock::time_point exit_time;
};

class Strategy {
public:
    virtual bool should_enter(const Asset& asset) = 0;
    virtual bool should_exit(const Asset& asset, const Position& position) = 0;
    virtual ~Strategy() = default;
};

class BacktestEngine {
private:
    std::unique_ptr<Strategy> strategy_;
    std::unordered_map<std::string, std::vector<Asset>> market_data_;
    double initial_capital_;
    double commission_rate_;
    double max_position_size_;
    double stop_loss_pct_;
    double max_drawdown_pct_;
    std::vector<TradeResult> results_;
    std::vector<double> equity_curve_;
    std::vector<TradeResult> trade_history_;
    double current_capital_;
    std::vector<Position> positions_;

public:
    BacktestEngine() : strategy_(nullptr), initial_capital_(100000.0), 
                      commission_rate_(0.001), max_position_size_(0.2),
                      stop_loss_pct_(0.02), max_drawdown_pct_(0.25) {}
    ~BacktestEngine() = default;
    
    void add_data(const std::string& symbol, const std::vector<Asset>& historical_data);
    void set_strategy(std::unique_ptr<Strategy> strategy);
    void set_initial_capital(double capital);
    void set_commission_rate(double rate);
    
    void run();
    
    // Performance metrics
    double get_total_return() const;
    double get_sharpe_ratio() const;
    double get_max_drawdown() const;
    std::vector<double> get_equity_curve() const;

    // Risk management settings
    void set_max_position_size(double size) { max_position_size_ = size; }
    void set_stop_loss(double pct) { stop_loss_pct_ = pct; }
    void set_max_drawdown(double pct) { max_drawdown_pct_ = pct; }
    
    // Additional analytics
    const std::vector<TradeResult>& get_trade_history() const { return trade_history_; }
    double get_win_rate() const;
    double get_profit_factor() const;
    double get_avg_trade() const;

    // Helper methods
    double calculate_position_size(const Asset& asset, double current_capital) const;
    bool check_risk_limits(const Position& pos, const Asset& asset, double current_capital) const;
};

} // namespace quant
