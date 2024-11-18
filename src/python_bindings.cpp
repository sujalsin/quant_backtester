#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "backtester.hpp"

namespace py = pybind11;

class PyStrategy : public quant::Strategy {
public:
    using quant::Strategy::Strategy;

    bool should_enter(const quant::Asset& asset) override {
        PYBIND11_OVERRIDE_PURE(
            bool,
            quant::Strategy,
            should_enter,
            asset
        );
    }

    bool should_exit(const quant::Asset& asset, const quant::Position& position) override {
        PYBIND11_OVERRIDE_PURE(
            bool,
            quant::Strategy,
            should_exit,
            asset,
            position
        );
    }
};

PYBIND11_MODULE(backtester_core, m) {
    m.doc() = "High-performance backtesting engine for quantitative trading strategies";

    py::class_<std::chrono::system_clock::time_point>(m, "TimePoint")
        .def(py::init<>())
        .def("__str__", [](const std::chrono::system_clock::time_point& t) {
            auto tt = std::chrono::system_clock::to_time_t(t);
            std::string ts = std::ctime(&tt);
            ts.pop_back();  // Remove trailing newline
            return ts;
        });

    py::class_<quant::Asset>(m, "Asset")
        .def(py::init<>())
        .def_readwrite("symbol", &quant::Asset::symbol)
        .def_readwrite("asset_class", &quant::Asset::asset_class)
        .def_readwrite("price", &quant::Asset::price)
        .def_readwrite("volume", &quant::Asset::volume)
        .def_readwrite("timestamp", &quant::Asset::timestamp);

    py::class_<quant::Position>(m, "Position")
        .def(py::init<>())
        .def_readwrite("symbol", &quant::Position::symbol)
        .def_readwrite("quantity", &quant::Position::quantity)
        .def_readwrite("entry_price", &quant::Position::entry_price)
        .def_readwrite("entry_time", &quant::Position::entry_time);

    py::class_<quant::TradeResult>(m, "TradeResult")
        .def(py::init<>())
        .def_readwrite("symbol", &quant::TradeResult::symbol)
        .def_readwrite("entry_price", &quant::TradeResult::entry_price)
        .def_readwrite("exit_price", &quant::TradeResult::exit_price)
        .def_readwrite("quantity", &quant::TradeResult::quantity)
        .def_readwrite("pnl", &quant::TradeResult::pnl)
        .def_readwrite("commission", &quant::TradeResult::commission)
        .def_readwrite("entry_time", &quant::TradeResult::entry_time)
        .def_readwrite("exit_time", &quant::TradeResult::exit_time);

    py::class_<quant::Strategy, PyStrategy>(m, "Strategy")
        .def(py::init<>())
        .def("should_enter", &quant::Strategy::should_enter)
        .def("should_exit", &quant::Strategy::should_exit);

    py::class_<quant::BacktestEngine>(m, "BacktestEngine")
        .def(py::init<>())
        .def("add_data", &quant::BacktestEngine::add_data)
        .def("set_strategy", [](quant::BacktestEngine& self, py::object strat) {
            if (py::isinstance<PyStrategy>(strat)) {
                auto* ptr = strat.cast<PyStrategy*>();
                self.set_strategy(std::unique_ptr<quant::Strategy>(new PyStrategy(*ptr)));
            } else {
                throw py::type_error("Expected a Strategy object");
            }
        })
        .def("set_initial_capital", &quant::BacktestEngine::set_initial_capital)
        .def("set_commission_rate", &quant::BacktestEngine::set_commission_rate)
        .def("set_max_position_size", &quant::BacktestEngine::set_max_position_size)
        .def("set_stop_loss", &quant::BacktestEngine::set_stop_loss)
        .def("run", &quant::BacktestEngine::run)
        .def("get_total_return", &quant::BacktestEngine::get_total_return)
        .def("get_sharpe_ratio", &quant::BacktestEngine::get_sharpe_ratio)
        .def("get_max_drawdown", &quant::BacktestEngine::get_max_drawdown)
        .def("get_equity_curve", &quant::BacktestEngine::get_equity_curve)
        .def("get_win_rate", &quant::BacktestEngine::get_win_rate)
        .def("get_profit_factor", &quant::BacktestEngine::get_profit_factor)
        .def("get_avg_trade", &quant::BacktestEngine::get_avg_trade)
        .def("get_trade_history", &quant::BacktestEngine::get_trade_history);
}
