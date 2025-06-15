# prop_firm_trading_bot/src/backtesting/__init__.py

"""
Backtesting module for the prop firm trading bot.

This module provides comprehensive backtesting capabilities that integrate
with the existing trading architecture while maintaining consistency with
live trading behavior.
"""

from .backtest_engine import BacktestEngine
from .backtest_config import BacktestConfiguration, DateRange, BacktestOutputSettings
from .data_sources import BacktestDataSource, CSVDataSource

__all__ = [
    'BacktestEngine',
    'BacktestConfiguration', 
    'DateRange',
    'BacktestOutputSettings',
    'BacktestDataSource',
    'CSVDataSource'
]


  
