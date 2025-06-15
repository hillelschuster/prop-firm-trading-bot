# prop_firm_trading_bot/src/backtesting/backtest_config.py

from typing import List, Optional, Dict, Any
from datetime import datetime, date
from pydantic import BaseModel, Field, field_validator
from pathlib import Path

from src.core.enums import Timeframe


class DateRange(BaseModel):
    """Date range specification for backtesting."""
    start_date: datetime
    end_date: datetime
    
    @field_validator('end_date')
    @classmethod
    def end_after_start(cls, v, info):
        if 'start_date' in info.data and v <= info.data['start_date']:
            raise ValueError('end_date must be after start_date')
        return v


class BacktestOutputSettings(BaseModel):
    """Configuration for backtest output and reporting."""
    generate_trade_log: bool = True
    generate_performance_report: bool = True
    export_format: str = Field(default="text", pattern="^(text|json|csv)$")
    output_directory: Optional[str] = None
    include_equity_curve: bool = True


class DataValidationSettings(BaseModel):
    """Settings for historical data validation."""
    require_complete_data: bool = True
    max_gap_tolerance_minutes: int = 60
    validate_ohlc_consistency: bool = True
    timezone_validation: bool = True


class BacktestConfiguration(BaseModel):
    """
    Comprehensive configuration for backtesting parameters and settings.
    Follows the Pydantic model pattern established in the architecture.
    """
    strategy_profile_key: str
    csv_file_path: str
    initial_balance: float = Field(default=10000.0, gt=0)
    date_range: Optional[DateRange] = None
    output_settings: BacktestOutputSettings = Field(default_factory=BacktestOutputSettings)
    validation_settings: DataValidationSettings = Field(default_factory=DataValidationSettings)
    
    @field_validator('csv_file_path')
    @classmethod
    def validate_csv_path(cls, v):
        if not Path(v).exists():
            raise ValueError(f'CSV file does not exist: {v}')
        return v
    
    @field_validator('strategy_profile_key')
    @classmethod
    def validate_strategy_profile(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError('strategy_profile_key must be a non-empty string')
        return v


class BacktestResults(BaseModel):
    """Results from a completed backtest."""
    strategy_profile_key: str
    initial_balance: float
    final_balance: float
    total_trades: int
    start_date: datetime
    end_date: datetime
    execution_time_seconds: float
    success: bool
    error_message: Optional[str] = None
    
    @property
    def total_return(self) -> float:
        """Calculate total return percentage."""
        if self.initial_balance == 0:
            return 0.0
        return ((self.final_balance - self.initial_balance) / self.initial_balance) * 100
    
    @property
    def net_profit(self) -> float:
        """Calculate net profit in currency units."""
        return self.final_balance - self.initial_balance


  
