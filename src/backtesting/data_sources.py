# prop_firm_trading_bot/src/backtesting/data_sources.py

from abc import ABC, abstractmethod
from typing import Optional, List, Tuple
from datetime import datetime, timezone
import pandas as pd
import logging
from pathlib import Path

from src.core.enums import Timeframe
from .backtest_config import DateRange, DataValidationSettings


class DataQualityReport(object):
    """Report on historical data quality and completeness."""
    def __init__(self, 
                 total_records: int,
                 missing_records: int,
                 data_gaps: List[Tuple[datetime, datetime]],
                 validation_errors: List[str],
                 is_valid: bool):
        self.total_records = total_records
        self.missing_records = missing_records
        self.data_gaps = data_gaps
        self.validation_errors = validation_errors
        self.is_valid = is_valid
    
    def __str__(self) -> str:
        return (f"DataQualityReport: {self.total_records} records, "
                f"{self.missing_records} missing, {len(self.data_gaps)} gaps, "
                f"Valid: {self.is_valid}")


class BacktestDataSource(ABC):
    """
    Abstract base class for historical data sources.
    Follows the platform abstraction pattern established in the architecture.
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    @abstractmethod
    def load_historical_data(self, 
                           symbol: str, 
                           timeframe: Timeframe, 
                           date_range: Optional[DateRange] = None) -> pd.DataFrame:
        """
        Load historical OHLCV data for specified parameters.
        
        Args:
            symbol: Trading instrument symbol
            timeframe: Timeframe enum for data granularity  
            date_range: Optional date range for data period
            
        Returns:
            pd.DataFrame with columns: timestamp, open, high, low, close, volume
        """
        pass
    
    @abstractmethod
    def validate_data_quality(self, 
                            data: pd.DataFrame, 
                            validation_settings: DataValidationSettings) -> DataQualityReport:
        """
        Validate historical data quality and completeness.
        
        Args:
            data: Historical data DataFrame to validate
            validation_settings: Validation criteria and settings
            
        Returns:
            DataQualityReport with validation results
        """
        pass


class CSVDataSource(BacktestDataSource):
    """
    CSV file-based historical data source.
    Handles loading and validation of CSV historical data files.
    """
    
    def __init__(self, csv_file_path: str, logger: logging.Logger):
        super().__init__(logger)
        self.csv_file_path = Path(csv_file_path)
        if not self.csv_file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_file_path}")
    
    def load_historical_data(self, 
                           symbol: str, 
                           timeframe: Timeframe, 
                           date_range: Optional[DateRange] = None) -> pd.DataFrame:
        """
        Load historical data from CSV file.
        
        Expected CSV format:
        timestamp,open,high,low,close,volume
        2023-01-01 00:00:00,1.2345,1.2350,1.2340,1.2348,1000
        """
        try:
            self.logger.info(f"Loading historical data from CSV: {self.csv_file_path}")
            
            # Load CSV with proper timestamp parsing
            df = pd.read_csv(self.csv_file_path, parse_dates=['timestamp'])
            
            # Validate required columns
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns in CSV: {missing_columns}")
            
            # Ensure timezone-aware timestamps
            if df['timestamp'].dt.tz is None:
                self.logger.info("Converting timestamps to UTC timezone")
                df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
            
            # Filter by date range if specified
            if date_range:
                mask = (df['timestamp'] >= date_range.start_date) & (df['timestamp'] <= date_range.end_date)
                df = df[mask].copy()
                self.logger.info(f"Filtered data to date range: {len(df)} records")
            
            # Sort by timestamp to ensure chronological order
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            self.logger.info(f"Successfully loaded {len(df)} records from {self.csv_file_path}")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load CSV data from {self.csv_file_path}: {e}")
            raise
    
    def validate_data_quality(self, 
                            data: pd.DataFrame, 
                            validation_settings: DataValidationSettings) -> DataQualityReport:
        """Validate CSV data quality according to settings."""
        validation_errors = []
        data_gaps = []
        
        # Basic data validation
        if data.empty:
            validation_errors.append("Data is empty")
            return DataQualityReport(0, 0, [], validation_errors, False)
        
        # OHLC consistency validation
        if validation_settings.validate_ohlc_consistency:
            invalid_ohlc = data[(data['high'] < data['low']) | 
                               (data['open'] > data['high']) | 
                               (data['open'] < data['low']) |
                               (data['close'] > data['high']) | 
                               (data['close'] < data['low'])]
            if not invalid_ohlc.empty:
                validation_errors.append(f"Found {len(invalid_ohlc)} records with invalid OHLC relationships")
        
        # Timezone validation
        if validation_settings.timezone_validation:
            if data['timestamp'].dt.tz is None:
                validation_errors.append("Timestamps are not timezone-aware")
        
        # Gap detection (simplified for MVP)
        if validation_settings.require_complete_data and len(data) > 1:
            time_diffs = data['timestamp'].diff().dt.total_seconds() / 60  # minutes
            large_gaps = time_diffs > validation_settings.max_gap_tolerance_minutes
            if large_gaps.any():
                gap_count = large_gaps.sum()
                validation_errors.append(f"Found {gap_count} data gaps larger than {validation_settings.max_gap_tolerance_minutes} minutes")
        
        is_valid = len(validation_errors) == 0
        
        return DataQualityReport(
            total_records=len(data),
            missing_records=0,  # Simplified for MVP
            data_gaps=data_gaps,
            validation_errors=validation_errors,
            is_valid=is_valid
        )


  
