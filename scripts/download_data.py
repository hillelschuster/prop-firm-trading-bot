#!/usr/bin/env python3
"""
Historical Data Download Script for Prop Firm Trading Bot

This script downloads historical market data from MetaTrader 5 for backtesting purposes.
It integrates with our existing MT5Adapter and follows established architectural patterns.

Usage:
    python scripts/download_data.py --symbol EURUSD --timeframe M15 --start-date 2023-05-01 --end-date 2023-05-31

Author: Prop Firm Trading Bot Team
"""

import sys
import os
import argparse
import logging
from datetime import datetime, timezone
from typing import Optional

# External library imports
import pandas as pd

# Environment variable loading (with graceful degradation)
try:
    from dotenv import load_dotenv
    _DOTENV_AVAILABLE = True
except ImportError:
    _DOTENV_AVAILABLE = False

# Add project root to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config_manager import load_and_validate_config, AppConfig
from src.api_connector.mt5_connector import MT5Adapter
from src.core.enums import Timeframe
from src.logging_service import setup_logging

# Load environment variables from .env file for MT5 credentials
# This must happen before ConfigManager attempts to load credentials from environment
if _DOTENV_AVAILABLE:
    try:
        load_dotenv()
        # Note: load_dotenv() will silently do nothing if .env file doesn't exist
        # This is the desired behavior for graceful degradation
    except Exception as e:
        # If dotenv loading fails for any reason, continue without it
        # The script will still work if environment variables are set manually
        print(f"Warning: Failed to load .env file: {e}")
else:
    print("Info: python-dotenv not installed. Environment variables must be set manually.")
    print("Install with: pip install python-dotenv")


class DataDownloader:
    """
    Historical data downloader using MT5Adapter integration.
    
    Follows our established patterns for configuration management,
    logging, and error handling.
    """
    
    def __init__(self, config: AppConfig, logger: logging.Logger):
        """
        Initialize the data downloader.
        
        Args:
            config: Application configuration
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        self.mt5_adapter: Optional[MT5Adapter] = None
        
        # Timeframe string to enum mapping
        self.timeframe_map = {
            "M1": Timeframe.M1,
            "M5": Timeframe.M5,
            "M15": Timeframe.M15,
            "M30": Timeframe.M30,
            "H1": Timeframe.H1,
            "H4": Timeframe.H4,
            "D1": Timeframe.D1,
            "W1": Timeframe.W1,
            "MN1": Timeframe.MN1,
        }
    
    def _initialize_mt5_adapter(self) -> bool:
        """
        Initialize MT5Adapter connection.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.logger.info("Initializing MT5Adapter for data download...")
            self.mt5_adapter = MT5Adapter(self.config, self.logger)
            
            if not self.mt5_adapter.connect():
                self.logger.error("Failed to connect to MetaTrader 5")
                return False
                
            self.logger.info("MT5Adapter connected successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing MT5Adapter: {e}", exc_info=True)
            return False
    
    def _cleanup_mt5_adapter(self) -> None:
        """Clean up MT5Adapter connection."""
        if self.mt5_adapter:
            try:
                self.mt5_adapter.disconnect()
                self.logger.info("MT5Adapter disconnected successfully")
            except Exception as e:
                self.logger.warning(f"Error during MT5Adapter cleanup: {e}")
    
    def _parse_date_string(self, date_str: str) -> Optional[datetime]:
        """
        Parse date string to timezone-aware datetime object.
        
        Args:
            date_str: Date string in YYYY-MM-DD format
            
        Returns:
            Timezone-aware datetime object or None if parsing fails
        """
        try:
            # Parse date and make it timezone-aware (UTC)
            naive_dt = datetime.strptime(date_str, "%Y-%m-%d")
            return naive_dt.replace(tzinfo=timezone.utc)
        except ValueError as e:
            self.logger.error(f"Invalid date format '{date_str}': {e}")
            return None
    
    def _convert_timeframe_string(self, timeframe_str: str) -> Optional[Timeframe]:
        """
        Convert timeframe string to Timeframe enum.
        
        Args:
            timeframe_str: Timeframe string (e.g., "M15", "H1")
            
        Returns:
            Timeframe enum or None if invalid
        """
        timeframe = self.timeframe_map.get(timeframe_str.upper())
        if not timeframe:
            self.logger.error(f"Unsupported timeframe: {timeframe_str}")
            self.logger.info(f"Supported timeframes: {list(self.timeframe_map.keys())}")
        return timeframe
    
    def _save_data_to_csv(self, data: list, output_path: str) -> bool:
        """
        Save OHLCV data to CSV file in our standard format.
        
        Args:
            data: List of OHLCVData objects
            output_path: Output CSV file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not data:
                self.logger.warning("No data to save")
                return False
            
            # Convert to DataFrame with our standard column format
            df_data = []
            for ohlcv in data:
                df_data.append({
                    'timestamp': ohlcv.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    'open': ohlcv.open,
                    'high': ohlcv.high,
                    'low': ohlcv.low,
                    'close': ohlcv.close,
                    'volume': ohlcv.volume
                })
            
            df = pd.DataFrame(df_data)
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save to CSV with our standard format
            df.to_csv(output_path, index=False)
            
            self.logger.info(f"Successfully saved {len(data)} bars to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving data to CSV: {e}", exc_info=True)
            return False

    def download_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        output_filename: str
    ) -> bool:
        """
        Download historical data from MT5 and save to CSV.

        Args:
            symbol: Trading instrument (e.g., "EURUSD")
            timeframe: Timeframe string (e.g., "M15", "H1")
            start_date: Start date in "YYYY-MM-DD" format
            end_date: End date in "YYYY-MM-DD" format
            output_filename: Output CSV filename

        Returns:
            True if successful, False otherwise
        """
        self.logger.info(f"Starting data download for {symbol} {timeframe} from {start_date} to {end_date}")

        # Validate and parse inputs
        timeframe_enum = self._convert_timeframe_string(timeframe)
        if not timeframe_enum:
            return False

        start_dt = self._parse_date_string(start_date)
        end_dt = self._parse_date_string(end_date)
        if not start_dt or not end_dt:
            return False

        if start_dt >= end_dt:
            self.logger.error("Start date must be before end date")
            return False

        # Initialize MT5 connection
        if not self._initialize_mt5_adapter():
            return False

        try:
            # Download historical data using MT5Adapter
            self.logger.info(f"Fetching historical data from MT5...")

            # Type check to ensure mt5_adapter is initialized
            if not self.mt5_adapter:
                self.logger.error("MT5Adapter not properly initialized")
                return False

            ohlcv_data = self.mt5_adapter.get_historical_ohlcv(
                symbol=symbol,
                timeframe=timeframe_enum,
                start_time=start_dt,
                end_time=end_dt
            )

            if not ohlcv_data:
                self.logger.error("No data returned from MT5. Check symbol, timeframe, and date range.")
                return False

            self.logger.info(f"Retrieved {len(ohlcv_data)} bars from MT5")

            # Save data to CSV
            success = self._save_data_to_csv(ohlcv_data, output_filename)

            if success:
                self.logger.info(f"Data download completed successfully!")
                self.logger.info(f"File saved: {output_filename}")
                self.logger.info(f"Data range: {ohlcv_data[0].timestamp} to {ohlcv_data[-1].timestamp}")
                self.logger.info(f"Total bars: {len(ohlcv_data)}")

            return success

        except Exception as e:
            self.logger.error(f"Error during data download: {e}", exc_info=True)
            return False

        finally:
            # Always cleanup MT5 connection
            self._cleanup_mt5_adapter()


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description="Download historical market data from MetaTrader 5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download EURUSD M15 data for May 2023
  python scripts/download_data.py --symbol EURUSD --timeframe M15 --start-date 2023-05-01 --end-date 2023-05-31

  # Download GBPUSD H1 data with custom output directory
  python scripts/download_data.py --symbol GBPUSD --timeframe H1 --start-date 2023-06-01 --end-date 2023-06-30 --output-dir custom_data/
        """
    )

    parser.add_argument(
        "--symbol",
        required=True,
        help="Trading instrument symbol (e.g., EURUSD, GBPUSD)"
    )

    parser.add_argument(
        "--timeframe",
        required=True,
        choices=["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1"],
        help="Timeframe for historical data"
    )

    parser.add_argument(
        "--start-date",
        required=True,
        help="Start date in YYYY-MM-DD format"
    )

    parser.add_argument(
        "--end-date",
        required=True,
        help="End date in YYYY-MM-DD format"
    )

    parser.add_argument(
        "--output-dir",
        default="data/",
        help="Output directory for CSV files (default: data/)"
    )

    parser.add_argument(
        "--config",
        default="config/main_config.yaml",
        help="Path to configuration file (default: config/main_config.yaml)"
    )

    return parser


def main():
    """Main function to handle command-line execution."""
    parser = create_argument_parser()
    args = parser.parse_args()

    try:
        # Load configuration first (needed for logging setup)
        config = load_and_validate_config()

        # Setup logging using config
        logger = setup_logging(config=config, logger_name="DataDownloader")
        logger.info("Data downloader started")

        # Create output filename
        output_filename = f"{args.symbol}_{args.timeframe}_{args.start_date.replace('-', '_')}_to_{args.end_date.replace('-', '_')}.csv"
        output_path = os.path.join(args.output_dir, output_filename)

        # Initialize downloader
        downloader = DataDownloader(config, logger)

        # Download data
        success = downloader.download_historical_data(
            symbol=args.symbol,
            timeframe=args.timeframe,
            start_date=args.start_date,
            end_date=args.end_date,
            output_filename=output_path
        )

        if success:
            logger.info("Data download completed successfully!")
            print(f"\n✅ SUCCESS: Historical data saved to {output_path}")
            return 0
        else:
            logger.error("Data download failed!")
            print(f"\n❌ FAILED: Data download unsuccessful. Check logs for details.")
            return 1

    except KeyboardInterrupt:
        logger.info("Data download interrupted by user")
        print("\n⚠️  Download interrupted by user")
        return 1

    except Exception as e:
        logger.error(f"Unexpected error during data download: {e}", exc_info=True)
        print(f"\n❌ ERROR: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)


  
