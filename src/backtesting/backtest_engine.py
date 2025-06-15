# prop_firm_trading_bot/src/backtesting/backtest_engine.py

import logging
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Set, List
from pathlib import Path

import pandas as pd

from src.config_manager import AppConfig
from src.orchestrator import Orchestrator
from src.api_connector.paper_trading_adapter import PaperTradingAdapter
from src.utils.performance_reporter import PerformanceReporter
from src.core.enums import Timeframe
from .backtest_config import BacktestConfiguration, BacktestResults, DateRange, DataValidationSettings
from .data_sources import CSVDataSource, DataQualityReport


class BacktestEngine:
    """
    Comprehensive backtesting engine that orchestrates historical data replay,
    strategy execution, and performance analysis.
    
    Follows the established architectural patterns and integrates with existing
    components (Orchestrator, RiskController, BaseStrategy) to ensure consistency
    between backtesting and live trading behavior.
    """
    
    def __init__(self, config: AppConfig, logger: logging.Logger):
        """
        Initialize the BacktestEngine with configuration and logging.
        
        Args:
            config: AppConfig instance with strategy and risk settings
            logger: Logger instance for backtest execution tracking
        """
        self.config = config
        self.logger = logger
        self.orchestrator: Optional[Orchestrator] = None
        self.paper_adapter: Optional[PaperTradingAdapter] = None

        self.logger.info("BacktestEngine initialized")

    def _detect_required_timeframes(self, strategy_profile_key: str) -> Set[Timeframe]:
        """
        Detect all timeframes required for backtesting based on strategy configuration
        and system components (e.g., MarketRegimeClassifier).

        Args:
            strategy_profile_key: Key of the strategy profile to analyze

        Returns:
            Set of required Timeframe enums
        """
        required_timeframes = set()

        # 1. Get strategy's primary timeframe
        strategy_profile = self.config.asset_strategy_profiles.get(strategy_profile_key)
        if not strategy_profile:
            raise ValueError(f"Strategy profile '{strategy_profile_key}' not found")

        strategy_params_key = strategy_profile.strategy_params_key
        if strategy_params_key not in self.config.loaded_strategy_parameters:
            raise ValueError(f"Strategy parameters '{strategy_params_key}' not found")

        strategy_params = self.config.loaded_strategy_parameters[strategy_params_key]
        timeframe_str = strategy_params.parameters.get("timeframe", "H1").upper()

        try:
            strategy_timeframe = getattr(Timeframe, timeframe_str)
            required_timeframes.add(strategy_timeframe)
            self.logger.info(f"Strategy primary timeframe: {strategy_timeframe.name}")
        except AttributeError:
            self.logger.warning(f"Invalid timeframe '{timeframe_str}' in strategy params. Defaulting to H1.")
            required_timeframes.add(Timeframe.H1)

        # 2. Add H4 timeframe for MarketRegimeClassifier (ADX calculation)
        required_timeframes.add(Timeframe.H4)
        self.logger.info("Added H4 timeframe for MarketRegimeClassifier")

        # 3. Future: Add additional timeframes for multi-timeframe strategies
        # This can be extended when we implement strategies that use multiple timeframes

        self.logger.info(f"Total required timeframes for backtesting: {[tf.name for tf in required_timeframes]}")
        return required_timeframes

    def _load_multi_timeframe_data(self,
                                 symbol: str,
                                 required_timeframes: Set[Timeframe],
                                 csv_file_path: str) -> Dict[Timeframe, pd.DataFrame]:
        """
        Load historical data for multiple timeframes.

        For now, this method assumes a single CSV file contains data for the primary timeframe,
        and attempts to load H4 data from a corresponding H4 file if it exists.

        Future enhancement: Support multi-timeframe CSV format or multiple file specification.

        Args:
            symbol: Trading symbol
            required_timeframes: Set of timeframes to load
            csv_file_path: Path to primary CSV file

        Returns:
            Dictionary mapping timeframes to their respective DataFrames
        """
        timeframe_data = {}
        data_source = CSVDataSource(csv_file_path, self.logger)

        # Determine primary timeframe (strategy's timeframe)
        primary_timeframe = None
        for tf in required_timeframes:
            if tf != Timeframe.H4:  # H4 is for regime classification, not primary strategy
                primary_timeframe = tf
                break

        if not primary_timeframe:
            # Fallback: if only H4 is required, use it as primary
            primary_timeframe = Timeframe.H4

        # Load primary timeframe data from the provided CSV file
        self.logger.info(f"Loading primary timeframe {primary_timeframe.name} data from {csv_file_path}")
        primary_data = data_source.load_historical_data(
            symbol=symbol,
            timeframe=primary_timeframe,
            date_range=None
        )

        if primary_data.empty:
            raise ValueError(f"No data loaded for primary timeframe {primary_timeframe.name}")

        timeframe_data[primary_timeframe] = primary_data

        # Load additional timeframes
        for timeframe in required_timeframes:
            if timeframe == primary_timeframe:
                continue  # Already loaded

            # Try to find corresponding file for this timeframe
            h4_file_path = self._find_timeframe_file(csv_file_path, symbol, timeframe)

            if h4_file_path and Path(h4_file_path).exists():
                self.logger.info(f"Loading {timeframe.name} data from {h4_file_path}")
                try:
                    tf_data_source = CSVDataSource(h4_file_path, self.logger)
                    tf_data = tf_data_source.load_historical_data(
                        symbol=symbol,
                        timeframe=timeframe,
                        date_range=None
                    )

                    if not tf_data.empty:
                        timeframe_data[timeframe] = tf_data
                        self.logger.info(f"Successfully loaded {len(tf_data)} {timeframe.name} bars")
                    else:
                        self.logger.warning(f"No data found in {h4_file_path} for {timeframe.name}")

                except Exception as e:
                    self.logger.error(f"Failed to load {timeframe.name} data from {h4_file_path}: {e}")
            else:
                self.logger.warning(f"No data file found for {timeframe.name}. MarketRegimeClassifier will be disabled during backtesting.")

        return timeframe_data

    def _find_timeframe_file(self, primary_csv_path: str, symbol: str, timeframe: Timeframe) -> Optional[str]:
        """
        Find the CSV file for a specific timeframe based on naming conventions.

        Expected naming patterns:
        - EURUSD_M15_2023_05_01_to_2023_05_31.csv -> EURUSD_H4_2023_FULL_YEAR.csv
        - EURUSD_M15.csv -> EURUSD_H4.csv
        - EURUSD_2023_M15.csv -> EURUSD_2023_H4.csv
        - data/EURUSD_M15.csv -> data/EURUSD_H4.csv

        Args:
            primary_csv_path: Path to the primary CSV file
            symbol: Trading symbol
            timeframe: Target timeframe to find

        Returns:
            Path to timeframe-specific CSV file if found, None otherwise
        """
        primary_path = Path(primary_csv_path)

        # Try common naming patterns
        patterns = [
            # Pattern 1: Replace timeframe in filename: EURUSD_M15_2023_05_01_to_2023_05_31.csv -> EURUSD_H4_2023_FULL_YEAR.csv
            primary_path.stem.replace("_M15", f"_{timeframe.name}").replace("_H1", f"_{timeframe.name}") + primary_path.suffix,

            # Pattern 2: For date-specific files, try FULL_YEAR version: EURUSD_M15_2023_05_01_to_2023_05_31.csv -> EURUSD_H4_2023_FULL_YEAR.csv
            f"{symbol}_{timeframe.name}_2023_FULL_YEAR{primary_path.suffix}",

            # Pattern 3: Simple symbol_timeframe format: EURUSD_H4.csv
            f"{symbol}_{timeframe.name}{primary_path.suffix}",

            # Pattern 4: Year-based format: EURUSD_H4_2023_01_01_to_2023_12_31.csv
            f"{symbol}_{timeframe.name}_2023_01_01_to_2023_12_31{primary_path.suffix}",

            # Pattern 5: Just symbol_timeframe.csv
            f"{symbol}_{timeframe.name}.csv"
        ]

        for pattern in patterns:
            candidate_path = primary_path.parent / pattern
            self.logger.debug(f"Checking for {timeframe.name} data file: {candidate_path}")
            if candidate_path.exists():
                self.logger.info(f"Found {timeframe.name} data file: {candidate_path}")
                return str(candidate_path)

        self.logger.warning(f"No {timeframe.name} data file found for {symbol}. Tried patterns: {patterns}")
        return None
    
    def run_single_strategy_backtest(self, 
                                   strategy_profile_key: str, 
                                   csv_file_path: str, 
                                   initial_balance: float) -> BacktestResults:
        """
        Execute complete backtesting simulation for a single strategy.
        
        Args:
            strategy_profile_key: Asset-strategy profile key to test (e.g., "EURUSD_SMA_H1")
            csv_file_path: Path to historical data CSV file
            initial_balance: Starting account balance for simulation
            
        Returns:
            BacktestResults with comprehensive results and performance metrics
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting backtest for strategy: {strategy_profile_key}")
            self.logger.info(f"Data source: {csv_file_path}")
            self.logger.info(f"Initial balance: {initial_balance:.2f}")
            
            # 1. Validate strategy profile exists
            if strategy_profile_key not in self.config.asset_strategy_profiles:
                raise ValueError(f"Strategy profile '{strategy_profile_key}' not found in configuration")

            strategy_profile = self.config.asset_strategy_profiles[strategy_profile_key]
            if not strategy_profile.enabled:
                raise ValueError(f"Strategy profile '{strategy_profile_key}' is disabled")

            # 2. Detect all required timeframes for this backtest
            required_timeframes = self._detect_required_timeframes(strategy_profile_key)

            # 3. Load multi-timeframe historical data
            timeframe_datasets = self._load_multi_timeframe_data(
                symbol=strategy_profile.symbol,
                required_timeframes=required_timeframes,
                csv_file_path=csv_file_path
            )

            if not timeframe_datasets:
                raise ValueError("No historical data loaded from CSV file(s)")

            # Get primary timeframe data for the strategy
            strategy_params_key = strategy_profile.strategy_params_key
            strategy_params = self.config.loaded_strategy_parameters[strategy_params_key]
            timeframe_str = strategy_params.parameters.get("timeframe", "H1").upper()
            try:
                strategy_timeframe = getattr(Timeframe, timeframe_str)
            except AttributeError:
                self.logger.warning(f"Invalid timeframe '{timeframe_str}' in strategy params. Defaulting to H1.")
                strategy_timeframe = Timeframe.H1

            # Ensure we have data for the strategy's primary timeframe
            if strategy_timeframe not in timeframe_datasets:
                raise ValueError(f"No data available for strategy's primary timeframe: {strategy_timeframe.name}")

            primary_historical_data = timeframe_datasets[strategy_timeframe]
            
            # 4. Validate data quality for primary timeframe
            validation_settings = DataValidationSettings()
            primary_data_source = CSVDataSource(csv_file_path, self.logger)
            quality_report = primary_data_source.validate_data_quality(primary_historical_data, validation_settings)

            if not quality_report.is_valid:
                self.logger.warning(f"Data quality issues detected: {quality_report.validation_errors}")
                # For MVP, continue with warnings rather than failing

            # 5. Initialize PaperTradingAdapter with multi-timeframe historical data
            paper_adapter_logger = logging.getLogger("PaperTradingAdapter.Backtest")
            self.paper_adapter = PaperTradingAdapter(
                config=self.config,
                logger=paper_adapter_logger,
                historical_data=primary_historical_data,  # Primary timeframe for strategy execution
                initial_balance=initial_balance
            )

            # Set multi-timeframe datasets in PaperTradingAdapter
            self.paper_adapter.set_timeframe_datasets(timeframe_datasets)

            # 5. Connect paper adapter BEFORE initializing Orchestrator
            if not self.paper_adapter.connect():
                raise RuntimeError("Failed to connect PaperTradingAdapter")

            # 6. Initialize Orchestrator with paper adapter override
            orchestrator_logger = logging.getLogger("Orchestrator.Backtest")
            self.orchestrator = Orchestrator(
                config_manager=self.config,
                main_logger=orchestrator_logger,
                platform_adapter_override=self.paper_adapter
            )
            
            self.logger.info("Components initialized successfully")

            # 7. Execute historical data replay
            processed_bars = self._execute_backtest_loop()
            
            # 8. Calculate results
            execution_time = time.time() - start_time

            # Get start and end dates from historical data
            try:
                # Use the first and last timestamps from the primary data
                timestamps = primary_historical_data['timestamp']
                start_date = timestamps.min().to_pydatetime() if hasattr(timestamps.min(), 'to_pydatetime') else timestamps.min()  # type: ignore
                end_date = timestamps.max().to_pydatetime() if hasattr(timestamps.max(), 'to_pydatetime') else timestamps.max()  # type: ignore
            except Exception as e:
                self.logger.warning(f"Failed to extract timestamps: {e}. Using current time as fallback.")
                start_date = datetime.now(timezone.utc)
                end_date = datetime.now(timezone.utc)

            results = BacktestResults(
                strategy_profile_key=strategy_profile_key,
                initial_balance=initial_balance,
                final_balance=self.paper_adapter.equity,
                total_trades=len(self.paper_adapter.trade_history),
                start_date=start_date,
                end_date=end_date,
                execution_time_seconds=execution_time,
                success=True
            )
            
            self.logger.info(f"Backtest completed successfully:")
            self.logger.info(f"  Processed bars: {processed_bars}")
            self.logger.info(f"  Final balance: {results.final_balance:.2f}")
            self.logger.info(f"  Total return: {results.total_return:.2f}%")
            self.logger.info(f"  Total trades: {results.total_trades}")
            self.logger.info(f"  Execution time: {results.execution_time_seconds:.2f}s")
            
            return results
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Backtest failed: {e}", exc_info=True)
            
            return BacktestResults(
                strategy_profile_key=strategy_profile_key,
                initial_balance=initial_balance,
                final_balance=initial_balance,  # No change if failed
                total_trades=0,
                start_date=datetime.now(timezone.utc),
                end_date=datetime.now(timezone.utc),
                execution_time_seconds=execution_time,
                success=False,
                error_message=str(e)
            )
    
    def _execute_backtest_loop(self) -> int:
        """
        Execute the main backtest loop using historical data replay.

        Returns:
            Number of bars processed
        """
        # Ensure components are properly initialized
        assert self.paper_adapter is not None, "PaperTradingAdapter not initialized"
        assert self.orchestrator is not None, "Orchestrator not initialized"

        processed_bars = 0

        self.logger.info("Starting historical data replay...")

        try:
            # Process historical data bar by bar
            paper_adapter = self.paper_adapter  # Type guard
            orchestrator = self.orchestrator    # Type guard

            while paper_adapter.next_bar():
                # Use Orchestrator's process_single_bar method for consistency
                if not orchestrator.process_single_bar():
                    self.logger.warning("Orchestrator indicated stop condition during backtest")
                    break

                processed_bars += 1

                # Progress logging every 1000 bars or at the end
                if (paper_adapter.current_bar_index % 1000 == 0 or
                    paper_adapter.current_bar_index == (len(paper_adapter.historical_data) - 1)):

                    current_timestamp = paper_adapter.historical_data.iloc[paper_adapter.current_bar_index]['timestamp']
                    self.logger.info(
                        f"Progress: {paper_adapter.current_bar_index + 1}/"
                        f"{len(paper_adapter.historical_data)} "
                        f"({current_timestamp}) - Equity: {paper_adapter.equity:.2f}"
                    )
            
            self.logger.info(f"Historical data replay completed. Processed {processed_bars} bars.")
            return processed_bars
            
        except Exception as e:
            self.logger.error(f"Error during backtest loop at bar {self.paper_adapter.current_bar_index}: {e}", exc_info=True)
            raise
    
    def generate_performance_report(self, results: BacktestResults, include_trade_log: bool = True) -> str:
        """
        Generate a comprehensive performance report using the enhanced PerformanceReporter.

        Args:
            results: BacktestResults from completed backtest
            include_trade_log: Whether to include detailed trade log in the report

        Returns:
            Formatted performance report string
        """
        if not self.paper_adapter:
            return "No backtest data available for reporting"

        try:
            # Always create reporter, even with no trades for proper "no trades" message
            reporter = PerformanceReporter(
                trade_history=self.paper_adapter.trade_history,
                equity_history=self.paper_adapter.equity_history,
                initial_balance=results.initial_balance
            )

            return reporter.generate_summary(include_trade_log=include_trade_log)

        except Exception as e:
            self.logger.error(f"Failed to generate performance report: {e}")
            return f"Error generating performance report: {e}"


  
