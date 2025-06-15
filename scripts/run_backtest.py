#!/usr/bin/env python

import argparse
import logging
import pandas as pd
from pathlib import Path
import sys

# Add project root to sys.path to allow for src imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from typing import List # Added for List type hint
from src.config_manager import load_and_validate_config, AppConfig
from src.backtesting.backtest_engine import BacktestEngine
from src.logging_service import setup_logging
from src.core.models import Order, TradeFill # For type hinting
from src.core.enums import OrderStatus, OrderType, OrderAction, StrategySignal # For logic

def main():
    parser = argparse.ArgumentParser(description="Run a backtest for a trading strategy using the enhanced BacktestEngine.")
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to the main_config.yaml file."
    )
    parser.add_argument(
        "--strategy-profile", "-p",
        type=str,
        required=True,
        help="Key of the asset strategy profile to run (e.g., 'EURUSD_SMA_H1')."
    )
    parser.add_argument(
        "--data", "-d",
        type=str,
        required=True,
        help="Path to the historical data CSV file. Columns: timestamp, open, high, low, close, volume."
    )
    parser.add_argument(
        "--initial-balance", "-b",
        type=float,
        default=10000.0,
        help="Initial balance for the backtest."
    )
    parser.add_argument(
        "--output-report", "-o",
        type=str,
        help="Optional path to save the performance report (e.g., report.txt)."
    )
    args = parser.parse_args()

    # 1. Load Configuration
    config_path = Path(args.config)
    app_config = load_and_validate_config(config_dir=str(config_path.parent), main_config_filename=config_path.name)

    # 2. Initialize Logging (after config is loaded)
    setup_logging(config=app_config)
    main_logger = logging.getLogger("BacktestRun")
    main_logger.info(f"Starting enhanced backtest for strategy profile: {args.strategy_profile}")

    # 3. Validate input arguments
    if not Path(args.data).exists():
        main_logger.error(f"Historical data file not found: {args.data}")
        sys.exit(1)

    # 4. Initialize BacktestEngine
    try:
        backtest_engine = BacktestEngine(config=app_config, logger=main_logger)
        main_logger.info("BacktestEngine initialized successfully")
    except Exception as e:
        main_logger.error(f"Failed to initialize BacktestEngine: {e}")
        sys.exit(1)

    # 5. Run the backtest
    try:
        main_logger.info("Executing backtest...")
        results = backtest_engine.run_single_strategy_backtest(
            strategy_profile_key=args.strategy_profile,
            csv_file_path=args.data,
            initial_balance=args.initial_balance
        )

        if not results.success:
            main_logger.error(f"Backtest failed: {results.error_message}")
            sys.exit(1)

    except Exception as e:
        main_logger.error(f"Backtest execution failed: {e}", exc_info=True)
        sys.exit(1)

    # 6. Generate and display performance report
    main_logger.info("Generating performance report...")
    try:
        performance_report = backtest_engine.generate_performance_report(results)

        print("\n" + "="*60)
        print("BACKTEST RESULTS SUMMARY")
        print("="*60)
        print(f"Strategy Profile: {results.strategy_profile_key}")
        print(f"Data File: {args.data}")
        print(f"Period: {results.start_date.strftime('%Y-%m-%d')} to {results.end_date.strftime('%Y-%m-%d')}")
        print(f"Initial Balance: {results.initial_balance:.2f}")
        print(f"Final Balance: {results.final_balance:.2f}")
        print(f"Net Profit: {results.net_profit:.2f}")
        print(f"Total Return: {results.total_return:.2f}%")
        print(f"Total Trades: {results.total_trades}")
        print(f"Execution Time: {results.execution_time_seconds:.2f}s")
        print("="*60)
        print("\nDETAILED PERFORMANCE METRICS:")
        print("-"*60)
        print(performance_report)
        print("="*60)

    except Exception as e:
        main_logger.error(f"Failed to generate performance report: {e}")
        print(f"\nBasic Results:")
        print(f"Final Balance: {results.final_balance:.2f}")
        print(f"Total Return: {results.total_return:.2f}%")
        print(f"Total Trades: {results.total_trades}")

    # 7. Save report to file if requested
    if args.output_report:
        try:
            with open(args.output_report, 'w') as f:
                f.write(f"Backtest Report for Strategy Profile: {results.strategy_profile_key}\n")
                f.write(f"Data File: {args.data}\n")
                f.write(f"Period: {results.start_date.strftime('%Y-%m-%d')} to {results.end_date.strftime('%Y-%m-%d')}\n")
                f.write(f"Initial Balance: {results.initial_balance:.2f}\n")
                f.write(f"Final Balance: {results.final_balance:.2f}\n")
                f.write(f"Net Profit: {results.net_profit:.2f}\n")
                f.write(f"Total Return: {results.total_return:.2f}%\n")
                f.write(f"Total Trades: {results.total_trades}\n")
                f.write(f"Execution Time: {results.execution_time_seconds:.2f}s\n")
                f.write("="*60 + "\n")
                f.write("DETAILED PERFORMANCE METRICS:\n")
                f.write("-"*60 + "\n")
                f.write(performance_report)
                f.write("\n" + "="*60 + "\n")
            main_logger.info(f"Performance report saved to: {args.output_report}")
        except Exception as e:
            main_logger.error(f"Failed to save performance report to {args.output_report}: {e}")

    main_logger.info("Enhanced backtest run completed successfully.")

if __name__ == "__main__":
    main()

  
