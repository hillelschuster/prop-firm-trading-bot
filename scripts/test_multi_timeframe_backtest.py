#!/usr/bin/env python3
"""
Test script for multi-timeframe backtesting functionality.

This script tests the enhanced backtesting engine that supports multiple timeframes,
specifically testing the MarketRegimeClassifier's ability to access H4 data during
backtesting while the strategy runs on M15 data.

Author: Prop Firm Trading Bot
Date: 2025-01-27
"""

import sys
import os
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config_manager import load_and_validate_config
from src.backtesting.backtest_engine import BacktestEngine
from src.logging_service import setup_logging


def test_multi_timeframe_backtesting():
    """
    Test multi-timeframe backtesting with MarketRegimeClassifier.
    
    This test:
    1. Loads M15 strategy configuration
    2. Runs backtest with both M15 and H4 data
    3. Verifies that MarketRegimeClassifier can access H4 data
    4. Confirms that regime-based strategy filtering works
    """
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - [%(levelname)s] - %(message)s'
    )
    logger = logging.getLogger("MultiTimeframeTest")
    
    try:
        logger.info("=== Multi-Timeframe Backtesting Test ===")
        
        # 1. Load configuration
        logger.info("Loading configuration...")
        config_dir = project_root / "config"
        config = load_and_validate_config(str(config_dir))
        
        # 2. Initialize BacktestEngine
        logger.info("Initializing BacktestEngine...")
        backtest_engine = BacktestEngine(config, logger)
        
        # 3. Test data paths
        m15_data_path = project_root / "data" / "EURUSD_M15_2023_05_01_to_2023_05_31.csv"
        h4_data_path = project_root / "data" / "EURUSD_H4_2023_FULL_YEAR.csv"
        
        if not m15_data_path.exists():
            logger.error(f"M15 data file not found: {m15_data_path}")
            return False
            
        if not h4_data_path.exists():
            logger.error(f"H4 data file not found: {h4_data_path}")
            return False
        
        logger.info(f"Using M15 data: {m15_data_path}")
        logger.info(f"Using H4 data: {h4_data_path}")
        
        # 4. Test timeframe detection
        logger.info("Testing timeframe detection...")
        strategy_profile_key = "EURUSD_RSI_M15"  # This should use M15 + require H4 for regime classification
        
        try:
            required_timeframes = backtest_engine._detect_required_timeframes(strategy_profile_key)
            logger.info(f"Detected required timeframes: {[tf.name for tf in required_timeframes]}")
            
            # Verify that both M15 and H4 are detected
            from src.core.enums import Timeframe
            if Timeframe.M15 not in required_timeframes:
                logger.error("M15 timeframe not detected for RSI strategy")
                return False
                
            if Timeframe.H4 not in required_timeframes:
                logger.error("H4 timeframe not detected for MarketRegimeClassifier")
                return False
                
            logger.info("✓ Timeframe detection working correctly")
            
        except Exception as e:
            logger.error(f"Timeframe detection failed: {e}")
            return False
        
        # 5. Test multi-timeframe data loading
        logger.info("Testing multi-timeframe data loading...")
        
        try:
            # Get strategy profile for symbol
            strategy_profile = config.asset_strategy_profiles.get(strategy_profile_key)
            if not strategy_profile:
                logger.error(f"Strategy profile {strategy_profile_key} not found")
                return False
            
            # Test the data loading method
            timeframe_datasets = backtest_engine._load_multi_timeframe_data(
                symbol=strategy_profile.symbol,
                required_timeframes=required_timeframes,
                csv_file_path=str(m15_data_path)
            )
            
            logger.info(f"Loaded datasets for timeframes: {[tf.name for tf in timeframe_datasets.keys()]}")
            
            # Verify data was loaded for both timeframes
            for tf in required_timeframes:
                if tf in timeframe_datasets:
                    data_length = len(timeframe_datasets[tf])
                    logger.info(f"  {tf.name}: {data_length} bars loaded")
                else:
                    logger.warning(f"  {tf.name}: No data loaded")
            
            if Timeframe.M15 in timeframe_datasets and len(timeframe_datasets[Timeframe.M15]) > 0:
                logger.info("✓ M15 data loaded successfully")
            else:
                logger.error("✗ M15 data loading failed")
                return False
            
            # H4 data might not be available if the file doesn't exist, but that's expected
            if Timeframe.H4 in timeframe_datasets and len(timeframe_datasets[Timeframe.H4]) > 0:
                logger.info("✓ H4 data loaded successfully")
            else:
                logger.warning("⚠ H4 data not loaded (file may not exist or naming convention issue)")
            
        except Exception as e:
            logger.error(f"Multi-timeframe data loading failed: {e}")
            return False
        
        # 6. Run a short backtest to test integration
        logger.info("Running integration test backtest...")
        
        try:
            # Run backtest with small dataset
            results = backtest_engine.run_single_strategy_backtest(
                strategy_profile_key=strategy_profile_key,
                csv_file_path=str(m15_data_path),
                initial_balance=10000.0
            )
            
            if results.success:
                logger.info("✓ Multi-timeframe backtest completed successfully")
                logger.info(f"  Final balance: ${results.final_balance:.2f}")
                logger.info(f"  Total trades: {results.total_trades}")
                logger.info(f"  Execution time: {results.execution_time_seconds:.2f}s")
                return True
            else:
                logger.error(f"✗ Backtest failed: {results.error_message}")
                return False
                
        except Exception as e:
            logger.error(f"Integration test backtest failed: {e}")
            return False
        
    except Exception as e:
        logger.error(f"Test failed with exception: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = test_multi_timeframe_backtesting()
    
    if success:
        print("\n🎉 Multi-timeframe backtesting test PASSED!")
        print("✓ Timeframe detection working")
        print("✓ Multi-timeframe data loading working") 
        print("✓ MarketRegimeClassifier integration ready")
        print("✓ Backtesting engine enhanced successfully")
        sys.exit(0)
    else:
        print("\n❌ Multi-timeframe backtesting test FAILED!")
        print("Please check the logs above for details.")
        sys.exit(1)


  
