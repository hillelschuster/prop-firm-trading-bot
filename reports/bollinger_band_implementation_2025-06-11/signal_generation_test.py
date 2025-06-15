#!/usr/bin/env python3
"""
Signal Generation Test
Tests the actual signal generation logic by simulating the exact conditions
that should generate a signal based on our diagnostic analysis.
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

# Add project root to sys.path to allow for src imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.strategies.mean_reversion_rsi import MeanReversionRSI
from src.core.models import TickData, Position
from src.core.enums import OrderAction, Timeframe
import logging

def load_strategy_config():
    """Load strategy configuration parameters"""
    config_path = Path("config/strategy_rsi_ranging_market_v1.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config['parameters']

def load_market_data():
    """Load and prepare market data"""
    data_path = Path("data/EURUSD_M15_2023_FULL_YEAR.csv")
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Ensure proper datetime index
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    # Sort by timestamp to ensure proper order
    df.sort_index(inplace=True)
    
    return df

def calculate_indicators(df, config):
    """Calculate technical indicators using same logic as strategy"""
    
    # RSI calculation
    rsi_period = config['rsi_period']
    df[f'RSI_{rsi_period}'] = ta.rsi(df['close'], length=rsi_period)
    
    # Bollinger Bands calculation
    bb_period = config['bollinger_period']
    bb_std = config['bollinger_std_dev']
    bb_result = ta.bbands(df['close'], length=bb_period, std=bb_std)
    
    # Extract Bollinger Band components
    df[f'BBL_{bb_period}_{bb_std}'] = bb_result[f'BBL_{bb_period}_{bb_std}']  # Lower band
    df[f'BBM_{bb_period}_{bb_std}'] = bb_result[f'BBM_{bb_period}_{bb_std}']  # Middle band (SMA)
    df[f'BBU_{bb_period}_{bb_std}'] = bb_result[f'BBU_{bb_period}_{bb_std}']  # Upper band
    
    # ATR calculation
    atr_period = config['stop_loss_atr_period']
    df[f'ATR_{atr_period}'] = ta.atr(df['high'], df['low'], df['close'], length=atr_period)
    
    return df

class MockPlatformAdapter:
    """Mock platform adapter for testing"""
    def get_symbol_info(self, symbol):
        class SymbolInfo:
            def __init__(self):
                self.digits = 5
                self.point = 0.00001
        return SymbolInfo()

class MockConfig:
    """Mock config for testing"""
    def __init__(self):
        class MockProfile:
            def __init__(self):
                self.symbol = "EURUSD"
                self.instrument_details_key = "EURUSD_FTMO"

        self.asset_strategy_profiles = {"TEST": MockProfile()}

        class MockInstrument:
            def __init__(self):
                self.platform_symbol = "EURUSD"
                self.digits = 5
                self.point = 0.00001

        self.loaded_instrument_details = {"EURUSD_FTMO": MockInstrument()}

class MockMarketDataManager:
    """Mock market data manager for testing"""
    pass

def test_signal_generation_for_date(df, config, test_date):
    """Test signal generation for a specific date"""
    
    print(f"\n{'='*60}")
    print(f"TESTING SIGNAL GENERATION FOR: {test_date}")
    print(f"{'='*60}")
    
    # Find the date in the dataframe
    if test_date not in df.index:
        print(f"❌ Date {test_date} not found in data")
        return None
    
    # Get data up to and including the test date
    test_data = df.loc[:test_date].copy()
    
    if len(test_data) < 50:  # Need enough data for indicators
        print(f"❌ Not enough data before {test_date}")
        return None
    
    # Create mock objects
    mock_platform = MockPlatformAdapter()
    mock_config = MockConfig()
    mock_mdm = MockMarketDataManager()
    logger = logging.getLogger("test")
    
    # Create strategy instance
    strategy = MeanReversionRSI(
        strategy_params=config,
        config=mock_config,
        platform_adapter=mock_platform,
        market_data_manager=mock_mdm,
        logger=logger,
        asset_profile_key="TEST"
    )
    
    # Set required attributes
    strategy.symbol = "EURUSD"
    strategy.timeframe = Timeframe.M15
    
    # Create mock tick data
    last_row = test_data.iloc[-1]
    mock_tick = TickData(
        symbol="EURUSD",
        timestamp=test_date,
        bid=last_row['close'] - 0.00001,
        ask=last_row['close'] + 0.00001
    )
    
    # Test signal generation
    try:
        signal = strategy.generate_signal(
            market_data_df=test_data,
            active_position=None,
            latest_tick=mock_tick
        )
        
        if signal:
            print(f"✅ SIGNAL GENERATED!")
            print(f"   Type: {signal['signal']}")
            print(f"   Price: {signal['price']}")
            print(f"   Comment: {signal['comment']}")
            print(f"   Stop Loss: {signal.get('stop_loss_price', 'N/A')}")
            print(f"   Take Profit: {signal.get('take_profit_price', 'N/A')}")
            return signal
        else:
            print(f"❌ NO SIGNAL GENERATED")
            
            # Analyze why no signal was generated
            rsi_period = config['rsi_period']
            bb_period = config['bollinger_period']
            bb_std = config['bollinger_std_dev']
            
            rsi_col = f'RSI_{rsi_period}'
            bbl_col = f'BBL_{bb_period}_{bb_std}'
            bbu_col = f'BBU_{bb_period}_{bb_std}'
            
            current_row = test_data.iloc[-1]
            prev_row = test_data.iloc[-2]
            
            print(f"   Current RSI: {current_row[rsi_col]:.2f}")
            print(f"   Previous RSI: {prev_row[rsi_col]:.2f}")
            print(f"   Current Close: {current_row['close']:.5f}")
            print(f"   Lower BB: {current_row[bbl_col]:.5f}")
            print(f"   Upper BB: {current_row[bbu_col]:.5f}")
            
            # Check conditions
            rsi_buy_cross = (prev_row[rsi_col] < 35) and (current_row[rsi_col] >= 35)
            rsi_sell_cross = (prev_row[rsi_col] > 65) and (current_row[rsi_col] <= 65)
            bb_buy_condition = current_row['close'] <= current_row[bbl_col]
            bb_sell_condition = current_row['close'] >= current_row[bbu_col]
            
            print(f"   RSI Buy Cross: {rsi_buy_cross}")
            print(f"   RSI Sell Cross: {rsi_sell_cross}")
            print(f"   BB Buy Condition: {bb_buy_condition}")
            print(f"   BB Sell Condition: {bb_sell_condition}")
            
            return None
            
    except Exception as e:
        print(f"❌ ERROR DURING SIGNAL GENERATION: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main test function"""
    print("Loading strategy configuration...")
    config = load_strategy_config()
    
    print("Loading market data...")
    df = load_market_data()
    
    print("Calculating technical indicators...")
    df = calculate_indicators(df, config)
    
    # Test dates that should generate signals based on our analysis
    test_dates = [
        pd.Timestamp('2023-02-08 23:30:00'),
        pd.Timestamp('2023-02-14 17:30:00'),
        pd.Timestamp('2023-01-09 11:15:00'),  # SELL signal
        pd.Timestamp('2023-03-09 12:30:00'),  # SELL signal
    ]
    
    print(f"\n{'='*60}")
    print("SIGNAL GENERATION TESTING")
    print(f"{'='*60}")
    
    successful_signals = 0
    
    for test_date in test_dates:
        signal = test_signal_generation_for_date(df, config, test_date)
        if signal:
            successful_signals += 1
    
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tested {len(test_dates)} dates")
    print(f"Successful signals: {successful_signals}")
    print(f"Failed signals: {len(test_dates) - successful_signals}")
    
    if successful_signals == 0:
        print("\n❌ CRITICAL ISSUE: No signals generated even for known valid dates")
        print("This indicates a fundamental problem with the signal generation logic")
    elif successful_signals == len(test_dates):
        print("\n✅ ALL TESTS PASSED: Signal generation working correctly")
        print("The issue must be in the backtesting execution pipeline")
    else:
        print(f"\n⚠️  PARTIAL SUCCESS: {successful_signals}/{len(test_dates)} signals generated")
        print("Some signal generation logic may have issues")

if __name__ == "__main__":
    main()
