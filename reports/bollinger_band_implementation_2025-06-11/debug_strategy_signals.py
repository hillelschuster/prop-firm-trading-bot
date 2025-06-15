#!/usr/bin/env python3
"""
Debug script to analyze why EURUSD_TrendFollowing_Test strategy isn't generating signals.
Part 1: Find potential trade setups (SMA crossovers)
Part 2: Debug strategy logic at those timestamps
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def calculate_sma(data, period):
    """Calculate Simple Moving Average"""
    return data.rolling(window=period, min_periods=period).mean()

def calculate_atr(high, low, close, period):
    """Calculate Average True Range"""
    # True Range calculation
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # ATR is the moving average of True Range
    atr = true_range.rolling(window=period, min_periods=period).mean()
    return atr

def find_crossovers(fast_sma, slow_sma):
    """Find SMA crossover points"""
    # Calculate crossover signals
    # BUY: fast_sma crosses above slow_sma (fast was below, now above)
    # SELL: fast_sma crosses below slow_sma (fast was above, now below)

    # Create boolean series for current and previous states
    fast_above_slow = (fast_sma > slow_sma).fillna(False)
    fast_above_slow_prev = fast_above_slow.shift(1).fillna(False)

    # BUY signal: fast_sma crosses above slow_sma (was below, now above)
    buy_signals = (~fast_above_slow_prev) & fast_above_slow

    # SELL signal: fast_sma crosses below slow_sma (was above, now below)
    sell_signals = fast_above_slow_prev & (~fast_above_slow)

    return buy_signals, sell_signals

def debug_strategy_logic(analysis_data):
    """Part 2: Debug the strategy logic at identified signal timestamps"""
    print(f"\n" + "="*80)
    print(f"PART 2: STRATEGY LOGIC DEBUG ANALYSIS")
    print(f"="*80)

    df_clean = analysis_data['df_clean']
    buy_timestamps = analysis_data['buy_timestamps']
    sell_timestamps = analysis_data['sell_timestamps']

    # Strategy parameters from EURUSD_TrendFollowing_Test config
    strategy_params = {
        'fast_sma_period': 5,
        'slow_sma_period': 10,
        'atr_period_for_sl': 5,
        'atr_multiplier_for_sl': 2.0,
        'min_reward_risk_ratio': 1.5,
        'use_trailing_stop': False,
        'trailing_stop_atr_period': 5,
        'trailing_stop_atr_multiplier': 1.5,
        'max_position_age_bars': 50
    }

    # EURUSD symbol properties (from PaperTradingAdapter defaults)
    point_value = 0.00001  # 5-digit broker

    print(f"\n📋 Strategy Parameters:")
    for key, value in strategy_params.items():
        print(f"   {key}: {value}")
    print(f"\n💱 Symbol Properties:")
    print(f"   point_value: {point_value}")

    # Debug first BUY signal
    if len(buy_timestamps) > 0:
        debug_signal_logic(df_clean, buy_timestamps[0], "BUY", strategy_params, point_value)

    # Debug first SELL signal
    if len(sell_timestamps) > 0:
        debug_signal_logic(df_clean, sell_timestamps[0], "SELL", strategy_params, point_value)

def debug_signal_logic(df_clean, timestamp, signal_type, strategy_params, point_value):
    """Debug the strategy logic for a specific signal timestamp"""
    print(f"\n" + "-"*60)
    print(f"🔍 DEBUGGING {signal_type} SIGNAL AT {timestamp}")
    print(f"-"*60)

    # Get current and previous data
    current_data = df_clean.loc[timestamp]
    signal_pos = df_clean.index.get_indexer([timestamp])[0]

    if signal_pos > 0:
        prev_data = df_clean.iloc[signal_pos - 1]
    else:
        print(f"❌ No previous data available for signal validation")
        return

    # Extract values
    close_price = current_data['close']
    fast_sma = current_data['SMA_5']
    slow_sma = current_data['SMA_10']
    atr_value = current_data['ATR_5']

    prev_fast_sma = prev_data['SMA_5']
    prev_slow_sma = prev_data['SMA_10']

    print(f"\n📊 Market Data:")
    print(f"   Close Price: {close_price:.5f}")
    print(f"   Current SMA_5: {fast_sma:.5f}, SMA_10: {slow_sma:.5f}")
    print(f"   Previous SMA_5: {prev_fast_sma:.5f}, SMA_10: {prev_slow_sma:.5f}")
    print(f"   ATR_5: {atr_value:.5f}")

    # Step 1: Verify crossover condition
    print(f"\n🔍 Step 1: Crossover Verification")
    if signal_type == "BUY":
        crossover_valid = (prev_fast_sma <= prev_slow_sma) and (fast_sma > slow_sma)
        print(f"   BUY Condition: (prev_fast <= prev_slow) AND (fast > slow)")
        print(f"   ({prev_fast_sma:.5f} <= {prev_slow_sma:.5f}) AND ({fast_sma:.5f} > {slow_sma:.5f})")
        print(f"   ({prev_fast_sma <= prev_slow_sma}) AND ({fast_sma > slow_sma}) = {crossover_valid}")
    else:  # SELL
        crossover_valid = (prev_fast_sma >= prev_slow_sma) and (fast_sma < slow_sma)
        print(f"   SELL Condition: (prev_fast >= prev_slow) AND (fast < slow)")
        print(f"   ({prev_fast_sma:.5f} >= {prev_slow_sma:.5f}) AND ({fast_sma:.5f} < {slow_sma:.5f})")
        print(f"   ({prev_fast_sma >= prev_slow_sma}) AND ({fast_sma < slow_sma}) = {crossover_valid}")

    if not crossover_valid:
        print(f"   ❌ CROSSOVER CONDITION FAILED!")
        return
    else:
        print(f"   ✅ Crossover condition passed")

    # Step 2: Calculate stop loss
    print(f"\n🔍 Step 2: Stop Loss Calculation")
    atr_multiplier = strategy_params['atr_multiplier_for_sl']
    stop_loss_pips = atr_value * atr_multiplier / point_value

    print(f"   stop_loss_pips = atr_value * atr_multiplier / point_value")
    print(f"   stop_loss_pips = {atr_value:.5f} * {atr_multiplier} / {point_value}")
    print(f"   stop_loss_pips = {stop_loss_pips:.2f} pips")

    # Step 3: Calculate take profit
    print(f"\n🔍 Step 3: Take Profit Calculation")
    min_rr_ratio = strategy_params['min_reward_risk_ratio']
    take_profit_pips = stop_loss_pips * min_rr_ratio

    print(f"   take_profit_pips = stop_loss_pips * min_reward_risk_ratio")
    print(f"   take_profit_pips = {stop_loss_pips:.2f} * {min_rr_ratio}")
    print(f"   take_profit_pips = {take_profit_pips:.2f} pips")

    # Step 4: Risk/Reward validation
    print(f"\n🔍 Step 4: Risk/Reward Validation")
    actual_rr_ratio = take_profit_pips / stop_loss_pips if stop_loss_pips > 0 else 0
    rr_valid = actual_rr_ratio >= min_rr_ratio

    print(f"   actual_rr_ratio = take_profit_pips / stop_loss_pips")
    print(f"   actual_rr_ratio = {take_profit_pips:.2f} / {stop_loss_pips:.2f} = {actual_rr_ratio:.2f}")
    print(f"   Required minimum: {min_rr_ratio}")
    print(f"   Risk/Reward valid: {rr_valid}")

    if not rr_valid:
        print(f"   ❌ RISK/REWARD RATIO TOO LOW!")
        return
    else:
        print(f"   ✅ Risk/Reward ratio acceptable")

    # Step 5: Final signal generation decision
    print(f"\n🔍 Step 5: Final Signal Decision")
    print(f"   All conditions passed - Signal should be generated!")
    print(f"   🎯 Expected Trade:")
    print(f"      Direction: {signal_type}")
    print(f"      Entry Price: {close_price:.5f}")
    print(f"      Stop Loss: {stop_loss_pips:.2f} pips")
    print(f"      Take Profit: {take_profit_pips:.2f} pips")
    print(f"      Risk/Reward: {actual_rr_ratio:.2f}")

    print(f"\n❓ WHY WASN'T THIS TRADE GENERATED?")
    print(f"   Possible reasons:")
    print(f"   1. Strategy not receiving the correct market data")
    print(f"   2. Additional filters in the strategy code not shown here")
    print(f"   3. Position management preventing new trades")
    print(f"   4. Risk management blocking the trade")
    print(f"   5. Timing issues in the backtesting engine")

def main():
    print("="*80)
    print("EURUSD_TrendFollowing_Test Strategy Signal Debug Analysis")
    print("="*80)
    
    # Load the CSV data
    csv_file = "data/EURUSD_M15_2023_05_01_to_2023_05_31.csv"
    print(f"\n📊 Loading data from: {csv_file}")
    
    try:
        df = pd.read_csv(csv_file)
        print(f"✅ Data loaded successfully: {len(df)} rows")
        print(f"📅 Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        print(f"📈 OHLC columns: {list(df.columns)}")
        print(f"📊 First few rows:")
        print(df.head())
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    print(f"\n🔧 Calculating indicators for EURUSD_TrendFollowing_Test strategy...")
    print(f"   - SMA_5 (fast_sma_period=5)")
    print(f"   - SMA_10 (slow_sma_period=10)")  
    print(f"   - ATR_5 (atr_period_for_sl=5)")
    
    # Calculate indicators exactly as the strategy does
    df['SMA_5'] = calculate_sma(df['close'], 5)
    df['SMA_10'] = calculate_sma(df['close'], 10)
    df['ATR_5'] = calculate_atr(df['high'], df['low'], df['close'], 5)
    
    # Remove rows with NaN values (insufficient data for indicators)
    df_clean = df.dropna()
    print(f"📊 Data after removing NaN rows: {len(df_clean)} rows (removed {len(df) - len(df_clean)} rows)")
    
    if len(df_clean) == 0:
        print("❌ No data available after calculating indicators!")
        return
    
    print(f"\n🔍 Searching for SMA crossover signals...")
    
    # Find crossovers
    buy_signals, sell_signals = find_crossovers(df_clean['SMA_5'], df_clean['SMA_10'])
    
    buy_timestamps = df_clean[buy_signals].index
    sell_timestamps = df_clean[sell_signals].index
    
    print(f"📈 BUY signals found: {len(buy_timestamps)}")
    print(f"📉 SELL signals found: {len(sell_timestamps)}")
    
    # Report first BUY signal
    if len(buy_timestamps) > 0:
        first_buy = buy_timestamps[0]
        buy_data = df_clean.loc[first_buy]
        # Find the previous row by getting the position in the index
        buy_pos = df_clean.index.get_indexer([first_buy])[0]
        if buy_pos > 0:
            prev_buy_data = df_clean.iloc[buy_pos - 1]
        else:
            prev_buy_data = None
        
        print(f"\n🟢 FIRST BUY SIGNAL DETECTED:")
        print(f"   📅 Timestamp: {first_buy}")
        print(f"   💰 OHLC: O={buy_data['open']:.5f}, H={buy_data['high']:.5f}, L={buy_data['low']:.5f}, C={buy_data['close']:.5f}")
        print(f"   📊 Current: SMA_5={buy_data['SMA_5']:.5f}, SMA_10={buy_data['SMA_10']:.5f}, ATR_5={buy_data['ATR_5']:.5f}")
        if prev_buy_data is not None:
            print(f"   📊 Previous: SMA_5={prev_buy_data['SMA_5']:.5f}, SMA_10={prev_buy_data['SMA_10']:.5f}")
            print(f"   ✅ Crossover: SMA_5 crossed ABOVE SMA_10 (was {prev_buy_data['SMA_5']:.5f} < {prev_buy_data['SMA_10']:.5f}, now {buy_data['SMA_5']:.5f} > {buy_data['SMA_10']:.5f})")
        else:
            print(f"   ⚠️  No previous data available for crossover confirmation")
    else:
        print(f"\n❌ NO BUY SIGNALS FOUND in the entire dataset!")
    
    # Report first SELL signal
    if len(sell_timestamps) > 0:
        first_sell = sell_timestamps[0]
        sell_data = df_clean.loc[first_sell]
        # Find the previous row by getting the position in the index
        sell_pos = df_clean.index.get_indexer([first_sell])[0]
        if sell_pos > 0:
            prev_sell_data = df_clean.iloc[sell_pos - 1]
        else:
            prev_sell_data = None

        print(f"\n🔴 FIRST SELL SIGNAL DETECTED:")
        print(f"   📅 Timestamp: {first_sell}")
        print(f"   💰 OHLC: O={sell_data['open']:.5f}, H={sell_data['high']:.5f}, L={sell_data['low']:.5f}, C={sell_data['close']:.5f}")
        print(f"   📊 Current: SMA_5={sell_data['SMA_5']:.5f}, SMA_10={sell_data['SMA_10']:.5f}, ATR_5={sell_data['ATR_5']:.5f}")
        if prev_sell_data is not None:
            print(f"   📊 Previous: SMA_5={prev_sell_data['SMA_5']:.5f}, SMA_10={prev_sell_data['SMA_10']:.5f}")
            print(f"   ✅ Crossover: SMA_5 crossed BELOW SMA_10 (was {prev_sell_data['SMA_5']:.5f} > {prev_sell_data['SMA_10']:.5f}, now {sell_data['SMA_5']:.5f} < {sell_data['SMA_10']:.5f})")
        else:
            print(f"   ⚠️  No previous data available for crossover confirmation")
    else:
        print(f"\n❌ NO SELL SIGNALS FOUND in the entire dataset!")
    
    # Save the analysis data for Part 2
    analysis_data = {
        'df_clean': df_clean,
        'buy_timestamps': buy_timestamps,
        'sell_timestamps': sell_timestamps
    }
    
    return analysis_data

if __name__ == "__main__":
    analysis_data = main()
    
    if analysis_data and (len(analysis_data['buy_timestamps']) > 0 or len(analysis_data['sell_timestamps']) > 0):
        print(f"\n🔧 Proceeding to Part 2: Strategy Logic Debug...")
        debug_strategy_logic(analysis_data)
    else:
        print(f"\n⚠️  No crossover signals found. This explains why no trades were generated!")
        print(f"   Possible reasons:")
        print(f"   1. SMA periods (5 vs 10) are too close - not enough separation for crossovers")
        print(f"   2. Market was trending in one direction without reversals")
        print(f"   3. Data period too short for meaningful crossovers")


  
