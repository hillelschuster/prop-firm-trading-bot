#!/usr/bin/env python3
"""
Debug Target Signal
Analyzes the specific conditions at our target signal date to understand
why it's not generating a signal in the backtest.
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from datetime import datetime
import json
from pathlib import Path

def load_strategy_config():
    """Load strategy configuration parameters"""
    config_path = Path("config/strategy_rsi_ranging_market_v1.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config['parameters']

def load_test_data():
    """Load test subset data"""
    data_path = Path("data/EURUSD_M15_TEST_SUBSET_2023-02-06_to_2023-02-10.csv")
    
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

def analyze_target_signal_conditions(df, config, target_date):
    """Analyze conditions at target signal date"""
    
    print(f"\n{'='*60}")
    print(f"ANALYZING TARGET SIGNAL CONDITIONS")
    print(f"Target Date: {target_date}")
    print(f"{'='*60}")
    
    # Check if target date exists in data
    if target_date not in df.index:
        print(f"❌ Target date {target_date} not found in data")
        # Find closest dates
        available_dates = df.index.tolist()
        closest_before = max([d for d in available_dates if d <= target_date], default=None)
        closest_after = min([d for d in available_dates if d >= target_date], default=None)
        print(f"   Closest before: {closest_before}")
        print(f"   Closest after: {closest_after}")
        return
    
    # Get target row and previous row
    target_idx = df.index.get_loc(target_date)
    if target_idx == 0:
        print(f"❌ Target date is first row - no previous data for RSI crossing analysis")
        return
    
    current_row = df.iloc[target_idx]
    prev_row = df.iloc[target_idx - 1]
    
    # Extract values
    rsi_period = config['rsi_period']
    bb_period = config['bollinger_period']
    bb_std = config['bollinger_std_dev']
    oversold_level = config['rsi_oversold']
    overbought_level = config['rsi_overbought']
    
    rsi_col = f'RSI_{rsi_period}'
    bbl_col = f'BBL_{bb_period}_{bb_std}'
    bbu_col = f'BBU_{bb_period}_{bb_std}'
    
    current_rsi = current_row[rsi_col]
    prev_rsi = prev_row[rsi_col]
    current_close = current_row['close']
    lower_bb = current_row[bbl_col]
    upper_bb = current_row[bbu_col]
    
    print(f"Current RSI: {current_rsi:.2f}")
    print(f"Previous RSI: {prev_rsi:.2f}")
    print(f"Current Close: {current_close:.5f}")
    print(f"Lower BB: {lower_bb:.5f}")
    print(f"Upper BB: {upper_bb:.5f}")
    
    # Check RSI crossing conditions
    rsi_buy_cross = (prev_rsi < oversold_level) and (current_rsi >= oversold_level)
    rsi_sell_cross = (prev_rsi > overbought_level) and (current_rsi <= overbought_level)
    
    print(f"\nRSI Crossing Analysis:")
    print(f"  BUY crossing (prev < 35 AND current >= 35): {rsi_buy_cross}")
    print(f"    Previous RSI < 35: {prev_rsi < oversold_level} ({prev_rsi:.2f} < {oversold_level})")
    print(f"    Current RSI >= 35: {current_rsi >= oversold_level} ({current_rsi:.2f} >= {oversold_level})")
    
    print(f"  SELL crossing (prev > 65 AND current <= 65): {rsi_sell_cross}")
    print(f"    Previous RSI > 65: {prev_rsi > overbought_level} ({prev_rsi:.2f} > {overbought_level})")
    print(f"    Current RSI <= 65: {current_rsi <= overbought_level} ({current_rsi:.2f} <= {overbought_level})")
    
    # Check Bollinger Band conditions
    bb_buy_condition = current_close <= lower_bb
    bb_sell_condition = current_close >= upper_bb
    
    print(f"\nBollinger Band Analysis:")
    print(f"  BUY condition (price <= lower BB): {bb_buy_condition}")
    print(f"    Price <= Lower BB: {current_close:.5f} <= {lower_bb:.5f}")
    
    print(f"  SELL condition (price >= upper BB): {bb_sell_condition}")
    print(f"    Price >= Upper BB: {current_close:.5f} >= {upper_bb:.5f}")
    
    # Final signal determination
    complete_buy_signal = rsi_buy_cross and bb_buy_condition
    complete_sell_signal = rsi_sell_cross and bb_sell_condition
    
    print(f"\nFinal Signal Analysis:")
    print(f"  Complete BUY signal: {complete_buy_signal}")
    print(f"  Complete SELL signal: {complete_sell_signal}")
    
    if complete_buy_signal:
        print(f"✅ Should generate BUY signal at {target_date}")
    elif complete_sell_signal:
        print(f"✅ Should generate SELL signal at {target_date}")
    else:
        print(f"❌ No signal should be generated at {target_date}")
        if rsi_buy_cross and not bb_buy_condition:
            print(f"   Reason: RSI BUY crossing occurred but price not at lower BB")
        elif rsi_sell_cross and not bb_sell_condition:
            print(f"   Reason: RSI SELL crossing occurred but price not at upper BB")
        elif not rsi_buy_cross and not rsi_sell_cross:
            print(f"   Reason: No RSI crossing occurred")
    
    return {
        'rsi_buy_cross': rsi_buy_cross,
        'rsi_sell_cross': rsi_sell_cross,
        'bb_buy_condition': bb_buy_condition,
        'bb_sell_condition': bb_sell_condition,
        'complete_buy_signal': complete_buy_signal,
        'complete_sell_signal': complete_sell_signal
    }

def find_actual_signals_in_subset(df, config):
    """Find all actual signals that should be generated in the test subset"""
    
    print(f"\n{'='*60}")
    print(f"FINDING ALL SIGNALS IN TEST SUBSET")
    print(f"{'='*60}")
    
    rsi_period = config['rsi_period']
    bb_period = config['bollinger_period']
    bb_std = config['bollinger_std_dev']
    oversold_level = config['rsi_oversold']
    overbought_level = config['rsi_overbought']
    
    rsi_col = f'RSI_{rsi_period}'
    bbl_col = f'BBL_{bb_period}_{bb_std}'
    bbu_col = f'BBU_{bb_period}_{bb_std}'
    
    # Calculate RSI crossings
    df['rsi_prev'] = df[rsi_col].shift(1)
    
    # BUY signals: RSI crosses up from oversold AND price <= lower BB
    rsi_buy_crossings = (df['rsi_prev'] < oversold_level) & (df[rsi_col] >= oversold_level)
    bb_buy_filter = df['close'] <= df[bbl_col]
    complete_buy_signals = rsi_buy_crossings & bb_buy_filter
    
    # SELL signals: RSI crosses down from overbought AND price >= upper BB
    rsi_sell_crossings = (df['rsi_prev'] > overbought_level) & (df[rsi_col] <= overbought_level)
    bb_sell_filter = df['close'] >= df[bbu_col]
    complete_sell_signals = rsi_sell_crossings & bb_sell_filter
    
    # Get signal dates
    buy_signal_dates = df[complete_buy_signals].index.tolist()
    sell_signal_dates = df[complete_sell_signals].index.tolist()
    
    print(f"Complete BUY signals in subset: {len(buy_signal_dates)}")
    for date in buy_signal_dates:
        row = df.loc[date]
        print(f"  {date}: RSI {row['rsi_prev']:.2f} → {row[rsi_col]:.2f}, Price {row['close']:.5f} <= BB {row[bbl_col]:.5f}")
    
    print(f"\nComplete SELL signals in subset: {len(sell_signal_dates)}")
    for date in sell_signal_dates:
        row = df.loc[date]
        print(f"  {date}: RSI {row['rsi_prev']:.2f} → {row[rsi_col]:.2f}, Price {row['close']:.5f} >= BB {row[bbu_col]:.5f}")
    
    total_signals = len(buy_signal_dates) + len(sell_signal_dates)
    print(f"\nTotal complete signals in subset: {total_signals}")
    
    return buy_signal_dates, sell_signal_dates

def main():
    """Main analysis function"""
    print("Loading strategy configuration...")
    config = load_strategy_config()
    
    print("Loading test data...")
    df = load_test_data()
    
    print("Calculating technical indicators...")
    df = calculate_indicators(df, config)
    
    # Analyze our target signal date
    target_date = pd.Timestamp('2023-02-08 23:30:00')
    target_analysis = analyze_target_signal_conditions(df, config, target_date)
    
    # Find all actual signals in the subset
    buy_dates, sell_dates = find_actual_signals_in_subset(df, config)
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    
    if target_analysis and (target_analysis['complete_buy_signal'] or target_analysis['complete_sell_signal']):
        print(f"✅ Target date should generate a signal")
    else:
        print(f"❌ Target date should NOT generate a signal")
    
    total_signals = len(buy_dates) + len(sell_dates)
    if total_signals > 0:
        print(f"✅ Test subset contains {total_signals} valid signals")
        print(f"   These signals should trigger our debug logging")
    else:
        print(f"❌ Test subset contains NO valid signals")
        print(f"   This explains why we see no orchestrator debug logs")
    
    print(f"\nNext steps:")
    if total_signals > 0:
        print(f"- Re-run backtest and check for [SIGNAL_DEBUG] logs")
        print(f"- If no debug logs appear, the issue is in signal generation")
    else:
        print(f"- Create a different test subset with actual signal dates")
        print(f"- Use dates from our previous analysis that confirmed 35 signals/year")

if __name__ == "__main__":
    main()
