#!/usr/bin/env python3
"""
Signal Flow Diagnostic
Analyzes the complete signal flow from strategy generation to trade execution
to identify why 35 generated signals resulted in 0 trades.
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

def find_valid_signals(df, config):
    """Find all signals that should pass the complete strategy filter"""
    
    rsi_period = config['rsi_period']
    oversold_level = config['rsi_oversold']
    overbought_level = config['rsi_overbought']
    bb_period = config['bollinger_period']
    bb_std = config['bollinger_std_dev']
    atr_period = config['stop_loss_atr_period']
    
    rsi_col = f'RSI_{rsi_period}'
    bbl_col = f'BBL_{bb_period}_{bb_std}'
    bbu_col = f'BBU_{bb_period}_{bb_std}'
    atr_col = f'ATR_{atr_period}'
    
    # Calculate RSI crossings
    df['rsi_prev'] = df[rsi_col].shift(1)
    
    # BUY signals: RSI crosses up from oversold AND price <= lower BB
    rsi_buy_crossings = (df['rsi_prev'] < oversold_level) & (df[rsi_col] >= oversold_level)
    bb_buy_filter = df['close'] <= df[bbl_col]
    valid_buy_signals = rsi_buy_crossings & bb_buy_filter
    
    # SELL signals: RSI crosses down from overbought AND price >= upper BB
    rsi_sell_crossings = (df['rsi_prev'] > overbought_level) & (df[rsi_col] <= overbought_level)
    bb_sell_filter = df['close'] >= df[bbu_col]
    valid_sell_signals = rsi_sell_crossings & bb_sell_filter
    
    # Check for data quality issues
    valid_data_mask = (
        pd.notna(df[rsi_col]) & 
        pd.notna(df['rsi_prev']) & 
        pd.notna(df[bbl_col]) & 
        pd.notna(df[bbu_col]) & 
        pd.notna(df[atr_col]) &
        pd.notna(df['close'])
    )
    
    # Apply data quality filter
    valid_buy_signals = valid_buy_signals & valid_data_mask
    valid_sell_signals = valid_sell_signals & valid_data_mask
    
    # Get signal details
    buy_signal_dates = df[valid_buy_signals].index.tolist()
    sell_signal_dates = df[valid_sell_signals].index.tolist()
    
    signal_details = []
    
    # Analyze BUY signals
    for date in buy_signal_dates:
        row = df.loc[date]
        signal_details.append({
            'date': date,
            'type': 'BUY',
            'close': row['close'],
            'rsi': row[rsi_col],
            'rsi_prev': row['rsi_prev'],
            'lower_bb': row[bbl_col],
            'upper_bb': row[bbu_col],
            'atr': row[atr_col],
            'bb_condition': row['close'] <= row[bbl_col],
            'rsi_condition': (row['rsi_prev'] < oversold_level) and (row[rsi_col] >= oversold_level),
            'data_quality': all([
                pd.notna(row[rsi_col]),
                pd.notna(row['rsi_prev']),
                pd.notna(row[bbl_col]),
                pd.notna(row[atr_col])
            ])
        })
    
    # Analyze SELL signals
    for date in sell_signal_dates:
        row = df.loc[date]
        signal_details.append({
            'date': date,
            'type': 'SELL',
            'close': row['close'],
            'rsi': row[rsi_col],
            'rsi_prev': row['rsi_prev'],
            'lower_bb': row[bbl_col],
            'upper_bb': row[bbu_col],
            'atr': row[atr_col],
            'bb_condition': row['close'] >= row[bbu_col],
            'rsi_condition': (row['rsi_prev'] > overbought_level) and (row[rsi_col] <= overbought_level),
            'data_quality': all([
                pd.notna(row[rsi_col]),
                pd.notna(row['rsi_prev']),
                pd.notna(row[bbu_col]),
                pd.notna(row[atr_col])
            ])
        })
    
    return signal_details

def analyze_potential_issues(signal_details):
    """Analyze potential issues that could prevent signal execution"""
    
    issues = {
        'data_quality_issues': 0,
        'atr_zero_or_negative': 0,
        'extreme_price_levels': 0,
        'weekend_signals': 0,
        'valid_signals': 0
    }
    
    for signal in signal_details:
        if not signal['data_quality']:
            issues['data_quality_issues'] += 1
            continue
            
        if signal['atr'] <= 0:
            issues['atr_zero_or_negative'] += 1
            continue
            
        # Check for weekend (Saturday = 5, Sunday = 6)
        if signal['date'].weekday() >= 5:
            issues['weekend_signals'] += 1
            continue
            
        # Check for extreme price levels (potential data errors)
        if signal['close'] <= 0 or signal['close'] > 10:  # Unrealistic for EURUSD
            issues['extreme_price_levels'] += 1
            continue
            
        issues['valid_signals'] += 1
    
    return issues

def main():
    """Main diagnostic function"""
    print("Loading strategy configuration...")
    config = load_strategy_config()
    
    print("Loading market data...")
    df = load_market_data()
    
    print("Calculating technical indicators...")
    df = calculate_indicators(df, config)
    
    print("Finding valid signals...")
    signal_details = find_valid_signals(df, config)
    
    print("Analyzing potential issues...")
    issues = analyze_potential_issues(signal_details)
    
    print("\n" + "="*60)
    print("SIGNAL FLOW DIAGNOSTIC REPORT")
    print("="*60)
    
    print(f"Total signals generated by strategy logic: {len(signal_details)}")
    print(f"  - BUY signals: {len([s for s in signal_details if s['type'] == 'BUY'])}")
    print(f"  - SELL signals: {len([s for s in signal_details if s['type'] == 'SELL'])}")
    
    print("\nSignal Quality Analysis:")
    print(f"  - Data quality issues: {issues['data_quality_issues']}")
    print(f"  - ATR zero/negative: {issues['atr_zero_or_negative']}")
    print(f"  - Weekend signals: {issues['weekend_signals']}")
    print(f"  - Extreme price levels: {issues['extreme_price_levels']}")
    print(f"  - Valid signals: {issues['valid_signals']}")
    
    if issues['valid_signals'] > 0:
        print(f"\n✅ {issues['valid_signals']} signals should be executable")
        print("❌ But backtest showed 0 trades - investigating further...")
        
        print("\nFirst 5 Valid Signal Examples:")
        valid_signals = [s for s in signal_details if s['data_quality'] and s['atr'] > 0]
        for i, signal in enumerate(valid_signals[:5]):
            print(f"\n{i+1}. {signal['type']} Signal at {signal['date']}")
            print(f"   Price: {signal['close']:.5f}")
            print(f"   RSI: {signal['rsi_prev']:.2f} → {signal['rsi']:.2f}")
            if signal['type'] == 'BUY':
                print(f"   Lower BB: {signal['lower_bb']:.5f} (price <= BB: {signal['bb_condition']})")
            else:
                print(f"   Upper BB: {signal['upper_bb']:.5f} (price >= BB: {signal['bb_condition']})")
            print(f"   ATR: {signal['atr']:.5f}")
            print(f"   Data Quality: {signal['data_quality']}")
        
        print("\n" + "="*60)
        print("POTENTIAL ROOT CAUSES FOR 0 TRADES:")
        print("="*60)
        print("1. Risk Controller blocking all trades (position limits, drawdown)")
        print("2. Platform adapter issues (tick data, symbol info)")
        print("3. Order execution failures")
        print("4. Concurrent position limits (max 2 positions)")
        print("5. Missing trend filter or other strategy filters")
        print("6. Backtesting engine configuration issues")
        
        print("\nRECOMMENDED INVESTIGATION:")
        print("- Check risk controller logs for trade rejections")
        print("- Verify platform adapter tick data availability")
        print("- Check for active positions blocking new entries")
        print("- Review orchestrator signal processing logs")
        
    else:
        print(f"\n❌ No valid signals found - all {len(signal_details)} signals have issues")
        print("This explains why backtest generated 0 trades")
    
    print("="*60)

if __name__ == "__main__":
    main()
