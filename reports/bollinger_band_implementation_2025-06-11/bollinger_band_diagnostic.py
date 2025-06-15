#!/usr/bin/env python3
"""
Bollinger Band Implementation Diagnostic
Analyzes why the complete strategy is generating 0 trades instead of expected ~35 trades.
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
    
    return df

def analyze_complete_strategy_signals(df, config):
    """Analyze complete strategy signal generation with all filters"""
    
    rsi_period = config['rsi_period']
    oversold_level = config['rsi_oversold']
    overbought_level = config['rsi_overbought']
    bb_period = config['bollinger_period']
    bb_std = config['bollinger_std_dev']
    
    rsi_col = f'RSI_{rsi_period}'
    bbl_col = f'BBL_{bb_period}_{bb_std}'
    bbu_col = f'BBU_{bb_period}_{bb_std}'
    
    # Calculate RSI crossings
    df['rsi_prev'] = df[rsi_col].shift(1)
    
    # BUY signals: RSI crosses up from oversold
    rsi_buy_signals = (df['rsi_prev'] < oversold_level) & (df[rsi_col] >= oversold_level)
    
    # SELL signals: RSI crosses down from overbought
    rsi_sell_signals = (df['rsi_prev'] > overbought_level) & (df[rsi_col] <= overbought_level)
    
    # Apply Bollinger Band filters
    # BUY: price <= lower BB
    bb_buy_filter = df['close'] <= df[bbl_col]
    complete_buy_signals = rsi_buy_signals & bb_buy_filter
    
    # SELL: price >= upper BB
    bb_sell_filter = df['close'] >= df[bbu_col]
    complete_sell_signals = rsi_sell_signals & bb_sell_filter
    
    # Count signals at each stage
    rsi_buy_count = rsi_buy_signals.sum()
    rsi_sell_count = rsi_sell_signals.sum()
    complete_buy_count = complete_buy_signals.sum()
    complete_sell_count = complete_sell_signals.sum()
    
    # Analyze blocked signals
    blocked_buy_signals = rsi_buy_signals & ~bb_buy_filter
    blocked_sell_signals = rsi_sell_signals & ~bb_sell_filter
    
    blocked_buy_count = blocked_buy_signals.sum()
    blocked_sell_count = blocked_sell_signals.sum()
    
    return {
        'rsi_buy_count': rsi_buy_count,
        'rsi_sell_count': rsi_sell_count,
        'complete_buy_count': complete_buy_count,
        'complete_sell_count': complete_sell_count,
        'blocked_buy_count': blocked_buy_count,
        'blocked_sell_count': blocked_sell_count,
        'complete_buy_dates': df[complete_buy_signals].index.tolist(),
        'complete_sell_dates': df[complete_sell_signals].index.tolist(),
        'blocked_buy_dates': df[blocked_buy_signals].index.tolist()[:10],  # First 10 for analysis
        'blocked_sell_dates': df[blocked_sell_signals].index.tolist()[:10]  # First 10 for analysis
    }

def analyze_signal_examples(df, analysis_results, config):
    """Analyze specific examples of blocked and passed signals"""
    
    bb_period = config['bollinger_period']
    bb_std = config['bollinger_std_dev']
    rsi_period = config['rsi_period']
    
    bbl_col = f'BBL_{bb_period}_{bb_std}'
    bbu_col = f'BBU_{bb_period}_{bb_std}'
    rsi_col = f'RSI_{rsi_period}'
    
    examples = {
        'blocked_buy_examples': [],
        'blocked_sell_examples': [],
        'passed_buy_examples': [],
        'passed_sell_examples': []
    }
    
    # Analyze blocked BUY signals
    for date in analysis_results['blocked_buy_dates']:
        if date in df.index:
            row = df.loc[date]
            examples['blocked_buy_examples'].append({
                'date': date,
                'close': row['close'],
                'lower_bb': row[bbl_col],
                'rsi': row[rsi_col],
                'price_vs_bb': row['close'] - row[bbl_col],
                'price_above_bb_pips': (row['close'] - row[bbl_col]) * 10000  # Approximate pips
            })
    
    # Analyze blocked SELL signals
    for date in analysis_results['blocked_sell_dates']:
        if date in df.index:
            row = df.loc[date]
            examples['blocked_sell_examples'].append({
                'date': date,
                'close': row['close'],
                'upper_bb': row[bbu_col],
                'rsi': row[rsi_col],
                'price_vs_bb': row['close'] - row[bbu_col],
                'price_below_bb_pips': (row[bbu_col] - row['close']) * 10000  # Approximate pips
            })
    
    # Analyze passed signals (if any)
    for date in analysis_results['complete_buy_dates']:
        if date in df.index:
            row = df.loc[date]
            examples['passed_buy_examples'].append({
                'date': date,
                'close': row['close'],
                'lower_bb': row[bbl_col],
                'rsi': row[rsi_col],
                'price_vs_bb': row['close'] - row[bbl_col]
            })
    
    for date in analysis_results['complete_sell_dates']:
        if date in df.index:
            row = df.loc[date]
            examples['passed_sell_examples'].append({
                'date': date,
                'close': row['close'],
                'upper_bb': row[bbu_col],
                'rsi': row[rsi_col],
                'price_vs_bb': row['close'] - row[bbu_col]
            })
    
    return examples

def main():
    """Main diagnostic function"""
    print("Loading strategy configuration...")
    config = load_strategy_config()
    
    print("Loading market data...")
    df = load_market_data()
    
    print("Calculating technical indicators...")
    df = calculate_indicators(df, config)
    
    print("Analyzing complete strategy signals...")
    analysis = analyze_complete_strategy_signals(df, config)
    
    print("Analyzing signal examples...")
    examples = analyze_signal_examples(df, analysis, config)
    
    print("\n" + "="*60)
    print("BOLLINGER BAND IMPLEMENTATION DIAGNOSTIC")
    print("="*60)
    
    print(f"RSI BUY signals (crossings): {analysis['rsi_buy_count']}")
    print(f"RSI SELL signals (crossings): {analysis['rsi_sell_count']}")
    print(f"Total RSI signals: {analysis['rsi_buy_count'] + analysis['rsi_sell_count']}")
    
    print(f"\nComplete BUY signals (RSI + BB): {analysis['complete_buy_count']}")
    print(f"Complete SELL signals (RSI + BB): {analysis['complete_sell_count']}")
    print(f"Total complete signals: {analysis['complete_buy_count'] + analysis['complete_sell_count']}")
    
    print(f"\nBlocked BUY signals: {analysis['blocked_buy_count']}")
    print(f"Blocked SELL signals: {analysis['blocked_sell_count']}")
    print(f"Total blocked signals: {analysis['blocked_buy_count'] + analysis['blocked_sell_count']}")
    
    # Calculate filtering efficiency
    total_rsi = analysis['rsi_buy_count'] + analysis['rsi_sell_count']
    total_complete = analysis['complete_buy_count'] + analysis['complete_sell_count']
    total_blocked = analysis['blocked_buy_count'] + analysis['blocked_sell_count']
    
    if total_rsi > 0:
        filter_rate = (total_blocked / total_rsi) * 100
        pass_rate = (total_complete / total_rsi) * 100
        print(f"\nFiltering efficiency: {filter_rate:.1f}% blocked, {pass_rate:.1f}% passed")
    
    print("\n" + "="*60)
    print("BLOCKED SIGNAL ANALYSIS")
    print("="*60)
    
    print("\nBlocked BUY Signal Examples:")
    for i, example in enumerate(examples['blocked_buy_examples'][:5]):
        print(f"  {i+1}. {example['date']}: Price {example['close']:.5f} > Lower BB {example['lower_bb']:.5f}")
        print(f"     Difference: {example['price_above_bb_pips']:.1f} pips above lower BB")
        print(f"     RSI: {example['rsi']:.2f}")
    
    print("\nBlocked SELL Signal Examples:")
    for i, example in enumerate(examples['blocked_sell_examples'][:5]):
        print(f"  {i+1}. {example['date']}: Price {example['close']:.5f} < Upper BB {example['upper_bb']:.5f}")
        print(f"     Difference: {example['price_below_bb_pips']:.1f} pips below upper BB")
        print(f"     RSI: {example['rsi']:.2f}")
    
    if examples['passed_buy_examples'] or examples['passed_sell_examples']:
        print("\n" + "="*60)
        print("PASSED SIGNAL ANALYSIS")
        print("="*60)
        
        for example in examples['passed_buy_examples']:
            print(f"PASSED BUY: {example['date']}: Price {example['close']:.5f} <= Lower BB {example['lower_bb']:.5f}")
        
        for example in examples['passed_sell_examples']:
            print(f"PASSED SELL: {example['date']}: Price {example['close']:.5f} >= Upper BB {example['upper_bb']:.5f}")
    else:
        print("\n❌ NO SIGNALS PASSED THE COMPLETE FILTER!")
        print("This explains why the backtest generated 0 trades.")
    
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    
    if total_complete == 0:
        print("❌ ISSUE IDENTIFIED: Bollinger Band filter is TOO RESTRICTIVE")
        print("   - All RSI signals are being blocked by BB filter")
        print("   - This suggests the BB filter logic may need adjustment")
        print("   - Consider using <= and >= instead of < and >")
        print("   - Or the BB parameters (period=20, std=2.0) may be too tight")
    else:
        print(f"✅ Strategy working as intended: {total_complete} signals passed filter")
    
    print("="*60)

if __name__ == "__main__":
    main()
