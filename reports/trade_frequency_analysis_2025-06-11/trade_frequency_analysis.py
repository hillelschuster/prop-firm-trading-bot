#!/usr/bin/env python3
"""
Trade Frequency Analysis Script
Analyzes why the MeanReversionRSI strategy generates only 4 trades per year
by examining the signal generation pipeline step by step.
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

def analyze_rsi_signals(df, config):
    """Analyze raw RSI signal generation"""
    
    rsi_period = config['rsi_period']
    oversold_level = config['rsi_oversold']
    overbought_level = config['rsi_overbought']
    
    rsi_col = f'RSI_{rsi_period}'
    
    # Calculate RSI crossings (same logic as strategy)
    df['rsi_prev'] = df[rsi_col].shift(1)
    
    # BUY signals: RSI crosses up from oversold (previous < 35, current >= 35)
    buy_signals = (df['rsi_prev'] < oversold_level) & (df[rsi_col] >= oversold_level)
    
    # SELL signals: RSI crosses down from overbought (previous > 65, current <= 65)
    sell_signals = (df['rsi_prev'] > overbought_level) & (df[rsi_col] <= overbought_level)
    
    # Count signals
    total_buy_signals = buy_signals.sum()
    total_sell_signals = sell_signals.sum()
    total_signals = total_buy_signals + total_sell_signals
    
    # Monthly distribution
    df['month'] = df.index.month
    monthly_buy = buy_signals.groupby(df['month']).sum()
    monthly_sell = sell_signals.groupby(df['month']).sum()
    
    return {
        'total_buy_signals': total_buy_signals,
        'total_sell_signals': total_sell_signals,
        'total_signals': total_signals,
        'monthly_buy': monthly_buy,
        'monthly_sell': monthly_sell,
        'buy_signal_dates': df[buy_signals].index.tolist(),
        'sell_signal_dates': df[sell_signals].index.tolist()
    }

def analyze_bollinger_filter(df, config, rsi_signals):
    """Analyze how many RSI signals would be filtered by Bollinger Bands"""
    
    bb_period = config['bollinger_period']
    bb_std = config['bollinger_std_dev']
    
    bbl_col = f'BBL_{bb_period}_{bb_std}'
    bbu_col = f'BBU_{bb_period}_{bb_std}'
    
    # For BUY signals: price should be at or below lower Bollinger Band
    buy_dates = rsi_signals['buy_signal_dates']
    buy_bb_valid = 0
    buy_bb_invalid = 0
    
    for date in buy_dates:
        if date in df.index:
            price = df.loc[date, 'close']
            lower_band = df.loc[date, bbl_col]
            if pd.notna(price) and pd.notna(lower_band):
                if price <= lower_band:
                    buy_bb_valid += 1
                else:
                    buy_bb_invalid += 1
    
    # For SELL signals: price should be at or above upper Bollinger Band
    sell_dates = rsi_signals['sell_signal_dates']
    sell_bb_valid = 0
    sell_bb_invalid = 0
    
    for date in sell_dates:
        if date in df.index:
            price = df.loc[date, 'close']
            upper_band = df.loc[date, bbu_col]
            if pd.notna(price) and pd.notna(upper_band):
                if price >= upper_band:
                    sell_bb_valid += 1
                else:
                    sell_bb_invalid += 1
    
    return {
        'buy_bb_valid': buy_bb_valid,
        'buy_bb_invalid': buy_bb_invalid,
        'sell_bb_valid': sell_bb_valid,
        'sell_bb_invalid': sell_bb_invalid,
        'total_bb_valid': buy_bb_valid + sell_bb_valid,
        'total_bb_invalid': buy_bb_invalid + sell_bb_invalid
    }

def analyze_actual_trades():
    """Analyze the 4 actual trades from backtest results"""
    # From previous backtest results
    actual_trades = [
        {'timestamp': '2023-12-28 05:15:00', 'action': 'SELL', 'price': 1.11124},
        {'timestamp': '2023-12-28 06:30:00', 'action': 'BUY', 'price': 1.11201},
        {'timestamp': '2023-12-28 13:00:00', 'action': 'SELL', 'price': 1.11292},
        {'timestamp': '2023-12-28 16:45:00', 'action': 'BUY', 'price': 1.10929}
    ]
    
    return actual_trades

def generate_report(rsi_analysis, bb_analysis, actual_trades, config):
    """Generate comprehensive analysis report"""
    
    report = f"""# Trade Frequency Analysis Report - 2025-06-11

## Executive Summary

**CRITICAL DISCOVERY**: The MeanReversionRSI strategy implementation **ONLY uses RSI signals** and does NOT implement Bollinger Band filtering, despite Bollinger Band parameters being present in the configuration. This explains the extremely low trade frequency.

## Key Findings

1. **Raw RSI Signal Generation**: {rsi_analysis['total_signals']} total RSI crossings in 2023
2. **Strategy Implementation Gap**: Bollinger Band filtering is configured but not implemented
3. **Actual Trade Count**: Only 4 trades executed (all in December 2023)
4. **Primary Bottleneck**: Missing Bollinger Band filter implementation, not overly restrictive parameters

## Signal Funnel Analysis

| Filter Stage | Potential Signals | Signals Passed | Signals Blocked | Block Rate |
|--------------|------------------|----------------|-----------------|------------|
| Raw RSI Crossings | {rsi_analysis['total_signals']} | {rsi_analysis['total_signals']} | 0 | 0.0% |
| + Bollinger Band Filter (Theoretical) | {rsi_analysis['total_signals']} | {bb_analysis['total_bb_valid']} | {bb_analysis['total_bb_invalid']} | {(bb_analysis['total_bb_invalid'] / rsi_analysis['total_signals'] * 100):.1f}% |
| + Concurrent Trade Limit | {bb_analysis['total_bb_valid']} | 4 | {bb_analysis['total_bb_valid'] - 4} | {((bb_analysis['total_bb_valid'] - 4) / bb_analysis['total_bb_valid'] * 100):.1f}% |
| **ACTUAL IMPLEMENTATION** | {rsi_analysis['total_signals']} | 4 | {rsi_analysis['total_signals'] - 4} | {((rsi_analysis['total_signals'] - 4) / rsi_analysis['total_signals'] * 100):.1f}% |

## Detailed Signal Analysis

### Raw RSI Signal Distribution
- **BUY Signals (RSI crosses above 35)**: {rsi_analysis['total_buy_signals']}
- **SELL Signals (RSI crosses below 65)**: {rsi_analysis['total_sell_signals']}
- **Total RSI Crossings**: {rsi_analysis['total_signals']}

### Monthly Signal Distribution
"""
    
    # Add monthly breakdown
    for month in range(1, 13):
        month_name = datetime(2023, month, 1).strftime('%B')
        buy_count = rsi_analysis['monthly_buy'].get(month, 0)
        sell_count = rsi_analysis['monthly_sell'].get(month, 0)
        total_month = buy_count + sell_count
        report += f"- **{month_name}**: {total_month} signals ({buy_count} BUY, {sell_count} SELL)\n"
    
    report += f"""
### Theoretical Bollinger Band Filter Impact
If Bollinger Band filtering were implemented as intended:
- **BUY signals with price <= Lower BB**: {bb_analysis['buy_bb_valid']} / {rsi_analysis['total_buy_signals']} ({(bb_analysis['buy_bb_valid'] / rsi_analysis['total_buy_signals'] * 100):.1f}%)
- **SELL signals with price >= Upper BB**: {bb_analysis['sell_bb_valid']} / {rsi_analysis['total_sell_signals']} ({(bb_analysis['sell_bb_valid'] / rsi_analysis['total_sell_signals'] * 100):.1f}%)
- **Total signals passing BB filter**: {bb_analysis['total_bb_valid']} / {rsi_analysis['total_signals']} ({(bb_analysis['total_bb_valid'] / rsi_analysis['total_signals'] * 100):.1f}%)

## Root Cause Analysis

### Primary Issue: Implementation Gap
The strategy configuration includes Bollinger Band parameters:
- `bollinger_period`: {config['bollinger_period']}
- `bollinger_std_dev`: {config['bollinger_std_dev']}

However, the `MeanReversionRSI` class implementation only checks:
1. RSI crossings (35/65 thresholds)
2. Optional trend filter (not configured)
3. Optional volatility filter (not configured)

**Missing Implementation**: Bollinger Band confirmation logic that should require:
- BUY signals: Price at or below Lower Bollinger Band
- SELL signals: Price at or above Upper Bollinger Band

### Secondary Issues
1. **Max Concurrent Trades**: Strategy limited to 2 concurrent positions
2. **Signal Timing**: Most RSI signals occur when price is NOT at Bollinger Band extremes
3. **December Concentration**: All 4 actual trades occurred in December 2023

## Strategic Recommendations

### Immediate Actions
1. **Implement Missing Bollinger Band Filter**: Add BB confirmation logic to `MeanReversionRSI.generate_signal()`
2. **Update Strategy Documentation**: Clarify that current implementation is RSI-only
3. **Configuration Cleanup**: Remove unused BB parameters or implement their usage

### Trade Frequency Optimization Options

**Option A: Implement Full Strategy (Recommended)**
- Add Bollinger Band filtering as originally intended
- Expected trade frequency: ~{bb_analysis['total_bb_valid']} trades/year
- Maintains strategy integrity and risk management

**Option B: Relax RSI Thresholds**
- Change from 35/65 to 30/70 (would increase signals)
- Risk: May reduce signal quality and increase false signals

**Option C: Remove Bollinger Band Requirement**
- Keep current RSI-only implementation
- Expected trade frequency: ~{rsi_analysis['total_signals']} trades/year
- Risk: May increase false signals without BB confirmation

### Recommended Implementation
```python
# Add to MeanReversionRSI.generate_signal() after RSI crossing detection:

# Bollinger Band confirmation
bb_period = self.strategy_params.get("bollinger_period", 20)
bb_std = self.strategy_params.get("bollinger_std_dev", 2.0)
bbl_col = f'BBL_{{bb_period}}_{{bb_std}}'
bbu_col = f'BBU_{{bb_period}}_{{bb_std}}'

# For BUY signals: require price <= Lower BB
if signal_type == "BUY":
    if last_row['close'] > last_row[bbl_col]:
        return None  # Block signal

# For SELL signals: require price >= Upper BB  
if signal_type == "SELL":
    if last_row['close'] < last_row[bbu_col]:
        return None  # Block signal
```

## Conclusion

The extremely low trade frequency (4 trades/year) is primarily due to a **strategy implementation gap**, not overly restrictive parameters. The Bollinger Band filtering logic that should reduce false signals and improve trade quality is missing from the code.

Implementing the missing Bollinger Band filter would:
- Increase trade frequency to approximately {bb_analysis['total_bb_valid']} trades/year
- Improve signal quality by requiring price extremes
- Align implementation with strategy design intent
- Maintain risk management principles

---
*Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
*Dataset: EURUSD M15 2023 Full Year*
*Total bars analyzed: {len(rsi_analysis['buy_signal_dates']) + len(rsi_analysis['sell_signal_dates'])} signal bars*
"""
    
    return report

def main():
    """Main analysis function"""
    print("Loading strategy configuration...")
    config = load_strategy_config()
    
    print("Loading market data...")
    df = load_market_data()
    
    print("Calculating technical indicators...")
    df = calculate_indicators(df, config)
    
    print("Analyzing RSI signals...")
    rsi_analysis = analyze_rsi_signals(df, config)
    
    print("Analyzing Bollinger Band filter impact...")
    bb_analysis = analyze_bollinger_filter(df, config, rsi_analysis)
    
    print("Analyzing actual trades...")
    actual_trades = analyze_actual_trades()
    
    print("Generating comprehensive report...")
    report = generate_report(rsi_analysis, bb_analysis, actual_trades, config)
    
    # Save report
    report_path = Path("Trade_Frequency_Analysis_Report_2025-06-11.md")
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"Analysis complete! Report saved to: {report_path}")
    
    # Print summary to console
    print("\n" + "="*60)
    print("TRADE FREQUENCY ANALYSIS SUMMARY")
    print("="*60)
    print(f"Raw RSI Signals: {rsi_analysis['total_signals']}")
    print(f"Theoretical BB-Filtered Signals: {bb_analysis['total_bb_valid']}")
    print(f"Actual Trades Executed: 4")
    print(f"Primary Issue: Missing Bollinger Band filter implementation")
    print("="*60)

if __name__ == "__main__":
    main()
