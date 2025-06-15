# Analysis Tools

This directory contains debugging and analysis tools for the prop firm trading bot.

## Tools

### debug_strategy_signals.py
Comprehensive two-part strategy signal analysis tool:

**Part 1: Signal Detection**
- Loads CSV market data
- Calculates strategy indicators (SMA, ATR, etc.)
- Identifies potential trade setups (crossovers)
- Reports first BUY and SELL signal timestamps

**Part 2: Strategy Logic Debug**
- Simulates strategy decision-making process
- Step-by-step condition evaluation
- Risk/reward calculations
- Root cause analysis for missing trades

**Usage:**
```bash
python analysis/debug_strategy_signals.py
```

**Purpose:**
- Debug strategies not generating expected trades
- Validate strategy logic before implementation
- Analyze market data for trading opportunities
- Performance tune strategy parameters

## Directory Structure
```
analysis/
├── README.md                    # This file
├── debug_strategy_signals.py    # Strategy signal analysis tool
└── [future analysis tools]      # Additional debugging utilities
```

## Guidelines
- Tools in this directory are for analysis/debugging only
- Not part of the core bot application (src/)
- Should not be imported by production code
- Can access src/ modules for testing purposes


  
