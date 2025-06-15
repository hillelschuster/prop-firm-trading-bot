# Prop Firm Trading Bot - Core Functions Reference

This document provides a comprehensive reference for all key functions and classes in the prop firm trading bot. Use this as a quick lookup guide for understanding component capabilities and interfaces.

## Orchestrator (Main Coordinator)

### `Orchestrator.__init__(config_manager, main_logger, platform_adapter_override=None)`
Initializes the main trading bot coordinator with all required components.

### `Orchestrator.run()`
**Purpose**: Main trading loop for live trading.
**Returns**: None (runs until stopped)
**Key Behavior**:
- Enters a loop that calls `process_single_bar()` repeatedly.
- Handles graceful shutdown on exceptions or stop signals.
- Manages a `time.sleep()` delay between cycles.

### `Orchestrator.process_single_bar()`
**Purpose**: Processes one logical tick or bar for all active strategies. This is the core of the trading cycle for both live trading and backtesting.
**Returns**: `bool` - `True` if the cycle completes, `False` if a halt is triggered.
**Key Behavior**:
- Performs daily reset and global risk checks.
- Iterates through all enabled strategies and calls `_process_strategy_cycle` for each.

### `Orchestrator._process_strategy_cycle(profile_key, strategy_instance)`
**Purpose**: Runs a full, orchestrated trading cycle for a single strategy instance.
**Process**:
1.  Calls `_is_strategy_active_for_regime()` to check market conditions.
2.  If active, calls `_process_strategy_signal()` to generate and execute new trade signals.
3.  Calls `_manage_strategy_positions()` to apply management logic to any open trades for the strategy's symbol.

### `Orchestrator._is_strategy_active_for_regime(profile_key, strategy_instance, current_timestamp)`
**Purpose**: Determines if a strategy should be active by checking the current market regime against the strategy's configured primary regimes.
**Returns**: `bool` - `True` if the strategy is suitable for the current market regime.

### `Orchestrator._process_strategy_signal(profile_key, strategy_instance, current_timestamp)`
**Purpose**: Handles the generation, validation, and execution of a new trade signal from a strategy.

### `Orchestrator._manage_strategy_positions(profile_key, strategy_instance)`
**Purpose**: Applies position management logic (e.g., trailing stops, take profit adjustments) to all open positions associated with a strategy's symbol.

### `Orchestrator.stop()`
**Purpose**: Graceful shutdown of all components.
**Behavior**: Stops main loop, disconnects platforms, saves final state.

## MarketDataManager (Data Management)

### `MarketDataManager.get_market_data(symbol, timeframe, up_to_timestamp=None)`
**Purpose**: Retrieve OHLCV data with calculated indicators for strategy use.
**Args**:
- `symbol`: Trading instrument.
- `timeframe`: Timeframe enum.
- `up_to_timestamp`: Optional timestamp filter for backtesting (prevents look-ahead bias).
**Returns**: `pd.DataFrame` - OHLCV data with all required indicators for the given symbol/timeframe.

### `MarketDataManager.ensure_data_subscription(symbol, timeframe)`
**Purpose**: Dynamically subscribe to tick and bar data for a symbol/timeframe pair.
**Behavior**: Sets up real-time data feeds via the platform adapter and fetches initial history. This is the primary method for activating data for a new instrument.

### `MarketDataManager._calculate_and_store_indicators(symbol, timeframe, df)`
**Purpose**: Orchestrates the calculation of all required indicators for a given symbol and timeframe.
**Process**:
1.  Calls `_find_asset_profile_keys()` to find all strategies active on the symbol/timeframe.
2.  Calls `_aggregate_indicator_configs()` to gather and de-duplicate all indicator requirements from those strategies.
3.  Calls `_apply_indicators_to_dataframe()` to perform the calculations using `pandas-ta`.
**Returns**: `pd.DataFrame` - Original data with added indicator columns.

## MarketRegimeClassifier (Market Condition Detection)

### `MarketRegimeClassifier.__init__(adx_period=14, trending_threshold=25.0, ranging_threshold=20.0)`
**Purpose**: Initialize the regime classifier with ADX parameters.
**Args**:
- `adx_period`: The lookback period for the ADX calculation.
- `trending_threshold`: The ADX level above which the market is considered "Trending".
- `ranging_threshold`: The ADX level below which the market is considered "Ranging".

### `MarketRegimeClassifier.classify_market_regime(symbol, h4_data)`
**Purpose**: Classify the current market regime based on ADX analysis of H4 data.
**Args**:
- `symbol`: Trading instrument for logging purposes.
- `h4_data`: DataFrame of H4 OHLCV data.
**Returns**: `str` - Market regime classification ("Trending", "Ranging", or "Ambiguous").
**Analysis**: Uses the `pandas-ta` library to calculate the ADX value and compares it against the configured thresholds.

## BaseStrategy (Abstract Strategy Interface)

### `BaseStrategy.__init__(symbol, timeframe, strategy_params, logger)`
**Purpose**: Initialize strategy with configuration and logging.
**Args**:
- `symbol`: Trading instrument (e.g., "EURUSD").
- `timeframe`: Timeframe enum (e.g., Timeframe.M15).
- `strategy_params`: Strategy-specific parameters dict.
- `logger`: Logger instance for this strategy.

### `BaseStrategy.generate_signal(market_data_df, active_position, latest_tick)`
**Purpose**: Core signal generation method (MUST be implemented by all strategies).
**Args**:
- `market_data_df`: DataFrame with OHLCV data and indicators.
- `active_position`: Current open position or None.
- `latest_tick`: Latest price data.
**Returns**: `Optional[Dict[str, Any]]` - Signal details or None.

**Enhanced Signal Generation with Multi-Filter Validation**:
Modern strategy implementations use layered filtering for signal quality:
1. **Primary Indicator Signal**: Base signal from main indicator (e.g., RSI crossing)
2. **Confirmation Filters**: Additional filters for signal validation (e.g., Bollinger Bands)
3. **Risk Filters**: Volatility and trend alignment checks

**Example - MeanReversionRSI with Bollinger Band Filtering**:
```python
# Primary RSI signal detection
rsi_buy_signal = (previous_rsi < oversold_level) and (current_rsi >= oversold_level)

# Bollinger Band confirmation filter
bb_confirmation = current_close <= lower_bollinger_band

# Combined signal validation
if rsi_buy_signal and bb_confirmation:
    return self._create_buy_signal(current_close, atr_value)
else:
    # Log filter blocking for debugging
    self.logger.info(f"Signal blocked by BB filter: price {current_close} > lower BB {lower_bollinger_band}")
    return None
```

### `BaseStrategy.manage_open_position(position, latest_bar, latest_tick)`
**Purpose**: Apply management logic to an existing open position (e.g., trailing stop loss).
**Args**:
- `position`: The `Position` object to manage.
- `latest_bar`: The most recent `OHLCVData` bar.
- `latest_tick`: The most recent `TickData`.
**Returns**: `Optional[Dict[str, Any]]` - A modification signal (e.g., `MODIFY_SLTP`) or `None`.

## RiskController (Risk Management Gatekeeper)

### `RiskController.validate_trade_proposal(symbol, action, strategy_type_name, stop_loss_pips, asset_profile_key)`
**Purpose**: Main gatekeeper for FTMO compliance - validates all trade proposals.
**Returns**: `Tuple[bool, str, Optional[float]]` - (can_trade, reason, calculated_lot_size).
**Critical**: ALL trades must pass through this function.

### `RiskController.check_all_risk_rules(current_equity=None)`
**Purpose**: Global risk state validation.
**Returns**: `Tuple[bool, str]` - (can_continue_trading, reason_if_halted).
**Checks**: Daily reset, drawdown limits, position limits, operational compliance.

## OrderExecutionManager (Trade Execution)

### `OrderExecutionManager.execute_trade_signal(trade_signal_details, calculated_lot_size, symbol_info, asset_profile_key)`
**Purpose**: Execute approved trade signals from strategies.
**Returns**: `Optional[Order]` - Placed order or None if failed.

---

**Usage Notes**:
- All functions follow the error handling patterns defined in CODING_STANDARDS.md.
- Type hints and docstrings are mandatory for all public methods.
- Integration between components follows the data flow patterns in ARCHITECTURE.md.

  
