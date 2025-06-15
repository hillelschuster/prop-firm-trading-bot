# Prop Firm Trading Bot - Architecture Documentation

This document defines the core architectural principles, patterns, and constraints for the prop firm trading bot. All development work must adhere to these guidelines to maintain system integrity, compliance, and extensibility.

## Core Design Principles

### Separation of Concerns
- Trading logic (strategies/) is pure signal generation with multi-layered filtering.
- Risk management (risk_controller/) handles all compliance.
- Platform interaction (api_connector/) abstracts broker APIs.
- Orchestration (orchestrator.py) coordinates the main trading cycle but contains no business logic itself.

### Data Flow Immutability
- Market data flows unidirectionally: Platform → MarketDataManager → Strategy.
- Signal flow is: Strategy → RiskController → OrderExecutionManager → Platform.
- No component should bypass the risk controller for trade execution.

### Error Handling Philosophy
- Fail fast on configuration errors.
- Graceful degradation on market data issues (e.g., classifier failure).
- Never trade when risk rules are violated.

## Account Type Specification
**Target Account Type**: Swing Trading Accounts
This system is designed and optimized for prop firm "Swing" trading accounts (e.g., FTMO Swing, MyForexFunds Swing) that permit overnight and weekend position holding. The system is **NOT compatible** with standard intraday-only accounts that require end-of-day position closure.

**Rationale**: Our MeanReversionRSI strategy requires up to 12 hours for trade maturation and uses time-based exits that extend beyond single trading sessions. The strategy employs Bollinger Band filtering to ensure trades are only executed at genuine price extremes, requiring patience for optimal entry conditions.

## Module Responsibilities

| Module | Primary Responsibility | What It Should NOT Do |
|---|---|---|
| orchestrator.py | Coordinate components, manage main trading cycle | Business logic, direct platform calls |
| strategies/ | Generate trading signals with multi-filter validation, manage open positions | Execute trades, manage risk |
| risk_controller/ | Validate trades, enforce limits | Generate signals, platform interaction |
| api_connector/ | Platform abstraction | Business logic, risk decisions |
| data_handler/ | Market data management and indicator calculation | Trading decisions, risk calculations |
| market_analysis/| Market regime classification | Signal generation, trade execution |

## Critical Data Flow Patterns

### Signal Generation Flow
1.  **Orchestrator** initiates the trading cycle.
2.  **MarketRegimeClassifier** analyzes H4 data to determine the current market regime.
3.  **Orchestrator** filters strategies, activating only those suitable for the current regime.
4.  **MarketDataManager** provides the active strategy with its required market data (OHLCV + indicators).
5.  **Strategy.generate_signal()** applies multi-layered filtering and returns a trade signal or `None`.
6.  **RiskController.validate_trade_proposal()** checks all compliance and risk rules.
7.  **OrderExecutionManager.execute_trade_signal()** places the approved order.
8.  **StateManager** persists the state periodically.

### Signal Quality Architecture
**Design Principle**: Prioritize signal quality over quantity through layered filtering.

**Multi-Filter Signal Validation**:
1. **Primary Signal Detection**: Base indicator crossing or pattern recognition
2. **Confirmation Filters**: Additional technical indicators for signal validation
3. **Market Context Filters**: Price position relative to key levels (e.g., Bollinger Bands)
4. **Risk Filters**: Volatility and trend alignment checks

**Example - MeanReversionRSI Implementation**:
- **Primary**: RSI crossing oversold/overbought thresholds
- **Confirmation**: Price must be at/beyond Bollinger Band extremes
- **Context**: Ensures trades only at genuine price extremes
- **Result**: Higher quality signals with lower frequency but better risk/reward

**Filter Logging Pattern**: All blocked signals are logged with specific filter reasons for strategy optimization and debugging.

### Risk Check Sequence (NEVER bypass)
1. Daily reset check (timezone-aware)
2. Drawdown limits (max + daily)
3. Position limits (per strategy + global)
4. **Portfolio-level risk validation (cross-strategy correlation, exposure limits)**
5. **Strategy allocation limits (per regime, per strategy type)**
6. News restrictions (if enabled)
7. Operational compliance (order frequency, etc.)

## Portfolio Management Architecture

**Reference**: This architecture implements the strategic pivot documented in ADR-014: Strategic Pivot to Portfolio-Based Trading Architecture.

### Design Principles
- **Market Regime Specialization**: Each strategy is optimized for specific market conditions (e.g., ranging, trending).
- **Dynamic Strategy Activation**: The Orchestrator, guided by the MarketRegimeClassifier, automatically activates or deactivates strategies based on real-time market conditions.
- **Risk Budget Distribution**: Total account risk is allocated across active strategies based on regime confidence and portfolio rules.
- **Correlation Management**: Monitor and limit cross-strategy correlation to prevent over-concentration.

### Multi-Strategy Coordination Patterns

#### Strategy Selection Flow
1.  **Market Regime Detection**: In each cycle, the Orchestrator uses the MarketRegimeClassifier to analyze current market conditions.
2.  **Strategy Filtering**: The Orchestrator filters the list of all strategies, allowing only those suitable for the detected regime to proceed.
3.  **Signal Generation**: Active strategies generate signals based on their logic.
4.  **Risk Allocation & Execution**: Signals are passed to the RiskController and OrderExecutionManager, which apply risk and execution logic.
5.  **Performance Monitoring**: Continuous tracking of strategy performance and regime accuracy.

#### Component Interactions
The `Orchestrator` sits at the center of the cycle. It first consults the `MarketRegimeClassifier` (which gets its data from the `MarketDataManager`). Based on the result, the `Orchestrator` then runs the trading cycle for each appropriate `Strategy`, which in turn gets its data from the `MarketDataManager` and sends signals through the `RiskController`.

## Market Regime Detection Architecture

### Regime Classification System
- **Ranging Markets**: Low volatility, mean-reverting price action (MeanReversionRSI specialist).
- **Trending Markets**: Sustained directional movement (TrendFollowingSMA specialist).
- **Ambiguous State**: A neutral zone between ranging and trending where no strategy is activated to manage risk during uncertain transitions.

### Detection Algorithms
- **Primary Indicator**: The Average Directional Index (ADX) is used to measure trend strength.
- **Implementation**: The calculation is performed using the optimized and validated ADX function from the `pandas-ta` library.
- **Analysis Timeframe**: H4 data is used for a stable, noise-filtered regime assessment.
- **Thresholds**: Configurable thresholds (e.g., ADX > 25 for trending, < 20 for ranging) determine the classification.

---
(The rest of the document remains unchanged as it is still accurate.)

### Configuration Architecture

#### Hierarchy (most specific wins)
1. Asset-strategy profile overrides
2. Strategy parameter files (.json)
3. Main config (main_config.yaml)
4. Default values in code

#### Configuration Loading Pattern
- ConfigManager loads all configs at startup
- Validates using Pydantic models
- Makes available through AppConfig object
- **Portfolio configurations loaded and validated for multi-strategy coordination**
- **Market regime mappings loaded for strategy-regime associations**
- No runtime config reloading (restart required)

### Extension Points

#### Adding New Strategies
1. Inherit from BaseStrategy
2. Implement generate_signal() and manage_open_position()
3. **Define market regime specialization in the strategy's JSON parameter file.**
4. Add strategy definition to main_config.yaml
5. Create parameter file in config/
6. Register in asset_strategy_profiles

### State Management

#### What Gets Persisted
- Open positions and their metadata
- Daily order counts and risk metrics
- Last processed timestamps
- Strategy-specific state (if any)

#### Persistence Rules
- State saved every N seconds (configurable)
- State loaded on startup
- Graceful handling of corrupted state files

### Error Handling Patterns

#### Configuration Errors (Fail Fast)
- Invalid config → immediate shutdown
- Missing required files → startup failure
- Bad strategy parameters → strategy disabled

#### Runtime Errors (Graceful Degradation)
- Platform connection lost → retry with backoff
- Market data missing → skip strategy cycle
- Single strategy error → continue with others

### Risk Violations (Hard Stops)
- Drawdown breach → emergency flatten + halt
- News restriction → reject trade
- Position limit → reject new positions

### Testing Strategy

#### Unit Test Boundaries
- Strategies: Mock market data, test signal logic
- RiskController: Mock account info, test rule enforcement
- Connectors: Mock platform APIs, test data conversion

#### Integration Test Scope
- Full trade cycle with paper trading adapter
- Configuration loading and validation
- State persistence and recovery

### Backtesting Architecture

#### Design Principles
- **Consistency**: Backtesting uses identical components as live trading (Orchestrator, RiskController, BaseStrategy).
- **Data Replay**: Historical data is replayed bar-by-bar via the `Orchestrator.process_single_bar()` method.
- **Component Reuse**: No modification of core trading logic; only the data source and main loop change.
- **Portfolio Support**: Multi-strategy backtesting with regime detection and dynamic activation.

#### Backtesting Data Flow
1. **Multi-Timeframe Data Loading**: The backtesting engine loads all necessary timeframes (e.g., M15 for a strategy, H4 for the classifier) into the PaperTradingAdapter.
2. **Component Integration**: Orchestrator is initialized with the PaperTradingAdapter.
3. **Chronological Processing**: The backtest engine calls `Orchestrator.process_single_bar()` for each step in the primary timeframe.
4. **Progressive Data Windows**: `MarketDataManager` ensures that components only receive data up to the current backtest timestamp, preventing look-ahead bias.
5. **Regime Classification**: `MarketRegimeClassifier` accesses H4 data during the backtest to perform its analysis, just as it would in live trading.

### Prop Firm Compliance Architecture

#### FTMO-Specific Rules (Hard-coded)
- Max drawdown: 10% (we use 9% buffer)
- Daily drawdown: 5% (we use 4.5% buffer)
- News trading: ±2 minutes restriction
- Weekend closure: mandatory for non-swing accounts

#### Extensibility for Other Prop Firms
- Risk rules configurable in main_config.yaml
- Platform-specific order limits
- Timezone-aware daily resets
- Audit trail through structured logging

---

**Note**: This architecture document is the definitive reference for all development work. Any changes to these patterns or principles must be discussed and documented here first.

  
