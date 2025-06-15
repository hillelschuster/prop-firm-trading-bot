# Prop Firm Trading Bot - Architectural Decisions

This document records all major architectural decisions made during the development of the prop firm trading bot. Each decision includes the context, options considered, decision made, and rationale. This serves as a historical record and guide for future architectural choices.

## Decision Format

Each decision follows this structure:
- **Decision ID**: Unique identifier (e.g., AD-001)
- **Date**: When the decision was made
- **Status**: Proposed | Accepted | Superseded | Deprecated
- **Context**: The situation that required a decision
- **Options Considered**: Alternative approaches evaluated
- **Decision**: What was chosen
- **Rationale**: Why this option was selected
- **Consequences**: Positive and negative outcomes
- **Related Decisions**: Links to other relevant decisions

---

## AD-001: Modular Architecture with Clear Separation of Concerns

**Date**: 2024-12-07
**Status**: Accepted

**Context**: Need to design a trading bot that can support multiple strategies, platforms, and compliance requirements while remaining maintainable and testable.

**Options Considered**:
1. Monolithic design with all logic in a single module
2. Modular design with clear component boundaries
3. Microservices architecture with separate processes

**Decision**: Modular design with clear component boundaries within a single process.

**Rationale**:
- **Maintainability**: Each module has a single responsibility
- **Testability**: Components can be tested in isolation with mocks
- **Extensibility**: New strategies and platforms can be added without affecting existing code
- **Performance**: Single process avoids inter-process communication overhead
- **Complexity**: Simpler than microservices for a single-user trading bot

**Consequences**:
- ✅ Easy to add new strategies and platforms
- ✅ Clear boundaries prevent scope creep
- ✅ Excellent testability with dependency injection
- ❌ Requires discipline to maintain boundaries
- ❌ Single point of failure (but acceptable for personal trading bot)

**Related Decisions**: AD-002, AD-003

---

## AD-002: Abstract Platform Interface for Multi-Broker Support

**Date**: 2024-12-07
**Status**: Accepted

**Context**: Need to support multiple trading platforms (MT5, cTrader) and paper trading without coupling business logic to specific platform APIs.

**Options Considered**:
1. Direct platform API calls throughout the codebase
2. Abstract interface with platform-specific adapters
3. Generic trading library wrapper

**Decision**: Abstract PlatformInterface with platform-specific adapters.

**Rationale**:
- **Platform Independence**: Business logic doesn't depend on specific platform APIs
- **Testing**: Paper trading adapter enables comprehensive backtesting
- **Future-Proofing**: Easy to add new platforms without changing core logic
- **Consistency**: Standardized data models across all platforms
- **Control**: Full control over platform interactions and error handling

**Consequences**:
- ✅ Platform-agnostic business logic
- ✅ Excellent testing capabilities with paper trading
- ✅ Easy to add new platforms
- ✅ Consistent error handling across platforms
- ❌ Additional abstraction layer adds complexity
- ❌ Platform-specific features may be harder to access

**Related Decisions**: AD-001, AD-007

---

## AD-003: Centralized Risk Controller as Trading Gatekeeper

**Date**: 2024-12-07
**Status**: Accepted

**Context**: Prop firm compliance requires strict adherence to risk rules. Need to ensure no trade can bypass risk validation.

**Options Considered**:
1. Risk checks distributed across strategies and execution components
2. Centralized risk controller that all trades must pass through
3. Risk validation at the platform adapter level

**Decision**: Centralized RiskController that validates all trade proposals before execution.

**Rationale**:
- **Compliance Guarantee**: Impossible to bypass risk checks
- **Consistency**: All risk rules applied uniformly across strategies
- **Auditability**: Single point for logging all risk decisions
- **Maintainability**: Risk rule changes only need to be made in one place
- **FTMO Requirements**: Strict adherence to prop firm rules is critical

**Consequences**:
- ✅ Guaranteed compliance with prop firm rules
- ✅ Consistent risk application across all strategies
- ✅ Clear audit trail for all trading decisions
- ✅ Easy to modify risk rules in one location
- ❌ Single point of failure for risk validation
- ❌ Potential performance bottleneck (but acceptable for trading frequency)

**Related Decisions**: AD-001, AD-008

---

## AD-004: Pydantic Models for All Data Structures

**Date**: 2024-12-07
**Status**: Accepted

**Context**: Need consistent data validation and serialization across all components, especially for configuration and inter-module communication.

**Options Considered**:
1. Plain Python dictionaries and classes
2. Pydantic models with validation
3. Dataclasses with manual validation
4. Protocol/TypedDict for type hints only

**Decision**: Pydantic models for all structured data in `src/core/models.py`.

**Rationale**:
- **Validation**: Automatic data validation with clear error messages
- **Serialization**: Built-in JSON serialization for logging and state persistence
- **Type Safety**: Runtime type checking prevents data corruption
- **Documentation**: Self-documenting with clear field definitions
- **IDE Support**: Excellent autocomplete and type checking

**Consequences**:
- ✅ Automatic data validation prevents runtime errors
- ✅ Consistent data structures across all modules
- ✅ Excellent IDE support and type checking
- ✅ Easy serialization for logging and persistence
- ❌ Slight performance overhead for validation
- ❌ Learning curve for developers unfamiliar with Pydantic

**Related Decisions**: AD-001, AD-009

---

## AD-005: Strategy Pattern with Abstract Base Class

**Date**: 2024-12-07
**Status**: Accepted

**Context**: Need to support multiple trading strategies while ensuring consistent interface and behavior.

**Options Considered**:
1. Strategy functions with common signature
2. Strategy classes with abstract base class
3. Plugin system with dynamic loading
4. Configuration-driven strategy engine

**Decision**: Strategy classes inheriting from BaseStrategy abstract class.

**Rationale**:
- **Consistency**: All strategies implement the same interface
- **Encapsulation**: Strategy state and logic are properly encapsulated
- **Extensibility**: Easy to add new strategies following established pattern
- **Type Safety**: Abstract methods ensure required methods are implemented
- **Testing**: Each strategy can be tested independently

**Consequences**:
- ✅ Consistent strategy interface across all implementations
- ✅ Easy to add new strategies following clear pattern
- ✅ Excellent testability with isolated strategy logic
- ✅ Type safety ensures proper implementation
- ❌ More verbose than simple functions
- ❌ Requires understanding of inheritance patterns

**Related Decisions**: AD-001, AD-004

---

## AD-006: YAML Configuration with JSON Strategy Parameters

**Date**: 2024-12-07
**Status**: Accepted

**Context**: Need flexible configuration system that supports main bot settings, strategy-specific parameters, and instrument details.

**Options Considered**:
1. Single YAML file for all configuration
2. Multiple JSON files for different configuration types
3. YAML for main config, JSON for strategy parameters
4. Environment variables for all configuration
5. Database-driven configuration

**Decision**: YAML for main configuration (`main_config.yaml`) with separate JSON files for strategy parameters.

**Rationale**:
- **Readability**: YAML is human-readable for main configuration
- **Flexibility**: JSON strategy files allow per-strategy customization
- **Version Control**: Text files work well with git
- **Validation**: Pydantic can validate both YAML and JSON
- **Separation**: Different configuration types are logically separated

**Consequences**:
- ✅ Human-readable main configuration
- ✅ Flexible strategy parameter management
- ✅ Good version control integration
- ✅ Clear separation of configuration concerns
- ❌ Multiple configuration files to manage
- ❌ Potential for configuration drift between files

**Related Decisions**: AD-004, AD-007

---

## AD-007: Environment Variables for Sensitive Credentials

**Date**: 2024-12-07
**Status**: Accepted

**Context**: Trading bot requires sensitive credentials (account numbers, passwords, API keys) that must never be committed to version control.

**Options Considered**:
1. Credentials in configuration files
2. Environment variables for all credentials
3. Encrypted credential files
4. External secret management service
5. Interactive credential prompts

**Decision**: Environment variables for all sensitive credentials with clear naming convention.

**Rationale**:
- **Security**: Credentials never stored in version control
- **Simplicity**: Standard approach for credential management
- **Deployment**: Easy to configure in different environments
- **Auditing**: Clear separation between public config and secrets
- **Convention**: Follows 12-factor app principles

**Consequences**:
- ✅ Credentials never accidentally committed to version control
- ✅ Easy to configure different environments
- ✅ Standard security practice
- ✅ Clear separation of public and private configuration
- ❌ Requires environment setup for each deployment
- ❌ Potential for missing environment variables at runtime

**Related Decisions**: AD-006, AD-002

---

## AD-008: Timezone-Aware Daily Reset for FTMO Compliance

**Date**: 2024-12-07
**Status**: Accepted

**Context**: FTMO daily drawdown limits reset at midnight in Prague timezone (CET/CEST), not UTC or local time.

**Options Considered**:
1. UTC-based daily reset
2. Local system timezone reset
3. Configurable timezone with FTMO default
4. Manual daily reset trigger

**Decision**: Timezone-aware daily reset using configurable timezone (default: Europe/Prague).

**Rationale**:
- **FTMO Compliance**: Matches FTMO's actual reset schedule
- **Accuracy**: Prevents incorrect risk calculations due to timezone differences
- **Flexibility**: Configurable for other prop firms with different timezones
- **Reliability**: Automatic reset without manual intervention

**Consequences**:
- ✅ Accurate compliance with FTMO daily reset schedule
- ✅ Flexible for other prop firms
- ✅ Automatic operation without manual intervention
- ✅ Prevents timezone-related compliance violations
- ❌ Complexity of timezone handling
- ❌ Potential issues with daylight saving time transitions

**Related Decisions**: AD-003, AD-009

---

## AD-009: Structured JSON Logging for Compliance Auditing

**Date**: 2024-12-07
**Status**: Accepted

**Context**: Prop firms may require detailed audit trails of all trading decisions and risk checks for compliance verification.

**Options Considered**:
1. Simple text logging
2. Structured JSON logging
3. Database logging
4. External logging service
5. No detailed logging

**Decision**: Structured JSON logging with configurable log rotation and retention.

**Rationale**:
- **Auditability**: Machine-readable logs for compliance analysis
- **Searchability**: Easy to query and filter log data
- **Structure**: Consistent log format across all components
- **Compliance**: Detailed record of all risk decisions and trade actions
- **Debugging**: Rich context for troubleshooting issues

**Consequences**:
- ✅ Excellent audit trail for compliance verification
- ✅ Machine-readable logs for analysis
- ✅ Consistent logging format across all components
- ✅ Rich debugging information
- ❌ Larger log files than simple text logging
- ❌ Requires log analysis tools for human reading

**Related Decisions**: AD-003, AD-004

---

## AD-010: Single-Threaded Main Loop with Background Data Polling

**Date**: 2024-12-07
**Status**: Accepted

**Context**: Need to balance simplicity with real-time data requirements for trading decisions.

**Options Considered**:
1. Fully single-threaded with blocking data requests
2. Multi-threaded with complex synchronization
3. Async/await event-driven architecture
4. Single main thread with background data polling thread

**Decision**: Single-threaded main trading loop with background thread for data polling only.

**Rationale**:
- **Simplicity**: Main trading logic remains single-threaded and predictable
- **Real-time Data**: Background polling ensures fresh market data
- **Debugging**: Easier to debug single-threaded trading logic
- **Safety**: No shared mutable state between threads
- **Performance**: Adequate for typical trading frequencies

**Consequences**:
- ✅ Simple, predictable main trading logic
- ✅ Real-time market data availability
- ✅ Easy debugging and testing
- ✅ No complex thread synchronization
- ❌ Limited scalability for high-frequency trading
- ❌ Background thread adds some complexity

**Related Decisions**: AD-001, AD-002

---

## AD-011: Backtesting Engine Integration with Existing Architecture

**Date**: 2024-12-08
**Status**: Accepted

**Context**: Need comprehensive backtesting capabilities that maintain consistency with live trading behavior while reusing existing components (Orchestrator, RiskController, BaseStrategy) without modification.

**Decision**: Implement BacktestEngine that integrates with existing architecture through:
- Enhanced PaperTradingAdapter with historical data replay capability
- Orchestrator reuse with platform adapter override pattern
- CSV data source abstraction for flexible historical data loading
- Pydantic models for backtest configuration and results

**Rationale**:
- Ensures backtesting behavior matches live trading exactly
- Validates strategy logic before live deployment
- Reuses existing risk management and execution logic
- Maintains architectural consistency and separation of concerns

**Consequences**:
- ✅ Consistent behavior between backtesting and live trading
- ✅ Comprehensive strategy validation before live deployment
- ✅ Reuses existing components without modification
- ✅ Flexible data source integration (CSV, future database support)
- ❌ Additional complexity in PaperTradingAdapter
- ❌ Historical data management requirements

**Related Decisions**: AD-001, AD-002, AD-012

---

## AD-012: Enhanced Performance Reporting with Comprehensive Metrics

**Date**: 2024-12-08
**Status**: Accepted

**Context**: Need detailed performance analysis for backtesting results and live trading monitoring, including industry-standard metrics for strategy evaluation and compliance reporting.

**Decision**: Implement enhanced PerformanceReporter with:
- Comprehensive trading metrics (profit factor, Sharpe ratio, drawdown analysis)
- Trade-by-trade logging and analysis
- Equity curve analysis and risk metrics
- Support for both trade history and equity history data sources
- Flexible output formats (string, dictionary, future JSON/CSV export)

**Rationale**:
- Enables thorough strategy performance evaluation
- Provides compliance-ready reporting for prop firm requirements
- Supports both backtesting and live trading analysis
- Industry-standard metrics for strategy comparison

**Consequences**:
- ✅ Comprehensive performance analysis capabilities
- ✅ Industry-standard metrics for strategy evaluation
- ✅ Detailed trade logging for compliance and debugging
- ✅ Flexible reporting formats for different use cases
- ❌ Increased computational overhead for metric calculations
- ❌ Additional memory usage for equity history tracking

**Related Decisions**: AD-009, AD-011

---

## AD-013: Progressive Data Windows for Backtesting Look-Ahead Prevention

**Date**: 2024-12-08
**Status**: Accepted

**Context**: Backtesting strategies were receiving complete historical datasets instead of progressive data windows, allowing look-ahead bias and preventing accurate simulation of real-time trading conditions.

**Options Considered**:
1. Modify strategy logic to handle time-aware data filtering
2. Create separate backtesting-specific strategy implementations
3. Implement progressive data windows in MarketDataManager
4. Pre-process data into time-sliced files for each backtest iteration

**Decision**: Implement progressive data windows in MarketDataManager with `up_to_timestamp` parameter.

**Rationale**:
- **Look-Ahead Prevention**: Strategies can only access data available at current timestamp
- **Architecture Consistency**: No changes required to existing strategy implementations
- **Real-Time Simulation**: Accurately simulates chronological data availability
- **Component Reuse**: Same MarketDataManager serves both live and backtest modes
- **Flexibility**: Timestamp filtering can be enabled/disabled per call

**Consequences**:
- ✅ Eliminates look-ahead bias in backtesting
- ✅ Maintains consistency between live and backtest strategy behavior
- ✅ No modification required to existing strategy implementations
- ✅ Accurate simulation of real-time data availability
- ❌ Additional complexity in MarketDataManager data filtering logic
- ❌ Slight performance overhead for timestamp filtering during backtesting

**Related Decisions**: AD-011, AD-002

---

## Decision Guidelines for Future Changes

### When to Create a New Decision Record
- Changes to core architectural patterns
- Technology stack modifications
- New compliance requirements
- Performance or scalability changes
- Security model updates

### Decision Review Process
1. Document the decision using the standard format
2. Consider impact on existing decisions
3. Update related documentation (ARCHITECTURE.md, CODING_STANDARDS.md)
4. Communicate changes to all developers
5. Update implementation to match decision

### Superseding Previous Decisions
- Mark old decision as "Superseded"
- Reference the new decision ID
- Explain why the change was necessary
- Document migration path if applicable

---

## AD-014: Strategic Pivot to Portfolio-Based Trading Architecture

**Date**: 2025-06-09
**Status**: Accepted
**Supersedes**: None
**Related**: AD-001 (Documentation-First Workflow)

### Context

Systematic optimization analysis (V1-V7) of the MeanReversionRSI strategy revealed:
- V3 configuration optimal for ranging markets (2 trades, +$25.60 P&L, 50% win rate on May 2023)
- Full-year 2023 backtest showed market regime dependency (4 trades, -$19.91 P&L, 0% win rate)
- Strategy correctly remains flat during unfavorable conditions but lacks consistent profitability
- Single-strategy approach insufficient for diverse market regimes throughout 2023

### Decision

1. **Portfolio Architecture Adoption**: Transition from single-strategy optimization to multi-strategy portfolio system
2. **Market Regime Specialization**: Designate MeanReversionRSI V3 as "Ranging Market Specialist" (`strategy_rsi_ranging_market_v1.json`)
3. **Complementary Strategy Development**: Prioritize TrendFollowingSMA strategy for trending market conditions
4. **Future Market Regime Classification**: Plan for intelligent strategy allocation based on market conditions

### Rationale

- Portfolio diversification reduces single-strategy risk and market regime dependency
- Specialist strategies outperform generalist approaches in specific conditions
- Maintains conservative risk management while improving market coverage
- Aligns with prop firm requirements for consistent performance across market cycles

### Consequences

- **Architecture**: Multi-strategy system requires enhanced Orchestrator capabilities
- **Risk Management**: Portfolio-level risk allocation and correlation management needed
- **Configuration**: Strategy-specific parameter files with clear naming conventions established
- **Development**: Focus shifts from single-strategy optimization to portfolio construction
- **Testing**: Requires multi-strategy backtesting and regime-specific validation protocols

### Implementation Status

- ✅ MeanReversionRSI V3 validated and renamed to production configuration
- ✅ Obsolete optimization iterations (V2, V4-V7) removed from configuration
- 🔄 TrendFollowingSMA strategy development in progress
- 📋 Market regime classifier design pending

---

---

## AD-015: Market Regime Classifier Design and Implementation Strategy

**Date**: 2025-06-09
**Status**: Accepted
**Supersedes**: None
**Related**: AD-007 (Portfolio Architecture), AD-009 (Strategy Coordination), AD-014 (Portfolio-Based Trading Architecture)

### **Context**

Phase 7 portfolio integration testing of our two-strategy portfolio (MeanReversionRSI on M15 + TrendFollowingSMA on H1) revealed critical limitations in our current approach:

- Both strategies were simultaneously active during the same volatile market period (December 2023)
- Portfolio performance was purely additive (-$10.06) with no diversification benefits
- No mechanism existed to selectively deploy strategies based on market suitability
- Risk concentration occurred when both strategies operated in unsuitable market conditions

This analysis confirmed the need for intelligent strategy activation based on market regime classification to achieve true portfolio diversification and performance optimization.

### **Decision**

We will implement a `MarketRegimeClassifier` component with the following specifications:

**Core Responsibility:**
- Analyze current market conditions and provide discrete market regime classification
- Enable selective strategy activation based on market suitability assessment

**Initial Implementation Design:**
- **Primary Classification**: Binary market state detection ('Trending' vs 'Ranging')
- **Technical Indicator**: Average Directional Index (ADX) with 14-period calculation
- **Analysis Timeframe**: H4 timeframe for stable, noise-filtered regime assessment
- **Classification Rules**:
  - ADX(14) > 25: Market state = **'Trending'** → Activate TrendFollowingSMA strategy
  - ADX(14) < 20: Market state = **'Ranging'** → Activate MeanReversionRSI strategy
  - ADX(14) between 20-25: **'Ambiguous'** state → No strategy activation (risk management)

**Integration Architecture:**
- Component will be called by the Orchestrator during each trading cycle
- Output will determine which strategies receive market data and can generate signals
- Maintains separation of concerns: classification logic isolated from strategy logic

### **Rationale**

**Technical Justification:**
- ADX provides non-directional trend strength measurement, avoiding bias toward bullish/bearish conditions
- H4 timeframe filters intraday noise while remaining responsive to genuine regime changes
- Binary classification with ambiguous zone prevents false signals during transitional periods

**Portfolio Optimization Benefits:**
- Addresses core issue identified in Phase 7 testing: simultaneous strategy activation in unsuitable conditions
- Enables true diversification by ensuring strategies operate only in optimal market regimes
- Reduces portfolio drawdown risk through selective strategy deployment

**Scalability Considerations:**
- Framework supports future expansion to additional regime types (e.g., 'Volatile', 'Consolidating')
- Component design allows for alternative classification algorithms without architectural changes

### **Consequences**

**Implementation Requirements:**
- New component: `src/market_analysis/market_regime_classifier.py`
- Data dependency: H4 timeframe data access for all traded instruments
- Orchestrator refactoring: Integration of regime classification into strategy selection logic

**Testing and Validation:**
- Unit tests for ADX calculation accuracy and classification rule logic
- Integration tests for Orchestrator-Classifier interaction
- Backtesting validation using Phase 7 dataset to measure diversification improvement

**Performance Impact:**
- Additional computational overhead: ADX calculation on H4 timeframe
- Data storage requirement: H4 historical data for regime analysis
- Latency consideration: Classification must complete within trading cycle timing constraints

**Future Evolution Path:**
- Foundation for advanced regime detection (machine learning, multi-indicator fusion)
- Enables dynamic strategy allocation based on regime confidence levels
- Supports portfolio optimization through regime-aware position sizing

---

## AD-016: Multi-Timeframe Backtesting Engine Architecture

**Date**: 2025-01-27
**Status**: Accepted
**Supersedes**: None
**Related**: AD-011 (Backtesting Engine Integration), AD-015 (Market Regime Classifier Design)

### **Context**

Phase 8 testing revealed a critical limitation in our backtesting engine: it only supports single-timeframe data loading per strategy. This prevents the MarketRegimeClassifier from accessing required H4 data during backtesting, forcing us to bypass regime classification entirely. This limitation undermines our core portfolio architecture that depends on intelligent strategy activation based on market regime detection.

**Current Limitation:**
- BacktestEngine loads only the strategy's primary timeframe (e.g., M15 for MeanReversionRSI)
- MarketRegimeClassifier requires H4 data for ADX-based regime classification
- Orchestrator detects backtesting mode and skips regime classification as a workaround
- This prevents validation of our core portfolio hypothesis during backtesting

### **Decision**

We will implement **Multi-Timeframe Backtesting Architecture** with the following specifications:

**Core Enhancement:**
1. **Multi-Timeframe Data Loading**: BacktestEngine will load multiple timeframes simultaneously
2. **Timeframe Registry**: Automatic detection of required timeframes from strategy configurations and system components
3. **Synchronized Data Replay**: Maintain proper chronological progression across all timeframes
4. **Enhanced PaperTradingAdapter**: Support multiple timeframe datasets with independent bar indexing

**Implementation Strategy:**
1. **Required Timeframes Detection**:
   - Strategy primary timeframes (from strategy parameters)
   - H4 timeframe (for MarketRegimeClassifier)
   - Additional timeframes (for future multi-timeframe strategies)

2. **Data Source Enhancement**:
   - CSVDataSource supports multiple timeframe files or multi-timeframe CSV format
   - Data validation across timeframes for consistency
   - Proper timestamp alignment and gap detection

3. **Backtesting Data Flow**:
   - Load all required timeframes during initialization
   - Maintain separate bar indices per timeframe
   - Provide timeframe-specific data access with proper `up_to_timestamp` filtering
   - Enable MarketRegimeClassifier to access H4 data during backtesting

### **Rationale**

**Technical Justification:**
- Enables full validation of portfolio architecture during backtesting
- Maintains consistency between live trading and backtesting behavior
- Supports future multi-timeframe strategies without architectural changes
- Preserves existing single-strategy backtesting functionality

**Portfolio Validation Benefits:**
- Allows testing of MarketRegimeClassifier accuracy in historical data
- Enables validation of regime-based strategy activation logic
- Provides comprehensive portfolio performance analysis with regime context
- Supports optimization of regime classification parameters

**Scalability Considerations:**
- Framework supports unlimited timeframes without core architecture changes
- Memory-efficient data management with configurable retention limits
- Extensible to database-backed data sources in future phases

### **Consequences**

**Implementation Requirements:**
- Enhanced BacktestEngine with multi-timeframe data loading
- Updated PaperTradingAdapter with timeframe-specific data management
- Modified CSVDataSource for multi-timeframe support
- MarketDataManager enhancements for backtesting multi-timeframe access

**Data Management:**
- Increased memory usage for multiple timeframe datasets
- Additional data validation complexity across timeframes
- Enhanced CSV data preparation requirements for backtesting

**Testing and Validation:**
- Comprehensive multi-timeframe data consistency validation
- Integration testing for MarketRegimeClassifier in backtesting mode
- Performance testing with large multi-timeframe datasets
- Validation of regime classification accuracy using historical data

**Performance Impact:**
- Moderate increase in backtesting initialization time (data loading)
- Minimal impact on backtesting execution speed (data already loaded)
- Memory usage scales linearly with number of timeframes
- Enhanced debugging capabilities with multi-timeframe context

**Future Evolution Path:**
- Foundation for advanced multi-timeframe strategies
- Supports real-time multi-timeframe analysis in live trading
- Enables sophisticated regime detection using multiple timeframe confluence
- Facilitates portfolio optimization across different time horizons

---

**Note**: This document should be updated whenever significant architectural decisions are made. All decisions should be discussed and agreed upon before implementation begins.
---

## AD-017: MarketRegimeClassifier Prototype Validation and Code Quality Refinement

**Date**: 2025-06-11
**Status**: Accepted
**Related**: AD-015 (Market Regime Classifier Design)

### Context
The MarketRegimeClassifier portfolio prototype is functionally complete and has been validated. A dedicated refactoring phase was initiated to enhance the quality of newly introduced code without altering its existing functionality. The focus was on the core components of the new portfolio system.

### Decision
A formal refactoring initiative was undertaken for the following files:
*   `src/orchestrator.py`
*   `src/data_handler/market_data_manager.py`
*   `src/market_analysis/market_regime_classifier.py`

The refactoring focused on four key areas:
1.  **Applying the DRY Principle**: Redundant code was identified and consolidated into reusable functions and methods. For example, the trading cycle logic in `Orchestrator` was unified, and the manual ADX calculation in `MarketRegimeClassifier` was replaced with a call to the `pandas-ta` library.
2.  **Adhering to the Single Responsibility Principle (SRP)**: Large, complex methods were decomposed into smaller, more focused components. For instance, the `_run_trading_cycle_for_strategy` method in `Orchestrator` was broken down into distinct methods for regime checking, signal processing, and position management.
3.  **Enhancing Clarity**: Variable, function, and method names were improved to be more descriptive and compliant with PEP 8. Code structure was simplified to improve readability.
4.  **Adding Strategic Comments**: Concise comments were added to explain the "why" behind complex logic or critical decision points, such as fallback behaviors.

### Consequences
- **Improved Code Quality**: The codebase is now more maintainable, readable, and easier to debug.
- **Better Adherence to Best Practices**: The code now more closely follows established software engineering principles like DRY and SRP.
- **No Functional Change**: The external behavior and functionality of the trading bot remain unchanged, as was the requirement.
- **Easier Future Development**: The cleaner, more modular code will facilitate future enhancements and collaboration.

---

## AD-018: Target Account Specification - Swing Trading Account Selection
**Date**: 2025-06-11
**Status**: Accepted
**Decision Maker**: Strategy Development Team

### Context
Our MeanReversionRSI strategy analysis revealed fundamental incompatibility with standard prop firm account rules:
- Strategy uses `max_position_age_bars: 48` (12 hours on M15 timeframe)
- Mean reversion trades require time to mature and reach take-profit targets
- Standard accounts (e.g., FTMO Standard) enforce "no overnight holding" rules requiring all positions closed before market close
- Our validated champion configuration (RSI 35/65 + TP 2.5x + 48-bar time limit) would be forced to close profitable positions prematurely

### Decision
**The project will officially target "Swing" style prop firm accounts** (e.g., FTMO Swing, MyForexFunds Swing) that permit:
- Overnight position holding
- Weekend position holding
- Extended trade duration up to several days
- No forced end-of-day position closure

All current and future strategy development, backtesting, and validation will be conducted under swing trading account constraints and rules.

### Consequences
**Positive**:
- `max_position_age_bars` parameter and time-based exits remain valid and useful
- No need to implement forced end-of-day exit mechanisms
- Strategy can fully utilize mean reversion cycles that span multiple sessions
- Aligns with our validated 12-hour position holding requirements

**Negative**:
- Excludes potential users with standard intraday-only accounts
- May require higher account minimums (swing accounts often have higher entry requirements)
- Weekend gap risk becomes a consideration for risk management

**Future Considerations**:
- A separate project branch may be developed for pure intraday strategies targeting standard accounts
- All risk management and drawdown calculations will be performed under swing trading assumptions
- Documentation and user guidance will specify swing account requirements


