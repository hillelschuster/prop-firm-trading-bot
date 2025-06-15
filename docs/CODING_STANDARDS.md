# Prop Firm Trading Bot - Coding Standards

This document defines the coding standards, quality requirements, and best practices for the prop firm trading bot project. All code must adhere to these standards to ensure consistency, maintainability, and compliance.

## Python Code Style

### Type Hints (Mandatory)
- All public methods and functions MUST have complete type hints
- Use `from typing import TYPE_CHECKING` for circular dependencies
- Example:
```python
def validate_trade_proposal(
    self,
    symbol: str,
    action: OrderAction,
    strategy_type_name: str
) -> Tuple[bool, str, Optional[float]]:
```

### Import Organization
- Standard library imports first
- Third-party imports second
- Local imports last
- Use absolute imports: `from src.core.models import Order`

```python
# Standard library
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

# Third-party
import pandas as pd
import numpy as np
from pydantic import BaseModel

# Local imports
from src.core.models import Order, Position
from src.core.enums import OrderType, StrategySignal
```

## Documentation Standards

### Docstring Format (Required for all public methods)
```python
def calculate_position_size(self, symbol: str, risk_pct: float, sl_pips: float) -> float:
    """
    Calculates position size based on risk percentage and stop loss distance.

    This is a critical risk management function that ensures position sizing
    adheres to prop firm requirements and account risk limits.

    Args:
        symbol: Trading instrument (e.g., "EURUSD")
        risk_pct: Risk as percentage of account (e.g., 0.01 for 1%)
        sl_pips: Stop loss distance in pips

    Returns:
        Position size in lots

    Raises:
        ValueError: If symbol not found or invalid risk parameters

    Example:
        >>> controller.calculate_position_size("EURUSD", 0.01, 20.0)
        0.05
    """
```

### Inline Comments (When Required)
- Complex business logic (especially prop firm rules)
- Non-obvious calculations (pip values, timezone conversions)
- Workarounds or temporary solutions

## Error Handling

### Exception Hierarchy
- Use specific exceptions from `src.core.enums` when available
- Never catch bare `Exception` in production code
- Always log errors with context before re-raising

### Error Logging Pattern
```python
try:
    result = self.platform_adapter.place_order(...)
except ConnectionError as e:
    self.logger.error(
        f"Failed to place order for {symbol}: Connection lost. "
        f"Will retry on next cycle. Error: {e}",
        extra={"symbol": symbol, "order_type": order_type.value}
    )
    return None
except ValueError as e:
    self.logger.error(
        f"Invalid order parameters for {symbol}: {e}",
        extra={"symbol": symbol, "volume": volume, "price": price}
    )
    raise  # Re-raise validation errors
```

### Signal Flow Debugging Patterns
**Purpose**: Comprehensive logging for signal generation and execution pipeline debugging.

**Strategy Signal Generation Logging**:
```python
# Log filter decisions for debugging
if current_close > lower_bb:
    self.logger.info(f"[{self.symbol}/{self.timeframe.name}] Bollinger Band filter blocked RSI Buy signal: price {current_close:.5f} > lower BB {lower_bb:.5f}")
    return None

# Log successful signal generation
self.logger.info(f"[{self.symbol}/{self.timeframe.name}] BUY signal generated: RSI {current_rsi:.2f}, price at BB extreme {current_close:.5f}")
```

**Orchestrator Signal Flow Logging**:
```python
# Use [SIGNAL_DEBUG] prefix for pipeline debugging (temporary debugging only)
if trade_signal_details:
    self.logger.info(f"[SIGNAL_DEBUG] Strategy {profile_key} generated signal: {trade_signal_details}")

# Log RiskController interaction (temporary debugging only)
self.logger.info(f"[SIGNAL_DEBUG] Orchestrator: RiskController response for {profile_key}: approved={is_approved}, lot_size={lot_size}")
```

**RiskController Validation Logging**:
```python
# Use [RISK_DEBUG] prefix for risk validation debugging (temporary debugging only)
self.logger.info(f"[RISK_DEBUG] RiskController: Position size calculated: {calculated_lot_size}")

# Log rejection reasons (temporary debugging only)
if calculated_lot_size <= 0.0:
    self.logger.info(f"[RISK_DEBUG] RiskController: REJECTED - Position size is zero or invalid: {calculated_lot_size}")
```

**Debug Logging Cleanup**: All `[*_DEBUG]` logging should be removed after debugging is complete to maintain clean production logs.

### Graceful Degradation Rules
- Platform connection issues → skip cycle, retry next iteration
- Single strategy errors → continue with other strategies
- Configuration errors → fail fast at startup

## Testing Standards

### Unit Test Requirements
- All strategy signal generation logic MUST have unit tests
- All risk controller validation methods MUST have unit tests
- **All portfolio coordination components MUST have unit tests (PortfolioManager, MarketRegimeClassifier)**
- **All regime detection algorithms MUST have unit tests with various market conditions**
- Test both success and failure scenarios
- Use descriptive test names: `test_rsi_strategy_generates_buy_signal_when_oversold_and_trend_allows`
- **Portfolio test names**: `test_portfolio_allocates_ranging_strategies_for_low_volatility_regime`

### Mock Patterns
```python
# Good: Mock external dependencies
@patch('src.api_connector.mt5_connector.mt5')
def test_mt5_connection_retry_on_failure(self, mock_mt5):
    mock_mt5.initialize.return_value = False
    mock_mt5.last_error.return_value = (123, "Connection failed")

    connector = MT5Adapter(config, logger)
    result = connector.connect()

    assert result is False
    assert mock_mt5.initialize.call_count == 1
```

### Integration Test Scope
- Full trade cycle with paper trading adapter
- **Portfolio coordination across multiple strategies**
- **Market regime detection and strategy switching**
- Configuration loading and validation (including portfolio configurations)
- State persistence and recovery (including portfolio state)
- **Multi-strategy backtesting with regime transitions**

## Data Handling

### Pydantic Model Usage (Mandatory)
- All data passed between modules MUST use models from `src.core.models`
- No raw dictionaries for structured data
- Validate data at module boundaries

### Pandas DataFrame Rules
- Always use `.copy()` when modifying DataFrames
- Use timezone-aware timestamps (UTC preferred)
- Clear column naming: `RSI_14`, `SMA_50`, `ATR_14`

### Timezone Handling
```python
# Good: Always timezone-aware
current_time = datetime.now(timezone.utc)
ftmo_time = current_time.astimezone(pytz.timezone('Europe/Prague'))

# Bad: Naive datetime
current_time = datetime.now()  # Ambiguous timezone
```

## Configuration Management

### Environment Variables (Security)
- ALL sensitive data MUST use environment variables
- Use descriptive names: `MT5_ACCOUNT`, `FINNHUB_API_KEY`
- Use .env files for development with python-dotenv library
- Ensure .env files are excluded from version control (.gitignore)
- Document required environment variables in README.md

**Environment Variable Loading Pattern**:
```python
# Standard pattern for scripts with environment variable loading
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file if available
except ImportError:
    # Graceful degradation if python-dotenv not installed
    pass
```

### Configuration Validation
- Use Pydantic models for all configuration sections
- Provide sensible defaults where possible
- Fail fast on invalid configuration at startup

### Configuration File Organization
```yaml
# Good: Clear hierarchy and comments
risk_management:
  global_max_account_drawdown_pct: 0.09  # 9% (FTMO buffer)
  global_daily_drawdown_limit_pct: 0.045 # 4.5% (FTMO buffer)

# Portfolio configuration example
portfolio_definitions:
  conservative_ranging:
    description: "Conservative portfolio for ranging market conditions"
    strategies:
      - strategy_key: "strategy_rsi_ranging_market_v1.json"
        max_allocation: 0.6
        regime_confidence_threshold: 70
    risk_allocation:
      max_portfolio_risk: 0.02  # 2% total portfolio risk
      correlation_limit: 0.7    # Maximum correlation between strategies

# Bad: No context or explanation
risk_management:
  max_dd: 0.09
  daily_dd: 0.045
```

## Portfolio Configuration Standards

### Portfolio Configuration File Naming
- **Portfolio Definitions**: Use descriptive names reflecting market conditions and risk profile
  - `portfolio_ranging_markets_v1.yaml` - Conservative ranging market portfolio
  - `portfolio_trending_markets_v1.yaml` - Aggressive trending market portfolio
  - `portfolio_mixed_regime_v1.yaml` - Balanced multi-regime portfolio
- **Version Control**: Always include version number for portfolio configurations
- **Regime Specificity**: Include regime type in filename for clarity

### YAML Structure Standards for Portfolio Definitions
```yaml
# Standard portfolio configuration structure in main_config.yaml
portfolio_definitions:
  ranging_market_conservative:
    description: "Conservative portfolio optimized for ranging market conditions"
    active: true
    strategies:
      - strategy_key: "strategy_rsi_ranging_market_v1.json"
        max_allocation: 0.6
        min_allocation: 0.3
        regime_confidence_threshold: 70
        correlation_limit: 0.8
      - strategy_key: "strategy_bollinger_ranging_v1.json"
        max_allocation: 0.4
        min_allocation: 0.2
        regime_confidence_threshold: 65
        correlation_limit: 0.7
    risk_allocation:
      max_portfolio_risk: 0.02        # 2% total account risk
      max_strategy_risk: 0.012        # 1.2% per strategy maximum
      correlation_limit: 0.7          # Portfolio-wide correlation limit
      rebalance_threshold: 0.1        # 10% allocation drift triggers rebalance
    regime_mapping:
      primary_regimes: ["ranging", "consolidating"]
      confidence_threshold: 70        # Minimum confidence to activate portfolio
      fallback_strategy: "strategy_rsi_ranging_market_v1.json"
```

### Regime-Strategy Mapping Configuration Patterns
```yaml
# Market regime detection and strategy mapping
market_regime_classifier:
  regimes:
    ranging:
      indicators:
        - name: "atr_ratio"
          weight: 0.3
          threshold_low: 0.8
          threshold_high: 1.2
        - name: "trend_strength"
          weight: 0.4
          threshold_low: 0.0
          threshold_high: 0.3
      suitable_strategies:
        - "strategy_rsi_ranging_market_v1.json"
        - "strategy_bollinger_ranging_v1.json"
      confidence_calculation: "weighted_average"
    trending:
      indicators:
        - name: "trend_strength"
          weight: 0.5
          threshold_low: 0.6
          threshold_high: 1.0
        - name: "momentum_persistence"
          weight: 0.3
          threshold_low: 0.7
          threshold_high: 1.0
      suitable_strategies:
        - "strategy_sma_trending_market_v1.json"
        - "strategy_breakout_trending_v1.json"
```

### Configuration Validation Requirements
- **Mandatory Fields**: All portfolio configurations MUST include description, strategies, risk_allocation
- **Allocation Validation**: Sum of max_allocation across strategies MUST NOT exceed 1.0
- **Correlation Constraints**: correlation_limit MUST be between 0.0 and 1.0
- **Confidence Thresholds**: regime_confidence_threshold MUST be between 0 and 100
- **Risk Limits**: max_portfolio_risk MUST be <= global risk limits from main configuration

## Strategy Interface Consistency Standards

### Mandatory Interface Requirements for Portfolio Compatibility
All strategies intended for portfolio use MUST implement the following interface:

```python
class PortfolioCompatibleStrategy(BaseStrategy):
    """
    Portfolio-compatible strategy interface requirements.
    All strategies must implement these methods for PortfolioManager integration.
    """

    @property
    def regime_specialization(self) -> List[str]:
        """
        Return list of market regimes this strategy is optimized for.

        Returns:
            List of regime names: ["ranging", "trending", "volatile"]
        """
        pass

    @property
    def correlation_constraints(self) -> Dict[str, float]:
        """
        Return correlation limits with other strategy types.

        Returns:
            Dict mapping strategy types to maximum correlation limits
        """
        pass

    def get_regime_confidence_score(self, market_data: pd.DataFrame) -> float:
        """
        Calculate confidence score for current market regime suitability.

        Args:
            market_data: Current market data with indicators

        Returns:
            Confidence score between 0.0 and 1.0
        """
        pass

    def get_allocation_preference(self, portfolio_context: Dict) -> float:
        """
        Return preferred allocation percentage based on current conditions.

        Args:
            portfolio_context: Current portfolio state and market conditions

        Returns:
            Preferred allocation as decimal (0.0 to 1.0)
        """
        pass
```

### Strategy Naming Conventions for Regime Specialization
- **Regime-Specific Naming**: `strategy_{technique}_{regime}_market_v{version}.json`
  - `strategy_rsi_ranging_market_v1.json` - RSI strategy for ranging markets
  - `strategy_sma_trending_market_v1.json` - SMA strategy for trending markets
  - `strategy_bollinger_volatile_market_v1.json` - Bollinger strategy for volatile markets
- **Technique Identification**: Use clear technique abbreviation (rsi, sma, macd, bollinger)
- **Market Regime**: Always specify primary regime (ranging, trending, volatile, mixed)
- **Version Control**: Increment version for parameter changes or optimizations

### Strategy Metadata Requirements
```json
{
  "strategy_params_key": "MeanReversion_RangingMarket_V1_Params",
  "description": "Mean Reversion RSI for Ranging Markets with Bollinger Band Filtering - Production V1",
  "strategy_definition_key": "MeanReversion_RSI_BB",
  "regime_metadata": {
    "primary_regimes": ["ranging", "consolidating"],
    "regime_confidence_indicators": ["atr_ratio", "rsi_oscillation"],
    "optimal_market_conditions": "Low volatility, mean-reverting price action at Bollinger Band extremes",
    "performance_attribution": "Specializes in sideways markets with clear support/resistance and price extremes"
  },
  "portfolio_integration": {
    "max_correlation_with_trend_strategies": 0.3,
    "max_correlation_with_volatility_strategies": 0.5,
    "preferred_allocation_range": [0.2, 0.6],
    "risk_scaling_factor": 1.0
  },
  "parameters": {
    "rsi_period": 14,
    "rsi_oversold": 35,
    "rsi_overbought": 65,
    "bollinger_period": 20,
    "bollinger_std_dev": 2.0,
    "stop_loss_atr_period": 10,
    "stop_loss_atr_multiplier": 1.8,
    "take_profit_atr_period": 10,
    "take_profit_atr_multiplier": 2.5,
    "max_position_age_bars": 48
  }
}
```

### Strategy Registration and Discovery Patterns
```python
# Strategy registration pattern for portfolio integration
@register_strategy(
    regime_specialization=["ranging", "consolidating"],
    correlation_constraints={"trending": 0.3, "volatility": 0.5},
    allocation_range=(0.2, 0.6)
)
class MeanReversionRSI(BaseStrategy):
    """Ranging market specialist using RSI mean reversion signals."""
    pass

# Discovery pattern in PortfolioManager
def discover_strategies_for_regime(self, regime: str) -> List[str]:
    """
    Discover all strategies suitable for given market regime.

    Args:
        regime: Market regime name

    Returns:
        List of strategy configuration keys suitable for regime
    """
    suitable_strategies = []
    for strategy_key, metadata in self.strategy_registry.items():
        if regime in metadata.get("regime_specialization", []):
            suitable_strategies.append(strategy_key)
    return suitable_strategies
```

## Multi-Strategy Testing Standards

### Unit Testing Requirements for Portfolio Components

#### PortfolioManager Testing
```python
class TestPortfolioManager:
    """
    Comprehensive unit tests for PortfolioManager component.
    Tests strategy allocation, coordination, and performance monitoring.
    """

    def test_allocate_strategies_for_ranging_regime_with_high_confidence(self):
        """Test strategy allocation for ranging regime with high confidence score."""
        portfolio_manager = PortfolioManager(mock_config, mock_risk_controller, mock_logger)

        # Test ranging regime with 85% confidence
        allocations = portfolio_manager.allocate_strategies("ranging", 85.0)

        assert "strategy_rsi_ranging_market_v1.json" in allocations
        assert allocations["strategy_rsi_ranging_market_v1.json"] >= 0.3  # Minimum allocation
        assert sum(allocations.values()) <= 1.0  # Total allocation constraint

    def test_coordinate_execution_prevents_correlated_signals(self):
        """Test that highly correlated strategies don't execute simultaneously."""
        # Mock strategies with high correlation
        mock_strategies = {
            "strategy_rsi_ranging_v1": MockStrategy(correlation=0.9),
            "strategy_bollinger_ranging_v1": MockStrategy(correlation=0.9)
        }

        signals = portfolio_manager.coordinate_execution(mock_strategies, mock_market_data)

        # Should limit execution due to high correlation
        assert len(signals) <= 1  # Only one correlated signal should execute
```

#### MarketRegimeClassifier Testing
```python
class TestMarketRegimeClassifier:
    """
    Unit tests for market regime detection and classification.
    """

    def test_classify_ranging_market_with_low_volatility(self):
        """Test ranging market detection with low volatility indicators."""
        classifier = MarketRegimeClassifier(mock_config, mock_data_manager, mock_logger)

        # Mock low volatility, mean-reverting market data
        mock_data = create_ranging_market_data(atr_ratio=0.9, trend_strength=0.2)

        regime, confidence = classifier.classify_current_regime("EURUSD", "M15")

        assert regime == "ranging"
        assert confidence >= 70.0  # High confidence for clear ranging conditions

    def test_detect_regime_transition_from_ranging_to_trending(self):
        """Test detection of regime transition from ranging to trending."""
        # Mock historical ranging data followed by trending data
        mock_historical_data = create_regime_transition_data()

        is_transitioning, new_regime, confidence = classifier.detect_regime_transition(
            "EURUSD", "M15", history_bars=50
        )

        assert is_transitioning is True
        assert new_regime == "trending"
        assert confidence >= 65.0  # Minimum confidence for regime change
```

### Integration Testing Standards for Multi-Strategy Scenarios

#### Portfolio Coordination Integration Tests
```python
class TestPortfolioIntegration:
    """
    Integration tests for complete portfolio coordination workflow.
    """

    @pytest.fixture
    def portfolio_test_environment(self):
        """Set up complete portfolio testing environment."""
        return {
            "orchestrator": MockOrchestrator(),
            "portfolio_manager": PortfolioManager(test_config, test_risk_controller, test_logger),
            "regime_classifier": MarketRegimeClassifier(test_config, test_data_manager, test_logger),
            "strategies": {
                "ranging": MockRangingStrategy(),
                "trending": MockTrendingStrategy()
            }
        }

    def test_complete_portfolio_trading_cycle(self, portfolio_test_environment):
        """Test complete trading cycle with regime detection and strategy coordination."""
        env = portfolio_test_environment

        # Simulate ranging market conditions
        mock_market_data = create_ranging_market_conditions()

        # Execute complete portfolio cycle
        regime, confidence = env["regime_classifier"].classify_current_regime("EURUSD", "M15")
        allocations = env["portfolio_manager"].allocate_strategies(regime, confidence)
        signals = env["portfolio_manager"].coordinate_execution(allocations, mock_market_data)

        # Verify ranging strategy was selected and executed
        assert regime == "ranging"
        assert "strategy_rsi_ranging_market_v1.json" in allocations
        assert len(signals) > 0  # Signals were generated
        assert all(signal["strategy_type"] == "ranging" for signal in signals)
```

#### Market Regime Transition Testing
```python
def test_strategy_switching_during_regime_transition():
    """Test graceful strategy switching when market regime changes."""
    # Start with ranging market and active ranging strategy
    initial_regime = "ranging"
    initial_allocations = {"strategy_rsi_ranging_market_v1.json": 0.6}

    # Simulate regime transition to trending
    transition_data = create_regime_transition_data(from_regime="ranging", to_regime="trending")

    # Process transition
    new_regime, confidence = regime_classifier.classify_current_regime("EURUSD", "M15")
    new_allocations = portfolio_manager.allocate_strategies(new_regime, confidence)

    # Verify smooth transition
    assert new_regime == "trending"
    assert "strategy_sma_trending_market_v1.json" in new_allocations
    assert new_allocations["strategy_rsi_ranging_market_v1.json"] < 0.3  # Reduced allocation
```

### Mock Patterns for Portfolio Testing

#### PortfolioManager Mock Pattern
```python
class MockPortfolioManager:
    """Mock PortfolioManager for testing strategy coordination."""

    def __init__(self, predefined_allocations=None):
        self.predefined_allocations = predefined_allocations or {}
        self.coordination_calls = []

    def allocate_strategies(self, regime: str, confidence: float) -> Dict[str, float]:
        """Mock strategy allocation with predefined responses."""
        if regime in self.predefined_allocations:
            return self.predefined_allocations[regime]
        return {"strategy_rsi_ranging_market_v1.json": 0.5}  # Default allocation

    def coordinate_execution(self, strategies: Dict, market_data: pd.DataFrame) -> List[Dict]:
        """Mock execution coordination with call tracking."""
        self.coordination_calls.append({
            "strategies": list(strategies.keys()),
            "timestamp": market_data.index[-1] if not market_data.empty else None
        })
        return [{"signal": "BUY", "strategy": "mock_strategy"}]
```

#### MarketRegimeClassifier Mock Pattern
```python
class MockMarketRegimeClassifier:
    """Mock MarketRegimeClassifier for testing regime detection."""

    def __init__(self, regime_sequence=None):
        self.regime_sequence = regime_sequence or [("ranging", 75.0)]
        self.call_index = 0

    def classify_current_regime(self, symbol: str, timeframe: str) -> Tuple[str, float]:
        """Mock regime classification with predefined sequence."""
        if self.call_index < len(self.regime_sequence):
            result = self.regime_sequence[self.call_index]
            self.call_index += 1
            return result
        return self.regime_sequence[-1]  # Return last regime if sequence exhausted
```

### Performance Testing Standards for Portfolio Backtesting

#### Portfolio Backtesting Performance Requirements
```python
class TestPortfolioBacktestingPerformance:
    """Performance tests for portfolio backtesting capabilities."""

    def test_portfolio_backtest_execution_time(self):
        """Test that portfolio backtesting completes within acceptable time limits."""
        start_time = time.time()

        # Execute portfolio backtest on 1-year dataset
        results = backtest_engine.run_portfolio_backtest(
            portfolio_config=test_portfolio_config,
            csv_file_path="data/EURUSD_M15_2023_full_year.csv",
            initial_balance=10000
        )

        execution_time = time.time() - start_time

        # Portfolio backtest should complete within 2 minutes for 1-year M15 data
        assert execution_time < 120.0
        assert results is not None
        assert len(results.trade_history) >= 0

    def test_portfolio_memory_usage_during_backtest(self):
        """Test memory usage remains within acceptable limits during portfolio backtesting."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Execute portfolio backtest
        results = backtest_engine.run_portfolio_backtest(
            portfolio_config=large_portfolio_config,
            csv_file_path="data/EURUSD_M15_2023_full_year.csv",
            initial_balance=10000
        )

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should not exceed 500MB for portfolio backtesting
        assert memory_increase < 500.0
```

## Performance Guidelines

### Memory Management
- Limit historical data retention (configurable window)
- Use generators for large datasets when possible
- Monitor memory usage in long-running processes

### Efficient Data Operations
```python
# Good: Vectorized operations
df['signal'] = np.where(
    (df['RSI_14'] < 30) & (df['close'] > df['SMA_200']),
    'BUY',
    'HOLD'
)

# Bad: Iterative operations
for i in range(len(df)):
    if df.iloc[i]['RSI_14'] < 30 and df.iloc[i]['close'] > df.iloc[i]['SMA_200']:
        df.iloc[i]['signal'] = 'BUY'
```

### Threading Considerations
- No shared mutable state between threads
- Use thread-safe logging
- Document any threading assumptions

## Security Requirements

### Credential Security (Critical)
- NEVER commit credentials to version control
- Use environment variables for all sensitive data
- Implement credential validation at startup

### API Security
- All external API calls MUST use HTTPS
- Implement proper timeout handling
- Log API errors without exposing credentials

### Audit Trail
- Log all trade decisions with structured data
- Include risk check results in logs
- Maintain immutable log records for compliance

## File and Module Organization

### Import Standards
```python
# Standard library
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

# Third-party
import pandas as pd
import numpy as np
from pydantic import BaseModel

# Local imports - Core
from src.core.models import Order, Position, PortfolioState
from src.core.enums import OrderType, StrategySignal, MarketRegime

# Local imports - Portfolio Management
from src.portfolio.portfolio_manager import PortfolioManager
from src.portfolio.regime_classifier import MarketRegimeClassifier
from src.portfolio.allocation_engine import AllocationEngine
```

### Portfolio Module Organization
```
src/
├── portfolio/
│   ├── __init__.py
│   ├── portfolio_manager.py      # Main portfolio coordination
│   ├── regime_classifier.py     # Market regime detection
│   ├── allocation_engine.py     # Risk allocation and sizing
│   ├── correlation_monitor.py   # Strategy correlation tracking
│   └── performance_attribution.py # Portfolio performance analysis
├── strategies/
│   ├── ranging/                  # Ranging market specialists
│   │   ├── mean_reversion_rsi.py
│   │   └── bollinger_reversion.py
│   ├── trending/                 # Trending market specialists
│   │   ├── trend_following_sma.py
│   │   └── momentum_breakout.py
│   └── volatile/                 # Volatility specialists
│       ├── volatility_breakout.py
│       └── adaptive_range.py
```

### Class Organization
- Public methods first
- Private methods last
- Group related methods together
- Use clear, descriptive method names
- **Portfolio classes**: Group regime detection, allocation, and coordination methods logically
- **Strategy classes**: Include regime specialization metadata and portfolio integration methods

## Prop Firm Compliance

### Code Comments for Compliance
- Document all prop firm rule implementations
- Include rule references in comments
- Example:
```python
# FTMO Rule: No trading within ±2 minutes of high-impact news
if self.news_filter.is_instrument_restricted(symbol):
    self.logger.warning(f"Trade rejected: News restriction active for {symbol}")
    return False, "News trading restriction", None
```

### Audit Requirements
- All risk decisions must be logged with reasoning
- Trade rejections must include specific rule violations
- **Portfolio allocation decisions must be logged with regime justification**
- **Strategy activation/deactivation must be logged with regime confidence scores**
- **Cross-strategy correlation breaches must be logged and addressed**
- Maintain structured logs for compliance review

### Portfolio-Specific Compliance Logging
```python
# Portfolio allocation logging pattern
self.logger.info(
    "Portfolio allocation updated for regime change",
    extra={
        "previous_regime": "ranging",
        "new_regime": "trending",
        "regime_confidence": 78.5,
        "previous_allocations": {"strategy_rsi_ranging_v1": 0.6},
        "new_allocations": {"strategy_sma_trending_v1": 0.7},
        "correlation_check": "passed",
        "risk_budget_utilization": 0.85
    }
)

# Strategy coordination compliance logging
self.logger.warning(
    "Strategy signal rejected due to correlation limit",
    extra={
        "rejected_strategy": "strategy_bollinger_ranging_v1",
        "active_strategy": "strategy_rsi_ranging_v1",
        "correlation_coefficient": 0.89,
        "correlation_limit": 0.8,
        "compliance_rule": "portfolio_correlation_limit"
    }
)
```

---

**Note**: These coding standards work in conjunction with ARCHITECTURE.md to ensure consistent, quality code that meets prop firm compliance requirements. All code reviews should verify adherence to both documents.

  
