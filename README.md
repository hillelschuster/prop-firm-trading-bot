# Prop Firm Trading Bot

## Project Overview

This is a sophisticated algorithmic trading bot specifically designed for proprietary trading firms like FTMO. The bot automates multi-strategy portfolio trading while maintaining strict compliance with prop firm rules, providing a flexible and extensible framework for developing, testing, and deploying diverse trading strategies with market regime detection, comprehensive risk management, and audit capabilities.

**Primary Goals:**
- Automate multi-strategy portfolio trading with guaranteed prop firm compliance
- Provide modular architecture for easy strategy development, portfolio coordination, and platform integration
- Ensure robust risk management with portfolio-level allocation and correlation management
- Enable intelligent market regime detection and strategy specialization
- Maintain detailed audit trails for compliance verification

## Core Features

Based on our architectural decisions, this bot provides:

- **Portfolio-Based Architecture**: Multi-strategy coordination with market regime detection and intelligent strategy allocation
- **Modular Architecture**: Clear separation of concerns with isolated components for strategies, portfolio management, risk management, platform interaction, and data handling
- **Centralized Risk Controller**: All trades must pass through portfolio-level risk validation - impossible to bypass compliance checks
- **Multi-Platform Support**: Abstract platform interface supporting MetaTrader 5, cTrader (planned), and paper trading for backtesting
- **Strategy Pattern Framework**: Extensible strategy system with regime specialization and portfolio integration interfaces
- **Pydantic Data Models**: All structured data uses validated models ensuring type safety and consistency across modules
- **Environment-Based Security**: All sensitive credentials loaded from environment variables - never stored in code
- **Timezone-Aware Compliance**: FTMO-specific daily reset handling with Prague timezone awareness
- **Structured JSON Logging**: Machine-readable audit trails for compliance verification and debugging
- **Single-Threaded Safety**: Predictable main trading logic with background data polling for real-time market data
- **Configuration Hierarchy**: YAML main config with JSON strategy parameters for flexible customization

## Project Structure

```
├── src/                    # Main source code
│   ├── api_connector/      # Platform adapters (MT5, cTrader, Paper Trading)
│   ├── core/              # Core models and enums (Pydantic data contracts)
│   ├── data_handler/      # Market data management and indicator calculation
│   ├── execution/         # Order execution and trade management
│   ├── portfolio/         # Portfolio management and market regime detection
│   ├── risk_controller/   # Risk management and compliance validation
│   ├── state_management/  # State persistence and recovery
│   ├── strategies/        # Trading strategy implementations (organized by regime)
│   ├── utils/             # Utility functions and performance reporting
│   ├── config_manager.py  # Configuration loading and validation
│   ├── logging_service.py # Centralized logging setup
│   └── orchestrator.py    # Main coordination engine with portfolio support
├── docs/                  # Project documentation (SINGLE SOURCE OF TRUTH)
│   ├── ARCHITECTURE.md    # System design and patterns
│   ├── CODING_STANDARDS.md # Implementation standards and quality
│   ├── CORE_FUNCTIONS.md  # Key function/class API reference
│   └── DECISIONS.md       # Architectural decisions and rationale
├── config/                # Configuration files
│   ├── main_config.yaml   # Main bot configuration with portfolio definitions
│   ├── instruments_ftmo.json # Trading instrument specifications
│   ├── portfolio_*.yaml   # Portfolio configuration files
│   └── strategy_*.json    # Strategy-specific parameters (regime-specialized)
├── tests/                 # Unit and integration tests
│   ├── unit/             # Component-level tests
│   └── integration/      # Full system tests
├── scripts/              # Entry point scripts
│   ├── run_bot.py        # Main bot execution
│   └── run_backtest.py   # Backtesting execution
└── logs/                 # Generated log files
```

## Setup and Installation

### Prerequisites
- Python 3.8 or higher
- MetaTrader 5 platform (if using MT5 connector)
- Git for version control

### Installation Steps

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd prop_firm_trading_bot
   ```

2. **Create and Activate Virtual Environment**
   ```bash
   python -m venv venv

   # On Windows
   venv\Scripts\activate

   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Development Dependencies (Optional)**
   ```bash
   # Uncomment pytest lines in requirements.txt or install directly
   pip install pytest pytest-mock coverage
   ```

## Configuration

### Main Configuration File

The bot uses `config/main_config.yaml` as the primary configuration file. This file contains:
- Bot settings (trading mode, loop delays, timezone)
- Platform configuration (MT5/cTrader settings)
- Risk management parameters (drawdown limits, position sizing)
- Portfolio definitions (strategy allocations and regime mappings)
- Asset-strategy profiles (which strategies to run on which instruments)
- Market regime classifier settings (regime detection parameters)
- News filter settings (economic calendar restrictions)
- Operational compliance settings (order limits, weekend closure)

### Required Environment Variables

**Critical Security Requirement**: All sensitive credentials MUST be set as environment variables and NEVER stored in configuration files or code.

#### MetaTrader 5 (Required if using MT5)
```bash
export MT5_ACCOUNT="your_account_number"
export MT5_PASSWORD="your_password"
export MT5_SERVER="your_broker_server"
```

#### News Provider APIs (Optional)
```bash
export FINNHUB_API_KEY="your_finnhub_api_key"  # If using Finnhub news
```

#### cTrader (Future Implementation)
```bash
export CTRADER_CLIENT_ID="your_client_id"
export CTRADER_CLIENT_SECRET="your_client_secret"
export CTRADER_ACCOUNT_ID="your_account_id"
```

### Configuration Validation

The bot validates all configuration at startup using Pydantic models. If any required settings are missing or invalid, the bot will fail fast with detailed error messages.

## Running the Bot

### Production Trading
```bash
python scripts/run_bot.py
```

### Backtesting
```bash
# Single strategy backtesting
python scripts/run_backtest.py --strategy-profile EURUSD_RSI_M15

# Portfolio backtesting (multiple strategies with regime detection)
python scripts/run_backtest.py --portfolio-mode
```

**Prerequisites for Running:**
1. Ensure all required environment variables are set
2. Verify `config/main_config.yaml` is properly configured with portfolio definitions
3. Confirm trading platform (MT5) is installed and accessible
4. Check that strategy parameter files exist in `config/` with regime-specific naming
5. Validate portfolio configurations and regime-strategy mappings

## Running Tests

The project uses `pytest` for comprehensive testing:

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src

# Run specific test categories
pytest tests/unit/          # Unit tests only
pytest tests/integration/   # Integration tests only

# Run with verbose output
pytest -v
```

## Development Workflow

### Fundamental Guideline for AI Collaboration

**For our work on the prop firm bot, you must always adhere to the rules defined in our project documentation, which is located in the docs/ directory. These files (especially docs/ARCHITECTURE.md and docs/CODING_STANDARDS.md) are our single source of truth. All code and suggestions must comply with them.**

**Crucially, if a change to the code requires a change to these rules, we will update the documentation first.**

### Documentation-First Development Process

1. **Before Making Changes**: Check if the change affects documented patterns in `docs/`
2. **Update Documentation First**: Modify relevant docs/ files before implementing code changes
3. **Implement Code**: Follow the updated documentation and established patterns
4. **Verify Compliance**: Use the project checklist to ensure all standards are met
5. **Test Thoroughly**: Run unit and integration tests to verify functionality

### Quality Assurance

This project follows comprehensive quality standards covering:
- Documentation compliance and updates (documentation-first workflow)
- Code quality and standards adherence (see `docs/CODING_STANDARDS.md`)
- Testing requirements and coverage (unit and integration tests)
- Prop firm compliance verification (risk validation and audit trails)
- Security and configuration validation (environment variables and credential management)

### Key Documentation Files

- **`docs/ARCHITECTURE.md`**: System design principles and patterns
- **`docs/CODING_STANDARDS.md`**: Implementation standards and quality requirements
- **`docs/CORE_FUNCTIONS.md`**: API reference for key functions and classes
- **`docs/DECISIONS.md`**: Architectural decisions and their rationale

## Compliance and Security

### Prop Firm Compliance
- **FTMO-Specific Rules**: Built-in compliance with FTMO requirements (drawdown limits, news restrictions, daily resets)
- **Risk Validation**: All trades must pass through centralized risk controller
- **Audit Trail**: Comprehensive structured logging for compliance verification
- **Timezone Awareness**: Proper handling of FTMO's Prague timezone for daily resets

### Security Requirements
- **Credential Management**: All sensitive data via environment variables only
- **API Security**: HTTPS-only external communications with proper timeout handling
- **Audit Logging**: Immutable log records for compliance and debugging
- **No Secrets in Code**: Zero tolerance for hardcoded credentials or API keys

## Support and Documentation

For detailed information about the system architecture, coding standards, and development guidelines, refer to the comprehensive documentation in the `docs/` directory. These files serve as the definitive reference for all development work and must be consulted before making any significant changes to the system.

---

**Note**: This README.md serves as the entry point to the project. For detailed technical information, always refer to the documentation in the `docs/` directory, which contains the complete architectural guidelines, coding standards, and development processes.
# salt 2025-06-11T11:28:25
  
