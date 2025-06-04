# Prop Firm Trading Bot

## Project Purpose

This project is a trading bot designed for proprietary trading firms. Its primary goal is to automate trading strategies, manage risk, and interact with various trading platforms (e.g., MetaTrader 5, cTrader) to execute trades based on predefined algorithms. The bot aims to provide a flexible and extensible framework for developing, testing, and deploying diverse trading strategies while adhering to prop firm requirements and risk management protocols.

## Setup

Follow these steps to set up the project environment:

1.  **Prerequisites:**
    *   Python 3.8 or higher is recommended.
    *   Ensure `pip` (Python package installer) is up to date.
    *   Certain trading platform connectors (e.g., MetaTrader 5) may require the platform software to be installed on your system.

2.  **Clone the Repository (if applicable):**
    ```bash
    git clone <repository-url>
    cd prop_firm_trading_bot
    ```

3.  **Create and Activate a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

4.  **Install Dependencies:**
    Install all required Python packages using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

5.  **Environment Variables & Configuration:**
    *   The bot uses a central configuration file located at `config/main_config.yaml`. You will need to review and potentially customize this file for your specific setup, including API keys, account credentials, and strategy parameters.
    *   Some sensitive information, like API keys or trading account credentials, might be configured to be read from environment variables. For example, for MetaTrader 5, you might need to set variables like `MT5_ACCOUNT`, `MT5_PASSWORD`, and `MT5_SERVER` in your environment. Refer to the comments within `scripts/run_bot.py` and the structure of `config/main_config.yaml` for details on required environment variables for your chosen platform.

## Running the Bot

To run the trading bot:

1.  **Ensure Configuration is Complete:**
    *   Verify that your `config/main_config.yaml` is correctly set up for the trading platform, strategies, and risk parameters you intend to use.
    *   Ensure any required environment variables (as mentioned in the Setup section) are set in your current terminal session.

2.  **Execute the Bot:**
    Navigate to the project root directory (`prop_firm_trading_bot`) in your terminal and run the main script:
    ```bash
    python scripts/run_bot.py
    ```
    The bot will initialize using the settings from `config/main_config.yaml` and start its trading operations. Logs will be generated according to the logging configuration.

## Running Tests

This project uses `pytest` for running unit and integration tests.

1.  **Ensure Test Dependencies are Installed:**
    If not already included in the main `requirements.txt` or if you have a separate `requirements-dev.txt`, make sure `pytest` and any related plugins (like `pytest-mock`) are installed.
    ```bash
    pip install pytest pytest-mock
    ```
    (Note: `pytest` and `pytest-mock` are commented out in the provided `requirements.txt`. Uncomment and install them if you intend to run tests.)

2.  **Run Tests:**
    Navigate to the project root directory (`prop_firm_trading_bot`) in your terminal and execute `pytest`:
    ```bash
    pytest
    ```
    Alternatively, you can run it as a module:
    ```bash
    python -m pytest
    ```
    Pytest will automatically discover and run tests located in the `tests/` directory. Ensure you are in the project root directory so that all paths and imports resolve correctly.