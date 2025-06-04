import sys
import os
import logging
from typing import Optional

# Adjust the Python path to include the 'src' directory from the project root
# This allows for absolute imports like 'from prop_firm_trading_bot.src.module import Class'
# Assumes 'scripts' is one level down from the project root 'prop_firm_trading_bot/'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now that the path is adjusted, we can use absolute imports from src
try:
    from prop_firm_trading_bot.src.config_manager import load_and_validate_config, AppConfig
    from prop_firm_trading_bot.src.logging_service import setup_logging
    from prop_firm_trading_bot.src.orchestrator import Orchestrator
except ImportError as e:
    # This basic logger will be used if the main logger setup fails
    logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.critical(
        "CRITICAL ERROR: Failed to import necessary modules. Ensure PYTHONPATH is correct or run from project root. Error: %s",
        e,
    )
    sys.exit(1)  # Exit if core modules can't be imported


def main():
    """Main function to initialize and run the trading bot."""
    # --- 1. Define Configuration Path ---
    # Assuming main_config.yaml is in the 'config' directory relative to the project root
    config_dir_path = os.path.join(project_root, "config") # Define directory for config
    main_config_filename_const = "main_config.yaml"      # Define main config filename

    # --- 2. Load Configuration ---
    try:
        app_config: AppConfig = load_and_validate_config(config_dir=config_dir_path, main_config_filename=main_config_filename_const)
    except Exception as e:  # noqa: BLE001
        # Use basic logging if app_config dependent logger setup fails
        logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.critical(
            "CRITICAL ERROR: Failed to load or validate configuration from '%s'. Bot cannot start. Error: %s",
            os.path.join(config_dir_path, main_config_filename_const), # Construct path for logging
            e,
            exc_info=True,
        )
        sys.exit(1)

    # --- 3. Setup Main Application Logger ---
    # The main_logger should be configured using settings from app_config.logging
    try:
        main_logger = setup_logging(config=app_config, logger_name=app_config.bot_settings.app_name)
    except Exception as e:  # noqa: BLE001
        # If logger setup fails, use a basic print and exit.
        sys.stderr.write(f"CRITICAL ERROR: Failed to set up main application logger. Error: {e}\n")
        sys.exit(1)

    main_logger.info("Configuration loaded. Bot Name: %s", app_config.bot_settings.app_name)
    main_logger.info("Log level set to: %s", app_config.logging.level)
    main_logger.info("Trading Mode: %s", app_config.bot_settings.trading_mode)
    main_logger.info("Platform: %s", app_config.platform.name)

    # --- 4. Instantiate and Run Orchestrator ---
    orchestrator_instance: Optional[Orchestrator] = None
    try:
        main_logger.info("Initializing Orchestrator...")
        orchestrator_instance = Orchestrator(config=app_config, main_logger=main_logger)
        main_logger.info("Orchestrator initialized. Starting bot run loop...")

        # The run() method should contain its own KeyboardInterrupt handling for graceful shutdown
        orchestrator_instance.run()

    except KeyboardInterrupt:
        main_logger.info("KeyboardInterrupt received by run_bot.py. Initiating shutdown...")
        # Orchestrator's run() method's finally block or its stop() method should handle this.
    except ValueError as ve:
        # Specific errors from our setup, e.g., bad config values not caught by Pydantic
        main_logger.critical("ValueError during Orchestrator setup or run: %s", ve, exc_info=True)
    except ImportError as ie:
        # e.g. if a strategy module is misconfigured
        main_logger.critical(
            "ImportError during Orchestrator setup (likely strategy loading): %s",
            ie,
            exc_info=True,
        )
    except Exception as e:  # noqa: BLE001
        main_logger.critical(
            "An unexpected critical error occurred in run_bot.py: %s",
            e,
            exc_info=True,
        )
    finally:
        main_logger.info("run_bot.py main function finished or exited.")
        if orchestrator_instance and orchestrator_instance.is_running:
            main_logger.info("Calling orchestrator stop due to main exit...")
            orchestrator_instance.stop()  # Ensure graceful shutdown if loop exited unexpectedly
        # Orchestrator.shutdown() is called within its own run() method's finally block.


if __name__ == "__main__":
    # This allows the script to be run directly.
    # To run from the command line (CLI):
    # 1. Ensure any required environment variables for your configured trading platform
    #    (e.g., MT5_ACCOUNT, MT5_PASSWORD, MT5_SERVER for MetaTrader5, as specified
    #    in your main_config.yaml under platform.mt5.*_env_var) are set in your terminal session.
    # 2. Navigate to the project root directory (prop_firm_trading_bot).
    # 3. Execute the script:
    #    python scripts/run_bot.py
    main()
