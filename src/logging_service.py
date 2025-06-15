# prop_firm_trading_bot/src/logging_service.py

import logging
import logging.handlers
import sys
import os
from typing import Optional, TYPE_CHECKING

# Attempt to use pythonjsonlogger if available for structured logging
try:
    from pythonjsonlogger import jsonlogger
    _JSON_LOGGER_AVAILABLE = True
except ImportError:
    _JSON_LOGGER_AVAILABLE = False

if TYPE_CHECKING:
    from src.config_manager import AppConfig, LoggingSettings, BotSettings

# Global variable to hold the app_name, can be set by setup_logging
_APP_NAME_FOR_LOGGING = "PropFirmAlgoBot"


class ContextualFilter(logging.Filter):
    """A filter to add contextual information like app_name to log records."""
    def filter(self, record):
        record.app_name = _APP_NAME_FOR_LOGGING
        # Ensure standard LogRecord attributes are present for JsonFormatter,
        # though JsonFormatter usually picks them up.
        # These are standard attributes, so they should already be on the record.
        # record.module = record.module
        # record.funcName = record.funcName
        # record.lineno = record.lineno
        return True

def setup_logging(
    config: Optional['AppConfig'] = None,
    logger_name: Optional[str] = None
) -> logging.Logger:
    """
    Sets up comprehensive logging for the application.
    Supports standard text logging and structured JSON logging based on configuration.

    Args:
        config (Optional[AppConfig]): The application configuration object.
                                      Expected to have 'logging' and 'bot_settings' sections.
        logger_name (Optional[str]): Name for the root logger. If None, uses app_name from config.

    Returns:
        logging.Logger: The configured root logger instance.
    """
    global _APP_NAME_FOR_LOGGING

    if config is None:
        # Fallback to basic logging if no config is provided (e.g., for config_manager itself during its tests)
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)s] - %(message)s")
        return logging.getLogger(logger_name or "DefaultApp")

    log_cfg: 'LoggingSettings' = config.logging
    bot_cfg: 'BotSettings' = config.bot_settings
    _APP_NAME_FOR_LOGGING = bot_cfg.app_name

    effective_logger_name = logger_name or _APP_NAME_FOR_LOGGING
    logger = logging.getLogger(effective_logger_name)

    # Prevent multiple handlers if setup_logging is called multiple times on the same logger
    if logger.hasHandlers():
        logger.handlers.clear()

    log_level_str = log_cfg.level.upper()
    numeric_level = getattr(logging, log_level_str, None)
    if not isinstance(numeric_level, int):
        # Use a temporary basicConfig for this warning if logger itself isn't fully set up
        logging.basicConfig()
        logging.warning(f"Invalid log level: {log_cfg.level}. Defaulting to INFO.")
        numeric_level = logging.INFO
    logger.setLevel(numeric_level)

    # Create logs directory if it doesn't exist
    log_directory = log_cfg.directory
    if not os.path.exists(log_directory):
        try:
            os.makedirs(log_directory, exist_ok=True)
        except OSError as e:
            # Use basic print/logging for this critical error as logger might not be fully functional
            sys.stderr.write(f"CRITICAL: Could not create log directory {log_directory}: {e}\n")
            log_directory = "." # Fallback to current directory

    context_filter = ContextualFilter() # Initialize the filter

    # --- Console Handler (always text for readability during interactive use) ---
    console_formatter = logging.Formatter(log_cfg.log_format)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(numeric_level) # Console logs at the same level as file by default
    console_handler.addFilter(context_filter)
    logger.addHandler(console_handler)

    # --- File Handler (Rotating, potentially JSON) ---
    use_json_for_file = log_cfg.structured_logging and _JSON_LOGGER_AVAILABLE
    
    if use_json_for_file:
        file_prefix = log_cfg.json_log_file_name_prefix or f"{log_cfg.file_name_prefix}_structured"
        log_file_path = os.path.join(log_directory, f"{file_prefix}.json.log")
        # Default JsonFormatter includes many standard LogRecord fields.
        # The format string defines which fields from LogRecord are included and their names in the JSON.
        # Adding 'app_name' as it's in our ContextualFilter.
        # The 'extra' dict in logging calls will be automatically included.
        log_format_for_json = (
            "%(asctime)s %(levelname)s %(name)s %(module)s %(funcName)s %(lineno)d %(app_name)s %(message)s"
        )
        file_formatter = jsonlogger.JsonFormatter(log_format_for_json)
    else:
        if log_cfg.structured_logging and not _JSON_LOGGER_AVAILABLE:
            logger.warning("Structured JSON logging was requested, but 'python-json-logger' library is not installed. Falling back to text logging for files.")
        file_prefix = log_cfg.file_name_prefix
        log_file_path = os.path.join(log_directory, f"{file_prefix}.log")
        file_formatter = logging.Formatter(log_cfg.log_format)

    try:
        # Using TimedRotatingFileHandler for daily rotation
        file_handler = logging.handlers.TimedRotatingFileHandler(
            log_file_path,
            when="midnight", # Rotate daily at midnight
            interval=1,
            backupCount=log_cfg.backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(file_formatter)
        file_handler.addFilter(context_filter) # Add filter to file handler too
        logger.addHandler(file_handler)
        
    except Exception as e:
        # If file logging setup fails, console logger should still work.
        logger.error(f"Could not set up file logging to {log_file_path}: {e}", exc_info=True)

    # Add filter to the logger itself, so all handlers inherit it unless they have their own
    # Although adding to handlers directly is often more explicit.
    # logger.addFilter(context_filter) # Already added to handlers, might be redundant here.

    logger.info(f"Logging setup complete for '{effective_logger_name}'. Level: {log_level_str}. File format for '{os.path.basename(log_file_path)}': {'JSON' if use_json_for_file else 'Text'}.")
    return logger


if __name__ == "__main__":
    # Example Usage (for testing the logger)
    # This requires the config_manager to be working to get a config object.
    if not os.path.exists("config"): os.makedirs("config", exist_ok=True)
    if not os.path.exists("logs"): os.makedirs("logs", exist_ok=True)
    if not os.path.exists("state"): os.makedirs("state", exist_ok=True)

    import yaml 
    # Assuming config_manager.py is now in src/
    # Adjust import path if running this test script from a different location
    try:
        from src.config_manager import load_and_validate_config
    except ImportError:
        # Fallback for direct script execution if PYTHONPATH isn't set up
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        from src.config_manager import load_and_validate_config


    dummy_config_path_for_logger = "config/test_logger_service_config.yaml"
    
    # Create a dummy config for testing JSON logging
    with open(dummy_config_path_for_logger, "w") as f:
        yaml.dump({
            "bot_settings": {"app_name": "LoggingServiceTestApp", "trading_mode": "paper", "main_loop_delay_seconds": 1, "ftmo_server_timezone": "UTC"},
            "logging": {
                "level": "DEBUG", "directory": "logs", 
                "file_name_prefix": "ls_text_test", 
                "structured_logging": True, 
                "json_log_file_name_prefix": "ls_json_test", 
                "max_bytes": 10240, "backup_count": 2,
                "log_format": "%(asctime)s - %(app_name)s - [%(levelname)s] - %(name)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s"
            },
            "platform": {"name": "Paper", # Using Paper for simple test config
                         "mt5": None, "ctrader": None}, # Ensure these can be null if not used by Paper
            "assets_to_trade": [],
            "asset_strategy_profiles": {},
            "strategy_definitions": {},
            "risk_management": {"global_max_account_drawdown_pct": 0.1, "global_daily_drawdown_limit_pct": 0.05, "default_risk_per_trade_idea_pct": 0.01, "max_concurrent_trades_per_strategy_type": 1, "max_total_concurrent_trades": 1},
            "operational_compliance": {"min_trade_duration_seconds": 1, "max_orders_per_second": 10, "max_total_orders_per_day": 1000, "max_order_modifications_per_minute_total": 100, "market_close_blackout_period_hours": 1, "enforce_weekend_closure": False, "is_swing_account": True},
            "news_filter": {"enabled": False, "api_provider": "Manual", "manual_news_file_path": "config/dummy_news.json"}, # Dummy for validation
            "state_management": {"persistence_file": "state/ls_test_state.json", "persistence_interval_seconds": 300}
        }, f)

    # Dummy environment variables for config_manager test (if platform was MT5/cTrader)
    # os.environ["TEST_LS_MT5_ACCOUNT"] = "ls_dummy_log" 
    # os.environ["TEST_LS_FINNHUB_API_KEY"] = "ls_dummy_key"

    try:
        print("--- Testing logging_service.py (with JSON logging enabled in config) ---")
        
        test_app_config: 'AppConfig' = load_and_validate_config(dummy_config_path_for_logger)
        
        # Test setting up the main application logger
        main_logger = setup_logging(config=test_app_config)
        
        if not _JSON_LOGGER_AVAILABLE and test_app_config.logging.structured_logging:
            print("NOTE: 'python-json-logger' is not installed. JSON logging test has fallen back to text.")

        main_logger.debug("This is a DEBUG message from main_logger.", extra={"trade_id": "debug123", "asset": "EURUSD"})
        main_logger.info("This is an INFO message from main_logger.", extra={"user_id": "info456", "action": "CONNECT"})
        main_logger.warning("This is a WARNING message from main_logger.", extra={"reason_code": "W001"})
        main_logger.error("This is an ERROR message from main_logger.", extra={"error_code": 500, "details": "Failed to connect to API."})
        main_logger.critical("This is a CRITICAL message from main_logger.", extra={"system_component": "Database"})

        # Test getting a logger for a specific module (it will inherit main logger's handlers)
        module_logger_name = f"{test_app_config.bot_settings.app_name}.MyModule"
        module_logger = logging.getLogger(module_logger_name)
        module_logger.info("Info message from MyModule's logger.")
        module_logger.debug("Debug message from MyModule's logger (should appear if root level is DEBUG).")

        log_file_to_check_json = os.path.join(
            test_app_config.logging.directory, 
            f"{test_app_config.logging.json_log_file_name_prefix}.json.log"
        )
        log_file_to_check_text = os.path.join(
            test_app_config.logging.directory,
            f"{test_app_config.logging.file_name_prefix}.log"
        )

        if test_app_config.logging.structured_logging and _JSON_LOGGER_AVAILABLE:
            print(f"JSON Logs should be in: {log_file_to_check_json}")
        else:
            print(f"Text Logs (fallback or configured) should be in: {log_file_to_check_text}")
        
        print("Please check the content of the log file(s) and console for correct formatting and messages.")

    except Exception as e:
        print(f"Error during logging_service test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("--- Finished testing logging_service.py ---")
        # if os.path.exists(dummy_config_path_for_logger):
        #     os.remove(dummy_config_path_for_logger) # Clean up test config file


  
