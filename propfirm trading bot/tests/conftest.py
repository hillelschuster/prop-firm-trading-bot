import os
import sys
import types
import yaml
import json
import logging
from unittest.mock import patch
import pytest

# Ensure project root is on sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Provide alias for imports expecting package name 'prop_firm_trading_bot'
if 'prop_firm_trading_bot' not in sys.modules:
    import src
    alias = types.ModuleType('prop_firm_trading_bot')
    alias.src = src
    sys.modules['prop_firm_trading_bot'] = alias
    sys.modules['prop_firm_trading_bot.src'] = src

from prop_firm_trading_bot.src.config_manager import AppConfig, load_and_validate_config
from prop_firm_trading_bot.src.logging_service import setup_logging

@pytest.fixture(scope="session")
def project_root_dir():
    """Returns the absolute path to the project root."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

@pytest.fixture(scope="session")
def dummy_config_dir(project_root_dir):
    """Creates and returns path to a temporary config dir for tests."""
    path = os.path.join(project_root_dir, "tests", "temp_config")
    os.makedirs(path, exist_ok=True)
    return path

@pytest.fixture(scope="function")
def sample_app_config(dummy_config_dir) -> AppConfig:
    """Provides a sample, validated AppConfig object for testing."""
    test_main_config_file = os.path.join(dummy_config_dir, "test_main_config_for_conftest.yaml")

    dummy_yaml_content = {
        "bot_settings": {
            "trading_mode": "paper", "main_loop_delay_seconds": 1,
            "app_name": "TestBotFixture", "ftmo_server_timezone": "Europe/Prague",
            "magic_number_default": 123
        },
        "logging": {
            "level": "DEBUG", "directory": os.path.join(project_root_dir, "tests", "temp_logs"),
            "file_name_prefix": "fixture_test_log", "structured_logging": False,
            "max_bytes": 1024, "backup_count": 1,
            "log_format": "%(asctime)s - %(name)s - [%(levelname)s] - %(message)s"
        },
        "platform": {
            "name": "Paper",
            "mt5": None,
            "ctrader": None
        },
        "assets_to_trade": [],
        "asset_strategy_profiles": {},
        "strategy_definitions": {},
        "risk_management": {
            "global_max_account_drawdown_pct": 0.1,
            "global_daily_drawdown_limit_pct": 0.05,
            "default_risk_per_trade_idea_pct": 0.01,
            "max_concurrent_trades_per_strategy_type": 1,
            "max_total_concurrent_trades": 1
        },
        "operational_compliance": {
            "min_trade_duration_seconds": 1, "max_orders_per_second": 10,
            "max_total_orders_per_day": 100, "max_order_modifications_per_minute_total": 10,
            "market_close_blackout_period_hours": 1, "enforce_weekend_closure": False,
            "is_swing_account": True
        },
        "news_filter": {
            "enabled": False, "api_provider": "Manual",
            "manual_news_file_path": os.path.join(dummy_config_dir, "dummy_news.json"),
            "calendar_fetch_interval_seconds": 3600,
            "min_impact_to_consider": "High",
            "pause_minutes_before_news": 2,
            "pause_minutes_after_news": 2,
            "high_impact_keywords": []
        },
        "state_management": {
            "persistence_file": os.path.join(project_root_dir, "tests", "temp_state", "fixture_test_state.json"),
            "persistence_interval_seconds": 300
        }
    }
    with open(test_main_config_file, "w") as f:
        yaml.dump(dummy_yaml_content, f, sort_keys=False)

    dummy_news_path = os.path.join(dummy_config_dir, "dummy_news.json")
    with open(dummy_news_path, "w") as f_news:
        json.dump([], f_news)

    os.makedirs(os.path.join(project_root_dir, "tests", "temp_state"), exist_ok=True)
    os.makedirs(os.path.join(project_root_dir, "tests", "temp_logs"), exist_ok=True)

    with patch.dict(os.environ, {}, clear=True):
        config = load_and_validate_config(config_dir=dummy_config_dir, main_config_filename="test_main_config_for_conftest.yaml")
    return config

@pytest.fixture(scope="function")
def test_logger(sample_app_config: AppConfig):
    """Provides a configured logger instance for tests."""
    log_dir = sample_app_config.logging.directory
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    logger = setup_logging(config=sample_app_config, logger_name="pytest_logger")
    return logger
