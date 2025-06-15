import os
import json
import yaml

import pytest

import prop_firm_trading_bot.src.config_manager as cm
from prop_firm_trading_bot.src.config_manager import load_and_validate_config, AppConfig


def create_temp_config(tmp_path):
    instruments = {
        "EURUSD_FTMO": {
            "platform_symbol": "EURUSD",
            "pip_value_in_account_currency_per_lot": 10.0,
            "point_value_in_account_currency_per_lot": 1.0,
            "digits": 5,
            "min_volume_lots": 0.01,
            "max_volume_lots": 50.0,
            "volume_step_lots": 0.01,
            "contract_size": 100000,
            "news_target_currencies": ["USD", "EUR"],
            "trade_allowed": True,
        }
    }
    with open(tmp_path / "instruments_ftmo.json", "w") as f:
        json.dump(instruments, f)

    params = {
        "strategy_params_key": "Test_Params",
        "description": "Test params",
        "strategy_definition_key": "TestStrategy",
        "parameters": {
            "timeframe": "M1",
            "sma_fast": {"length": 3},
            "atr": {"length": 14},
        },
    }
    with open(tmp_path / "Test_Params.json", "w") as f:
        json.dump(params, f)

    main_cfg = {
        "bot_settings": {
            "trading_mode": "paper",
            "main_loop_delay_seconds": 1,
            "app_name": "T",
            "ftmo_server_timezone": "Europe/Prague",
        },
        "logging": {
            "level": "DEBUG",
            "directory": str(tmp_path),
            "file_name_prefix": "t",
            "structured_logging": False,
            "max_bytes": 1024,
            "backup_count": 1,
            "log_format": "%(message)s",
        },
        "platform": {"name": "Paper", "mt5": None, "ctrader": None},
        "assets_to_trade": ["TEST_PROFILE"],
        "asset_strategy_profiles": {
            "TEST_PROFILE": {
                "symbol": "EURUSD",
                "enabled": True,
                "instrument_details_key": "EURUSD_FTMO",
                "strategy_params_key": "Test_Params",
            }
        },
        "strategy_definitions": {
            "TestStrategy": {
                "strategy_module": "prop_firm_trading_bot.src.strategies.trend_following_sma",
                "strategy_class": "TrendFollowingSMA",
            }
        },
        "risk_management": {
            "global_max_account_drawdown_pct": 0.1,
            "global_daily_drawdown_limit_pct": 0.05,
            "default_risk_per_trade_idea_pct": 0.01,
        },
        "operational_compliance": {
            "min_trade_duration_seconds": 1,
            "max_orders_per_second": 10,
            "max_total_orders_per_day": 100,
            "max_order_modifications_per_minute_total": 10,
            "market_close_blackout_period_hours": 1,
            "enforce_weekend_closure": False,
            "is_swing_account": True,
        },
        "news_filter": {
            "enabled": False,
            "api_provider": "Manual",
            "manual_news_file_path": str(tmp_path / "news.json"),
            "calendar_fetch_interval_seconds": 3600,
            "min_impact_to_consider": "High",
            "pause_minutes_before_news": 2,
            "pause_minutes_after_news": 2,
            "high_impact_keywords": [],
        },
        "state_management": {
            "persistence_file": str(tmp_path / "state.json"),
            "persistence_interval_seconds": 300,
        },
    }

    with open(tmp_path / "news.json", "w") as f:
        json.dump([], f)
    with open(tmp_path / "main.yaml", "w") as f:
        yaml.dump(main_cfg, f, sort_keys=False)

    return tmp_path


def test_load_config_success(tmp_path):
    cfg_dir = create_temp_config(tmp_path)
    cm._config_instance = None
    config = load_and_validate_config(config_dir=str(cfg_dir), main_config_filename="main.yaml")

    assert isinstance(config, AppConfig)
    assert "EURUSD_FTMO" in config.loaded_instrument_details
    assert "Test_Params" in config.loaded_strategy_parameters


def test_missing_strategy_file(tmp_path):
    cfg_dir = create_temp_config(tmp_path)
    os.remove(cfg_dir / "Test_Params.json")
    cm._config_instance = None
    config = load_and_validate_config(config_dir=str(cfg_dir), main_config_filename="main.yaml")

    assert config.loaded_strategy_parameters == {}


def test_invalid_strategy_params(tmp_path):
    cfg_dir = create_temp_config(tmp_path)
    with open(cfg_dir / "Test_Params.json", "w") as f:
        json.dump({"strategy_params_key": "Test_Params"}, f)
    cm._config_instance = None
    config = load_and_validate_config(config_dir=str(cfg_dir), main_config_filename="main.yaml")

    assert config.loaded_strategy_parameters == {}


  
