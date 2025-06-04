# prop_firm_trading_bot/src/config_manager.py

import yaml
import json  # For loading strategy parameter JSON files
import os
import logging
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, validator, ValidationError, Field

# --- Pydantic Models for Configuration Structure ---

class LoggingSettings(BaseModel):
    level: str = "INFO"
    directory: str = "logs"
    file_name_prefix: str = "trading_bot_text"
    structured_logging: bool = False
    json_log_file_name_prefix: Optional[str] = "trading_bot_structured"
    max_bytes: int = 10485760
    backup_count: int = 7
    log_format: str = "%(asctime)s - %(app_name)s - [%(levelname)s] - %(name)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s"

class MT5PlatformSettings(BaseModel):
    account_env_var: str
    password_env_var: str
    server_env_var: str
    path: Optional[str] = None
    timeout_ms: int = 10000
    magic_number_default: int = 12345
    slippage_default_points: int = 20

class CTraderPlatformSettings(BaseModel):
    client_id_env_var: str
    client_secret_env_var: str
    account_id_env_var: str
    host_type: str = "demo"

class PlatformSettings(BaseModel):
    name: str = "MetaTrader5"
    mt5: Optional[MT5PlatformSettings] = None
    ctrader: Optional[CTraderPlatformSettings] = None

    @validator('name')
    def name_must_be_supported(cls, v):
        if v not in ["MetaTrader5", "cTrader", "Paper"]:
            raise ValueError("Platform name must be 'MetaTrader5', 'cTrader', or 'Paper'")
        return v

    @validator('ctrader', always=True)
    def check_ctrader_settings(cls, v, values):
        if values.get('name') == "cTrader" and v is None:
            raise ValueError("cTrader settings are required when platform name is 'cTrader'")
        return v

    @validator('mt5', always=True)
    def check_mt5_settings(cls, v, values):
        if values.get('name') == "MetaTrader5" and v is None:
            raise ValueError("MT5 settings are required when platform name is 'MetaTrader5'")
        return v

class AssetStrategyProfile(BaseModel):
    symbol: str
    enabled: bool = True
    instrument_details_key: str
    strategy_params_key: str
    risk_per_trade_idea_pct: Optional[float] = None

    @validator('risk_per_trade_idea_pct', always=True)
    def risk_per_trade_must_be_fractional_or_none(cls, v):
        if v is not None and (v <= 0 or v >= 1):
            raise ValueError('risk_per_trade_idea_pct must be a fraction between 0 and 1 or null')
        return v

class StrategyDefinition(BaseModel):
    strategy_module: str
    strategy_class: str
    description: Optional[str] = ""

class StrategyParameterSet(BaseModel):
    description: Optional[str] = ""
    strategy_definition_key: str
    parameters: Dict[str, Any]

class RiskManagementSettings(BaseModel):
    global_max_account_drawdown_pct: float
    global_daily_drawdown_limit_pct: float
    default_risk_per_trade_idea_pct: float
    max_concurrent_trades_per_strategy_type: int = 3
    max_total_concurrent_trades: int = 5

    @validator('global_max_account_drawdown_pct', 'global_daily_drawdown_limit_pct', 'default_risk_per_trade_idea_pct', pre=True)
    def percentages_must_be_fractional(cls, v, field):
        if v <= 0 or v >= 1:
            raise ValueError(f"{field.name} must be a fraction between 0 and 1")
        return v

    @validator('max_concurrent_trades_per_strategy_type', 'max_total_concurrent_trades')
    def counts_must_be_positive_or_zero(cls, v, field):
        if v < 0:
            raise ValueError(f"'{v}' for field {field.name} must be positive or zero")
        return v

class OperationalComplianceSettings(BaseModel):
    min_trade_duration_seconds: int = 60
    max_orders_per_second: int = 4
    max_total_orders_per_day: int = 1800
    max_order_modifications_per_minute_total: int = 10
    market_close_blackout_period_hours: int = 4
    enforce_weekend_closure: bool = True
    is_swing_account: bool = False

class NewsFilterSettings(BaseModel):
    enabled: bool = True
    api_provider: str = "ForexFactoryJSON"
    api_key_env_var: Optional[str] = None
    ff_json_url: Optional[str] = "https://cdn-nfs.faireconomy.media/ff_calendar_thisweek.json"
    manual_news_file_path: Optional[str] = None
    min_impact_to_consider: str = "High"
    pause_minutes_before_news: int = 3
    pause_minutes_after_news: int = 3
    high_impact_keywords: List[str] = Field(default_factory=list)
    calendar_fetch_interval_seconds: int = 3600

    @validator('api_provider')
    def api_provider_supported(cls, v):
        if v not in ["ForexFactoryJSON", "Finnhub", "EODHD", "JBlanked_News", "Manual"]:
            raise ValueError("Unsupported news API provider")
        return v

    @validator('api_key_env_var', always=True)
    def api_key_required_for_provider(cls, v, values):
        provider = values.get('api_provider')
        if provider and provider not in ["ForexFactoryJSON", "Manual"] and not v:
            raise ValueError(f"api_key_env_var is required for news provider: {provider}")
        return v

    @validator('manual_news_file_path', always=True)
    def manual_file_path_required(cls, v, values):
        if values.get('api_provider') == "Manual" and not v:
            raise ValueError("manual_news_file_path is required when api_provider is 'Manual'")
        return v

class StateManagementSettings(BaseModel):
    persistence_file: str = "state/bot_state.json"
    persistence_interval_seconds: int = 300

class BotSettings(BaseModel):
    trading_mode: str
    main_loop_delay_seconds: int
    app_name: str = "PropFirmAlgoBot"
    ftmo_server_timezone: str = "Europe/Prague"
    magic_number_default: int = 67890

    @validator('trading_mode')
    def trading_mode_supported(cls, v):
        if v not in ["paper", "live"]:
            raise ValueError("trading_mode must be 'paper' or 'live'")
        return v

class AppConfig(BaseModel):
    bot_settings: BotSettings
    logging: LoggingSettings
    platform: PlatformSettings
    assets_to_trade: List[str] = Field(default_factory=list)
    asset_strategy_profiles: Dict[str, AssetStrategyProfile] = Field(default_factory=dict)
    strategy_definitions: Dict[str, StrategyDefinition] = Field(default_factory=dict)
    risk_management: RiskManagementSettings
    operational_compliance: OperationalComplianceSettings
    news_filter: NewsFilterSettings
    state_management: StateManagementSettings
    loaded_strategy_parameters: Dict[str, StrategyParameterSet] = Field(default_factory=dict, exclude=True)
    loaded_instrument_details: Dict[str, Dict[str, Any]] = Field(default_factory=dict, exclude=True)

    class Config:
        extra = 'ignore'
        platform_credentials: Dict[str, Any] = {}
        news_api_key_actual: Optional[str] = None

_config_manager_logger = logging.getLogger(__name__)
if not _config_manager_logger.hasHandlers():
    _config_manager_logger.addHandler(logging.StreamHandler())
    _config_manager_logger.setLevel(logging.INFO)

_config_instance: Optional[AppConfig] = None

def load_and_validate_config(config_dir: str = "config", main_config_filename: str = "main_config.yaml") -> AppConfig:
    global _config_instance
    if _config_instance is not None:
        _config_manager_logger.debug("Returning cached config instance.")
        return _config_instance

    main_config_path = os.path.join(config_dir, main_config_filename)
    _config_manager_logger.info(f"Attempting to load main configuration from: {main_config_path}")

    try:
        with open(main_config_path, 'r', encoding='utf-8') as f:
            raw_config_main = yaml.safe_load(f)
            if raw_config_main is None:
                err_msg = f"Main configuration file '{main_config_path}' is empty or invalid YAML."
                _config_manager_logger.error(err_msg)
                raise ValueError(err_msg)
    except FileNotFoundError:
        _config_manager_logger.error(f"Main configuration file not found at: {main_config_path}")
        raise
    except yaml.YAMLError as e:
        _config_manager_logger.error(f"Error parsing main YAML configuration from {main_config_path}: {e}")
        raise

    try:
        app_config = AppConfig(**raw_config_main)
        _config_manager_logger.info("Main configuration successfully parsed and validated.")

        app_config.Config.platform_credentials = {}
        if app_config.platform.name == "MetaTrader5" and app_config.platform.mt5:
            mt5_cfg = app_config.platform.mt5
            try:
                app_config.Config.platform_credentials['mt5_account'] = os.environ[mt5_cfg.account_env_var]
                app_config.Config.platform_credentials['mt5_password'] = os.environ[mt5_cfg.password_env_var]
                app_config.Config.platform_credentials['mt5_server'] = os.environ[mt5_cfg.server_env_var]
                _config_manager_logger.info("MT5 credentials loaded from environment variables.")
            except KeyError as e:
                err_msg = f"Env var for MT5 credentials not set: {e}. Set {mt5_cfg.account_env_var}, etc."
                _config_manager_logger.error(err_msg)
                raise KeyError(err_msg)
        elif app_config.platform.name == "cTrader" and app_config.platform.ctrader:
            ctrader_cfg = app_config.platform.ctrader
            try:
                app_config.Config.platform_credentials['ctrader_client_id'] = os.environ[ctrader_cfg.client_id_env_var]
                app_config.Config.platform_credentials['ctrader_client_secret'] = os.environ[ctrader_cfg.client_secret_env_var]
                app_config.Config.platform_credentials['ctrader_account_id'] = os.environ[ctrader_cfg.account_id_env_var]
                _config_manager_logger.info("cTrader credentials loaded from environment variables.")
            except KeyError as e:
                err_msg = f"Env var for cTrader credentials not set: {e}."
                _config_manager_logger.error(err_msg)
                raise KeyError(err_msg)

        if app_config.news_filter.enabled and \
           app_config.news_filter.api_provider not in ["ForexFactoryJSON", "Manual"] and \
           app_config.news_filter.api_key_env_var:
            try:
                app_config.Config.news_api_key_actual = os.environ[app_config.news_filter.api_key_env_var]
                _config_manager_logger.info(f"{app_config.news_filter.api_provider} API key loaded.")
            except KeyError as e:
                err_msg = f"Env var for News API key not set: {e}."
                _config_manager_logger.error(err_msg)
                raise KeyError(err_msg)

        app_config.loaded_strategy_parameters = {}
        for profile_key, asset_profile in app_config.asset_strategy_profiles.items():
            if asset_profile.enabled:
                strategy_param_filename = asset_profile.strategy_params_key
                if not strategy_param_filename.endswith(".json"):
                    strategy_param_filename_json = f"{strategy_param_filename}.json"
                else:
                    strategy_param_filename_json = strategy_param_filename

                strategy_param_path = os.path.join(config_dir, strategy_param_filename_json)

                try:
                    with open(strategy_param_path, 'r', encoding='utf-8') as f_strat:
                        raw_strategy_params = json.load(f_strat)
                    validated_params = StrategyParameterSet(**raw_strategy_params)
                    app_config.loaded_strategy_parameters[asset_profile.strategy_params_key] = validated_params
                    _config_manager_logger.info(f"Loaded and validated strategy parameters for '{asset_profile.strategy_params_key}' from '{strategy_param_path}'.")
                except FileNotFoundError:
                    _config_manager_logger.error(f"Strategy parameter file not found: {strategy_param_path} for profile '{profile_key}'.")
                except json.JSONDecodeError as e:
                    _config_manager_logger.error(f"Error decoding JSON from strategy parameter file {strategy_param_path}: {e}")
                except ValidationError as e:
                    _config_manager_logger.error(f"Validation error in strategy parameter file {strategy_param_path}: {e}")
                except Exception as e_strat:
                    _config_manager_logger.error(f"Unexpected error loading strategy parameters from {strategy_param_path}: {e_strat}", exc_info=True)

        instrument_file_name = "instruments_ftmo.json"
        instrument_file_path = os.path.join(config_dir, instrument_file_name)
        app_config.loaded_instrument_details = {}
        try:
            with open(instrument_file_path, 'r', encoding='utf-8') as f_instr:
                raw_instrument_details = json.load(f_instr)
            app_config.loaded_instrument_details = raw_instrument_details
            _config_manager_logger.info(f"Loaded instrument details from '{instrument_file_path}'.")
        except FileNotFoundError:
            _config_manager_logger.warning(f"Instrument details file not found: {instrument_file_path}. Some functionalities like precise pip value might be limited.")
        except json.JSONDecodeError as e:
            _config_manager_logger.error(f"Error decoding JSON from instrument details file {instrument_file_path}: {e}")
        except Exception as e_instr:
            _config_manager_logger.error(f"Unexpected error loading instrument details from {instrument_file_path}: {e_instr}", exc_info=True)

        _config_instance = app_config
        _config_manager_logger.info("Configuration loading and processing complete.")
        return app_config

    except ValidationError as e:
        _config_manager_logger.error(f"Main configuration validation error from {main_config_path}: {e}")
        _config_manager_logger.error(f"Detailed Pydantic errors: {e.errors()}")
        raise
    except Exception as e:
        _config_manager_logger.error(f"An unexpected error occurred during configuration loading from {main_config_path}: {e}", exc_info=True)
        raise

def get_config() -> AppConfig:
    if _config_instance is None:
        _config_manager_logger.info("Config accessed via get_config() before explicit load. Loading with default path.")
        return load_and_validate_config()
    return _config_instance

if __name__ == "__main__":
    if not os.path.exists("config"): os.makedirs("config", exist_ok=True)
    if not os.path.exists("logs"): os.makedirs("logs", exist_ok=True)
    if not os.path.exists("state"): os.makedirs("state", exist_ok=True)

    test_main_config_file = "config/test_main_for_strat_loading.yaml"
    dummy_main_yaml_content = {
        "bot_settings": {"trading_mode": "paper", "main_loop_delay_seconds":10, "app_name": "StratLoadTest", "ftmo_server_timezone": "Europe/Prague"},
        "logging": {"level": "DEBUG", "directory": "logs", "file_name_prefix": "strat_load_test", "structured_logging": False, "max_bytes":1024, "backup_count":1, "log_format":"%(message)s"},
        "platform": {"name": "Paper", "mt5": None, "ctrader": None},
        "assets_to_trade": ["EURUSD_SMA_H1_Profile"],
        "asset_strategy_profiles": {
            "EURUSD_SMA_H1_Profile": {
                "symbol": "EURUSD", "enabled": True,
                "instrument_details_key": "EURUSD_FTMO_Test",
                "strategy_params_key": "strategy_sma_eurusd_h1_test.json"
            }
        },
        "strategy_definitions": {
            "SMACrossDef": {"strategy_module": "prop_firm_trading_bot.src.strategies.trend_following_sma", "strategy_class": "TrendFollowingSMA"}
        },
        "risk_management": {"global_max_account_drawdown_pct": 0.1, "global_daily_drawdown_limit_pct": 0.05, "default_risk_per_trade_idea_pct": 0.01},
        "operational_compliance": {"is_swing_account": False},
        "news_filter": {"enabled": False},
        "state_management": {"persistence_file": "state/strat_load_test_state.json", "persistence_interval_seconds": 300}
    }
    with open(test_main_config_file, "w") as f:
        yaml.dump(dummy_main_yaml_content, f, sort_keys=False)

    dummy_strategy_params_file = "config/strategy_sma_eurusd_h1_test.json"
    dummy_strategy_params_content = {
        "strategy_params_key": "strategy_sma_eurusd_h1_test.json",
        "description": "Test SMA params for EURUSD H1",
        "strategy_definition_key": "SMACrossDef",
        "parameters": {
            "timeframe": "H1",
            "fast_sma_period": 15,
            "slow_sma_period": 45,
            "atr_period_for_sl": 10,
            "atr_multiplier_for_sl": 2.2
        }
    }
    with open(dummy_strategy_params_file, "w") as f:
        json.dump(dummy_strategy_params_content, f, indent=4)

    dummy_instruments_file = "config/instruments_ftmo_test.json"
    dummy_instruments_content = {
        "EURUSD_FTMO_Test": { "platform_symbol": "EURUSD", "pip_value_in_account_currency_per_lot": 10.0 }
    }
    with open(dummy_instruments_file, "w") as f_instr:
        json.dump(dummy_instruments_content, f_instr, indent=4)

    try:
        _config_manager_logger.info("--- Testing ConfigManager with strategy param loading ---")
        app_config = load_and_validate_config(main_config_filename="test_main_for_strat_loading.yaml")

        if app_config.loaded_strategy_parameters:
            _config_manager_logger.info("Successfully loaded strategy parameters:")
            for key, params_set in app_config.loaded_strategy_parameters.items():
                _config_manager_logger.info(f"  Key: {key}, Params: {params_set.parameters}")
        else:
            _config_manager_logger.warning("No strategy parameters were loaded.")

        if app_config.loaded_instrument_details:
            _config_manager_logger.info(f"Instrument details loaded: {app_config.loaded_instrument_details}")
        else:
            _config_manager_logger.warning("Instrument details not loaded (check filename or logic if this is unexpected).")

    except Exception as e:
        _config_manager_logger.error(f"Error during ConfigManager test: {e}", exc_info=True)
    finally:
        _config_manager_logger.info("--- Finished ConfigManager test ---")
