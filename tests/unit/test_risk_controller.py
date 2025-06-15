import time
from datetime import datetime, timedelta, timezone
from typing import Optional # Added Optional

import pytest

from prop_firm_trading_bot.src.config_manager import AppConfig, AssetStrategyProfile
from prop_firm_trading_bot.src.risk_controller.risk_controller import RiskController
from prop_firm_trading_bot.src.core.models import AccountInfo, SymbolInfo
from prop_firm_trading_bot.src.core.enums import OrderAction
import logging
import pytz


class DummyPlatform:
    def __init__(self, initial_balance=10000, initial_equity=10000):
        self.account = AccountInfo(
            account_id="1",
            balance=initial_balance,
            equity=initial_equity,
            margin=0,
            margin_free=initial_balance, # Should be based on balance/equity
            margin_level_pct=0,
            currency="USD",
            server_time=datetime.utcnow().replace(tzinfo=timezone.utc),
            platform_specific_details={},
        )

    def get_account_info(self):
        return self.account

    def update_account_info(self, balance: Optional[float] = None, equity: Optional[float] = None):
        if balance is not None:
            self.account.balance = balance
        if equity is not None:
            self.account.equity = equity
        return self.account

    def get_symbol_info(self, symbol):
        return SymbolInfo(
            name=symbol,
            description=symbol,
            digits=5,
            point=0.00001,
            min_volume_lots=0.01,
            max_volume_lots=100,
            volume_step_lots=0.01,
            contract_size=100000,
            currency_base="EUR",
            currency_profit="USD",
            currency_margin="USD",
            trade_allowed=True,
            trade_tick_value=1.0,
            trade_tick_size=0.00001,
            platform_specific_details={},
        )

    def get_open_positions(self):
        return []


def build_config(tmp_path, *, weekend=False, swing=False):
    cfg = AppConfig(
        bot_settings={
            "trading_mode": "paper",
            "main_loop_delay_seconds": 1,
            "app_name": "T",
            "ftmo_server_timezone": "Europe/Prague",
        },
        logging={"level": "DEBUG", "directory": str(tmp_path), "file_name_prefix": "t"},
        platform={"name": "Paper"},
        assets_to_trade=["TEST"],
        asset_strategy_profiles={
            "TEST": AssetStrategyProfile(
                symbol="EURUSD",
                enabled=True,
                instrument_details_key="EURUSD_FTMO",
                strategy_params_key="Test_Params",
            )
        },
        strategy_definitions={},
        risk_management={
            "global_max_account_drawdown_pct": 0.2,
            "global_daily_drawdown_limit_pct": 0.1,
            "default_risk_per_trade_idea_pct": 0.01,
            "max_concurrent_trades_per_strategy_type": 2,
            "max_total_concurrent_trades": 3,
        },
        operational_compliance={
            "min_trade_duration_seconds": 60,
            "max_orders_per_second": 2,
            "max_total_orders_per_day": 5,
            "max_order_modifications_per_minute_total": 10,
            "market_close_blackout_period_hours": 1,
            "enforce_weekend_closure": weekend,
            "is_swing_account": swing,
        },
        news_filter={"enabled": False},
        state_management={"persistence_file": str(tmp_path / "state.json"), "persistence_interval_seconds": 300},
    )
    cfg.loaded_strategy_parameters = {
        "Test_Params": {"parameters": {"timeframe": "M1"}}
    }
    cfg.loaded_instrument_details = {
        "EURUSD_FTMO": {
            "platform_symbol": "EURUSD",
            "pip_value_in_account_currency_per_lot": 10.0,
            "point_value_in_account_currency_per_lot": 1.0,
            "digits": 5,
            "min_volume_lots": 0.01,
            "max_volume_lots": 50.0,
            "volume_step_lots": 0.01,
            "contract_size": 100000,
        }
    }
    return cfg


class DummyNewsFilter:
    def __init__(self, enabled=False, restricted=False):
        class NC:
            pass
        self.news_config = NC()
        self.news_config.enabled = enabled
        self.restricted = restricted

    def _fetch_economic_calendar_with_retry(self):
        # No-op for tests
        return None

    def is_instrument_restricted(self, symbol, current_time_utc=None):
        return self.restricted


@pytest.fixture
def risk_controller(tmp_path):
    cfg = build_config(tmp_path)
    platform = DummyPlatform() # Uses default 10000 balance/equity
    news_filter = DummyNewsFilter()
    # Initialize RiskController, which calls _check_and_perform_daily_reset via constructor
    rc = RiskController(cfg, platform, news_filter, market_data_manager=None, logger=logging.getLogger("test"))
    # rc.initial_challenge_balance should be 10000
    # rc.start_of_day_equity should be 10000
    # rc.initial_balance_today should be 10000
    # rc.todays_closed_pnl should be 0.0
    return rc


def test_calculate_compliant_position_size_basic(risk_controller):
    # This test now implicitly tests that if pip_value_in_account_currency_per_lot is present, it's used.
    size = risk_controller.calculate_compliant_position_size(
        "EURUSD", 20, 10000, "TEST" # TEST profile uses EURUSD_FTMO key
    )
    assert pytest.approx(size, 0.01) == 0.5


def test_calculate_compliant_position_size_invalid_sl_pips(risk_controller):
    # Test for invalid stop_loss_pips (negative)
    size = risk_controller.calculate_compliant_position_size(
        "EURUSD", -5, 10000, "TEST"
    )
    assert size == 0.0


def test_get_symbol_properties_missing_pip_value(risk_controller, tmp_path):
    # Modify config to remove pip_value_in_account_currency_per_lot
    cfg = risk_controller.config
    del cfg.loaded_instrument_details["EURUSD_FTMO"]["pip_value_in_account_currency_per_lot"]

    with pytest.raises(ValueError) as excinfo: # Changed KeyError to ValueError
        risk_controller._get_symbol_properties_for_risk("EURUSD")
    assert "Mandatory configuration 'pip_value_in_account_currency_per_lot' is missing" in str(excinfo.value)
    assert "EURUSD_FTMO" in str(excinfo.value) # Ensure instrument key is mentioned

    # Test that calculate_compliant_position_size also fails as expected
    with pytest.raises(ValueError): # Changed KeyError to ValueError
         risk_controller.calculate_compliant_position_size("EURUSD", 20, 10000, "TEST")


def test_get_symbol_properties_invalid_pip_value_zero(risk_controller, tmp_path):
    cfg = risk_controller.config
    cfg.loaded_instrument_details["EURUSD_FTMO"]["pip_value_in_account_currency_per_lot"] = 0

    with pytest.raises(ValueError) as excinfo:
        risk_controller._get_symbol_properties_for_risk("EURUSD")
    assert "Invalid value for 'pip_value_in_account_currency_per_lot'" in str(excinfo.value)
    assert "Expected a positive number, got: 0" in str(excinfo.value)

    # Test that calculate_compliant_position_size also fails as expected
    with pytest.raises(ValueError):
         risk_controller.calculate_compliant_position_size("EURUSD", 20, 10000, "TEST")


def test_get_symbol_properties_invalid_pip_value_negative(risk_controller, tmp_path):
    cfg = risk_controller.config
    cfg.loaded_instrument_details["EURUSD_FTMO"]["pip_value_in_account_currency_per_lot"] = -10.0

    with pytest.raises(ValueError) as excinfo:
        risk_controller._get_symbol_properties_for_risk("EURUSD")
    assert "Invalid value for 'pip_value_in_account_currency_per_lot'" in str(excinfo.value)
    assert "Expected a positive number, got: -10.0" in str(excinfo.value)


def test_get_symbol_properties_invalid_pip_value_type(risk_controller, tmp_path):
    cfg = risk_controller.config
    cfg.loaded_instrument_details["EURUSD_FTMO"]["pip_value_in_account_currency_per_lot"] = "not-a-number"

    with pytest.raises(ValueError) as excinfo:
        risk_controller._get_symbol_properties_for_risk("EURUSD")
    assert "Invalid value for 'pip_value_in_account_currency_per_lot'" in str(excinfo.value)
    assert "Expected a positive number, got: not-a-number" in str(excinfo.value)


def test_get_symbol_properties_point_value_derivation_if_missing(risk_controller, mocker):
    cfg = risk_controller.config
    # Ensure pip_value is present and valid
    cfg.loaded_instrument_details["EURUSD_FTMO"]["pip_value_in_account_currency_per_lot"] = 10.0
    # Remove point_value to test derivation
    if "point_value_in_account_currency_per_lot" in cfg.loaded_instrument_details["EURUSD_FTMO"]:
        del cfg.loaded_instrument_details["EURUSD_FTMO"]["point_value_in_account_currency_per_lot"]
    # Ensure digits are present for derivation
    cfg.loaded_instrument_details["EURUSD_FTMO"]["digits"] = 5

    # Mock platform_adapter.get_symbol_info in case digits were also missing from config (not this test case)
    mock_get_symbol_info = mocker.patch.object(risk_controller.platform_adapter, 'get_symbol_info')

    props = risk_controller._get_symbol_properties_for_risk("EURUSD")
    
    assert props is not None
    assert props["pip_value_in_account_currency_per_lot"] == 10.0
    assert props.get("point_value_in_account_currency_per_lot") == 1.0 # Derived: 10.0 / 10 for 5 digits
    mock_get_symbol_info.assert_not_called() # Digits were in config, so no live call needed


def test_get_symbol_properties_point_value_derivation_needs_live_digits(risk_controller, mocker):
    cfg = risk_controller.config
    cfg.loaded_instrument_details["EURUSD_FTMO"]["pip_value_in_account_currency_per_lot"] = 10.0
    if "point_value_in_account_currency_per_lot" in cfg.loaded_instrument_details["EURUSD_FTMO"]:
        del cfg.loaded_instrument_details["EURUSD_FTMO"]["point_value_in_account_currency_per_lot"]
    if "digits" in cfg.loaded_instrument_details["EURUSD_FTMO"]:
        del cfg.loaded_instrument_details["EURUSD_FTMO"]["digits"] # Force live fetch for digits

    # Mock platform_adapter.get_symbol_info to return digits
    mock_symbol_info = SymbolInfo(name="EURUSD", description="EURUSD", digits=3, point=0.001, min_volume_lots=0.01, max_volume_lots=100, volume_step_lots=0.01, contract_size=100000, currency_base="EUR", currency_profit="USD", currency_margin="USD", trade_allowed=True, trade_tick_value=1.0, trade_tick_size=0.00001, platform_specific_details={})
    mocker.patch.object(risk_controller.platform_adapter, 'get_symbol_info', return_value=mock_symbol_info)

    props = risk_controller._get_symbol_properties_for_risk("EURUSD")
    
    assert props is not None
    assert props["pip_value_in_account_currency_per_lot"] == 10.0
    assert props.get("point_value_in_account_currency_per_lot") == 1.0 # Derived: 10.0 / 10 for 3 digits
    risk_controller.platform_adapter.get_symbol_info.assert_called_once_with("EURUSD")
    assert props.get("digits") == 3 # Should have been stored back


def test_order_frequency_limits(tmp_path):
    cfg = build_config(tmp_path)
    platform = DummyPlatform()
    news_filter = DummyNewsFilter()
    rc = RiskController(cfg, platform, news_filter, market_data_manager=None, logger=logging.getLogger("test"))

    rc.last_order_timestamp = time.time()
    assert rc._check_order_frequency_limits() is False

    rc.last_order_timestamp = time.time() - 1
    assert rc._check_order_frequency_limits() is True

    rc.daily_order_count = rc.compliance_config.max_total_orders_per_day
    rc.last_order_timestamp = time.time() - 1
    assert rc._check_order_frequency_limits() is False


def test_concurrent_trade_limits(tmp_path):
    cfg = build_config(tmp_path)
    platform = DummyPlatform()
    news_filter = DummyNewsFilter()
    rc = RiskController(cfg, platform, news_filter, market_data_manager=None, logger=logging.getLogger("test"))

    rc.open_trades_count = rc.risk_config.max_total_concurrent_trades
    assert rc._check_concurrent_trade_limits("TEST") is False

    rc.open_trades_count = 0
    rc.open_trades_per_strategy_type["TEST"] = rc.risk_config.max_concurrent_trades_per_strategy_type
    assert rc._check_concurrent_trade_limits("TEST") is False

    rc.open_trades_per_strategy_type["TEST"] = 0
    assert rc._check_concurrent_trade_limits("TEST") is True


def test_should_enforce_weekend_closure(tmp_path):
    cfg = build_config(tmp_path, weekend=True, swing=False)
    platform = DummyPlatform()
    news_filter = DummyNewsFilter()
    rc = RiskController(cfg, platform, news_filter, market_data_manager=None, logger=logging.getLogger("test"))

    friday = datetime(2023, 9, 1, 21, 30, tzinfo=timezone.utc)
    rc._get_current_ftmo_time = lambda: friday
    assert rc.should_enforce_weekend_closure() is True

    rc._get_current_ftmo_time = lambda: friday.replace(hour=20)
    assert rc.should_enforce_weekend_closure() is False

    rc.compliance_config.is_swing_account = True
    rc._get_current_ftmo_time = lambda: friday.replace(hour=22)
    assert rc.should_enforce_weekend_closure() is False


def test_check_min_trade_duration(tmp_path):
    cfg = build_config(tmp_path)
    platform = DummyPlatform()
    news_filter = DummyNewsFilter()
    rc = RiskController(cfg, platform, news_filter, market_data_manager=None, logger=logging.getLogger("test"))

    now = datetime.now(timezone.utc)
    short_time = now - timedelta(seconds=30)
    long_time = now - timedelta(seconds=120)

    assert rc.check_min_trade_duration(short_time) is False
    assert rc.check_min_trade_duration(long_time) is True


def test_validate_trade_proposal_news_block(tmp_path):
    cfg = build_config(tmp_path)
    cfg.operational_compliance.is_swing_account = False
    platform = DummyPlatform()
    news_filter = DummyNewsFilter(enabled=True, restricted=True)
    rc = RiskController(cfg, platform, news_filter, market_data_manager=None, logger=logging.getLogger("test"))

    rc._check_and_perform_daily_reset = lambda force_update=False: None
    rc.check_all_risk_rules = lambda current_equity=None: (True, "")
    rc._check_order_frequency_limits = lambda: True
    rc._check_concurrent_trade_limits = lambda strategy_type_name: True
    rc.calculate_compliant_position_size = lambda *args, **kwargs: 1.0

    allowed, reason, size = rc.validate_trade_proposal(
        "EURUSD", OrderAction.BUY, "TEST", 20, "TEST"
    )
    assert allowed is False
    assert "News restriction" in reason
    assert size is None


def test_daily_reset_logic_for_pnl_tracking(tmp_path):
    cfg = build_config(tmp_path)
    platform = DummyPlatform(initial_balance=10000, initial_equity=10000)
    news_filter = DummyNewsFilter()
    logger = logging.getLogger("test_daily_reset")
    rc = RiskController(cfg, platform, news_filter, market_data_manager=None, logger=logger)

    # Initial state after __init__ which calls _check_and_perform_daily_reset
    assert rc.initial_balance_today == 10000
    assert rc.start_of_day_equity == 10000
    assert rc.todays_closed_pnl == 0.0
    initial_reset_date = rc.date_of_last_daily_reset

    # Simulate some P&L
    rc.todays_closed_pnl = -500

    # Simulate time passing to next day
    # Mock _get_current_ftmo_time to return a time on the next day
    current_ftmo_time_mock = rc._get_current_ftmo_time()
    next_day_ftmo_time = current_ftmo_time_mock + timedelta(days=1)
    
    # Update platform account info for the reset
    platform.update_account_info(balance=9500, equity=9400) # Equity might be different due to open trades

    # Manually trigger reset for "next day"
    original_get_time = rc._get_current_ftmo_time
    rc._get_current_ftmo_time = lambda: next_day_ftmo_time
    
    rc._check_and_perform_daily_reset() # This should trigger the reset
    
    rc._get_current_ftmo_time = original_get_time # Restore mock

    assert rc.date_of_last_daily_reset == next_day_ftmo_time.date()
    assert rc.date_of_last_daily_reset > initial_reset_date
    assert rc.initial_balance_today == 9500 # Updated to new balance
    assert rc.start_of_day_equity == 9400 # Updated to new equity
    assert rc.todays_closed_pnl == 0.0 # Reset for the new day


def test_record_trade_closure_pnl(risk_controller):
    risk_controller.todays_closed_pnl = 0.0 # Ensure starting at 0 for test
    
    risk_controller.record_trade_closure_pnl(100.50)
    assert risk_controller.todays_closed_pnl == 100.50
    
    risk_controller.record_trade_closure_pnl(-50.25)
    assert risk_controller.todays_closed_pnl == 50.25
    
    risk_controller.record_trade_closure_pnl(-200)
    assert risk_controller.todays_closed_pnl == -149.75


def test_ftmo_daily_loss_check_pnl_based(tmp_path):
    cfg = build_config(tmp_path) # daily_drawdown_limit_pct = 0.1 (10%)
    platform = DummyPlatform(initial_balance=10000, initial_equity=10000)
    news_filter = DummyNewsFilter()
    logger = logging.getLogger("test_ftmo_daily_loss")
    rc = RiskController(cfg, platform, news_filter, market_data_manager=None, logger=logger)

    # rc.initial_balance_today is 10000 after init
    # rc.risk_config.global_daily_drawdown_limit_pct is 0.1
    # Allowed daily loss value = 10000 * 0.1 = 1000

    # Scenario 1: No P&L loss, current equity doesn't matter for this specific check
    rc.todays_closed_pnl = 0.0
    can_trade, reason = rc.check_all_risk_rules(current_equity=9500) # Equity drop, but P&L is 0
    assert can_trade is True
    assert "All drawdown rules passed" in reason

    # Scenario 2: P&L loss within limit
    rc.todays_closed_pnl = -500 # Loss of 500
    can_trade, reason = rc.check_all_risk_rules(current_equity=9500)
    assert can_trade is True
    assert "All drawdown rules passed" in reason
    assert rc.trading_halted_due_to_dd is False

    # Scenario 3: P&L loss exactly at limit (should be allowed, > check)
    rc.todays_closed_pnl = -1000 # Loss of 1000
    can_trade, reason = rc.check_all_risk_rules(current_equity=9000)
    assert can_trade is True # (self.todays_closed_pnl * -1) > limit_value
    assert "All drawdown rules passed" in reason
    assert rc.trading_halted_due_to_dd is False

    # Scenario 4: P&L loss breaches limit
    rc.todays_closed_pnl = -1000.01 # Loss of 1000.01
    can_trade, reason = rc.check_all_risk_rules(current_equity=8999.99)
    assert can_trade is False
    assert "FTMO DAILY LOSS LIMIT (P&L BASED) BREACHED" in reason
    assert rc.trading_halted_due_to_dd is True
    
    # Reset halt for next test part
    rc.trading_halted_due_to_dd = False
    rc.todays_closed_pnl = 0 # Reset PNL

    # Scenario 5: Positive P&L, should not trigger loss
    rc.todays_closed_pnl = 200
    can_trade, reason = rc.check_all_risk_rules(current_equity=10200)
    assert can_trade is True
    assert "All drawdown rules passed" in reason
    assert rc.trading_halted_due_to_dd is False

    # Scenario 6: Max overall loss still respected (takes precedence if equity drops significantly)
    # initial_challenge_balance = 10000, global_max_account_drawdown_pct = 0.2
    # max_loss_equity_level = 10000 * (1 - 0.2) = 8000
    rc.todays_closed_pnl = 50 # Small profit today
    # platform.update_account_info(equity=7999) # This update is for the platform, rc fetches fresh
    can_trade, reason = rc.check_all_risk_rules(current_equity=7999) # Simulate equity drop
    assert can_trade is False
    assert "MAX OVERALL LOSS LIMIT BREACHED" in reason
    assert rc.trading_halted_due_to_dd is True


def test_ftmo_daily_loss_check_first_run_warning(tmp_path, caplog):
    cfg = build_config(tmp_path)
    platform = DummyPlatform(initial_balance=10000, initial_equity=10000)
    news_filter = DummyNewsFilter()
    logger = logging.getLogger("test_ftmo_warning")
    rc = RiskController(cfg, platform, news_filter, market_data_manager=None, logger=logger)

    # Simulate state where initial_balance_today might be 0 before first proper daily reset
    # or if account info fetch failed during a reset.
    rc.initial_balance_today = 0.0
    rc.initial_challenge_balance = 10000 # Ensure this is set
    rc.todays_closed_pnl = -10 # A small loss

    with caplog.at_level(logging.WARNING):
        can_trade, reason = rc.check_all_risk_rules(current_equity=9990)
    
    assert "Initial balance today is zero" in caplog.text
    # With initial_balance_today = 0, daily_loss_limit_value = 0.
    # Any negative P&L (e.g., -10) means current_day_loss_from_pnl (10) > 0.
    assert can_trade is False
    assert "FTMO DAILY LOSS LIMIT (P&L BASED) BREACHED" in reason
    assert "Allowed Daily Loss Value: 0.00" in reason # Because initial_balance_today was 0

# salt 2025-06-11T11:27:26
  
