import time
from datetime import datetime, timedelta, timezone

import pytest

from prop_firm_trading_bot.src.config_manager import AppConfig, AssetStrategyProfile
from prop_firm_trading_bot.src.risk_controller.risk_controller import RiskController
from prop_firm_trading_bot.src.core.models import AccountInfo, SymbolInfo
from prop_firm_trading_bot.src.core.enums import OrderAction
import logging


class DummyPlatform:
    def __init__(self):
        self.account = AccountInfo(
            account_id="1",
            balance=10000,
            equity=10000,
            margin=0,
            margin_free=10000,
            margin_level_pct=0,
            currency="USD",
            server_time=datetime.utcnow().replace(tzinfo=timezone.utc),
            platform_specific_details={},
        )

    def get_account_info(self):
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
    platform = DummyPlatform()
    news_filter = DummyNewsFilter()
    rc = RiskController(cfg, platform, news_filter, market_data_manager=None, logger=logging.getLogger("test"))
    return rc


def test_calculate_compliant_position_size_basic(risk_controller):
    size = risk_controller.calculate_compliant_position_size(
        "EURUSD", 20, 10000, "TEST"
    )
    assert pytest.approx(size, 0.01) == 0.5


def test_calculate_compliant_position_size_invalid(risk_controller):
    size = risk_controller.calculate_compliant_position_size(
        "EURUSD", -5, 10000, "TEST"
    )
    assert size == 0.0


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
