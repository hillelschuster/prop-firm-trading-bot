import logging
from datetime import datetime, timedelta, timezone

import pandas as pd
import numpy as np

if not hasattr(np, "NaN"):
    np.NaN = np.nan

from prop_firm_trading_bot.src.data_handler.market_data_manager import MarketDataManager
from prop_firm_trading_bot.src.core.enums import Timeframe
from prop_firm_trading_bot.src.core.models import OHLCVData, AccountInfo, SymbolInfo
from prop_firm_trading_bot.src.config_manager import AppConfig, AssetStrategyProfile, StrategyParameterSet


class DummyPlatform:
    def __init__(self):
        self.subscribed_bars = []
        self.subscribed_ticks = []
        self._subscribed_bar_symbols_tf = {}

    def subscribe_ticks(self, symbol, callback):
        self.subscribed_ticks.append(symbol)
        return True

    def subscribe_bars(self, symbol, timeframe, callback):
        self.subscribed_bars.append((symbol, timeframe))
        self._subscribed_bar_symbols_tf.setdefault(symbol, {})[timeframe] = datetime.fromtimestamp(0, tz=timezone.utc)
        return True

    def get_historical_ohlcv(self, symbol, timeframe, count=200):
        base = datetime(2020, 1, 1, tzinfo=timezone.utc)
        data = []
        for i in range(5):
            data.append(
                OHLCVData(
                    timestamp=base + timedelta(minutes=i),
                    symbol=symbol,
                    timeframe=timeframe,
                    open=1 + i * 0.01,
                    high=1.1,
                    low=0.9,
                    close=1,
                    volume=100,
                )
            )
        return data

    def get_account_info(self):
        return AccountInfo(
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


def build_config(tmp_path):
    cfg = AppConfig(
        bot_settings={"trading_mode": "paper", "main_loop_delay_seconds": 1, "app_name": "T", "ftmo_server_timezone": "Europe/Prague"},
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
        risk_management={"global_max_account_drawdown_pct": 0.1, "global_daily_drawdown_limit_pct": 0.05, "default_risk_per_trade_idea_pct": 0.01},
        operational_compliance={"min_trade_duration_seconds": 1, "max_orders_per_second": 10, "max_total_orders_per_day": 100, "max_order_modifications_per_minute_total": 10, "market_close_blackout_period_hours": 1, "enforce_weekend_closure": False, "is_swing_account": True},
        news_filter={"enabled": False},
        state_management={"persistence_file": str(tmp_path / "state.json"), "persistence_interval_seconds": 300},
    )
    cfg.loaded_strategy_parameters = {
        "Test_Params": {
            "parameters": {"timeframe": "M1", "sma_fast": {"length": 3}, "atr": {"length": 3}}
        }
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


def test_get_indicator_params(tmp_path):
    cfg = build_config(tmp_path)
    adapter = DummyPlatform()
    mdm = MarketDataManager(config=cfg, platform_adapter=adapter, logger=logging.getLogger("test"))

    params = mdm._get_indicator_params_for_profile("TEST")
    assert params["sma_fast"]["length"] == 3


def test_initialize_subscriptions(tmp_path):
    cfg = build_config(tmp_path)
    adapter = DummyPlatform()
    mdm = MarketDataManager(config=cfg, platform_adapter=adapter, logger=logging.getLogger("test"))

    assert ("EURUSD", Timeframe.M1) in adapter.subscribed_bars


def test_calculate_and_store_indicators(tmp_path):
    cfg = build_config(tmp_path)
    adapter = DummyPlatform()
    mdm = MarketDataManager(config=cfg, platform_adapter=adapter, logger=logging.getLogger("test"))

    df = mdm.ohlcv_data["EURUSD"][Timeframe.M1]
    assert "SMA_3" in df.columns
    assert "ATR_3" in df.columns
