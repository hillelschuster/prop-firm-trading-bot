import pytest
from prop_firm_trading_bot.src.core.enums import Timeframe, OrderType, OrderAction, OrderStatus, PositionStatus, BrokerType, StrategySignal


def test_timeframe_to_mt5_string():
    assert Timeframe.M1.to_mt5_timeframe_string() == "TIMEFRAME_M1"
    assert Timeframe.H4.to_mt5_timeframe_string() == "TIMEFRAME_H4"
    with pytest.raises(ValueError):
        Timeframe.TICK.to_mt5_timeframe_string()


def test_timeframe_to_ctrader_string():
    assert Timeframe.M5.to_ctrader_timeframe() == "m5"
    assert Timeframe.D1.to_ctrader_timeframe() == "d1"
    with pytest.raises(ValueError):
        Timeframe.TICK.to_ctrader_timeframe()


def test_timeframe_to_seconds():
    assert Timeframe.M1.to_seconds() == 60
    assert Timeframe.H1.to_seconds() == 3600
    assert Timeframe.TICK.to_seconds() == 0


def test_enum_values_are_strings():
    for enum_cls in [OrderType, OrderAction, OrderStatus, PositionStatus, BrokerType, StrategySignal]:
        for member in enum_cls:
            assert isinstance(member.value, str)


  
