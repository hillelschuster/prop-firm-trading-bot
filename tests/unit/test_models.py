import pytest
from pydantic import ValidationError
from datetime import datetime, timezone
from prop_firm_trading_bot.src.core.models import TickData, OHLCVData, Order, Position, AccountInfo, SymbolInfo, TradeFill, MarketEvent
from prop_firm_trading_bot.src.core.enums import Timeframe, OrderType, OrderAction, OrderStatus, PositionStatus


def test_tick_data_creation():
    now = datetime.now(timezone.utc)
    tick = TickData(timestamp=now, symbol="EURUSD", bid=1.0, ask=1.0002)
    assert tick.symbol == "EURUSD"
    assert tick.bid == 1.0


def test_ohlcv_data_creation():
    now = datetime.now(timezone.utc)
    bar = OHLCVData(timestamp=now, symbol="EURUSD", timeframe=Timeframe.M5, open=1.1, high=1.105, low=1.095, close=1.102, volume=100)
    assert bar.timeframe == Timeframe.M5
    assert bar.open == 1.1


def test_order_creation_required_fields():
    now = datetime.now(timezone.utc)
    with pytest.raises(ValidationError):
        Order(order_id="ord1")

    order = Order(
        order_id="ord1", client_order_id="client_ord_abc", symbol="GBPUSD",
        order_type=OrderType.MARKET, action=OrderAction.BUY, volume=0.1,
        status=OrderStatus.NEW, created_at=now
    )
    assert order.status == OrderStatus.NEW
    assert order.volume == 0.1


def test_position_creation():
    now = datetime.now(timezone.utc)
    pos = Position(
        position_id="pos123", symbol="EURUSD", action=OrderAction.BUY,
        volume=0.5, open_price=1.0800, open_time=now
    )
    assert pos.status == PositionStatus.OPEN
    assert pos.action == OrderAction.BUY


def test_account_info_creation():
    acc = AccountInfo(account_id="acc1", balance=100000, equity=99500, margin=500, margin_free=99000, currency="USD")
    assert acc.currency == "USD"
    assert acc.equity == 99500


def test_symbol_info_creation():
    sym = SymbolInfo(
        name="XAUUSD", digits=2, point=0.01,
        min_volume_lots=0.01, max_volume_lots=10.0, volume_step_lots=0.01,
        contract_size=100, currency_base="XAU", currency_profit="USD", currency_margin="USD"
    )
    assert sym.contract_size == 100
    assert sym.currency_base == "XAU"


def test_trade_fill_creation():
    now = datetime.now(timezone.utc)
    fill = TradeFill(
        fill_id="fill1", order_id="ord1", timestamp=now, symbol="EURUSD",
        action=OrderAction.BUY, volume=0.1, price=1.0850
    )
    assert fill.price == 1.0850


def test_market_event_creation():
    now = datetime.now(timezone.utc)
    event = MarketEvent(
        timestamp=now, event_type="NEWS_HIGH_IMPACT",
        description="FOMC Statement", symbols_affected=["US30.cash", "EURUSD"]
    )
    assert "EURUSD" in event.symbols_affected
    assert event.event_type == "NEWS_HIGH_IMPACT"


  
