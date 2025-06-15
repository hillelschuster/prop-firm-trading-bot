# This is the tests/unit/test_base_connector.py file.
import pytest
from unittest import mock
import logging
from datetime import datetime, timezone

# Imports from your project
from prop_firm_trading_bot.src.api_connector.base_connector import PlatformInterface, TickCallback, BarCallback, OrderUpdateCallback, PositionUpdateCallback, AccountUpdateCallback, ErrorCallback, MarketEventCallback
from prop_firm_trading_bot.src.core.models import TickData, OHLCVData, Order, Position, AccountInfo, MarketEvent
from prop_firm_trading_bot.src.core.enums import Timeframe, OrderType, OrderAction, OrderStatus, PositionStatus
# We might need a minimal AppConfig mock if the PlatformInterface constructor uses it.
# Based on your base_connector.py, it takes (config, logger).

# --- Fixtures ---

@pytest.fixture
def mock_logger_base_conn(mocker): # Renamed for clarity
    return mocker.MagicMock(spec=logging.Logger)

@pytest.fixture
def mock_config_base_conn(mocker): # Minimal config mock
    return mocker.MagicMock()

# A minimal concrete implementation of PlatformInterface for testing its callback system
class DummyConcreteConnector(PlatformInterface):
    def __init__(self, config, logger):
        super().__init__(config, logger)
        # Dummy implementations for abstract methods
        self.connect_called = False
        self.disconnect_called = False
        # Add more flags or simple implementations as needed by specific tests

    def connect(self) -> bool:
        self.connect_called = True
        return True
    def disconnect(self) -> None:
        self.disconnect_called = True
    def is_connected(self) -> bool: return self.connect_called and not self.disconnect_called
    def get_account_info(self): return None
    def get_symbol_info(self, symbol: str): return None
    def get_all_tradable_symbols(self): return []
    def get_latest_tick(self, symbol: str): return None
    def get_historical_ohlcv(self, symbol: str, timeframe: Timeframe, start_time=None, end_time=None, count=None): return []
    def subscribe_ticks(self, symbol: str, callback: TickCallback) -> bool:
        super().register_tick_subscriber(symbol, callback) # Test base class method
        return True
    def unsubscribe_ticks(self, symbol: str, callback = None) -> bool:
        if callback:
            super().unregister_tick_subscriber(symbol, callback)
        # Simplified: for full test, handle callback=None case to remove all for symbol
        return True
    def subscribe_bars(self, symbol: str, timeframe: Timeframe, callback: BarCallback) -> bool:
        super().register_bar_subscriber(symbol, timeframe, callback)
        return True
    def unsubscribe_bars(self, symbol: str, timeframe: Timeframe, callback = None) -> bool:
        if callback:
            super().unregister_bar_subscriber(symbol, timeframe, callback)
        return True
    def place_order(self, symbol: str, order_type: OrderType, action: OrderAction, volume: float, price=None, stop_loss=None, take_profit=None, client_order_id=None, slippage_points=None, comment=None, expiration_time=None): return None
    def modify_order(self, order_id: str, new_price=None, new_stop_loss=None, new_take_profit=None): return None
    def cancel_order(self, order_id: str): return None
    def get_order_status(self, order_id: str): return None
    def get_open_orders(self, symbol = None): return []
    def get_order_history(self, symbol=None, start_time=None, end_time=None, count=None): return []
    def get_open_positions(self, symbol = None): return []
    def get_position(self, position_id: str): return None
    def close_position(self, position_id: str, volume_to_close=None, price=None, comment=None): return None
    def modify_position_sl_tp(self, position_id: str, stop_loss=None, take_profit=None): return None
    def get_trade_history(self, symbol=None, start_time=None, end_time=None, count=None): return []
    def get_server_time(self): return datetime.now(timezone.utc)


@pytest.fixture
def concrete_connector(mock_config_base_conn, mock_logger_base_conn):
    return DummyConcreteConnector(config=mock_config_base_conn, logger=mock_logger_base_conn)

# --- Test Cases ---

class TestPlatformInterfaceCallbacks:

    def test_register_and_dispatch_tick_callback(self, concrete_connector, mocker):
        mock_callback1 = mocker.MagicMock()
        mock_callback2 = mocker.MagicMock()
        symbol = "EURUSD"
        
        concrete_connector.register_tick_subscriber(symbol, mock_callback1)
        concrete_connector.register_tick_subscriber(symbol, mock_callback2)

        test_tick_data = TickData(timestamp=datetime.now(timezone.utc), symbol=symbol, bid=1.1, ask=1.2)
        concrete_connector._on_tick(test_tick_data)

        mock_callback1.assert_called_once_with(test_tick_data)
        mock_callback2.assert_called_once_with(test_tick_data)

    def test_unregister_tick_callback(self, concrete_connector, mocker):
        mock_callback1 = mocker.MagicMock()
        mock_callback2 = mocker.MagicMock()
        symbol = "EURUSD"

        concrete_connector.register_tick_subscriber(symbol, mock_callback1)
        concrete_connector.register_tick_subscriber(symbol, mock_callback2)
        concrete_connector.unregister_tick_subscriber(symbol, mock_callback1)

        test_tick_data = TickData(timestamp=datetime.now(timezone.utc), symbol=symbol, bid=1.1, ask=1.2)
        concrete_connector._on_tick(test_tick_data)

        mock_callback1.assert_not_called()
        mock_callback2.assert_called_once_with(test_tick_data)

    def test_register_and_dispatch_bar_callback(self, concrete_connector, mocker):
        mock_callback = mocker.MagicMock()
        symbol = "GBPUSD"
        tf = Timeframe.H1
        
        concrete_connector.register_bar_subscriber(symbol, tf, mock_callback)
        test_bar_data = OHLCVData(timestamp=datetime.now(timezone.utc), symbol=symbol, timeframe=tf, open=1.2, high=1.3, low=1.1, close=1.25, volume=100)
        concrete_connector._on_bar(test_bar_data)
        
        mock_callback.assert_called_once_with(test_bar_data)

    def test_dispatch_order_update_callback(self, concrete_connector, mocker):
        mock_callback = mocker.MagicMock()
        concrete_connector.register_order_update_callback(mock_callback)
        
        test_order_data = Order(order_id="123", symbol="EURUSD", order_type=OrderType.MARKET, action=OrderAction.BUY, volume=0.1, status=OrderStatus.FILLED, created_at=datetime.now(timezone.utc))
        concrete_connector._on_order_update(test_order_data)
        
        mock_callback.assert_called_once_with(test_order_data)

    def test_dispatch_position_update_callback(self, concrete_connector, mocker):
        mock_callback = mocker.MagicMock()
        concrete_connector.register_position_update_callback(mock_callback)
        
        test_position_data = Position(position_id="p1", symbol="EURUSD", action=OrderAction.BUY, volume=0.1, open_price=1.1, open_time=datetime.now(timezone.utc))
        concrete_connector._on_position_update(test_position_data)
        
        mock_callback.assert_called_once_with(test_position_data)

    def test_dispatch_account_update_callback(self, concrete_connector, mocker):
        mock_callback = mocker.MagicMock()
        concrete_connector.register_account_update_callback(mock_callback)
        
        test_account_data = AccountInfo(account_id="acc1", balance=1000, equity=1000, margin=0, margin_free=1000, currency="USD")
        concrete_connector._on_account_update(test_account_data)
        
        mock_callback.assert_called_once_with(test_account_data)

    def test_dispatch_error_callback(self, concrete_connector, mocker):
        mock_callback = mocker.MagicMock()
        concrete_connector.register_error_callback(mock_callback)
        
        error_msg = "Test error"
        exception_obj = ValueError("Something went wrong")
        concrete_connector._on_error(error_msg, exception_obj)
        
        mock_callback.assert_called_once_with(error_msg, exception_obj)

    def test_dispatch_market_event_callback(self, concrete_connector, mocker):
        mock_callback = mocker.MagicMock()
        concrete_connector.register_market_event_callback(mock_callback)
        
        test_market_event = MarketEvent(timestamp=datetime.now(timezone.utc), event_type="NEWS_HIGH", description="Test news")
        concrete_connector._on_market_event(test_market_event)
        
        mock_callback.assert_called_once_with(test_market_event)

    def test_error_in_one_callback_does_not_stop_others(self, concrete_connector, mocker, mock_logger_base_conn):
        mock_good_callback1 = mocker.MagicMock()
        mock_bad_callback = mocker.MagicMock(side_effect=RuntimeError("Callback failed"))
        mock_good_callback2 = mocker.MagicMock()
        symbol = "USDJPY"
        
        concrete_connector.register_tick_subscriber(symbol, mock_good_callback1)
        concrete_connector.register_tick_subscriber(symbol, mock_bad_callback)
        concrete_connector.register_tick_subscriber(symbol, mock_good_callback2)

        test_tick_data = TickData(timestamp=datetime.now(timezone.utc), symbol=symbol, bid=150.0, ask=150.02)
        concrete_connector._on_tick(test_tick_data)

        mock_good_callback1.assert_called_once_with(test_tick_data)
        mock_bad_callback.assert_called_once_with(test_tick_data)
        mock_good_callback2.assert_called_once_with(test_tick_data)
        
        # Check that the error from mock_bad_callback was logged
        mock_logger_base_conn.error.assert_any_call(
            f"Error in tick callback for {symbol}: Callback failed", exc_info=True
        )
        # Check that the _on_error dispatcher was also called (which logs via logger.debug if no error_callbacks are registered,
        # or calls error_callbacks which might then log)
        mock_logger_base_conn.debug.assert_any_call(
             f"Dispatching error to subscribers: Tick callback error for {symbol} via {getattr(mock_bad_callback, '__name__', repr(mock_bad_callback))}"
        )


    # Add tests for unregistering other callback types if their logic is more complex
    # Add tests for calling _on_* methods when no callbacks are registered.

  
