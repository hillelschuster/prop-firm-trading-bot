# This is the tests/unit/test_mt5_connector.py file.
import pytest
from unittest import mock
from datetime import datetime, timezone, timedelta

# Imports from your project
from prop_firm_trading_bot.src.api_connector.mt5_connector import MT5Adapter
from prop_firm_trading_bot.src.config_manager import AppConfig, PlatformSettings, MT5PlatformSettings, BotSettings
from prop_firm_trading_bot.src.core.models import AccountInfo, SymbolInfo, TickData, OHLCVData, Order, Position
from prop_firm_trading_bot.src.core.enums import Timeframe, OrderType, OrderAction, OrderStatus

# Mock AppConfig for the adapter
@pytest.fixture
def mock_app_config_for_mt5(mocker):
    config_mock = mocker.MagicMock(spec=AppConfig)

    # Mocking nested attributes directly
    config_mock.platform.name = "MetaTrader5"
    config_mock.platform.mt5.path = "C:/Program Files/MetaTrader 5/terminal64.exe"
    config_mock.platform.mt5.timeout_ms = 10000
    config_mock.platform.mt5.magic_number_default = 123456
    config_mock.platform.mt5.slippage_default_points = 10
    
    config_mock.bot_settings.app_name = "TestBot"
    config_mock.bot_settings.ftmo_server_timezone = "Europe/Prague"
    config_mock.bot_settings.magic_number_default = 67890 # Matches your Orchestrator use

    # Mock the nested 'Config' class attribute that holds credentials
    # This needs to be an object that can have attributes assigned.
    # We can't directly mock AppConfig.Config as a class attribute easily with MagicMock instance
    # So, we prepare 'platform_credentials' and 'news_api_key_actual' to be accessed
    # as if they were loaded into a dynamic Config attribute by ConfigManager.
    # The MT5Adapter accesses config.Config.platform_credentials
    
    # Simulate the structure MT5Adapter expects for credentials after ConfigManager processing
    # In MT5Adapter: self.credentials = config.Config.platform_credentials
    # So, config.Config needs to be an object with a 'platform_credentials' attribute.
    
    mock_internal_config_attrs = mocker.MagicMock()
    mock_internal_config_attrs.platform_credentials = {
        'mt5_account': "12345",
        'mt5_password': "password",
        'mt5_server': "TestServer"
    }
    mock_internal_config_attrs.news_api_key_actual = "dummy_news_key"
    config_mock.Config = mock_internal_config_attrs

    return config_mock

@pytest.fixture
def mt5_mock(mocker):
    """Fixture to mock the MetaTrader5 library."""
    # Patch the mt5 library import within the mt5_adapter module
    # This ensures that when MT5Adapter tries to 'import MetaTrader5 as mt5',
    # it gets our mock instead.
    return mocker.patch('prop_firm_trading_bot.src.api_connector.mt5_connector.mt5')

@pytest.fixture
def mock_logger(mocker):
    return mocker.MagicMock()

@pytest.fixture
def mt5_adapter_fix(mock_app_config_for_mt5, mt5_mock, mock_logger):
    adapter = MT5Adapter(config=mock_app_config_for_mt5, logger=mock_logger)
    # Assign the global mt5_mock to the instance's mt5 attribute for easier access in tests
    # if the adapter stores it. If it imports it directly, mt5_mock (from patch) will be used.
    # Adapter code uses 'import MetaTrader5 as mt5', so the patch on mt5_mock fixture handles it.
    return adapter

# --- Test Cases ---

class TestMT5AdapterConnection:
    def test_initialization(self, mt5_adapter_fix, mock_app_config_for_mt5):
        adapter = mt5_adapter_fix
        assert adapter is not None
        assert adapter._is_connected is False
        assert adapter.platform_config == mock_app_config_for_mt5.platform.mt5
        assert adapter.credentials['mt5_account'] == "12345"
        adapter.logger.info.assert_not_called() # No logging on init typically

    def test_connect_successful(self, mt5_adapter_fix, mt5_mock, mock_logger):
        adapter = mt5_adapter_fix
        
        mt5_mock.initialize.return_value = True
        
        mock_term_info = mock.MagicMock()
        mock_term_info.connected = True
        mt5_mock.terminal_info.return_value = mock_term_info
        
        mock_acc_info = mock.MagicMock()
        mock_acc_info.login = 12345
        mock_acc_info.server = "TestServer"
        mt5_mock.account_info.return_value = mock_acc_info

        assert adapter.connect() is True
        assert adapter.is_connected() is True
        mt5_mock.initialize.assert_called_once_with(
            path=adapter.platform_config.path,
            login=int(adapter.credentials['mt5_account']),
            password=adapter.credentials['mt5_password'],
            server=adapter.credentials['mt5_server'],
            timeout=adapter.platform_config.timeout_ms
        )
        mock_logger.info.assert_any_call("Attempting to initialize MT5: Path='C:/Program Files/MetaTrader 5/terminal64.exe', Login='12345', Server='TestServer'")
        mock_logger.info.assert_any_call("MT5 initialized and logged in successfully to account 12345 on server TestServer.")
        assert adapter._polling_active is False # Polling starts if subscriptions exist

    def test_connect_initialize_fails(self, mt5_adapter_fix, mt5_mock, mock_logger):
        adapter = mt5_adapter_fix
        mt5_mock.initialize.return_value = False
        mt5_mock.last_error.return_value = (1001, "Initialization failed")

        assert adapter.connect() is False
        assert adapter.is_connected() is False
        mock_logger.error.assert_called_with("MT5 initialize() failed. Error 1001: Initialization failed")
        # Check if _on_error was called
        mock_logger.debug.assert_any_call("Dispatching error to subscribers: MT5 Connection Failed (1001)")


    def test_connect_login_fails_after_init(self, mt5_adapter_fix, mt5_mock, mock_logger):
        adapter = mt5_adapter_fix
        mt5_mock.initialize.return_value = True
        
        mock_term_info = mock.MagicMock()
        mock_term_info.connected = True # Terminal connected
        mt5_mock.terminal_info.return_value = mock_term_info
        
        mt5_mock.account_info.return_value = None # Login failed
        mt5_mock.last_error.return_value = (1002, "Account info failed")

        assert adapter.connect() is False
        assert adapter.is_connected() is False
        mt5_mock.shutdown.assert_called_once()
        mock_logger.error.assert_called_with("MT5 connected to terminal but not logged into trade account or no account info. Error 1002: Account info failed")
        mock_logger.debug.assert_any_call("Dispatching error to subscribers: MT5 Login/Account Info Failed (1002)")

    def test_disconnect_when_connected(self, mt5_adapter_fix, mt5_mock, mock_logger):
        adapter = mt5_adapter_fix
        adapter._is_connected = True # Simulate connected state
        
        adapter.disconnect()
        
        mt5_mock.shutdown.assert_called_once()
        assert adapter.is_connected() is False
        mock_logger.info.assert_any_call("MT5 connection shut down.")

    def test_disconnect_when_not_connected(self, mt5_adapter_fix, mt5_mock, mock_logger):
        adapter = mt5_adapter_fix
        adapter._is_connected = False
        
        adapter.disconnect()
        
        mt5_mock.shutdown.assert_not_called()
        assert adapter.is_connected() is False
        mock_logger.info.assert_any_call("MT5Adapter was not connected.")

    def test_is_connected_true_and_valid(self, mt5_adapter_fix, mt5_mock):
        adapter = mt5_adapter_fix
        adapter._is_connected = True

        mock_term_info = mock.MagicMock()
        mock_term_info.connected = True
        mt5_mock.terminal_info.return_value = mock_term_info
        
        mock_acc_info = mock.MagicMock()
        mock_acc_info.login = 12345
        mt5_mock.account_info.return_value = mock_acc_info
        
        assert adapter.is_connected() is True

    def test_is_connected_false_if_terminal_disconnects(self, mt5_adapter_fix, mt5_mock, mock_logger):
        adapter = mt5_adapter_fix
        adapter._is_connected = True # Was connected

        mock_term_info = mock.MagicMock()
        mock_term_info.connected = False # Now terminal is not connected
        mt5_mock.terminal_info.return_value = mock_term_info
        
        mock_acc_info = mock.MagicMock() # Account info might still be there but terminal is key
        mock_acc_info.login = 12345
        mt5_mock.account_info.return_value = mock_acc_info
        
        assert adapter.is_connected() is False # Should detect and update
        mock_logger.warning.assert_called_with("MT5 connection check failed: terminal or account not properly connected.")
        mock_logger.debug.assert_any_call("Dispatching error to subscribers: MT5 connection lost (terminal/account status)")


class TestMT5AdapterDataFetching:
    def test_get_account_info_successful(self, mt5_adapter_fix, mt5_mock):
        adapter = mt5_adapter_fix
        adapter._is_connected = True # Assume connected

        mock_mt5_account = mock.MagicMock()
        mock_mt5_account.login = 12345
        mock_mt5_account.balance = 10000.0
        mock_mt5_account.equity = 9500.0
        mock_mt5_account.margin = 500.0
        mock_mt5_account.margin_free = 9000.0
        mock_mt5_account.margin_level = 1900.0
        mock_mt5_account.currency = "USD"
        mock_mt5_account.name = "Test Account"
        mock_mt5_account.server = "TestServer"
        mock_mt5_account.leverage = 100
        mock_mt5_account.trade_mode = 0 # Example
        mt5_mock.account_info.return_value = mock_mt5_account
        
        # Mock terminal_info for server_time
        mock_term_info = mock.MagicMock()
        mock_term_info.time_server = int(datetime.now(timezone.utc).timestamp())
        mt5_mock.terminal_info.return_value = mock_term_info

        acc_info = adapter.get_account_info()

        assert isinstance(acc_info, AccountInfo)
        assert acc_info.account_id == "12345"
        assert acc_info.balance == 10000.0
        assert acc_info.equity == 9500.0
        assert acc_info.currency == "USD"
        assert acc_info.platform_specific_details['leverage'] == 100
        # Check if _on_account_update was called (adapter's logger or a separate callback mock)
        adapter.logger.debug.assert_any_call(f"Dispatching account update to subscribers: Account ID 12345")


    def test_get_account_info_not_connected(self, mt5_adapter_fix, mock_logger):
        adapter = mt5_adapter_fix
        adapter._is_connected = False
        assert adapter.get_account_info() is None
        mock_logger.error.assert_called_with("Cannot get account info: Not connected to MT5.")

    def test_get_symbol_info_successful(self, mt5_adapter_fix, mt5_mock):
        adapter = mt5_adapter_fix
        adapter._is_connected = True
        
        mock_mt5_sym_info = mock.MagicMock()
        mock_mt5_sym_info.name = "EURUSD"
        mock_mt5_sym_info.description = "Euro vs US Dollar"
        mock_mt5_sym_info.digits = 5
        mock_mt5_sym_info.point = 0.00001
        mock_mt5_sym_info.volume_min = 0.01
        mock_mt5_sym_info.volume_max = 100.0
        mock_mt5_sym_info.volume_step = 0.01
        mock_mt5_sym_info.trade_contract_size = 100000.0
        mock_mt5_sym_info.currency_base = "EUR"
        mock_mt5_sym_info.currency_profit = "USD"
        mock_mt5_sym_info.currency_margin = "USD"
        mock_mt5_sym_info.trade_mode = mt5_mock.SYMBOL_TRADE_MODE_FULL # Use the mock's constant
        mock_mt5_sym_info.spread = 2
        mock_mt5_sym_info.bid = 1.08000
        mock_mt5_sym_info.ask = 1.08002
        mt5_mock.symbol_info.return_value = mock_mt5_sym_info
        mt5_mock.symbol_select.return_value = True

        sym_info = adapter.get_symbol_info("EURUSD")

        assert isinstance(sym_info, SymbolInfo)
        assert sym_info.name == "EURUSD"
        assert sym_info.digits == 5
        assert sym_info.min_volume_lots == 0.01
        assert sym_info.max_volume_lots == 100.0
        assert sym_info.volume_step_lots == 0.01
        assert sym_info.trade_allowed is True
        assert sym_info.platform_specific_details['spread'] == 2

    def test_get_latest_tick_successful(self, mt5_adapter_fix, mt5_mock):
        adapter = mt5_adapter_fix
        adapter._is_connected = True

        mock_mt5_tick = mock.MagicMock()
        mock_mt5_tick.time = int(datetime.now(timezone.utc).timestamp())
        mock_mt5_tick.bid = 1.08000
        mock_mt5_tick.ask = 1.08002
        mock_mt5_tick.last = 0.0 # Forex typically no last
        mock_mt5_tick.volume = 0 # Tick volume
        mt5_mock.symbol_info_tick.return_value = mock_mt5_tick

        tick_data = adapter.get_latest_tick("EURUSD")

        assert isinstance(tick_data, TickData)
        assert tick_data.symbol == "EURUSD"
        assert tick_data.bid == 1.08000
        assert tick_data.ask == 1.08002

    def test_get_historical_ohlcv_successful(self, mt5_adapter_fix, mt5_mock):
        adapter = mt5_adapter_fix
        adapter._is_connected = True
        symbol = "EURUSD"
        timeframe = Timeframe.H1
        count = 10

        # Prepare mock rates (NumPy structured array format)
        rates_data = []
        start_dt = datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        for i in range(count):
            ts = int((start_dt + timedelta(hours=i)).timestamp())
            rates_data.append((ts, 1.1+i*0.001, 1.105+i*0.001, 1.095+i*0.001, 1.102+i*0.001, 100+i, 2, 1000+i*10))
        
        # dtype matching mt5.copy_rates_* output
        dt = [('time', '<i8'), ('open', '<f8'), ('high', '<f8'), ('low', '<f8'), 
              ('close', '<f8'), ('tick_volume', '<u8'), ('spread', '<i4'), ('real_volume', '<u8')]
        mock_rates_array = mock.MagicMock() # Simulate numpy array if direct construction is tricky
        mock_rates_array.__iter__.return_value = iter(rates_data) # Make it iterable
        # To properly mock a NumPy array that can be indexed like `rate['time']` etc.,
        # you'd need a more sophisticated mock or actually create a NumPy array.
        # For simplicity here, let's assume the conversion logic in adapter handles dict-like access
        # or we mock the conversion if it's complex.
        # Let's refine to directly return what `_convert_mt5_rates_to_common_ohlcv` expects
        # which means it should look like a list of dicts or structured array elements.
        
        # Simpler: Mock the return of copy_rates_from_pos
        mt5_rates_output = []
        for i in range(count):
            mt5_rates_output.append({
                'time': int((start_dt + timedelta(hours=i)).timestamp()),
                'open': 1.1 + i * 0.001, 'high': 1.105 + i * 0.001,
                'low': 1.095 + i * 0.001, 'close': 1.102 + i * 0.001,
                'tick_volume': 100 + i
            })
        mt5_mock.copy_rates_from_pos.return_value = mt5_rates_output
        mt5_mock.TIMEFRAME_H1 = 16408 # Example MT5 constant value

        ohlcv_list = adapter.get_historical_ohlcv(symbol, timeframe, count=count)

        assert len(ohlcv_list) == count
        assert all(isinstance(bar, OHLCVData) for bar in ohlcv_list)
        assert ohlcv_list[0].symbol == symbol
        assert ohlcv_list[0].timeframe == timeframe
        assert ohlcv_list[0].open == 1.1
        mt5_mock.copy_rates_from_pos.assert_called_once_with(symbol, mt5_mock.TIMEFRAME_H1, 0, count)


class TestMT5AdapterOrderManagement:
    @pytest.fixture
    def mock_symbol_info_for_order(self, mt5_mock):
        # Common symbol info mock for order tests
        symbol_info_raw = mock.MagicMock()
        symbol_info_raw.volume_min = 0.01
        symbol_info_raw.volume_max = 100.0
        symbol_info_raw.volume_step = 0.01
        mt5_mock.symbol_info.return_value = symbol_info_raw
        
        tick = mock.MagicMock()
        tick.bid = 1.08000
        tick.ask = 1.08002
        mt5_mock.symbol_info_tick.return_value = tick
        return symbol_info_raw, tick

    def test_place_market_order_successful(self, mt5_adapter_fix, mt5_mock, mock_symbol_info_for_order, mock_logger):
        adapter = mt5_adapter_fix
        adapter._is_connected = True
        symbol_info_raw, tick = mock_symbol_info_for_order

        order_result_mock = mock.MagicMock()
        order_result_mock.retcode = mt5_mock.TRADE_RETCODE_DONE # Success
        order_result_mock.deal = 777 # Deal ID
        order_result_mock.order = 888 # Order ID that resulted in this deal
        order_result_mock.price = 1.08002 # Fill price
        order_result_mock.volume = 0.1
        order_result_mock.comment = "Market Executed"
        order_result_mock.request.magic = adapter.platform_config.magic_number_default # from request
        mt5_mock.order_send.return_value = order_result_mock

        # Mock history_deals_get to return info about the deal
        deal_info_mock = mock.MagicMock()
        deal_info_mock.ticket = 777
        deal_info_mock.order = 888
        deal_info_mock.time = int(datetime.now(timezone.utc).timestamp())
        deal_info_mock.type = mt5_mock.DEAL_TYPE_BUY # assuming buy
        deal_info_mock.entry = mt5_mock.DEAL_ENTRY_IN
        deal_info_mock.symbol = "EURUSD"
        deal_info_mock.volume = 0.1
        deal_info_mock.price = 1.08002
        deal_info_mock.commission = -0.5
        deal_info_mock.swap = 0.0
        deal_info_mock.profit = 0.0 # Just opened
        deal_info_mock.magic = adapter.platform_config.magic_number_default
        mt5_mock.history_deals_get.return_value = [deal_info_mock]


        order_details = adapter.place_order(
            symbol="EURUSD", order_type=OrderType.MARKET, action=OrderAction.BUY,
            volume=0.1, comment="Test Market Buy"
        )

        assert isinstance(order_details, Order)
        assert order_details.status == OrderStatus.FILLED
        assert order_details.order_id == "888" # From deal_info.order
        assert order_details.filled_price == 1.08002
        assert order_details.filled_volume == 0.1
        assert "Market Executed (Deal: 777)" in order_details.comment
        mt5_mock.order_send.assert_called_once()
        request_arg = mt5_mock.order_send.call_args[0][0]
        assert request_arg['action'] == mt5_mock.TRADE_ACTION_DEAL
        assert request_arg['type'] == mt5_mock.ORDER_TYPE_BUY
        assert request_arg['price'] == tick.ask # Market order uses current ask for buy
        assert request_arg['magic'] == adapter.platform_config.magic_number_default
        adapter.logger.debug.assert_any_call(f"Dispatching order update to subscribers: Order ID {order_details.order_id}")

    def test_place_limit_order_successful(self, mt5_adapter_fix, mt5_mock, mock_symbol_info_for_order, mock_logger):
        adapter = mt5_adapter_fix
        adapter._is_connected = True
        symbol_info_raw, _ = mock_symbol_info_for_order # tick not used for limit price

        order_result_mock = mock.MagicMock()
        order_result_mock.retcode = mt5_mock.TRADE_RETCODE_PLACED # Success for pending
        order_result_mock.order = 999 # Order ID for pending
        order_result_mock.request.magic = adapter.platform_config.magic_number_default
        mt5_mock.order_send.return_value = order_result_mock

        # Mock orders_get for fetching the pending order details
        mt5_order_info_mock = mock.MagicMock()
        mt5_order_info_mock.ticket = 999
        mt5_order_info_mock.symbol = "EURUSD"
        mt5_order_info_mock.type = mt5_mock.ORDER_TYPE_BUY_LIMIT
        mt5_order_info_mock.volume_current = 0.1
        mt5_order_info_mock.price_open = 1.07000
        mt5_order_info_mock.sl = 1.06500
        mt5_order_info_mock.tp = 1.07500
        mt5_order_info_mock.state = mt5_mock.ORDER_STATE_PLACED # Pending
        mt5_order_info_mock.time_setup = int(datetime.now(timezone.utc).timestamp())
        mt5_order_info_mock.time_update = mt5_order_info_mock.time_setup
        mt5_order_info_mock.magic = adapter.platform_config.magic_number_default
        mt5_order_info_mock.comment = "Test Buy Limit"
        mt5_mock.orders_get.return_value = [mt5_order_info_mock]
        mt5_mock.history_orders_get.return_value = [] # Not in history yet


        order_details = adapter.place_order(
            symbol="EURUSD", order_type=OrderType.LIMIT, action=OrderAction.BUY,
            volume=0.1, price=1.07000, stop_loss=1.06500, take_profit=1.07500,
            comment="Test Buy Limit"
        )

        assert isinstance(order_details, Order)
        assert order_details.order_id == "999"
        assert order_details.status == OrderStatus.OPEN # Mapped from PLACED
        assert order_details.price == 1.07000
        mt5_mock.order_send.assert_called_once()
        request_arg = mt5_mock.order_send.call_args[0][0]
        assert request_arg['action'] == mt5_mock.TRADE_ACTION_PENDING
        assert request_arg['type'] == mt5_mock.ORDER_TYPE_BUY_LIMIT
        assert request_arg['price'] == 1.07000
        adapter.logger.debug.assert_any_call(f"Dispatching order update to subscribers: Order ID {order_details.order_id}")


    def test_place_order_fails_retcode(self, mt5_adapter_fix, mt5_mock, mock_symbol_info_for_order, mock_logger):
        adapter = mt5_adapter_fix
        adapter._is_connected = True
        symbol_info_raw, tick = mock_symbol_info_for_order

        order_result_mock = mock.MagicMock()
        order_result_mock.retcode = mt5_mock.TRADE_RETCODE_REJECT # Failure
        order_result_mock.comment = "Rejected by dealer"
        order_result_mock.request.symbol = "EURUSD" # Add request details for logging
        order_result_mock.request.volume = 0.1
        order_result_mock.request.type = mt5_mock.ORDER_TYPE_BUY
        mt5_mock.order_send.return_value = order_result_mock

        order_details = adapter.place_order(
            symbol="EURUSD", order_type=OrderType.MARKET, action=OrderAction.BUY, volume=0.1
        )

        assert isinstance(order_details, Order) # Should return a synthetic REJECTED order
        assert order_details.status == OrderStatus.REJECTED
        assert "MT5 Reject: Rejected by dealer" in order_details.comment
        mock_logger.error.assert_called_with(
            f"Order send for EURUSD not successful. Retcode: {mt5_mock.TRADE_RETCODE_REJECT}, Comment: Rejected by dealer. Request: {order_result_mock.request}"
        )
        adapter.logger.debug.assert_any_call(f"Dispatching order update to subscribers: Order ID {order_details.order_id}")


    def test_close_position_successful(self, mt5_adapter_fix, mt5_mock, mock_logger):
        adapter = mt5_adapter_fix
        adapter._is_connected = True
        position_id_to_close = "12345"

        # Mock positions_get
        pos_info_mock = mock.MagicMock()
        pos_info_mock.ticket = int(position_id_to_close)
        pos_info_mock.symbol = "EURUSD"
        pos_info_mock.volume = 0.1
        pos_info_mock.type = mt5_mock.POSITION_TYPE_BUY # Closing a buy position
        pos_info_mock.magic = 111
        mt5_mock.positions_get.return_value = [pos_info_mock]

        # Mock symbol_info_tick for current price
        tick_mock = mock.MagicMock()
        tick_mock.bid = 1.08500 # Price for selling to close a buy
        tick_mock.ask = 1.08502
        mt5_mock.symbol_info_tick.return_value = tick_mock

        # Mock order_send for closing
        close_order_result_mock = mock.MagicMock()
        close_order_result_mock.retcode = mt5_mock.TRADE_RETCODE_DONE
        close_order_result_mock.deal = 999 # Deal ID for the close
        mt5_mock.order_send.return_value = close_order_result_mock
        
        # Mock history_deals_get for the closing deal
        closing_deal_info_mock = mock.MagicMock()
        closing_deal_info_mock.ticket = 999
        closing_deal_info_mock.order = 777 # The order that executed this closing deal
        closing_deal_info_mock.time = int(datetime.now(timezone.utc).timestamp())
        closing_deal_info_mock.symbol = "EURUSD"
        closing_deal_info_mock.type = mt5_mock.DEAL_TYPE_SELL # Closing a buy position by selling
        closing_deal_info_mock.volume = 0.1
        closing_deal_info_mock.price = 1.08500
        closing_deal_info_mock.commission = -0.5
        closing_deal_info_mock.swap = 0.0
        mt5_mock.history_deals_get.return_value = [closing_deal_info_mock]


        closing_order = adapter.close_position(position_id_to_close, comment="Test Close")

        assert isinstance(closing_order, Order)
        assert closing_order.status == OrderStatus.FILLED
        assert closing_order.action == OrderAction.SELL # Opposite of position
        assert closing_order.volume == 0.1
        assert closing_order.filled_price == 1.08500
        assert f"Close Position {position_id_to_close} (Deal: 999) Test Close" in closing_order.comment
        
        mt5_mock.order_send.assert_called_once()
        request_arg = mt5_mock.order_send.call_args[0][0]
        assert request_arg['action'] == mt5_mock.TRADE_ACTION_DEAL
        assert request_arg['position'] == int(position_id_to_close)
        assert request_arg['type'] == mt5_mock.ORDER_TYPE_SELL # Closing a BUY
        adapter.logger.debug.assert_any_call(f"Dispatching order update to subscribers: Order ID {closing_order.order_id}")

    # Add more tests for:
    # - modify_order (success, failure, order not found)
    # - cancel_order (success, failure, order not found)
    # - get_order_status (active, historical, not found)
    # - get_open_orders
    # - get_order_history
    # - get_open_positions
    # - get_position (found, not found)
    # - modify_position_sl_tp (success, failure, position not found)
    # - get_trade_history
    # - get_server_time
    # - subscribe_ticks / unsubscribe_ticks and _on_tick callback via polling simulation (more complex)
    # - subscribe_bars / unsubscribe_bars and _on_bar callback via polling simulation (more complex)
    # - _convert methods if they have complex logic not covered by testing the public methods
    # - Edge cases for volume rounding in place_order
    # - Different error codes from mt5.order_send and mt5.last_error()
