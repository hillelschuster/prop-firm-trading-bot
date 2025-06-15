# This is the tests/unit/test_mt5_connector.py file.
import pytest
from unittest import mock
from datetime import datetime, timezone, timedelta
import pytz

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

    def test_get_account_info_mt5_returns_none(self, mt5_adapter_fix, mt5_mock, mock_logger):
        adapter = mt5_adapter_fix
        adapter._is_connected = True
        mt5_mock.account_info.return_value = None
        mt5_mock.last_error.return_value = (1, "Generic error")

        acc_info = adapter.get_account_info()

        assert acc_info is None
        mock_logger.error.assert_called_with("Failed to retrieve MT5 account info. Error 1: Generic error")
        adapter.logger.debug.assert_any_call("Dispatching error to subscribers: MT5 Get Account Info Failed (1)")


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
        mock_mt5_sym_info.trade_mode = mt5_mock.SYMBOL_TRADE_MODE_FULL
        mock_mt5_sym_info.spread = 2
        mock_mt5_sym_info.bid = 1.08000
        mock_mt5_sym_info.ask = 1.08002
        mt5_mock.symbol_info.return_value = mock_mt5_sym_info
        mt5_mock.symbol_select.return_value = True

        sym_info = adapter.get_symbol_info("EURUSD")
        
        assert isinstance(sym_info, SymbolInfo)
        assert sym_info.name == "EURUSD"
        assert sym_info.digits == 5
        assert sym_info.trade_allowed is True
        assert sym_info.platform_specific_details['spread'] == 2

    def test_get_symbol_info_mt5_returns_none(self, mt5_adapter_fix, mt5_mock, mock_logger):
        adapter = mt5_adapter_fix
        adapter._is_connected = True
        mt5_mock.symbol_info.return_value = None
        mt5_mock.symbol_select.return_value = True # Assume select works
        mt5_mock.last_error.return_value = (2, "Symbol not found error")

        sym_info = adapter.get_symbol_info("INVALID")

        assert sym_info is None
        mock_logger.error.assert_called_with("Failed to retrieve symbol info for INVALID. Error 2: Symbol not found error")
        adapter.logger.debug.assert_any_call("Dispatching error to subscribers: MT5 Get Symbol Info Failed for INVALID (2)")

    def test_get_symbol_info_select_fails(self, mt5_adapter_fix, mt5_mock, mock_logger):
        adapter = mt5_adapter_fix
        adapter._is_connected = True
        mt5_mock.symbol_select.return_value = False # Symbol select fails
        mt5_mock.symbol_info.return_value = None # Consequently, symbol_info might also fail or return None
        mt5_mock.last_error.return_value = (3, "Cannot select symbol")


        with mock.patch('prop_firm_trading_bot.src.api_connector.mt5_connector.time.sleep') as mock_sleep:
            sym_info = adapter.get_symbol_info("UNSELECTABLE")
        
        assert sym_info is None
        mock_logger.warning.assert_any_call("Symbol UNSELECTABLE could not be selected in MarketWatch. Info might be partial or unavailable.")
        # Depending on implementation, it might still call symbol_info. If so, an error for that would also be logged.
        # Based on current adapter code, it proceeds to call symbol_info, so we expect that error too.
        mock_logger.error.assert_any_call("Failed to retrieve symbol info for UNSELECTABLE. Error 3: Cannot select symbol")
        adapter.logger.debug.assert_any_call("Dispatching error to subscribers: MT5 Get Symbol Info Failed for UNSELECTABLE (3)")
        assert mock_sleep.call_count == 1 # It tries to sleep and select again


    def test_get_latest_tick_successful(self, mt5_adapter_fix, mt5_mock, mock_app_config_for_mt5):
        adapter = mt5_adapter_fix
        adapter._is_connected = True
        
        # Ensure ftmo_timezone is set for the adapter instance for correct datetime conversion
        adapter.ftmo_timezone = pytz.timezone(mock_app_config_for_mt5.bot_settings.ftmo_server_timezone)


        mock_mt5_tick = mock.MagicMock()
        now_utc_ts = int(datetime.now(timezone.utc).timestamp())
        mock_mt5_tick.time = now_utc_ts
        mock_mt5_tick.bid = 1.08000
        mock_mt5_tick.ask = 1.08002
        mock_mt5_tick.last = 0.0
        mock_mt5_tick.volume = 10
        mt5_mock.symbol_info_tick.return_value = mock_mt5_tick

        tick_data = adapter.get_latest_tick("EURUSD")

        assert isinstance(tick_data, TickData)
        assert tick_data.symbol == "EURUSD"
        assert tick_data.bid == 1.08000
        assert tick_data.ask == 1.08002
        assert tick_data.volume == 10
        expected_dt = datetime.fromtimestamp(now_utc_ts, tz=timezone.utc).astimezone(adapter.ftmo_timezone)
        assert tick_data.timestamp == expected_dt


    def test_get_latest_tick_mt5_returns_none(self, mt5_adapter_fix, mt5_mock, mock_logger):
        adapter = mt5_adapter_fix
        adapter._is_connected = True
        mt5_mock.symbol_info_tick.return_value = None
        mt5_mock.last_error.return_value = (4, "Tick not available")

        tick_data = adapter.get_latest_tick("EURUSD")

        assert tick_data is None
        mock_logger.warning.assert_called_with("Failed to get latest tick for EURUSD. Error 4: Tick not available")
        # _on_error is not called for individual tick failures in get_latest_tick by design in adapter

    def test_get_historical_ohlcv_successful(self, mt5_adapter_fix, mt5_mock, mock_app_config_for_mt5):
        adapter = mt5_adapter_fix
        adapter._is_connected = True
        adapter.ftmo_timezone = pytz.timezone(mock_app_config_for_mt5.bot_settings.ftmo_server_timezone)
        symbol = "EURUSD"
        timeframe = Timeframe.H1
        count = 10

        rates_data = []
        start_dt_utc = datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        for i in range(count):
            ts_utc = int((start_dt_utc + timedelta(hours=i)).timestamp())
            rates_data.append({ # Simulate dict-like access for structured array elements
                'time': ts_utc,
                'open': 1.1 + i * 0.001, 'high': 1.105 + i * 0.001,
                'low': 1.095 + i * 0.001, 'close': 1.102 + i * 0.001,
                'tick_volume': 100 + i, 'spread': 2, 'real_volume': 1000 + i * 10
            })
        
        mt5_mock.copy_rates_from_pos.return_value = rates_data # Return list of dicts
        mt5_mock.TIMEFRAME_H1 = 16408

        ohlcv_list = adapter.get_historical_ohlcv(symbol, timeframe, count=count)

        assert len(ohlcv_list) == count
        assert all(isinstance(bar, OHLCVData) for bar in ohlcv_list)
        assert ohlcv_list[0].symbol == symbol
        assert ohlcv_list[0].timeframe == timeframe
        assert ohlcv_list[0].open == 1.1
        expected_ts_0 = datetime.fromtimestamp(rates_data[0]['time'], tz=timezone.utc).astimezone(adapter.ftmo_timezone)
        assert ohlcv_list[0].timestamp == expected_ts_0
        mt5_mock.copy_rates_from_pos.assert_called_once_with(symbol, mt5_mock.TIMEFRAME_H1, 0, count)

    def test_get_historical_ohlcv_no_data(self, mt5_adapter_fix, mt5_mock, mock_logger):
        adapter = mt5_adapter_fix
        adapter._is_connected = True
        symbol = "EURUSD"
        timeframe = Timeframe.M1
        mt5_mock.TIMEFRAME_M1 = 1
        mt5_mock.copy_rates_from_pos.return_value = [] # No data
        mt5_mock.last_error.return_value = (0, "No error, just no data")


        ohlcv_list = adapter.get_historical_ohlcv(symbol, timeframe, count=5)

        assert len(ohlcv_list) == 0
        mock_logger.info.assert_called_with(f"No historical data returned for {symbol}/{timeframe.name} for the specified criteria. MT5 Error (if any): 0: No error, just no data")

    def test_get_historical_ohlcv_mt5_error(self, mt5_adapter_fix, mt5_mock, mock_logger):
        adapter = mt5_adapter_fix
        adapter._is_connected = True
        symbol = "EURUSD"
        timeframe = Timeframe.M1
        mt5_mock.TIMEFRAME_M1 = 1
        mt5_mock.copy_rates_from_pos.return_value = None # MT5 error
        mt5_mock.last_error.return_value = (500, "Server error fetching rates")

        ohlcv_list = adapter.get_historical_ohlcv(symbol, timeframe, count=5)

        assert len(ohlcv_list) == 0
        mock_logger.info.assert_called_with(f"No historical data returned for {symbol}/{timeframe.name} for the specified criteria. MT5 Error (if any): 500: Server error fetching rates")
        # _on_error is not called directly by get_historical_ohlcv for "no data" or "None" return from copy_rates

    def test_get_historical_ohlcv_unsupported_timeframe(self, mt5_adapter_fix, mock_logger):
        adapter = mt5_adapter_fix
        adapter._is_connected = True
        
        # Create a dummy Timeframe value not in mt5_timeframe_map
        class UnsupportedTimeframe(Timeframe):
            UNSUPPORTED = "Unsupported"
        
        unsupported_tf = UnsupportedTimeframe.UNSUPPORTED
        
        ohlcv_list = adapter.get_historical_ohlcv("EURUSD", unsupported_tf, count=5)
        assert len(ohlcv_list) == 0
        mock_logger.error.assert_called_with(f"Unsupported timeframe {unsupported_tf} for MT5.")
        adapter.logger.debug.assert_any_call(f"Dispatching error to subscribers: MT5 Get Historical OHLCV: Unsupported timeframe {unsupported_tf}")


    def test_get_open_orders_no_orders(self, mt5_adapter_fix, mt5_mock):
        adapter = mt5_adapter_fix
        adapter._is_connected = True
        mt5_mock.orders_get.return_value = []

        orders = adapter.get_open_orders()
        assert len(orders) == 0
        mt5_mock.orders_get.assert_called_once_with(symbol=None)

    def test_get_open_orders_with_data(self, mt5_adapter_fix, mt5_mock):
        adapter = mt5_adapter_fix
        adapter._is_connected = True

        mt5_order1_raw = mock.MagicMock(
            ticket=101, symbol="EURUSD", type=mt5_mock.ORDER_TYPE_BUY_LIMIT, volume_current=0.1,
            price_open=1.1000, sl=1.0900, tp=1.1100, state=mt5_mock.ORDER_STATE_PLACED,
            time_setup=int(datetime.now(timezone.utc).timestamp()), time_update=int(datetime.now(timezone.utc).timestamp()),
            magic=123, comment="Limit Buy 1"
        )
        mt5_order2_raw = mock.MagicMock(
            ticket=102, symbol="GBPUSD", type=mt5_mock.ORDER_TYPE_SELL_STOP, volume_current=0.2,
            price_open=1.2000, sl=1.2100, tp=1.1900, state=mt5_mock.ORDER_STATE_PLACED,
            time_setup=int(datetime.now(timezone.utc).timestamp()), time_update=int(datetime.now(timezone.utc).timestamp()),
            magic=123, comment="Stop Sell 1"
        )
        mt5_mock.orders_get.return_value = [mt5_order1_raw, mt5_order2_raw]

        orders = adapter.get_open_orders()
        assert len(orders) == 2
        assert orders[0].order_id == "101"
        assert orders[0].symbol == "EURUSD"
        assert orders[0].order_type == OrderType.LIMIT
        assert orders[0].action == OrderAction.BUY
        assert orders[1].order_id == "102"
        assert orders[1].symbol == "GBPUSD"
        assert orders[1].order_type == OrderType.STOP
        assert orders[1].action == OrderAction.SELL
        mt5_mock.orders_get.assert_called_once_with(symbol=None)

    def test_get_open_orders_filtered_by_symbol(self, mt5_adapter_fix, mt5_mock):
        adapter = mt5_adapter_fix
        adapter._is_connected = True
        
        mt5_order_eurusd_raw = mock.MagicMock(ticket=101, symbol="EURUSD", type=mt5_mock.ORDER_TYPE_BUY_LIMIT, state=mt5_mock.ORDER_STATE_PLACED, time_setup=0, time_update=0, magic=0, comment="", volume_current=0.1, price_open=1.1)
        mt5_mock.orders_get.return_value = [mt5_order_eurusd_raw]

        orders = adapter.get_open_orders(symbol="EURUSD")
        assert len(orders) == 1
        assert orders[0].symbol == "EURUSD"
        mt5_mock.orders_get.assert_called_once_with(symbol="EURUSD")

    def test_get_open_positions_no_positions(self, mt5_adapter_fix, mt5_mock):
        adapter = mt5_adapter_fix
        adapter._is_connected = True
        mt5_mock.positions_get.return_value = []

        positions = adapter.get_open_positions()
        assert len(positions) == 0
        mt5_mock.positions_get.assert_called_once_with(symbol=None)

    def test_get_open_positions_with_data(self, mt5_adapter_fix, mt5_mock):
        adapter = mt5_adapter_fix
        adapter._is_connected = True

        mt5_pos1_raw = mock.MagicMock(
            ticket=201, symbol="USDJPY", type=mt5_mock.POSITION_TYPE_BUY, volume=0.5,
            price_open=130.00, price_current=130.50, sl=129.00, tp=131.00,
            time=int(datetime.now(timezone.utc).timestamp()), commission=0, swap=0, profit=25.0,
            magic=456, comment="Buy JPY", identifier=1001
        )
        mt5_pos2_raw = mock.MagicMock(
            ticket=202, symbol="AUDCAD", type=mt5_mock.POSITION_TYPE_SELL, volume=1.0,
            price_open=0.9000, price_current=0.8950, sl=0.9100, tp=0.8900,
            time=int(datetime.now(timezone.utc).timestamp()), commission=-1, swap=-0.5, profit=50.0,
            magic=456, comment="Sell AUDCAD", identifier=1002
        )
        mt5_mock.positions_get.return_value = [mt5_pos1_raw, mt5_pos2_raw]

        positions = adapter.get_open_positions()
        assert len(positions) == 2
        assert positions[0].position_id == "201"
        assert positions[0].symbol == "USDJPY"
        assert positions[0].action == OrderAction.BUY
        assert positions[1].position_id == "202"
        assert positions[1].symbol == "AUDCAD"
        assert positions[1].action == OrderAction.SELL
        mt5_mock.positions_get.assert_called_once_with(symbol=None)
        # Check if _on_position_update was called for each
        adapter.logger.debug.assert_any_call(f"Dispatching position update to subscribers: Position ID 201")
        adapter.logger.debug.assert_any_call(f"Dispatching position update to subscribers: Position ID 202")


    def test_get_open_positions_filtered_by_symbol(self, mt5_adapter_fix, mt5_mock):
        adapter = mt5_adapter_fix
        adapter._is_connected = True
        
        mt5_pos_usdjpy_raw = mock.MagicMock(ticket=201, symbol="USDJPY", type=mt5_mock.POSITION_TYPE_BUY, time=0, magic=0, comment="", volume=0.1, price_open=130.0, price_current=130.5, identifier=0)
        mt5_mock.positions_get.return_value = [mt5_pos_usdjpy_raw]

        positions = adapter.get_open_positions(symbol="USDJPY")
        assert len(positions) == 1
        assert positions[0].symbol == "USDJPY"
        mt5_mock.positions_get.assert_called_once_with(symbol="USDJPY")


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


    @mock.patch('prop_firm_trading_bot.src.api_connector.mt5_connector.uuid.uuid4')
    @mock.patch('prop_firm_trading_bot.src.api_connector.mt5_connector.time.sleep')
    def test_place_order_fails_retcode_reject(self, mock_sleep, mock_uuid4, mt5_adapter_fix, mt5_mock, mock_symbol_info_for_order, mock_logger):
        mock_uuid4.return_value = "mocked-reject-order-id"
        adapter = mt5_adapter_fix
        adapter._is_connected = True
        symbol_info_raw, tick = mock_symbol_info_for_order
        mt5_mock.TRADE_RETCODE_REJECT = 10004 # Ensure constant is available

        order_result_mock = mock.MagicMock()
        order_result_mock.retcode = mt5_mock.TRADE_RETCODE_REJECT
        order_result_mock.comment = "Rejected by dealer"
        order_result_mock.request = mock.MagicMock() # Ensure request is a MagicMock
        order_result_mock.request.symbol = "EURUSD"
        order_result_mock.request.volume = 0.1
        order_result_mock.request.type = mt5_mock.ORDER_TYPE_BUY
        mt5_mock.order_send.return_value = order_result_mock

        order_details = adapter.place_order(
            symbol="EURUSD", order_type=OrderType.MARKET, action=OrderAction.BUY, volume=0.1
        )

        assert isinstance(order_details, Order)
        assert order_details.order_id == "mocked-reject-order-id"
        assert order_details.status == OrderStatus.REJECTED
        assert "MT5 Reject: Rejected by dealer (retcode 10004)" in order_details.comment
        assert order_details.symbol == "EURUSD"
        assert order_details.volume == 0.1
        assert order_details.order_type == OrderType.MARKET
        assert order_details.action == OrderAction.BUY
        mock_logger.error.assert_called_with(
            f"Order send for EURUSD not successful. Retcode: {mt5_mock.TRADE_RETCODE_REJECT}, Comment: Rejected by dealer. Request: {order_result_mock.request}"
        )
        adapter.logger.debug.assert_any_call(f"Dispatching order update to subscribers: Order ID {order_details.order_id}")
        mock_sleep.assert_not_called() # Sleep is after successful send, before fetching details

    @mock.patch('prop_firm_trading_bot.src.api_connector.mt5_connector.uuid.uuid4')
    @mock.patch('prop_firm_trading_bot.src.api_connector.mt5_connector.time.sleep')
    def test_place_order_fails_retcode_no_money(self, mock_sleep, mock_uuid4, mt5_adapter_fix, mt5_mock, mock_symbol_info_for_order, mock_logger):
        mock_uuid4.return_value = "mocked-nomoney-order-id"
        adapter = mt5_adapter_fix
        adapter._is_connected = True
        symbol_info_raw, tick = mock_symbol_info_for_order
        mt5_mock.TRADE_RETCODE_NO_MONEY = 10019

        order_result_mock = mock.MagicMock()
        order_result_mock.retcode = mt5_mock.TRADE_RETCODE_NO_MONEY
        order_result_mock.comment = "Not enough money"
        order_result_mock.request = mock.MagicMock()
        order_result_mock.request.symbol = "EURUSD"
        order_result_mock.request.volume = 0.1
        order_result_mock.request.type = mt5_mock.ORDER_TYPE_BUY
        mt5_mock.order_send.return_value = order_result_mock

        order_details = adapter.place_order(
            symbol="EURUSD", order_type=OrderType.MARKET, action=OrderAction.BUY, volume=0.1
        )

        assert isinstance(order_details, Order)
        assert order_details.order_id == "mocked-nomoney-order-id"
        assert order_details.status == OrderStatus.REJECTED
        assert "MT5 Reject: Not enough money (retcode 10019)" in order_details.comment
        assert order_details.symbol == "EURUSD"
        assert order_details.volume == 0.1
        assert order_details.order_type == OrderType.MARKET
        assert order_details.action == OrderAction.BUY
        mock_logger.error.assert_called_with(
            f"Order send for EURUSD not successful. Retcode: {mt5_mock.TRADE_RETCODE_NO_MONEY}, Comment: Not enough money. Request: {order_result_mock.request}"
        )
        adapter.logger.debug.assert_any_call(f"Dispatching order update to subscribers: Order ID {order_details.order_id}")

    @mock.patch('prop_firm_trading_bot.src.api_connector.mt5_connector.uuid.uuid4')
    @mock.patch('prop_firm_trading_bot.src.api_connector.mt5_connector.time.sleep')
    def test_place_order_fails_retcode_connection(self, mock_sleep, mock_uuid4, mt5_adapter_fix, mt5_mock, mock_symbol_info_for_order, mock_logger):
        mock_uuid4.return_value = "mocked-connection-order-id"
        adapter = mt5_adapter_fix
        adapter._is_connected = True
        symbol_info_raw, tick = mock_symbol_info_for_order
        mt5_mock.TRADE_RETCODE_CONNECTION = 10012

        order_result_mock = mock.MagicMock()
        order_result_mock.retcode = mt5_mock.TRADE_RETCODE_CONNECTION
        order_result_mock.comment = "No connection to trade server"
        order_result_mock.request = mock.MagicMock()
        order_result_mock.request.symbol = "EURUSD"
        order_result_mock.request.volume = 0.1
        order_result_mock.request.type = mt5_mock.ORDER_TYPE_BUY
        mt5_mock.order_send.return_value = order_result_mock

        order_details = adapter.place_order(
            symbol="EURUSD", order_type=OrderType.MARKET, action=OrderAction.BUY, volume=0.1
        )
        assert isinstance(order_details, Order)
        assert order_details.order_id == "mocked-connection-order-id"
        assert order_details.status == OrderStatus.REJECTED
        assert "MT5 Reject: No connection to trade server (retcode 10012)" in order_details.comment
        assert order_details.symbol == "EURUSD"
        assert order_details.volume == 0.1
        assert order_details.order_type == OrderType.MARKET
        assert order_details.action == OrderAction.BUY
        mock_logger.error.assert_called_with(
            f"Order send for EURUSD not successful. Retcode: {mt5_mock.TRADE_RETCODE_CONNECTION}, Comment: No connection to trade server. Request: {order_result_mock.request}"
        )
        adapter.logger.debug.assert_any_call(f"Dispatching order update to subscribers: Order ID {order_details.order_id}")

    @mock.patch('prop_firm_trading_bot.src.api_connector.mt5_connector.uuid.uuid4')
    @mock.patch('prop_firm_trading_bot.src.api_connector.mt5_connector.time.sleep')
    def test_place_order_fails_retcode_invalid_stops(self, mock_sleep, mock_uuid4, mt5_adapter_fix, mt5_mock, mock_symbol_info_for_order, mock_logger):
        mock_uuid4.return_value = "mocked-stops-order-id"
        adapter = mt5_adapter_fix
        adapter._is_connected = True
        symbol_info_raw, tick = mock_symbol_info_for_order
        mt5_mock.TRADE_RETCODE_INVALID_STOPS = 10016

        order_result_mock = mock.MagicMock()
        order_result_mock.retcode = mt5_mock.TRADE_RETCODE_INVALID_STOPS
        order_result_mock.comment = "Invalid stops"
        order_result_mock.request = mock.MagicMock()
        order_result_mock.request.symbol = "EURUSD"
        order_result_mock.request.volume = 0.1
        order_result_mock.request.type = mt5_mock.ORDER_TYPE_BUY
        mt5_mock.order_send.return_value = order_result_mock

        order_details = adapter.place_order(
            symbol="EURUSD", order_type=OrderType.MARKET, action=OrderAction.BUY, volume=0.1,
            stop_loss=1.00000, take_profit=2.00000 # Example SL/TP that might be invalid
        )
        assert isinstance(order_details, Order)
        assert order_details.order_id == "mocked-stops-order-id"
        assert order_details.status == OrderStatus.REJECTED
        assert "MT5 Reject: Invalid stops (retcode 10016)" in order_details.comment
        assert order_details.symbol == "EURUSD"
        assert order_details.volume == 0.1
        assert order_details.order_type == OrderType.MARKET
        assert order_details.action == OrderAction.BUY
        mock_logger.error.assert_called_with(
            f"Order send for EURUSD not successful. Retcode: {mt5_mock.TRADE_RETCODE_INVALID_STOPS}, Comment: Invalid stops. Request: {order_result_mock.request}"
        )
        adapter.logger.debug.assert_any_call(f"Dispatching order update to subscribers: Order ID {order_details.order_id}")

    @mock.patch('prop_firm_trading_bot.src.api_connector.mt5_connector.time.sleep')
    def test_place_order_order_send_returns_none(self, mock_sleep, mt5_adapter_fix, mt5_mock, mock_symbol_info_for_order, mock_logger):
        adapter = mt5_adapter_fix
        adapter._is_connected = True
        symbol_info_raw, tick = mock_symbol_info_for_order
        mt5_mock.order_send.return_value = None
        mt5_mock.last_error.return_value = (10000, "Some internal error") # Example error

        order_details = adapter.place_order(
            symbol="EURUSD", order_type=OrderType.MARKET, action=OrderAction.BUY, volume=0.1
        )

        assert order_details is None
        mock_logger.error.assert_called_with(
            f"Order send call failed for EURUSD (returned None). MT5 Error 10000: Some internal error. Request: {mt5_mock.order_send.call_args[0][0]}"
        )
        mock_logger.debug.assert_any_call("Dispatching error to subscribers: MT5 Place Order Failed (10000) for EURUSD")
        mock_sleep.assert_not_called()

    @mock.patch('prop_firm_trading_bot.src.api_connector.mt5_connector.time.sleep')
    def test_modify_order_successful(self, mock_sleep, mt5_adapter_fix, mt5_mock, mock_logger):
        adapter = mt5_adapter_fix
        adapter._is_connected = True
        order_id = "12345"
        new_price = 1.07500
        new_sl = 1.07000
        new_tp = 1.08000

        # Mock existing order
        existing_order_mt5 = mock.MagicMock()
        existing_order_mt5.ticket = int(order_id)
        existing_order_mt5.symbol = "EURUSD"
        existing_order_mt5.price_open = 1.07400
        existing_order_mt5.sl = 1.06900
        existing_order_mt5.tp = 1.07900
        existing_order_mt5.type_time = mt5_mock.ORDER_TIME_GTC
        existing_order_mt5.time_expiration = 0
        existing_order_mt5.type = mt5_mock.ORDER_TYPE_BUY_LIMIT
        existing_order_mt5.volume_current = 0.1
        existing_order_mt5.state = mt5_mock.ORDER_STATE_PLACED
        existing_order_mt5.time_setup = int(datetime.now(timezone.utc).timestamp())
        existing_order_mt5.time_update = existing_order_mt5.time_setup
        existing_order_mt5.magic = 12345
        existing_order_mt5.comment = "Original"

        mt5_mock.orders_get.side_effect = [
            [existing_order_mt5], # First call for fetching existing
            [mock.MagicMock( # Second call after modification
                ticket=int(order_id), symbol="EURUSD", price_open=new_price, sl=new_sl, tp=new_tp,
                type=mt5_mock.ORDER_TYPE_BUY_LIMIT, volume_current=0.1, state=mt5_mock.ORDER_STATE_PLACED,
                time_setup=existing_order_mt5.time_setup, time_update=int(datetime.now(timezone.utc).timestamp()),
                magic=12345, comment="Original" # Comment usually not changed by modify
            )]
        ]

        # Mock order_send result for modification
        modify_result_mock = mock.MagicMock()
        modify_result_mock.retcode = mt5_mock.TRADE_RETCODE_DONE
        modify_result_mock.order = int(order_id)
        mt5_mock.order_send.return_value = modify_result_mock

        modified_order = adapter.modify_order(order_id, new_price=new_price, new_stop_loss=new_sl, new_take_profit=new_tp)

        assert modified_order is not None
        assert modified_order.order_id == order_id
        assert modified_order.price == new_price
        assert modified_order.stop_loss == new_sl
        assert modified_order.take_profit == new_tp
        mt5_mock.order_send.assert_called_once()
        request_arg = mt5_mock.order_send.call_args[0][0]
        assert request_arg['action'] == mt5_mock.TRADE_ACTION_MODIFY
        assert request_arg['order'] == int(order_id)
        assert request_arg['price'] == new_price
        assert request_arg['sl'] == new_sl
        assert request_arg['tp'] == new_tp
        mock_logger.info.assert_any_call(f"Order {order_id} modified successfully. Ticket: {int(order_id)}, Retcode: {mt5_mock.TRADE_RETCODE_DONE}")
        adapter.logger.debug.assert_any_call(f"Dispatching order update to subscribers: Order ID {order_id}")
        mock_sleep.assert_called_once_with(0.2)

    @mock.patch('prop_firm_trading_bot.src.api_connector.mt5_connector.time.sleep')
    def test_modify_order_not_found(self, mock_sleep, mt5_adapter_fix, mt5_mock, mock_logger):
        adapter = mt5_adapter_fix
        adapter._is_connected = True
        order_id = "nonexistent"
        mt5_mock.orders_get.return_value = [] # Order not found

        modified_order = adapter.modify_order(order_id, new_price=1.1)

        assert modified_order is None
        mock_logger.error.assert_called_with(f"Cannot modify order {order_id}: Order not found in active orders.")
        adapter.logger.debug.assert_any_call(f"Dispatching error to subscribers: MT5 Modify Order: Order {order_id} not found")
        mt5_mock.order_send.assert_not_called()

    @mock.patch('prop_firm_trading_bot.src.api_connector.mt5_connector.time.sleep')
    def test_modify_order_fails_retcode(self, mock_sleep, mt5_adapter_fix, mt5_mock, mock_logger):
        adapter = mt5_adapter_fix
        adapter._is_connected = True
        order_id = "12345"
        mt5_mock.TRADE_RETCODE_INVALID_PRICE = 10015 # Example failure retcode

        existing_order_mt5 = mock.MagicMock(ticket=int(order_id), symbol="EURUSD", price_open=1.07400, sl=0, tp=0, type_time=mt5_mock.ORDER_TIME_GTC, time_expiration=0)
        
        # Mock for get_order_status fallback
        failed_order_status_mock = mock.MagicMock(spec=Order, status=OrderStatus.OPEN, order_id=order_id)
        
        # Side effect for orders_get: first for initial fetch, second for get_order_status
        mt5_mock.orders_get.side_effect = [
            [existing_order_mt5],
            [existing_order_mt5] # For the get_order_status call
        ]
        
        modify_result_mock = mock.MagicMock()
        modify_result_mock.retcode = mt5_mock.TRADE_RETCODE_INVALID_PRICE
        modify_result_mock.comment = "Invalid price for modification"
        modify_result_mock.request = mock.MagicMock() # Ensure request is a MagicMock
        mt5_mock.order_send.return_value = modify_result_mock
        
        # Mock the adapter's get_order_status to control its return value
        with mock.patch.object(adapter, 'get_order_status', return_value=failed_order_status_mock) as mock_get_status:
            returned_order = adapter.modify_order(order_id, new_price=0.1) # Invalid price

        assert returned_order == failed_order_status_mock
        mock_logger.error.assert_called_with(
            f"Failed to modify order {order_id}. MT5 Error/RetCode {mt5_mock.TRADE_RETCODE_INVALID_PRICE}: Invalid price for modification. Request: {modify_result_mock.request}"
        )
        adapter.logger.debug.assert_any_call(f"Dispatching error to subscribers: MT5 Modify Order Failed ({mt5_mock.TRADE_RETCODE_INVALID_PRICE}) for {order_id}")
        mock_get_status.assert_called_once_with(order_id)


    @mock.patch('prop_firm_trading_bot.src.api_connector.mt5_connector.time.sleep')
    def test_cancel_order_successful(self, mock_sleep, mt5_adapter_fix, mt5_mock, mock_logger):
        adapter = mt5_adapter_fix
        adapter._is_connected = True
        order_id = "67890"

        existing_order_mt5 = mock.MagicMock(ticket=int(order_id), symbol="GBPUSD")
        mt5_mock.orders_get.return_value = [existing_order_mt5]

        cancel_result_mock = mock.MagicMock()
        cancel_result_mock.retcode = mt5_mock.TRADE_RETCODE_DONE
        cancel_result_mock.order = int(order_id)
        mt5_mock.order_send.return_value = cancel_result_mock

        # Mock history_orders_get for the cancelled order
        cancelled_order_mt5 = mock.MagicMock(
            ticket=int(order_id), symbol="GBPUSD", state=mt5_mock.ORDER_STATE_CANCELED,
            type=mt5_mock.ORDER_TYPE_SELL_LIMIT, volume_current=0.05, price_open=1.2500,
            time_setup=int(datetime.now(timezone.utc).timestamp()), time_update=int(datetime.now(timezone.utc).timestamp()),
            magic=123, comment="Cancelled by user"
        )
        mt5_mock.history_orders_get.return_value = [cancelled_order_mt5]

        cancelled_order = adapter.cancel_order(order_id)

        assert cancelled_order is not None
        assert cancelled_order.order_id == order_id
        assert cancelled_order.status == OrderStatus.CANCELLED
        mt5_mock.order_send.assert_called_once()
        request_arg = mt5_mock.order_send.call_args[0][0]
        assert request_arg['action'] == mt5_mock.TRADE_ACTION_REMOVE
        assert request_arg['order'] == int(order_id)
        mock_logger.info.assert_any_call(f"Order {order_id} cancelled successfully. Ticket: {int(order_id)}, Retcode: {mt5_mock.TRADE_RETCODE_DONE}")
        adapter.logger.debug.assert_any_call(f"Dispatching order update to subscribers: Order ID {order_id}")
        mock_sleep.assert_called_once_with(0.2)

    @mock.patch('prop_firm_trading_bot.src.api_connector.mt5_connector.time.sleep')
    def test_cancel_order_not_found(self, mock_sleep, mt5_adapter_fix, mt5_mock, mock_logger):
        adapter = mt5_adapter_fix
        adapter._is_connected = True
        order_id = "nonexistent"
        mt5_mock.orders_get.return_value = []
        mt5_mock.history_orders_get.return_value = []

        cancelled_order = adapter.cancel_order(order_id)

        assert cancelled_order is None
        mock_logger.warning.assert_called_with(f"Cannot cancel order {order_id}: Order not found in active orders. It might have been filled or already cancelled.")
        mt5_mock.order_send.assert_not_called()

    @mock.patch('prop_firm_trading_bot.src.api_connector.mt5_connector.time.sleep')
    def test_cancel_order_fails_retcode(self, mock_sleep, mt5_adapter_fix, mt5_mock, mock_logger):
        adapter = mt5_adapter_fix
        adapter._is_connected = True
        order_id = "67890"
        mt5_mock.TRADE_RETCODE_INVALID_REQUEST = 10013 # Example failure

        existing_order_mt5 = mock.MagicMock(ticket=int(order_id), symbol="GBPUSD")
        mt5_mock.orders_get.return_value = [existing_order_mt5]
        
        cancel_result_mock = mock.MagicMock()
        cancel_result_mock.retcode = mt5_mock.TRADE_RETCODE_INVALID_REQUEST
        cancel_result_mock.comment = "Invalid request for cancel"
        cancel_result_mock.request = mock.MagicMock()
        mt5_mock.order_send.return_value = cancel_result_mock

        # Mock for get_order_status fallback
        failed_order_status_mock = mock.MagicMock(spec=Order, status=OrderStatus.OPEN, order_id=order_id)
        with mock.patch.object(adapter, 'get_order_status', return_value=failed_order_status_mock) as mock_get_status:
            returned_order = adapter.cancel_order(order_id)
        
        assert returned_order == failed_order_status_mock
        mock_logger.error.assert_called_with(
            f"Failed to cancel order {order_id}. MT5 Error/RetCode {mt5_mock.TRADE_RETCODE_INVALID_REQUEST}: Invalid request for cancel. Request: {cancel_result_mock.request}"
        )
        adapter.logger.debug.assert_any_call(f"Dispatching error to subscribers: MT5 Cancel Order Failed ({mt5_mock.TRADE_RETCODE_INVALID_REQUEST}) for {order_id}")
        mock_get_status.assert_called_once_with(order_id)


    @mock.patch('prop_firm_trading_bot.src.api_connector.mt5_connector.time.sleep')
    def test_modify_position_sl_tp_successful(self, mock_sleep, mt5_adapter_fix, mt5_mock, mock_logger):
        adapter = mt5_adapter_fix
        adapter._is_connected = True
        position_id = "55555"
        new_sl = 1.20000
        new_tp = 1.30000

        existing_pos_mt5 = mock.MagicMock(
            ticket=int(position_id), symbol="AUDCAD", type=mt5_mock.POSITION_TYPE_BUY,
            volume=0.5, price_open=1.25000, sl=0, tp=0, time=int(datetime.now(timezone.utc).timestamp()),
            price_current=1.25100, commission=0, swap=0, profit=0, magic=123, identifier=444,
            time_msc=0, time_update_msc=0, comment="Open Pos"
        )
        
        updated_pos_mt5 = mock.MagicMock(
            ticket=int(position_id), symbol="AUDCAD", type=mt5_mock.POSITION_TYPE_BUY,
            volume=0.5, price_open=1.25000, sl=new_sl, tp=new_tp, time=existing_pos_mt5.time,
            price_current=1.25100, commission=0, swap=0, profit=0, magic=123, identifier=444,
            time_msc=0, time_update_msc=int(datetime.now(timezone.utc).timestamp() * 1000), comment="Open Pos"
        )

        mt5_mock.positions_get.side_effect = [[existing_pos_mt5], [updated_pos_mt5]]

        sltp_result_mock = mock.MagicMock()
        sltp_result_mock.retcode = mt5_mock.TRADE_RETCODE_DONE
        sltp_result_mock.order = 0 # Often 0 for SLTP actions
        mt5_mock.order_send.return_value = sltp_result_mock

        modified_position = adapter.modify_position_sl_tp(position_id, stop_loss=new_sl, take_profit=new_tp)

        assert modified_position is not None
        assert modified_position.position_id == position_id
        assert modified_position.stop_loss == new_sl
        assert modified_position.take_profit == new_tp
        mt5_mock.order_send.assert_called_once()
        request_arg = mt5_mock.order_send.call_args[0][0]
        assert request_arg['action'] == mt5_mock.TRADE_ACTION_SLTP
        assert request_arg['position'] == int(position_id)
        assert request_arg['symbol'] == "AUDCAD"
        assert request_arg['sl'] == new_sl
        assert request_arg['tp'] == new_tp
        mock_logger.info.assert_any_call(f"SL/TP for position {position_id} modified successfully. Request ID (if any): 0, Retcode: {mt5_mock.TRADE_RETCODE_DONE}")
        adapter.logger.debug.assert_any_call(f"Dispatching position update to subscribers: Position ID {position_id}")
        mock_sleep.assert_called_once_with(0.2)

    @mock.patch('prop_firm_trading_bot.src.api_connector.mt5_connector.time.sleep')
    def test_modify_position_sl_tp_not_found(self, mock_sleep, mt5_adapter_fix, mt5_mock, mock_logger):
        adapter = mt5_adapter_fix
        adapter._is_connected = True
        position_id = "nonexistent_pos"
        mt5_mock.positions_get.return_value = []

        modified_position = adapter.modify_position_sl_tp(position_id, new_sl=1.1)

        assert modified_position is None
        mock_logger.error.assert_called_with(f"Position {position_id} not found for SL/TP modification.")
        adapter.logger.debug.assert_any_call(f"Dispatching error to subscribers: MT5 Modify Position SL/TP: Position {position_id} not found")
        mt5_mock.order_send.assert_not_called()

    @mock.patch('prop_firm_trading_bot.src.api_connector.mt5_connector.time.sleep')
    def test_modify_position_sl_tp_fails_retcode(self, mock_sleep, mt5_adapter_fix, mt5_mock, mock_logger):
        adapter = mt5_adapter_fix
        adapter._is_connected = True
        position_id = "55555"
        mt5_mock.TRADE_RETCODE_INVALID_STOPS = 10016

        existing_pos_mt5 = mock.MagicMock(ticket=int(position_id), symbol="AUDCAD", sl=0, tp=0)
        
        # Mock for get_position fallback
        failed_pos_status_mock = mock.MagicMock(spec=Position, position_id=position_id, stop_loss=0, take_profit=0)

        mt5_mock.positions_get.side_effect = [
            [existing_pos_mt5], # Initial fetch
            [existing_pos_mt5]  # For get_position fallback
        ]
        
        sltp_result_mock = mock.MagicMock()
        sltp_result_mock.retcode = mt5_mock.TRADE_RETCODE_INVALID_STOPS
        sltp_result_mock.comment = "Invalid SL/TP values"
        sltp_result_mock.request = mock.MagicMock()
        mt5_mock.order_send.return_value = sltp_result_mock

        with mock.patch.object(adapter, 'get_position', return_value=failed_pos_status_mock) as mock_get_pos:
            returned_position = adapter.modify_position_sl_tp(position_id, stop_loss=0.1, take_profit=0.1) # Invalid values

        assert returned_position == failed_pos_status_mock
        mock_logger.error.assert_called_with(
            f"Failed to modify SL/TP for position {position_id}. MT5 Error/RetCode {mt5_mock.TRADE_RETCODE_INVALID_STOPS}: Invalid SL/TP values. Request: {sltp_result_mock.request}"
        )
        adapter.logger.debug.assert_any_call(f"Dispatching error to subscribers: MT5 Modify Position SL/TP Failed ({mt5_mock.TRADE_RETCODE_INVALID_STOPS}) for {position_id}")
        mock_get_pos.assert_called_once_with(position_id)


    @mock.patch('prop_firm_trading_bot.src.api_connector.mt5_connector.time.sleep')
    def test_close_position_successful(self, mock_sleep, mt5_adapter_fix, mt5_mock, mock_logger):
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
        mock_sleep.assert_called_once_with(0.2)

    # Add more tests for:
    # - get_order_status (active, historical, not found)
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


  
