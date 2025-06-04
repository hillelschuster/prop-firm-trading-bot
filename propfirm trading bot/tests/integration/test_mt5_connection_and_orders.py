# This is the tests/integration/test_mt5_connection_and_orders.py file.
import pytest
import os
import time
from datetime import datetime, timezone, timedelta
import logging

# Imports from your project
from prop_firm_trading_bot.src.api_connector.mt5_adapter import MT5Adapter
from prop_firm_trading_bot.src.config_manager import AppConfig, load_and_validate_config # For loading real config
from prop_firm_trading_bot.src.core.enums import Timeframe, OrderType, OrderAction
from prop_firm_trading_bot.src.core.models import AccountInfo, SymbolInfo, TickData, OHLCVData, Order, Position

# Configure logging for tests (optional, pytest might handle it)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - [%(levelname)s] - %(message)s')

# --- Fixtures ---

@pytest.fixture(scope="module") # Scope to module to connect/disconnect once per test module
def app_config_for_integration():
    """Loads the main configuration for integration tests.
    Ensure your config/main_config.yaml points to a DEMO MT5 account
    and necessary environment variables (MT5_ACCOUNT, MT5_PASSWORD, MT5_SERVER) are set.
    """
    try:
        # Assuming run_bot.py and config are in standard locations relative to tests
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        config_dir = os.path.join(project_root, "config")
        # Ensure environment variables for MT5 are set before this runs!
        # Example:
        # os.environ["MT5_ACCOUNT"] = "your_demo_account"
        # os.environ["MT5_PASSWORD"] = "your_demo_password"
        # os.environ["MT5_SERVER"] = "your_demo_server"
        if not all(k in os.environ for k in ["MT5_ACCOUNT", "MT5_PASSWORD", "MT5_SERVER"]):
            pytest.skip("MT5_ACCOUNT, MT5_PASSWORD, and MT5_SERVER environment variables must be set for MT5 integration tests.")
        
        config = load_and_validate_config(config_dir=config_dir, main_config_filename="main_config.yaml")
        if config.platform.name != "MetaTrader5":
            pytest.skip("main_config.yaml is not configured for MetaTrader5 platform. Skipping MT5 integration tests.")
        return config
    except Exception as e:
        pytest.skip(f"Failed to load configuration for MT5 integration tests: {e}")
    return None


@pytest.fixture(scope="module")
def mt5_adapter_live(app_config_for_integration, request): # request is a pytest fixture
    """Provides a connected MT5Adapter instance for integration tests."""
    if not app_config_for_integration: # If config loading skipped
        pytest.skip("Skipping MT5 adapter fixture due to config load issue.")
        return None

    adapter_logger = logging.getLogger("MT5AdapterIntegrationTest")
    adapter = MT5Adapter(config=app_config_for_integration, logger=adapter_logger)
    
    is_connected = adapter.connect()
    if not is_connected:
        pytest.fail("Failed to connect to MT5 for integration tests. Check credentials, server, and MT5 terminal.")

    def finalizer():
        logger.info("Disconnecting MT5 adapter after integration tests.")
        if adapter and adapter.is_connected():
            adapter.disconnect()
    request.addfinalizer(finalizer) # Ensures disconnect happens after all tests in module
    
    return adapter

# --- Test Cases ---
# These tests will interact with a live (DEMO) MT5 terminal.
# Use with caution and on a non-critical demo account.

@pytest.mark.integration_mt5
class TestMT5AdapterLiveInteraction:

    TEST_SYMBOL = "EURUSD" # A common symbol, ensure it's available on your demo
    SMALL_VOLUME = 0.01

    def test_live_is_connected(self, mt5_adapter_live: MT5Adapter):
        assert mt5_adapter_live is not None
        assert mt5_adapter_live.is_connected() is True

    def test_live_get_account_info(self, mt5_adapter_live: MT5Adapter):
        acc_info = mt5_adapter_live.get_account_info()
        assert acc_info is not None
        assert isinstance(acc_info, AccountInfo)
        assert acc_info.balance > 0
        logger.info(f"Live Account Info: ID {acc_info.account_id}, Bal {acc_info.balance}, Eq {acc_info.equity} {acc_info.currency}")

    def test_live_get_symbol_info(self, mt5_adapter_live: MT5Adapter):
        sym_info = mt5_adapter_live.get_symbol_info(self.TEST_SYMBOL)
        assert sym_info is not None
        assert isinstance(sym_info, SymbolInfo)
        assert sym_info.name == self.TEST_SYMBOL
        logger.info(f"Live Symbol Info for {self.TEST_SYMBOL}: Digits {sym_info.digits}, Point {sym_info.point}")

    def test_live_get_latest_tick(self, mt5_adapter_live: MT5Adapter):
        tick = mt5_adapter_live.get_latest_tick(self.TEST_SYMBOL)
        assert tick is not None
        assert isinstance(tick, TickData)
        assert tick.symbol == self.TEST_SYMBOL
        assert tick.bid > 0 and tick.ask > 0
        logger.info(f"Live Tick for {self.TEST_SYMBOL}: Bid {tick.bid}, Ask {tick.ask} at {tick.timestamp}")

    def test_live_get_historical_ohlcv(self, mt5_adapter_live: MT5Adapter):
        bars = mt5_adapter_live.get_historical_ohlcv(self.TEST_SYMBOL, Timeframe.M1, count=5)
        assert bars is not None
        assert len(bars) == 5
        assert all(isinstance(b, OHLCVData) for b in bars)
        assert bars[0].symbol == self.TEST_SYMBOL
        assert bars[0].timeframe == Timeframe.M1
        logger.info(f"Fetched {len(bars)} M1 bars for {self.TEST_SYMBOL}. First bar close: {bars[0].close}")

    @pytest.mark.flaky(reruns=1, reruns_delay=2) # Market orders can sometimes fail due to price changes
    def test_live_market_order_cycle(self, mt5_adapter_live: MT5Adapter):
        """Tests placing a market order, checking position, then closing it."""
        logger.info(f"Starting market order cycle for {self.TEST_SYMBOL}")
        
        # Ensure no existing positions for this symbol by this bot's magic number
        # This might require enhancing adapter or having a cleanup step
        initial_positions = mt5_adapter_live.get_open_positions(symbol=self.TEST_SYMBOL)
        magic = mt5_adapter_live.platform_config.magic_number_default
        for pos in initial_positions:
            if pos.platform_specific_details.get("magic") == magic:
                logger.warning(f"Closing pre-existing position {pos.position_id} for cleanup.")
                mt5_adapter_live.close_position(pos.position_id)
                time.sleep(1) # Allow time for closure

        # 1. Place Market Buy Order
        buy_order = mt5_adapter_live.place_order(
            symbol=self.TEST_SYMBOL,
            order_type=OrderType.MARKET,
            action=OrderAction.BUY,
            volume=self.SMALL_VOLUME,
            comment="integration_test_market_buy"
        )
        assert buy_order is not None, "Market buy order placement failed"
        assert buy_order.status == OrderStatus.FILLED, f"Market buy order not filled: {buy_order.comment}"
        logger.info(f"Market BUY order placed and filled: {buy_order.order_id}, Deal: {buy_order.platform_specific_details.get('deal_ticket')}")
        
        time.sleep(1) # Allow platform to update positions

        # 2. Check Open Positions
        open_positions = mt5_adapter_live.get_open_positions(symbol=self.TEST_SYMBOL)
        assert open_positions is not None
        
        test_position = next((p for p in open_positions if p.platform_specific_details.get("magic") == magic and \
                                                        p.action == OrderAction.BUY and \
                                                        abs(p.volume - self.SMALL_VOLUME) < 0.001 # Check volume
                                                        ), None) # Compare deal_ticket if available on position?
        assert test_position is not None, f"Position for magic {magic} and BUY not found after market order. Open positions: {open_positions}"
        logger.info(f"Found open position: {test_position.position_id}, Volume: {test_position.volume}, Price: {test_position.open_price}")
        position_id_to_close = test_position.position_id

        # 3. Close the Position
        close_order = mt5_adapter_live.close_position(
            position_id=position_id_to_close,
            comment="integration_test_market_close"
        )
        assert close_order is not None, "Market close order placement failed"
        assert close_order.status == OrderStatus.FILLED, f"Market close order not filled: {close_order.comment}"
        logger.info(f"Market CLOSE order placed and filled: {close_order.order_id}")

        time.sleep(1)
        final_positions = mt5_adapter_live.get_open_positions(symbol=self.TEST_SYMBOL)
        test_position_after_close = next((p for p in final_positions if p.position_id == position_id_to_close), None)
        assert test_position_after_close is None, f"Position {position_id_to_close} still found after attempting to close."
        logger.info(f"Market order cycle for {self.TEST_SYMBOL} completed successfully.")

    # Add more live interaction tests:
    # - Placing, modifying, cancelling pending orders
    # - Modifying SL/TP of open positions
    # - Test with different symbols if available and configured
    # - Test error handling for invalid order parameters (e.g., too small volume, SL too close)
