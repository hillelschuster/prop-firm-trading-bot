# This is the tests/integration/test_mt5_connection_and_orders.py file.
import pytest
import os
import time
from datetime import datetime, timezone, timedelta
import logging

# Imports from your project
from prop_firm_trading_bot.src.api_connector.mt5_adapter import MT5Adapter
from prop_firm_trading_bot.src.config_manager import AppConfig, load_and_validate_config # For loading real config
from prop_firm_trading_bot.src.core.enums import Timeframe, OrderType, OrderAction, OrderStatus
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
@pytest.fixture(scope="function")
def cleanup_orders_positions(mt5_adapter_live: MT5Adapter, request):
    """
    Ensures no open orders or positions for the test symbol associated with the bot's magic number
    exist before and after a test. Uses request.node.cls.TEST_SYMBOL.
    """
    adapter = mt5_adapter_live
    if not adapter or not adapter.is_connected():
        logger.warning("MT5 adapter not available or not connected in cleanup_orders_positions. Skipping cleanup.")
        yield
        return

    try:
        # Access TEST_SYMBOL from the class using the fixture
        test_symbol = request.node.cls.TEST_SYMBOL
    except AttributeError:
        logger.error("cleanup_orders_positions: Could not get TEST_SYMBOL from request.node.cls. Defaulting to 'EURUSD'.")
        test_symbol = "EURUSD" # Fallback

    magic = None
    if hasattr(adapter, 'platform_config') and adapter.platform_config and \
       hasattr(adapter.platform_config, 'magic_number_default'):
        magic = adapter.platform_config.magic_number_default
    else:
        logger.error("cleanup_orders_positions: Cannot determine magic number. Cleanup for specific magic will be skipped.")

    def perform_cleanup(phase="Pre-test"):
        if magic is None:
            logger.warning(f"{phase} cleanup for {test_symbol} skipped: magic number unavailable.")
            return

        logger.info(f"{phase} cleanup started for symbol {test_symbol}, magic {magic}.")

        # Clean up open positions
        try:
            open_positions = adapter.get_open_positions(symbol=test_symbol)
            if open_positions:
                for pos in open_positions:
                    psd = pos.platform_specific_details
                    if isinstance(psd, dict) and psd.get("magic") == magic:
                        logger.warning(f"{phase} cleanup: Closing position {pos.position_id} (vol: {pos.volume}) for {test_symbol}")
                        close_result = adapter.close_position(pos.position_id, volume=pos.volume)
                        if close_result:
                            logger.info(f"{phase} cleanup: Close order {close_result.order_id} status {close_result.status} for pos {pos.position_id}")
                        else:
                            logger.warning(f"{phase} cleanup: Failed to place close order for pos {pos.position_id}")
                        time.sleep(0.5) # Allow processing
        except Exception as e:
            logger.error(f"{phase} cleanup: Error closing positions for {test_symbol}: {e}", exc_info=True)

        # Clean up open orders (pending orders)
        try:
            open_orders = adapter.get_open_orders(symbol=test_symbol)
            if open_orders:
                for order in open_orders:
                    psd = order.platform_specific_details
                    if isinstance(psd, dict) and psd.get("magic") == magic and \
                       order.status in [OrderStatus.OPEN, OrderStatus.PENDING_OPEN]: # Key statuses for active pending orders
                        logger.warning(f"{phase} cleanup: Cancelling pending order {order.order_id} for {test_symbol}")
                        cancel_result = adapter.cancel_order(order.order_id)
                        if cancel_result:
                            logger.info(f"{phase} cleanup: Cancelled order {cancel_result.order_id} status {cancel_result.status}")
                        else:
                            logger.warning(f"{phase} cleanup: Failed to cancel order {order.order_id}")
                        time.sleep(0.5) # Allow processing
        except Exception as e:
            logger.error(f"{phase} cleanup: Error cancelling orders for {test_symbol}: {e}", exc_info=True)
        logger.info(f"{phase} cleanup finished for symbol {test_symbol}, magic {magic}.")

    perform_cleanup(phase="Pre-test")
    yield
    perform_cleanup(phase="Post-test")

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
    def test_live_market_order_cycle(self, mt5_adapter_live: MT5Adapter, cleanup_orders_positions):
        """Tests placing a market order, checking position, then closing it."""
        logger.info(f"Starting market order cycle for {self.TEST_SYMBOL} (with cleanup fixture)")
        magic = mt5_adapter_live.platform_config.magic_number_default
        
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

    @pytest.mark.flaky(reruns=1, reruns_delay=2) # Pending orders can also be affected by market conditions
    def test_live_pending_order_cycle(self, mt5_adapter_live: MT5Adapter, cleanup_orders_positions):
        """Tests placing, modifying, and cancelling a pending order."""
        logger.info(f"Starting pending order cycle for {self.TEST_SYMBOL} (with cleanup fixture)")
        magic = mt5_adapter_live.platform_config.magic_number_default
        
        # 1. Place Pending Buy Limit Order
        symbol_info = mt5_adapter_live.get_symbol_info(self.TEST_SYMBOL)
        assert symbol_info is not None, "Failed to get symbol info"
        current_tick = mt5_adapter_live.get_latest_tick(self.TEST_SYMBOL)
        assert current_tick is not None, "Failed to get latest tick"

        # Place limit order significantly below current market
        limit_price = round(current_tick.ask * 0.95, symbol_info.digits) # 5% below ask
        
        logger.info(f"Attempting to place BUY LIMIT for {self.TEST_SYMBOL} at {limit_price} (current ask: {current_tick.ask})")

        pending_order = mt5_adapter_live.place_order(
            symbol=self.TEST_SYMBOL,
            order_type=OrderType.LIMIT,
            action=OrderAction.BUY,
            volume=self.SMALL_VOLUME,
            price=limit_price,
            comment="integration_test_pending_buy_limit"
        )
        assert pending_order is not None, "Pending order placement failed"
        # MT5 might return FILLED if the price was somehow hittable, or OPEN if placed.
        # For a limit far from market, it should be OPEN.
        assert pending_order.status == OrderStatus.OPEN or pending_order.status == OrderStatus.PENDING_OPEN, \
            f"Pending order not in expected initial state (OPEN/PENDING_OPEN): {pending_order.status}, Comment: {pending_order.comment}"
        
        # If PENDING_OPEN, wait a bit and re-check status
        if pending_order.status == OrderStatus.PENDING_OPEN:
            time.sleep(1)
            pending_order = mt5_adapter_live.get_order_status(pending_order.order_id)
            assert pending_order is not None and pending_order.status == OrderStatus.OPEN, \
                f"Pending order did not transition to OPEN. Current status: {pending_order.status if pending_order else 'Not Found'}"

        logger.info(f"Pending BUY LIMIT order placed: {pending_order.order_id}, Price: {pending_order.price}, Status: {pending_order.status}")
        order_id_to_manage = pending_order.order_id

        # 2. Verify Order Exists
        open_orders = mt5_adapter_live.get_open_orders(symbol=self.TEST_SYMBOL)
        found_order = next((o for o in open_orders if o.order_id == order_id_to_manage), None)
        assert found_order is not None, f"Placed pending order {order_id_to_manage} not found in open orders."
        assert found_order.price == limit_price
        assert found_order.order_type == OrderType.LIMIT
        assert found_order.action == OrderAction.BUY

        # 3. Modify Order Price
        modified_price = round(limit_price * 0.99, symbol_info.digits) # Move further down
        logger.info(f"Modifying pending order {order_id_to_manage} to price {modified_price}")
        modified_order = mt5_adapter_live.modify_order(
            order_id=order_id_to_manage,
            new_price=modified_price
        )
        assert modified_order is not None, "Pending order modification failed"
        # Modification might take a moment to reflect or could be rejected if too close to market / invalid.
        # Assuming it's accepted, status should remain OPEN.
        assert modified_order.status == OrderStatus.OPEN, f"Modified order not OPEN: {modified_order.status}"
        assert abs(modified_order.price - modified_price) < (10**-symbol_info.digits)/2, \
             f"Modified order price {modified_order.price} does not match expected {modified_price}"
        logger.info(f"Pending order modified: {modified_order.order_id}, New Price: {modified_order.price}")

        # 4. Verify Modification
        time.sleep(0.5) # Allow server to update
        verified_modified_order = mt5_adapter_live.get_order_status(order_id_to_manage)
        assert verified_modified_order is not None, "Failed to retrieve modified order status"
        assert abs(verified_modified_order.price - modified_price) < (10**-symbol_info.digits)/2, \
            f"Verified modified order price {verified_modified_order.price} does not match expected {modified_price}"

        # 5. Cancel Order
        logger.info(f"Cancelling pending order {order_id_to_manage}")
        cancelled_order = mt5_adapter_live.cancel_order(order_id_to_manage)
        assert cancelled_order is not None, "Pending order cancellation failed"
        assert cancelled_order.status == OrderStatus.CANCELLED, f"Cancelled order not in CANCELLED state: {cancelled_order.status}"
        logger.info(f"Pending order cancelled: {cancelled_order.order_id}, Status: {cancelled_order.status}")

        # 6. Verify Cancellation
        time.sleep(0.5) # Allow server to update
        final_check_order = mt5_adapter_live.get_order_status(order_id_to_manage)
        assert final_check_order is not None, "Failed to get status of (should be) cancelled order"
        assert final_check_order.status == OrderStatus.CANCELLED, \
            f"Order {order_id_to_manage} not confirmed as CANCELLED. Final status: {final_check_order.status}"
        
        open_orders_after_cancel = mt5_adapter_live.get_open_orders(symbol=self.TEST_SYMBOL)
        found_after_cancel = next((o for o in open_orders_after_cancel if o.order_id == order_id_to_manage), None)
        assert found_after_cancel is None, f"Cancelled order {order_id_to_manage} still found in open orders."
        
        logger.info(f"Pending order cycle for {self.TEST_SYMBOL} completed successfully.")
    @pytest.mark.flaky(reruns=2, reruns_delay=3) # Invalid volume tests can be sensitive
    def test_live_place_invalid_order_volume(self, mt5_adapter_live: MT5Adapter):
        """Tests that the platform correctly rejects an order with a volume below the minimum."""
        logger.info(f"Starting test for placing an invalid order volume for {self.TEST_SYMBOL}")
        
        symbol_info = mt5_adapter_live.get_symbol_info(self.TEST_SYMBOL)
        assert symbol_info is not None, "Failed to get symbol info for invalid volume test."
        assert hasattr(symbol_info, 'min_volume_lots'), "SymbolInfo must have 'min_volume_lots' attribute."
        assert hasattr(symbol_info, 'volume_step_lots'), "SymbolInfo must have 'volume_step_lots' for precise volume calculation."

        min_vol = float(symbol_info.min_volume_lots)
        vol_step = float(symbol_info.volume_step_lots)
        # Ensure lot_digits is an int for round()
        lot_digits_raw = symbol_info.digits_lot if hasattr(symbol_info, 'digits_lot') else (str(vol_step)[::-1].find('.') if '.' in str(vol_step) else 0)
        lot_digits = int(lot_digits_raw)


        # Try to create a volume that is smaller than min_vol but not zero.
        invalid_volume_candidate = round(min_vol / 2, lot_digits + 1) # Use extra precision initially

        if float(invalid_volume_candidate) <= 0: # If min_vol is tiny, half might be zero
            if min_vol > vol_step: # If min_vol is larger than the smallest step, try something between step and min_vol
                invalid_volume_candidate = round(vol_step / 2, lot_digits + 1)
            elif min_vol > 0 : # If min_vol is the smallest step, try even smaller
                 invalid_volume_candidate = round(min_vol / 10, lot_digits + 1) # e.g. 0.0001 if min_vol is 0.001
            else: # min_vol is 0 or less, which is unusual
                pytest.skip(f"min_volume_lots ({min_vol}) is zero or negative, cannot test invalid small volume.")

        # Ensure it's not zero after rounding, and less than min_vol
        invalid_volume = round(float(invalid_volume_candidate), lot_digits) # Round to typical lot digits for sending
        
        if float(invalid_volume) <= 0 and min_vol > 0: # If it rounded to 0, try smallest possible non-zero based on step
            invalid_volume = round(vol_step / 2, lot_digits) # Try half of a step, rounded
            if float(invalid_volume) <= 0: # If even half a step rounds to 0
                 invalid_volume = round(vol_step / 10, lot_digits) # Try a tenth of a step
                 if float(invalid_volume) <= 0: # If still zero, this is problematic
                      pytest.skip(f"Cannot create a non-zero invalid volume. min_vol={min_vol}, vol_step={vol_step}, lot_digits={lot_digits}")
        
        if float(invalid_volume) >= min_vol: # If after all attempts, it's still not smaller
            if min_vol > vol_step: # Only if min_vol is not the smallest step itself
                invalid_volume = round(min_vol - (vol_step / 2), lot_digits)
                if float(invalid_volume) >= min_vol or float(invalid_volume) <= 0:
                     pytest.skip(f"Could not reliably create an invalid volume ({invalid_volume}) smaller than min_volume {min_vol} but positive. vol_step={vol_step}")
            else: # min_vol is likely the smallest step
                 pytest.skip(f"min_volume {min_vol} is likely the smallest step ({vol_step}), difficult to create a smaller valid non-zero volume for this test type.")


        if float(invalid_volume) <= 0:
             pytest.skip(f"Final calculated invalid volume ({invalid_volume}) is not positive. min_vol={min_vol}, vol_step={vol_step}")
        if float(invalid_volume) >= min_vol:
             pytest.skip(f"Final calculated invalid volume ({invalid_volume}) is not smaller than min_volume ({min_vol}).")


        logger.info(f"Attempting to place BUY order for {self.TEST_SYMBOL} with invalid volume: {invalid_volume} (min is {min_vol}, step is {vol_step}, lot_digits: {lot_digits})")
        
        rejected_order = mt5_adapter_live.place_order(
            symbol=self.TEST_SYMBOL,
            order_type=OrderType.MARKET,
            action=OrderAction.BUY,
            volume=float(invalid_volume), # Ensure volume is float
            comment="integration_test_invalid_volume"
        )
        
        assert rejected_order is not None, "place_order should return a synthetic Order object on rejection."
        assert rejected_order.status == OrderStatus.REJECTED, \
            f"Order status was not REJECTED. Status: {rejected_order.status}, " \
            f"Comment: {rejected_order.comment}, " \
            f"Platform Comment: {rejected_order.platform_specific_details.get('comment', 'N/A') if rejected_order.platform_specific_details else 'N/A'}, " \
            f"Platform Retcode: {rejected_order.platform_specific_details.get('retcode', 'N/A') if rejected_order.platform_specific_details else 'N/A'}"
        logger.info(f"Successfully received REJECTED status for invalid volume order. Platform Comment: {rejected_order.platform_specific_details.get('comment', 'N/A') if rejected_order.platform_specific_details else 'N/A'}")

    @pytest.mark.flaky(reruns=2, reruns_delay=10) # SL/TP modifications can be sensitive
    def test_live_modify_open_position_sltp(self, mt5_adapter_live: MT5Adapter, cleanup_orders_positions):
        """Tests placing a market order and then modifying the SL/TP of the resulting position."""
        logger.info(f"Starting SL/TP modification test for {self.TEST_SYMBOL} (with cleanup)")
        magic = mt5_adapter_live.platform_config.magic_number_default
        
        # 1. Place a market order to create a position
        buy_order = mt5_adapter_live.place_order(
            symbol=self.TEST_SYMBOL,
            order_type=OrderType.MARKET,
            action=OrderAction.BUY,
            volume=self.SMALL_VOLUME,
            comment="integration_test_modify_sltp"
        )
        assert buy_order is not None, "Market order placement failed to return an order object."
        assert buy_order.status == OrderStatus.FILLED, f"Market order was not FILLED. Status: {buy_order.status}, Comment: {buy_order.comment}, Platform Retcode: {buy_order.platform_specific_details.get('retcode', 'N/A') if buy_order.platform_specific_details else 'N/A'}"
        logger.info(f"Market BUY order placed successfully: {buy_order.order_id}, Deal: {buy_order.platform_specific_details.get('deal_ticket') if buy_order.platform_specific_details else 'N/A'}")
        
        delay_seconds = getattr(mt5_adapter_live.platform_config, 'api_request_delay_seconds', 0.2) * 15 # Increased delay
        logger.info(f"Waiting {delay_seconds}s for position to appear and stabilize...")
        time.sleep(delay_seconds)

        # 2. Find the open position
        position_ticket_to_find = buy_order.platform_order_id
        
        test_position = mt5_adapter_live.get_position(position_ticket_to_find)
        
        if test_position is None:
            logger.warning(f"Could not find position by order ticket {position_ticket_to_find}, attempting to find by symbol, magic, and volume.")
            open_positions = mt5_adapter_live.get_open_positions(self.TEST_SYMBOL)
            # Find most recent position matching criteria if multiple
            candidate_positions = [
                p for p in open_positions
                if p.platform_specific_details and p.platform_specific_details.get("magic") == magic and \
                    p.symbol == self.TEST_SYMBOL and \
                    abs(p.volume - self.SMALL_VOLUME) < 0.00001 and \
                    p.position_type == OrderAction.BUY
            ]
            if candidate_positions:
                # Sort by open time descending if available, otherwise take first found
                if hasattr(candidate_positions[0], 'open_time_dt') and candidate_positions[0].open_time_dt is not None:
                    candidate_positions.sort(key=lambda p: p.open_time_dt, reverse=True)
                test_position = candidate_positions[0]
                logger.info(f"Found position {test_position.position_id} via fallback search.")
            else:
                 # One last attempt: check if the original buy_order's deal_ticket matches any position's platform_position_id
                deal_ticket_from_order = buy_order.platform_specific_details.get('deal_ticket') if buy_order.platform_specific_details else None
                if deal_ticket_from_order:
                    logger.info(f"Attempting to find position by deal ticket {deal_ticket_from_order} from order {buy_order.order_id}")
                    test_position = mt5_adapter_live.get_position(str(deal_ticket_from_order)) # Ensure it's a string


        assert test_position is not None, f"Failed to find the newly created BUY position for symbol {self.TEST_SYMBOL}, volume {self.SMALL_VOLUME}, magic {magic}. Order ticket: {position_ticket_to_find}. Open positions: {open_positions if 'open_positions' in locals() else 'not fetched by fallback'}"
        logger.info(f"Found position: {test_position.position_id} (Ticket: {test_position.platform_position_id}), Volume: {test_position.volume}, Type: {test_position.position_type}, Open Price: {test_position.open_price}")

        # 3. Modify the position's SL and TP
        tick = mt5_adapter_live.get_latest_tick(self.TEST_SYMBOL)
        assert tick is not None, f"Failed to get latest tick for {self.TEST_SYMBOL}"
        
        symbol_info = mt5_adapter_live.get_symbol_info(self.TEST_SYMBOL)
        assert symbol_info is not None, f"Failed to get symbol info for {self.TEST_SYMBOL}"
        assert hasattr(symbol_info, 'min_stop_level_points'), "SymbolInfo must have 'min_stop_level_points'."
        assert hasattr(symbol_info, 'point'), "SymbolInfo must have 'point'."
        assert hasattr(symbol_info, 'digits'), "SymbolInfo must have 'digits'."

        min_stop_points = symbol_info.min_stop_level_points
        point_val = symbol_info.point
        price_digits = symbol_info.digits

        sl_pips_offset = 50
        tp_pips_offset = 100

        # For BUY position: SL is below open_price, TP is above open_price.
        # SL must be <= current_bid - (min_stop_points * point_val)
        # TP must be >= current_ask + (min_stop_points * point_val)
        
        # Base calculations on position's open price
        pos_open_price = test_position.open_price

        # Desired SL/TP based on open price and pips offset
        desired_sl = round(pos_open_price - sl_pips_offset * point_val, price_digits)
        desired_tp = round(pos_open_price + tp_pips_offset * point_val, price_digits)

        # Minimum distance from current market prices (add a small buffer, e.g., 1-2 points)
        stop_buffer_points = 2
        min_sl_from_market = round(tick.bid - (min_stop_points + stop_buffer_points) * point_val, price_digits)
        min_tp_from_market = round(tick.ask + (min_stop_points + stop_buffer_points) * point_val, price_digits)

        # Adjust SL: must be <= min_sl_from_market AND < pos_open_price
        new_sl = min(desired_sl, min_sl_from_market)
        if new_sl >= pos_open_price: # Ensure SL is strictly below open price for BUY
            new_sl = round(pos_open_price - (min_stop_points + stop_buffer_points) * point_val, price_digits)
            if new_sl >= pos_open_price: # If still not below, make it more aggressive
                 new_sl = round(pos_open_price - (sl_pips_offset / 2) * point_val, price_digits) # Fallback to half offset
            logger.warning(f"Adjusted SL from {desired_sl} to {new_sl} to be below open price {pos_open_price} and respect market limits (bid: {tick.bid}, min_stop_points: {min_stop_points}).")

        # Adjust TP: must be >= min_tp_from_market AND > pos_open_price
        new_tp = max(desired_tp, min_tp_from_market)
        if new_tp <= pos_open_price: # Ensure TP is strictly above open price for BUY
            new_tp = round(pos_open_price + (min_stop_points + stop_buffer_points) * point_val, price_digits)
            if new_tp <= pos_open_price: # If still not above, make it more aggressive
                new_tp = round(pos_open_price + (tp_pips_offset / 2) * point_val, price_digits) # Fallback to half offset
            logger.warning(f"Adjusted TP from {desired_tp} to {new_tp} to be above open price {pos_open_price} and respect market limits (ask: {tick.ask}, min_stop_points: {min_stop_points}).")

        if new_sl == 0 or new_tp == 0:
             pytest.skip(f"Calculated SL ({new_sl}) or TP ({new_tp}) is zero. This is usually invalid. PosOpen: {pos_open_price}, Tick: {tick.bid}/{tick.ask}")
        if new_sl >= new_tp:
            pytest.skip(f"Calculated SL {new_sl} is not less than TP {new_tp}. Market conditions or symbol config might be unfavorable. PosOpen: {pos_open_price}, Tick: {tick.bid}/{tick.ask}")

        logger.info(f"Attempting to modify position {test_position.platform_position_id} (Open: {pos_open_price}) with new SL: {new_sl} and TP: {new_tp}. Current Tick (Bid/Ask): {tick.bid}/{tick.ask}")
        
        modification_successful = mt5_adapter_live.modify_position_sl_tp(
            position_id=test_position.platform_position_id,
            stop_loss=new_sl,
            take_profit=new_tp
        )
        assert modification_successful, f"SL/TP modification call reported failure for position {test_position.platform_position_id}. Attempted SL: {new_sl}, TP: {new_tp}. Platform may have rejected due to proximity or other rules."
        
        logger.info(f"Waiting {delay_seconds}s for SL/TP modification to process...")
        time.sleep(delay_seconds)
        
        verified_position = mt5_adapter_live.get_position(test_position.platform_position_id)
        assert verified_position is not None, f"Failed to retrieve position {test_position.platform_position_id} after modification attempt."
        
        sl_tp_tolerance = point_val * 5 # Allow for a few points difference due to server adjustments/slippage/spread

        assert verified_position.stop_loss is not None and verified_position.stop_loss > 0, \
            f"Stop loss not set or is zero after modification. Got: {verified_position.stop_loss}"
        assert verified_position.take_profit is not None and verified_position.take_profit > 0, \
            f"Take profit not set or is zero after modification. Got: {verified_position.take_profit}"

        # Check if SL/TP are reasonably close to what we sent.
        # MT5 might adjust to specific price steps or if our price was slightly off due to market movement.
        assert verified_position.stop_loss == pytest.approx(new_sl, abs=sl_tp_tolerance), \
            f"SL not set correctly. Expected around: {new_sl}, Got: {verified_position.stop_loss} (Position Open: {verified_position.open_price}, Tick Bid: {tick.bid})"
        assert verified_position.take_profit == pytest.approx(new_tp, abs=sl_tp_tolerance), \
            f"TP not set correctly. Expected around: {new_tp}, Got: {verified_position.take_profit} (Position Open: {verified_position.open_price}, Tick Ask: {tick.ask})"
        
        logger.info(f"Successfully verified SL/TP modification on position {verified_position.platform_position_id}. SL: {verified_position.stop_loss}, TP: {verified_position.take_profit}")
    # Add more live interaction tests:
    # - Placing, modifying, cancelling pending orders
    # - Modifying SL/TP of open positions
    # - Test with different symbols if available and configured
    # - Test error handling for invalid order parameters (e.g., too small volume, SL too close)


  
