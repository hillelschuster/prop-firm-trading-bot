import unittest
from unittest.mock import MagicMock, patch, call
from typing import Optional # Added Optional
import logging
import uuid
from datetime import datetime

# Assuming correct relative imports based on project structure
# If 'src' is a top-level package recognized by Python path:
from prop_firm_trading_bot.src.execution.order_execution_manager import OrderExecutionManager
from prop_firm_trading_bot.src.core.enums import StrategySignal, OrderAction, OrderType, OrderStatus
from prop_firm_trading_bot.src.core.models import Order, Position
from prop_firm_trading_bot.src.api_connector.base_connector import PlatformInterface
# from prop_firm_trading_bot.src.config_manager import AppConfig # Mocked
from prop_firm_trading_bot.src.risk_controller.risk_controller import RiskController # For mocking

class TestOrderExecutionManager(unittest.TestCase):

    def setUp(self):
        self.mock_config = MagicMock()
        # Mocking platform config part, e.g., self.mock_config.platform.mt5.slippage_default_points
        self.mock_config.platform.mt5.slippage_default_points = 20
        self.mock_config.asset_strategy_profiles = {
            "EURUSD_Profile": MagicMock(symbol="EURUSD")
        }

        self.mock_platform_adapter = MagicMock(spec=PlatformInterface)
        self.mock_logger = MagicMock(spec=logging.Logger)
        self.mock_risk_controller = MagicMock(spec=RiskController)
        
        self.manager = OrderExecutionManager(
            config=self.mock_config,
            platform_adapter=self.mock_platform_adapter,
            logger=self.mock_logger,
            risk_controller=self.mock_risk_controller
        )
        
        self.symbol_info_mock = MagicMock()
        self.symbol_info_mock.digits = 5

        self.base_trade_signal_details = {
            "symbol": "EURUSD",
            "price": 1.10000,
            "stop_loss_price": 1.09000,
            "take_profit_price": 1.11000,
            "comment": "Test trade",
            "client_order_id": "test_client_id_123"
        }
        self.calculated_lot_size = 0.1
        self.asset_profile_key = "EURUSD_Profile"

    def _get_mock_order(self, order_id="broker_order_1", client_order_id="client_order_1", symbol="EURUSD", action=OrderAction.BUY, status=OrderStatus.OPEN, pnl: Optional[float] = None):
        return Order(
            order_id=order_id,
            client_order_id=client_order_id,
            symbol=symbol,
            order_type=OrderType.MARKET, # Assuming market for simplicity in tests
            action=action,
            volume=0.1,
            price=1.10000,
            status=status,
            created_at=datetime.utcnow(),
            pnl=pnl
        )

    def _get_mock_position(self, position_id="broker_pos_1", symbol="EURUSD", action=OrderAction.BUY):
        return Position(
            position_id=position_id,
            symbol=symbol,
            action=action,
            volume=0.1,
            open_price=1.10000,
            open_time=datetime.utcnow(),
            status=OrderStatus.OPEN # Using OrderStatus for simplicity, should be PositionStatus
        )

    def test_execute_buy_signal_no_existing_orders_or_positions(self):
        self.mock_platform_adapter.get_open_orders.return_value = []
        self.mock_platform_adapter.get_open_positions.return_value = []
        mock_placed_order = self._get_mock_order(status=OrderStatus.FILLED) # Assume it gets filled
        self.mock_platform_adapter.place_order.return_value = mock_placed_order

        trade_signal = {**self.base_trade_signal_details, "signal": StrategySignal.BUY}
        
        result = self.manager.execute_trade_signal(
            trade_signal_details=trade_signal,
            calculated_lot_size=self.calculated_lot_size,
            symbol_info=self.symbol_info_mock,
            asset_profile_key=self.asset_profile_key
        )

        self.mock_platform_adapter.get_open_orders.assert_called_once_with(symbol="EURUSD")
        self.mock_platform_adapter.get_open_positions.assert_called_once_with(symbol="EURUSD")
        self.mock_platform_adapter.place_order.assert_called_once()
        self.assertEqual(result, mock_placed_order)
        self.mock_logger.warning.assert_not_called()

    def test_execute_sell_signal_no_existing_orders_or_positions(self):
        self.mock_platform_adapter.get_open_orders.return_value = []
        self.mock_platform_adapter.get_open_positions.return_value = []
        mock_placed_order = self._get_mock_order(action=OrderAction.SELL, status=OrderStatus.FILLED)
        self.mock_platform_adapter.place_order.return_value = mock_placed_order

        trade_signal = {**self.base_trade_signal_details, "signal": StrategySignal.SELL}

        result = self.manager.execute_trade_signal(
            trade_signal_details=trade_signal,
            calculated_lot_size=self.calculated_lot_size,
            symbol_info=self.symbol_info_mock,
            asset_profile_key=self.asset_profile_key
        )
        self.mock_platform_adapter.get_open_orders.assert_called_once_with(symbol="EURUSD")
        self.mock_platform_adapter.get_open_positions.assert_called_once_with(symbol="EURUSD")
        self.mock_platform_adapter.place_order.assert_called_once()
        self.assertEqual(result, mock_placed_order)
        self.mock_logger.warning.assert_not_called()

    def test_execute_buy_signal_existing_open_buy_order_prevents_new_order(self):
        existing_order = self._get_mock_order(symbol="EURUSD", action=OrderAction.BUY)
        self.mock_platform_adapter.get_open_orders.return_value = [existing_order]
        self.mock_platform_adapter.get_open_positions.return_value = []

        trade_signal = {**self.base_trade_signal_details, "signal": StrategySignal.BUY}
        
        result = self.manager.execute_trade_signal(
            trade_signal_details=trade_signal,
            calculated_lot_size=self.calculated_lot_size,
            symbol_info=self.symbol_info_mock,
            asset_profile_key=self.asset_profile_key
        )

        self.mock_platform_adapter.get_open_orders.assert_called_once_with(symbol="EURUSD")
        self.mock_platform_adapter.get_open_positions.assert_not_called() # Should return after finding order
        self.mock_platform_adapter.place_order.assert_not_called()
        self.assertIsNone(result)
        self.mock_logger.warning.assert_called_once_with(
            "Prevented duplicate BUY order for EURUSD. Existing open order found: ID broker_order_1, Client ID client_order_1"
        )

    def test_execute_sell_signal_existing_open_sell_position_prevents_new_order(self):
        existing_position = self._get_mock_position(symbol="EURUSD", action=OrderAction.SELL)
        self.mock_platform_adapter.get_open_orders.return_value = []
        self.mock_platform_adapter.get_open_positions.return_value = [existing_position]

        trade_signal = {**self.base_trade_signal_details, "signal": StrategySignal.SELL}

        result = self.manager.execute_trade_signal(
            trade_signal_details=trade_signal,
            calculated_lot_size=self.calculated_lot_size,
            symbol_info=self.symbol_info_mock,
            asset_profile_key=self.asset_profile_key
        )
        self.mock_platform_adapter.get_open_orders.assert_called_once_with(symbol="EURUSD")
        self.mock_platform_adapter.get_open_positions.assert_called_once_with(symbol="EURUSD")
        self.mock_platform_adapter.place_order.assert_not_called()
        self.assertIsNone(result)
        self.mock_logger.warning.assert_called_once_with(
            "Prevented duplicate SELL order for EURUSD. Existing open SELL position found: ID broker_pos_1"
        )

    def test_execute_buy_signal_existing_open_sell_order_allows_new_order(self):
        existing_order = self._get_mock_order(symbol="EURUSD", action=OrderAction.SELL)
        self.mock_platform_adapter.get_open_orders.return_value = [existing_order]
        self.mock_platform_adapter.get_open_positions.return_value = []
        mock_placed_order = self._get_mock_order(action=OrderAction.BUY, status=OrderStatus.FILLED)
        self.mock_platform_adapter.place_order.return_value = mock_placed_order
        
        trade_signal = {**self.base_trade_signal_details, "signal": StrategySignal.BUY}

        result = self.manager.execute_trade_signal(
            trade_signal_details=trade_signal,
            calculated_lot_size=self.calculated_lot_size,
            symbol_info=self.symbol_info_mock,
            asset_profile_key=self.asset_profile_key
        )
        self.mock_platform_adapter.get_open_orders.assert_called_once_with(symbol="EURUSD")
        self.mock_platform_adapter.get_open_positions.assert_called_once_with(symbol="EURUSD")
        self.mock_platform_adapter.place_order.assert_called_once()
        self.assertEqual(result, mock_placed_order)
        self.mock_logger.warning.assert_not_called()

    def test_execute_buy_signal_existing_open_sell_position_allows_new_order(self):
        existing_position = self._get_mock_position(symbol="EURUSD", action=OrderAction.SELL)
        self.mock_platform_adapter.get_open_orders.return_value = []
        self.mock_platform_adapter.get_open_positions.return_value = [existing_position]
        mock_placed_order = self._get_mock_order(action=OrderAction.BUY, status=OrderStatus.FILLED)
        self.mock_platform_adapter.place_order.return_value = mock_placed_order

        trade_signal = {**self.base_trade_signal_details, "signal": StrategySignal.BUY}

        result = self.manager.execute_trade_signal(
            trade_signal_details=trade_signal,
            calculated_lot_size=self.calculated_lot_size,
            symbol_info=self.symbol_info_mock,
            asset_profile_key=self.asset_profile_key
        )
        self.mock_platform_adapter.get_open_orders.assert_called_once_with(symbol="EURUSD")
        self.mock_platform_adapter.get_open_positions.assert_called_once_with(symbol="EURUSD")
        self.mock_platform_adapter.place_order.assert_called_once()
        self.assertEqual(result, mock_placed_order)
        self.mock_logger.warning.assert_not_called()

    def test_execute_buy_signal_platform_adapter_error_during_check_prevents_order(self):
        self.mock_platform_adapter.get_open_orders.side_effect = Exception("Connection error")

        trade_signal = {**self.base_trade_signal_details, "signal": StrategySignal.BUY}

        result = self.manager.execute_trade_signal(
            trade_signal_details=trade_signal,
            calculated_lot_size=self.calculated_lot_size,
            symbol_info=self.symbol_info_mock,
            asset_profile_key=self.asset_profile_key
        )
        self.mock_platform_adapter.get_open_orders.assert_called_once_with(symbol="EURUSD")
        self.mock_platform_adapter.get_open_positions.assert_not_called()
        self.mock_platform_adapter.place_order.assert_not_called()
        self.assertIsNone(result)
        self.mock_logger.error.assert_any_call("Error during pre-order check for EURUSD BUY: Connection error", exc_info=True)
        # The specific warning about proceeding with caution might or might not be hit depending on where the exception is caught
        # and if the logic decides to return immediately. Based on current diff, it returns None.
        # self.mock_logger.warning.assert_any_call("Proceeding with caution for EURUSD BUY as pre-order check failed. This might lead to duplicates if connection issues persist.")

    def test_execute_close_long_signal_records_pnl(self):
        mock_pnl = 100.50
        # Ensure the mock order returned by close_position has P&L
        mock_closed_order = self._get_mock_order(status=OrderStatus.FILLED, pnl=mock_pnl)
        self.mock_platform_adapter.close_position.return_value = mock_closed_order
        
        trade_signal = {
            "signal": StrategySignal.CLOSE_LONG,
            "position_id": "pos_to_close_123",
            "symbol": "EURUSD",
            "comment": "Closing long",
            "price": 1.10500 # Price might be used by close_position
        }

        result = self.manager.execute_trade_signal(
            trade_signal_details=trade_signal,
            calculated_lot_size=0,
            symbol_info=self.symbol_info_mock,
            asset_profile_key=self.asset_profile_key
        )
        self.mock_platform_adapter.close_position.assert_called_once_with(
            position_id="pos_to_close_123",
            volume_to_close=None,
            price=1.10500,
            comment="Closing long"
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.status, OrderStatus.FILLED)
        self.mock_risk_controller.record_trade_closure_pnl.assert_called_once_with(mock_pnl)
        self.mock_logger.info.assert_any_call(f"Recorded P&L {mock_pnl:.2f} for closed position pos_to_close_123 via order {mock_closed_order.order_id}")

    def test_execute_close_short_signal_records_pnl(self):
        mock_pnl = -50.25
        mock_closed_order = self._get_mock_order(status=OrderStatus.FILLED, pnl=mock_pnl, action=OrderAction.SELL) # Action for context if needed
        self.mock_platform_adapter.close_position.return_value = mock_closed_order
        
        trade_signal = {
            "signal": StrategySignal.CLOSE_SHORT,
            "position_id": "pos_to_close_456",
            "symbol": "GBPUSD",
            "comment": "Closing short",
            "volume_to_close": 0.05 # Example specific volume
        }

        # Update asset profile for GBPUSD if needed for symbol derivation, or ensure symbol is in trade_signal
        self.mock_config.asset_strategy_profiles["GBPUSD_Profile"] = MagicMock(symbol="GBPUSD")

        result = self.manager.execute_trade_signal(
            trade_signal_details=trade_signal,
            calculated_lot_size=0,
            symbol_info=self.symbol_info_mock, # Assuming it's generic enough or mock specific one
            asset_profile_key="GBPUSD_Profile" # Match the symbol
        )
        self.mock_platform_adapter.close_position.assert_called_once_with(
            position_id="pos_to_close_456",
            volume_to_close=0.05,
            price=None, # Not provided in this signal
            comment="Closing short"
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.status, OrderStatus.FILLED)
        self.mock_risk_controller.record_trade_closure_pnl.assert_called_once_with(mock_pnl)
        self.mock_logger.info.assert_any_call(f"Recorded P&L {mock_pnl:.2f} for closed position pos_to_close_456 via order {mock_closed_order.order_id}")

    def test_execute_close_signal_no_pnl_on_order_does_not_record(self):
        # Order has no P&L attribute or it's None
        mock_closed_order_no_pnl = self._get_mock_order(status=OrderStatus.FILLED, pnl=None)
        self.mock_platform_adapter.close_position.return_value = mock_closed_order_no_pnl
        
        trade_signal = {"signal": StrategySignal.CLOSE_LONG, "position_id": "pos_no_pnl", "symbol": "EURUSD"}

        self.manager.execute_trade_signal(trade_signal, 0, self.symbol_info_mock, self.asset_profile_key)
        
        self.mock_risk_controller.record_trade_closure_pnl.assert_not_called()
        self.mock_logger.error.assert_not_called() # Should not error, just not record

    def test_execute_close_signal_pnl_recording_fails_logs_error(self):
        mock_pnl = 75.00
        mock_closed_order = self._get_mock_order(status=OrderStatus.FILLED, pnl=mock_pnl)
        self.mock_platform_adapter.close_position.return_value = mock_closed_order
        self.mock_risk_controller.record_trade_closure_pnl.side_effect = Exception("DB error")

        trade_signal = {"signal": StrategySignal.CLOSE_LONG, "position_id": "pos_fail_rec", "symbol": "EURUSD"}
        
        self.manager.execute_trade_signal(trade_signal, 0, self.symbol_info_mock, self.asset_profile_key)
        
        self.mock_risk_controller.record_trade_closure_pnl.assert_called_once_with(mock_pnl)
        self.mock_logger.error.assert_called_once_with(
            f"Failed to record P&L for order {mock_closed_order.order_id} from position pos_fail_rec: DB error",
            exc_info=True
        )

    def test_execute_modify_sltp_signal_skips_duplicate_check(self):
        # For MODIFY_SLTP, execute_trade_signal returns None upon success/request sent
        self.mock_platform_adapter.modify_position_sl_tp.return_value = self._get_mock_position() 
        
        trade_signal = {
            "signal": StrategySignal.MODIFY_SLTP,
            "position_id": "pos_to_modify_456",
            "new_stop_loss": 1.09500,
            "new_take_profit": 1.10500,
            "symbol": "EURUSD" # Symbol needed for logging
        }

        result = self.manager.execute_trade_signal(
            trade_signal_details=trade_signal,
            calculated_lot_size=0, # Not relevant
            symbol_info=self.symbol_info_mock,
            asset_profile_key=self.asset_profile_key
        )
        self.mock_platform_adapter.get_open_orders.assert_not_called()
        self.mock_platform_adapter.get_open_positions.assert_not_called()
        self.mock_platform_adapter.modify_position_sl_tp.assert_called_once_with(
            position_id="pos_to_modify_456",
            stop_loss=1.09500,
            take_profit=1.10500
        )
        self.assertIsNone(result) # As per current OrderExecutionManager logic for MODIFY_SLTP

    def test_invalid_lot_size_prevents_order(self):
        trade_signal = {**self.base_trade_signal_details, "signal": StrategySignal.BUY}
        result = self.manager.execute_trade_signal(
            trade_signal_details=trade_signal,
            calculated_lot_size=0, # Invalid lot size
            symbol_info=self.symbol_info_mock,
            asset_profile_key=self.asset_profile_key
        )
        self.assertIsNone(result)
        self.mock_platform_adapter.place_order.assert_not_called()
        self.mock_logger.error.assert_called_with("OrderExecutionManager: Invalid lot size 0 for EURUSD.")

    def test_missing_signal_type_prevents_order(self):
        trade_signal = {**self.base_trade_signal_details}
        del trade_signal["signal"] # Remove signal
        
        result = self.manager.execute_trade_signal(
            trade_signal_details=trade_signal,
            calculated_lot_size=self.calculated_lot_size,
            symbol_info=self.symbol_info_mock,
            asset_profile_key=self.asset_profile_key
        )
        self.assertIsNone(result)
        self.mock_platform_adapter.place_order.assert_not_called()
        self.mock_logger.error.assert_called_with("OrderExecutionManager: 'signal' type missing in trade_signal_details.")

    def test_emergency_close_all_positions_records_pnl_for_each_closed_position(self):
        pos1_pnl = 100.0
        pos2_pnl = -50.0
        
        mock_position1 = self._get_mock_position(position_id="pos1", symbol="EURUSD")
        mock_position2 = self._get_mock_position(position_id="pos2", symbol="GBPUSD", action=OrderAction.SELL)
        
        self.mock_platform_adapter.get_open_positions.return_value = [mock_position1, mock_position2]
        
        # Mock return values for close_position for each position
        closed_order_pos1 = self._get_mock_order(order_id="closed_ord1", status=OrderStatus.FILLED, pnl=pos1_pnl)
        closed_order_pos2 = self._get_mock_order(order_id="closed_ord2", status=OrderStatus.FILLED, pnl=pos2_pnl)
        
        self.mock_platform_adapter.close_position.side_effect = [closed_order_pos1, closed_order_pos2]
        
        self.manager.execute_emergency_close_all_positions(reason="Test emergency")
        
        self.assertEqual(self.mock_platform_adapter.close_position.call_count, 2)
        self.mock_platform_adapter.close_position.assert_any_call(
            position_id="pos1", comment="Emergency Close: Test emergency"
        )
        self.mock_platform_adapter.close_position.assert_any_call(
            position_id="pos2", comment="Emergency Close: Test emergency"
        )
        
        self.assertEqual(self.mock_risk_controller.record_trade_closure_pnl.call_count, 2)
        self.mock_risk_controller.record_trade_closure_pnl.assert_any_call(pos1_pnl)
        self.mock_risk_controller.record_trade_closure_pnl.assert_any_call(pos2_pnl)

        self.mock_logger.info.assert_any_call(f"Recorded P&L {pos1_pnl:.2f} for emergency closed position pos1 via order closed_ord1")
        self.mock_logger.info.assert_any_call(f"Recorded P&L {pos2_pnl:.2f} for emergency closed position pos2 via order closed_ord2")

    def test_emergency_close_all_positions_handles_pnl_recording_failure(self):
        pos1_pnl = 120.0
        mock_position1 = self._get_mock_position(position_id="pos_err_rec", symbol="AUDUSD")
        self.mock_platform_adapter.get_open_positions.return_value = [mock_position1]
        
        closed_order_pos1 = self._get_mock_order(order_id="closed_ord_err", status=OrderStatus.FILLED, pnl=pos1_pnl)
        self.mock_platform_adapter.close_position.return_value = closed_order_pos1
        
        self.mock_risk_controller.record_trade_closure_pnl.side_effect = Exception("Risk DB unavailable")
        
        self.manager.execute_emergency_close_all_positions(reason="Test emergency with PNL error")
        
        self.mock_risk_controller.record_trade_closure_pnl.assert_called_once_with(pos1_pnl)
        self.mock_logger.error.assert_called_once_with(
            f"Failed to record P&L for emergency closing order closed_ord_err from position pos_err_rec: Risk DB unavailable",
            exc_info=True
        )

    def test_emergency_close_all_positions_no_pnl_on_order(self):
        mock_position1 = self._get_mock_position(position_id="pos_no_pnl_emerg", symbol="EURUSD")
        self.mock_platform_adapter.get_open_positions.return_value = [mock_position1]
        
        # Order has no P&L attribute or it's None
        closed_order_no_pnl = self._get_mock_order(order_id="closed_no_pnl_emerg", status=OrderStatus.FILLED, pnl=None)
        self.mock_platform_adapter.close_position.return_value = closed_order_no_pnl
        
        self.manager.execute_emergency_close_all_positions(reason="Test emergency no PNL")
        
        self.mock_risk_controller.record_trade_closure_pnl.assert_not_called()
        # Ensure no error logged for *missing* PNL, only for *failed recording* of existing PNL
        # Check that the specific error message for failed recording is NOT present
        for call_args in self.mock_logger.error.call_args_list:
            self.assertNotIn("Failed to record P&L", call_args[0][0])


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

  
