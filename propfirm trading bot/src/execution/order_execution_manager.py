# prop_firm_trading_bot/src/execution/order_execution_manager.py

import logging
from typing import Optional, TYPE_CHECKING, Dict, Any
import uuid

from prop_firm_trading_bot.src.core.enums import OrderType, OrderAction, StrategySignal, OrderStatus
from prop_firm_trading_bot.src.core.models import Order, Position
# Assuming PlatformInterface is correctly in base_connector
from prop_firm_trading_bot.src.api_connector.base_connector import PlatformInterface

if TYPE_CHECKING:
    from prop_firm_trading_bot.src.config_manager import AppConfig


class OrderExecutionManager:
    """
    Handles the actual placement, modification, and cancellation of orders
    on the trading platform via the PlatformInterface.
    It receives fully validated and specified trade instructions.
    """

    def __init__(self, 
                 config: 'AppConfig', 
                 platform_adapter: PlatformInterface, 
                 logger: logging.Logger):
        self.config = config
        self.platform_adapter = platform_adapter
        self.logger = logger
        # Default slippage might be defined in platform config or operational compliance
        self.default_slippage_points = config.platform.mt5.slippage_default_points if config.platform.mt5 else 20 # Example

    def generate_client_order_id(self) -> str:
        """Generates a unique client order ID if one is not provided."""
        return str(uuid.uuid4())

    def execute_trade_signal(self, 
                             trade_signal_details: Dict[str, Any], 
                             calculated_lot_size: float,
                             symbol_info: Any, # Expects SymbolInfo model or similar with digits
                             asset_profile_key: str # For logging/comments
                            ) -> Optional[Order]:
        """
        Takes a trade signal (from Strategy, approved & sized by RiskController)
        and executes it.
        
        trade_signal_details: Expected to contain 'signal' (StrategySignal), 'price',
                              'stop_loss_price', 'take_profit_price', 'comment'.
        calculated_lot_size: The final lot size approved by RiskController.
        symbol_info: Contains digits for price rounding.
        """
        signal_type: Optional[StrategySignal] = trade_signal_details.get("signal") # Make it Optional
        # Ensure symbol is present, fallback to asset_profile_key if needed, though signal should provide it.
        symbol: Optional[str] = trade_signal_details.get("symbol")
        if not symbol and asset_profile_key in self.config.asset_strategy_profiles:
            symbol = self.config.asset_strategy_profiles[asset_profile_key].symbol
        
        price: Optional[float] = trade_signal_details.get("price") 
        stop_loss_price: Optional[float] = trade_signal_details.get("stop_loss_price")
        take_profit_price: Optional[float] = trade_signal_details.get("take_profit_price")
        custom_comment: Optional[str] = trade_signal_details.get("comment", f"{asset_profile_key}")
        client_order_id = trade_signal_details.get("client_order_id", self.generate_client_order_id())

        if not signal_type:
            self.logger.error("OrderExecutionManager: 'signal' type missing in trade_signal_details.")
            return None
        if not symbol:
            self.logger.error("OrderExecutionManager: Symbol missing in trade signal details and could not be derived from asset_profile_key.")
            return None
        if calculated_lot_size <= 0:
            self.logger.error(f"OrderExecutionManager: Invalid lot size {calculated_lot_size} for {symbol}.")
            return None

        order_to_place: Optional[Order] = None

        if signal_type == StrategySignal.BUY:
            self.logger.info(f"Executing BUY signal for {symbol}, Size: {calculated_lot_size}, SL: {stop_loss_price}, TP: {take_profit_price}")
            order_to_place = self.platform_adapter.place_order(
                symbol=symbol,
                order_type=OrderType.MARKET, 
                action=OrderAction.BUY,
                volume=calculated_lot_size,
                price=price, 
                stop_loss=stop_loss_price,
                take_profit=take_profit_price,
                client_order_id=client_order_id,
                slippage_points=self.default_slippage_points,
                comment=custom_comment
            )
        elif signal_type == StrategySignal.SELL:
            self.logger.info(f"Executing SELL signal for {symbol}, Size: {calculated_lot_size}, SL: {stop_loss_price}, TP: {take_profit_price}")
            order_to_place = self.platform_adapter.place_order(
                symbol=symbol,
                order_type=OrderType.MARKET, 
                action=OrderAction.SELL,
                volume=calculated_lot_size,
                price=price,
                stop_loss=stop_loss_price,
                take_profit=take_profit_price,
                client_order_id=client_order_id,
                slippage_points=self.default_slippage_points,
                comment=custom_comment
            )
        elif signal_type == StrategySignal.CLOSE_LONG or signal_type == StrategySignal.CLOSE_SHORT:
            position_id_to_close = trade_signal_details.get("position_id")
            volume_to_close = trade_signal_details.get("volume_to_close") 
            if position_id_to_close:
                self.logger.info(f"Executing {signal_type.name} for position {position_id_to_close} on {symbol}")
                order_to_place = self.platform_adapter.close_position(
                    position_id=position_id_to_close,
                    volume_to_close=volume_to_close, # Corrected parameter name
                    price=price, 
                    comment=custom_comment or f"Close {signal_type.name}"
                )
            else:
                self.logger.error(f"Cannot execute {signal_type.name}: position_id missing in signal details.")
        
        elif signal_type == StrategySignal.MODIFY_SLTP: 
            position_id_to_modify = trade_signal_details.get("position_id")
            new_sl = trade_signal_details.get("new_stop_loss")
            new_tp = trade_signal_details.get("new_take_profit")
            if position_id_to_modify:
                self.logger.info(f"Executing MODIFY_SLTP for position {position_id_to_modify} on {symbol}. New SL: {new_sl}, New TP: {new_tp}")
                updated_position = self.platform_adapter.modify_position_sl_tp(
                    position_id=position_id_to_modify,
                    stop_loss=new_sl,
                    take_profit=new_tp
                )
                if updated_position:
                     self.logger.info(f"Position {position_id_to_modify} SL/TP modification successful (or request sent).")
                     return None 
                else:
                    self.logger.error(f"Failed to modify SL/TP for position {position_id_to_modify}.")
                    return None
            else:
                self.logger.error("Cannot execute MODIFY_SLTP: position_id missing.")

        else:
            self.logger.warning(f"OrderExecutionManager received unhandled signal type: {signal_type}")

        if order_to_place:
            self.logger.info(f"Order execution result for {symbol} ({signal_type.name}): ID {order_to_place.order_id}, Status {order_to_place.status.name}")
        else:
            self.logger.warning(f"No order placed for signal {signal_type.name} on {symbol} based on details: {trade_signal_details}")
            
        return order_to_place

    def execute_emergency_close_all_positions(self, reason: str):
        self.logger.critical(f"Executing EMERGENCY CLOSE ALL POSITIONS. Reason: {reason}")
        open_positions = self.platform_adapter.get_open_positions()
        closed_count = 0
        failed_count = 0

        if not open_positions:
            self.logger.info("Emergency Close: No open positions found.")
            return

        for position in open_positions:
            self.logger.warning(f"Emergency closing position: {position.position_id} ({position.symbol} {position.volume} {position.action.name})")
            
            closing_order = self.platform_adapter.close_position(
                position_id=position.position_id,
                comment=f"Emergency Close: {reason}"
            )
            if closing_order and closing_order.status == OrderStatus.FILLED: 
                self.logger.info(f"Emergency close successful for position {position.position_id}. Closing Order ID: {closing_order.order_id}")
                closed_count +=1
            else:
                status_name = closing_order.status.name if closing_order and hasattr(closing_order, 'status') and hasattr(closing_order.status, 'name') else 'None/Unknown'
                self.logger.error(f"Emergency close FAILED or UNCONFIRMED for position {position.position_id}. Status: {status_name}")
                failed_count +=1
        
        self.logger.critical(f"Emergency close all positions complete. Closed: {closed_count}, Failed/Unconfirmed: {failed_count}.")
