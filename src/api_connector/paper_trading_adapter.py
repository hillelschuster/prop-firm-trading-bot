from datetime import datetime, timezone
from typing import List, Dict, Optional, Any, Tuple, Set
import pandas as pd
import logging

# Project-specific imports
from src.api_connector.base_connector import PlatformInterface
from src.core.models import Order, Position, TickData, TradeFill, AccountInfo, SymbolInfo, OHLCVData
from src.core.enums import OrderStatus, OrderType, OrderAction, PositionType, Timeframe
# from src.config_manager import AppConfig # For type hinting config if available

class PaperTradingAdapter(PlatformInterface):
    def __init__(self, config: Any, logger: logging.Logger, historical_data: pd.DataFrame, initial_balance: float):
        super().__init__(config, logger)
        self.initial_balance: float = initial_balance
        self.balance: float = initial_balance
        self.equity: float = initial_balance
        
        self.open_positions: Dict[str, Position] = {}
        self.pending_orders: Dict[str, Order] = {} # Active pending orders
        self.all_orders: Dict[str, Order] = {} # Archive of all orders ever created
        self.trade_history: List[TradeFill] = [] # Record of actual fills/closures
        self.equity_history: List[Tuple[datetime, float]] = []
        self.market_orders_pending_fill_next_bar: List[str] = [] # Order IDs

        self.next_order_id_counter: int = 1
        self.next_position_id_counter: int = 1

        self.historical_data: pd.DataFrame = historical_data.copy() # Use a copy
        if not self.historical_data.empty and 'timestamp' in self.historical_data.columns:
            if not pd.api.types.is_datetime64_any_dtype(self.historical_data['timestamp']):
                self.historical_data['timestamp'] = pd.to_datetime(self.historical_data['timestamp'])
            # Ensure timezone-aware (UTC) if not already, or make consistent
            if self.historical_data['timestamp'].dt.tz is None:
                self.historical_data['timestamp'] = self.historical_data['timestamp'].dt.tz_localize(timezone.utc)
        else:
            self.logger.error("Historical data is empty or missing 'timestamp' column.")
            # Potentially raise an error or handle this state appropriately

        self.current_bar_index: int = -1 # Start before the first bar
        self._is_connected: bool = False

        # Multi-timeframe support for backtesting
        self.timeframe_datasets: Dict[Timeframe, pd.DataFrame] = {}
        self.current_bar_indices: Dict[Timeframe, int] = {}

        # Initialize with primary timeframe data (backward compatibility)
        # The primary timeframe will be determined when timeframe_datasets is set
        self.primary_timeframe: Optional[Timeframe] = None

        # Determine symbol, point, and lot_size
        self.symbol: str = "EURUSD" # Default
        self.point: float = 0.00001 # Default pip size for EURUSD-like pairs
        self.lot_size: float = 100000.0 # Default lot size (e.g., 100,000 units for 1 standard lot)

        if not self.historical_data.empty and 'symbol' in self.historical_data.columns:
            # Assuming a 'symbol' column in historical_data if data is for multiple symbols,
            # or that the historical_data provided is for a single, specific symbol.
            # For this adapter, we'll assume it operates on ONE primary symbol.
            unique_symbols = self.historical_data['symbol'].unique()
            if len(unique_symbols) == 1:
                self.symbol = unique_symbols[0]
            elif len(unique_symbols) > 1:
                self.logger.warning("Historical data contains multiple symbols. PaperTradingAdapter will use the first one found or config setting.")
                # Prioritize config if available, otherwise take the first one.
        
        # Override with config if available
        # Example: config.trading_settings.symbol or config.strategy_profile.symbol
        primary_symbol_from_config = None
        if hasattr(config, 'asset_strategy_profiles') and config.asset_strategy_profiles:
            # Assuming the first profile is the relevant one for this paper trader instance
            first_profile_key = next(iter(config.asset_strategy_profiles))
            profile = config.asset_strategy_profiles[first_profile_key]
            if hasattr(profile, 'symbol'):
                primary_symbol_from_config = profile.symbol
        
        if primary_symbol_from_config:
            self.symbol = primary_symbol_from_config
        elif 'symbol' not in self.historical_data.columns and not primary_symbol_from_config:
             self.logger.warning(f"PaperTradingAdapter: Symbol could not be determined from historical_data or config. Defaulting to {self.symbol}.")


        # Get point and lot_size from config if available for the determined symbol
        if hasattr(config, 'platform') and hasattr(config.platform, 'symbol_settings') and \
           self.symbol in config.platform.symbol_settings:
            symbol_config = config.platform.symbol_settings[self.symbol]
            self.point = symbol_config.get('point', self.point)
            self.lot_size = symbol_config.get('lot_size', self.lot_size)
            # self.digits = symbol_config.get('digits', 5 if self.point == 0.00001 else 3) # Example
        else:
            self.logger.warning(f"PaperTradingAdapter: Symbol-specific settings (point, lot_size) for {self.symbol} not found in config. Using defaults.")
        
        self.logger.info(f"PaperTradingAdapter initialized for symbol: {self.symbol}, Point: {self.point}, Lot Size: {self.lot_size}, Initial Balance: {self.initial_balance}")

    def set_timeframe_datasets(self, timeframe_datasets: Dict[Timeframe, pd.DataFrame]) -> None:
        """
        Set multi-timeframe datasets for backtesting.

        Args:
            timeframe_datasets: Dictionary mapping timeframes to their respective DataFrames
        """
        self.timeframe_datasets = timeframe_datasets.copy()

        # Initialize bar indices for each timeframe
        for timeframe, data in self.timeframe_datasets.items():
            self.current_bar_indices[timeframe] = -1

            # Ensure timezone-aware timestamps for each dataset
            if not data.empty and 'timestamp' in data.columns:
                if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
                    data['timestamp'] = pd.to_datetime(data['timestamp'])
                if data['timestamp'].dt.tz is None:
                    data['timestamp'] = data['timestamp'].dt.tz_localize(timezone.utc)

        # Determine primary timeframe (the one that matches historical_data)
        if self.timeframe_datasets:
            # Find timeframe with same length as historical_data
            for timeframe, data in self.timeframe_datasets.items():
                if len(data) == len(self.historical_data):
                    self.primary_timeframe = timeframe
                    self.logger.info(f"Primary timeframe detected: {timeframe.name}")
                    break

            if not self.primary_timeframe:
                # Fallback: use the first timeframe
                self.primary_timeframe = next(iter(self.timeframe_datasets.keys()))
                self.logger.warning(f"Could not detect primary timeframe. Using {self.primary_timeframe.name} as fallback.")

        self.logger.info(f"Multi-timeframe datasets configured: {[tf.name for tf in self.timeframe_datasets.keys()]}")

    def get_timeframe_data(self, timeframe: Timeframe, up_to_timestamp: Optional[datetime] = None) -> Optional[pd.DataFrame]:
        """
        Get historical data for a specific timeframe with optional timestamp filtering.

        Args:
            timeframe: Requested timeframe
            up_to_timestamp: Optional timestamp to filter data up to (for progressive data windows)

        Returns:
            DataFrame with data for the requested timeframe, or None if not available
        """
        if timeframe not in self.timeframe_datasets:
            self.logger.warning(f"Timeframe {timeframe.name} not available in timeframe datasets")
            return None

        data = self.timeframe_datasets[timeframe]

        if up_to_timestamp is None:
            # Return all data up to current bar index for this timeframe
            current_index = self.current_bar_indices.get(timeframe, -1)
            if current_index < 0:
                return pd.DataFrame()  # No data available yet
            return data.iloc[:current_index + 1].copy()
        else:
            # Filter data up to the specified timestamp
            if up_to_timestamp.tzinfo is None:
                up_to_timestamp = up_to_timestamp.replace(tzinfo=timezone.utc)

            filtered_data = data[data['timestamp'] <= up_to_timestamp].copy()
            return filtered_data

    def _generate_order_id(self) -> str:
        order_id = f"paper_{self.next_order_id_counter}"
        self.next_order_id_counter += 1
        return order_id

    def _generate_position_id(self) -> str:
        pos_id = f"paper_pos_{self.next_position_id_counter}"
        self.next_position_id_counter += 1
        return pos_id

    def connect(self) -> bool:
        self._is_connected = True
        self.logger.info("PaperTradingAdapter connected.")
        return True

    def disconnect(self) -> None:
        self._is_connected = False
        self.logger.info("PaperTradingAdapter disconnected.")

    def is_connected(self) -> bool:
        return self._is_connected

    def get_historical_ohlcv(
        self,
        symbol: str,
        timeframe: Timeframe, # Changed from timeframe_str
        start_time: Optional[datetime] = None, # Changed from from_timestamp, added end_time
        end_time: Optional[datetime] = None,
        count: Optional[int] = None # Changed to Optional[int]
    ) -> List[OHLCVData]: # Return type changed to List[OHLCVData]
        if not self._is_connected:
            self.logger.error("Not connected. Cannot get historical OHLCV.")
            return []
        if symbol != self.symbol:
            self.logger.warning(f"Requested symbol {symbol} does not match adapter's symbol {self.symbol}.")
            return []

        self.logger.debug(f"get_historical_ohlcv called for {symbol}, timeframe {timeframe.name if timeframe else 'Any'}, count {count}, start {start_time}, end {end_time}")

        # Multi-timeframe support: try to get data for the specific timeframe
        effective_data = None

        if timeframe in self.timeframe_datasets:
            # Use timeframe-specific data
            timeframe_data = self.timeframe_datasets[timeframe]
            current_tf_index = self.current_bar_indices.get(timeframe, -1)

            if current_tf_index < 0:
                # During initialization, provide all available data for indicator calculation
                effective_data = timeframe_data.copy()
            else:
                # During normal backtesting, consider data up to and including the current bar index for this timeframe
                effective_data = timeframe_data.iloc[:current_tf_index + 1]
        else:
            # Fallback to primary historical data (backward compatibility)
            if self.historical_data.empty:
                return []

            # For backtesting initial history fetch (when current_bar_index is -1),
            # provide access to historical data for indicator calculation
            if self.current_bar_index < 0:
                # During initialization, provide the requested count of historical bars
                # This allows strategies to calculate initial indicators
                effective_data = self.historical_data.copy()
            else:
                # During normal backtesting, consider data up to and including the current_bar_index
                effective_data = self.historical_data.iloc[:self.current_bar_index + 1]

        # Apply time filters if specified
        if start_time:
            if start_time.tzinfo is None: start_time = start_time.replace(tzinfo=timezone.utc)
            effective_data = effective_data[effective_data['timestamp'] >= start_time].copy()

        if end_time:
            if end_time.tzinfo is None: end_time = end_time.replace(tzinfo=timezone.utc)
            effective_data = effective_data[effective_data['timestamp'] <= end_time].copy()

        # Check if we have any data after filtering
        if effective_data.empty:
            return []

        # Apply count limit if specified
        if count is not None and count > 0:
            effective_data = effective_data.tail(count)
        
        # Convert DataFrame rows to List[OHLCVData]
        ohlcv_list: List[OHLCVData] = []
        for _, row in effective_data.iterrows():
            ts = row['timestamp']
            if not isinstance(ts, datetime): # Ensure it's datetime
                ts = pd.to_datetime(ts)
            if ts.tzinfo is None: # Ensure timezone-aware
                ts = ts.replace(tzinfo=timezone.utc) # Corrected from tz_localize

            ohlcv_list.append(OHLCVData(
                timestamp=ts,
                symbol=symbol, # Use requested symbol
                timeframe=timeframe, # Use requested timeframe
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row.get('volume') # Use .get for optional column
            ))
        return ohlcv_list


    def get_latest_tick(self, symbol: str) -> Optional[TickData]:
        if not self._is_connected:
            self.logger.error("Not connected. Cannot get latest tick.")
            return None
        if symbol != self.symbol:
            self.logger.warning(f"Requested symbol {symbol} does not match adapter's symbol {self.symbol}.")
            return None
        if self.current_bar_index < 0 or self.current_bar_index >= len(self.historical_data):
            self.logger.warning("Current bar index is out of bounds. No tick data available.")
            return None

        current_bar = self.historical_data.iloc[self.current_bar_index]
        return TickData(
            timestamp=current_bar['timestamp'],
            bid=current_bar['close'], # Simplification: bid/ask spread is zero
            ask=current_bar['close'],
            symbol=symbol # Return the requested symbol
        )

    def get_account_info(self) -> Optional[AccountInfo]:
        if not self._is_connected:
            self.logger.error("Not connected. Cannot get account info.")
            return None
        
        server_time = datetime.now(timezone.utc) # Default to current time
        if self.current_bar_index >= 0 and self.current_bar_index < len(self.historical_data):
            server_time = self.historical_data.iloc[self.current_bar_index]['timestamp']
            if server_time.tzinfo is None: server_time = server_time.replace(tzinfo=timezone.utc)

        return AccountInfo(
            account_id=str(getattr(self.config.platform, 'account_id', "paper_account_001")),
            balance=self.balance,
            equity=self.equity,
            currency=str(getattr(self.config.platform, 'account_currency', "USD")),
            server_time=server_time,
            margin=0.0, # Simplified for paper trading
            margin_free=self.equity, # Simplified - use correct field name
            margin_level_pct=100.0 if self.equity > 0 else 0.0, # Simplified
            platform_specific_details={
                "leverage": int(getattr(self.config.platform, 'leverage', 100)),
                "name": "Paper Trading Account"
            }
        )

    def get_symbol_info(self, symbol: str) -> Optional[SymbolInfo]:
        if not self._is_connected:
            self.logger.error("Not connected. Cannot get symbol info.")
            return None
        if symbol != self.symbol: # This adapter is configured for one symbol's specifics
            self.logger.warning(f"Requested symbol info for {symbol}, but adapter configured for {self.symbol}.")
            # Could try to provide generic info or info from config if available for 'symbol'
            # For now, only provide for the adapter's main symbol.
            return None

        # Defaults, should ideally come from config or be more dynamic
        digits = 5 if self.point == 0.00001 else (3 if self.point == 0.001 else 2) # Infer digits from point
        min_volume_lots = 0.01
        volume_step_lots = 0.01
        max_volume_lots = 1000.0 # Default

        # Try to get from config for the specific symbol self.symbol
        if hasattr(self.config, 'platform') and hasattr(self.config.platform, 'symbol_settings') and \
           self.symbol in self.config.platform.symbol_settings:
            settings = self.config.platform.symbol_settings[self.symbol]
            digits = settings.get('digits', digits)
            min_volume_lots = settings.get('min_volume_lots', min_volume_lots)
            max_volume_lots = settings.get('max_volume_lots', max_volume_lots)
            volume_step_lots = settings.get('volume_step_lots', volume_step_lots)
            # self.point and self.lot_size are already set in __init__

        return SymbolInfo(
            name=self.symbol,  # Required field
            point=self.point,
            digits=digits,
            min_volume_lots=min_volume_lots,
            max_volume_lots=max_volume_lots,
            volume_step_lots=volume_step_lots,
            contract_size=self.lot_size,  # Required field (renamed from lot_size)
            description=f"Paper trading symbol {self.symbol}",
            currency_base=self.symbol[:3] if len(self.symbol) >= 6 else "BASE", # Heuristic
            currency_profit=self.symbol[3:6] if len(self.symbol) >= 6 else "PROFIT", # Heuristic
            currency_margin=str(getattr(self.config.platform, 'account_currency', "USD"))
        )

    def place_order(self, symbol: str, order_type: OrderType, action: OrderAction, volume: float,
                    price: Optional[float] = None, stop_loss: Optional[float] = None,
                    take_profit: Optional[float] = None,
                    client_order_id: Optional[str] = None, # Added from interface
                    slippage_points: Optional[int] = None,  # Added from interface (deviation maps here)
                    comment: Optional[str] = None,
                    expiration_time: Optional[datetime] = None, # Added from interface
                    magic_number: Optional[int] = None # Kept as specific to paper trader logic
                    ) -> Optional[Order]:
        if not self._is_connected:
            self.logger.error("Not connected. Cannot place order.")
            return None
        if symbol != self.symbol:
            self.logger.warning(f"Order for symbol {symbol} does not match adapter's symbol {self.symbol}. Rejecting.")
            return None
        if self.current_bar_index < 0 or self.current_bar_index >= len(self.historical_data):
            self.logger.error("No current market data (no bar processed yet or end of data). Cannot place order.")
            return None

        order_id = self._generate_order_id() # This is platform_order_id
        current_time = self.historical_data.iloc[self.current_bar_index]['timestamp']
        if current_time.tzinfo is None: # Ensure timezone aware
            current_time = current_time.replace(tzinfo=timezone.utc)


        order_status_initial = OrderStatus.NEW # Default for non-market

        if order_type == OrderType.MARKET:
            order_status_initial = OrderStatus.PENDING_OPEN # Mark as pending, to be filled in next_bar
        elif order_type in [OrderType.LIMIT, OrderType.STOP]:
            if price is None:
                self.logger.error(f"Order {order_id} ({order_type.name}) requires a price. Rejecting.")
                # Create a rejected order object to return
                rejected_order = Order(
                    order_id=order_id, client_order_id=client_order_id, symbol=symbol,
                    order_type=order_type, action=action, volume=volume, price=price,
                    stop_loss=stop_loss, take_profit=take_profit, status=OrderStatus.REJECTED,
                    created_at=current_time, comment=comment or "Rejected: Price required for pending order",
                    magic_number=magic_number
                )
                self.all_orders[order_id] = rejected_order
                return rejected_order
            order_status_initial = OrderStatus.OPEN # Pending orders are 'OPEN' once accepted

        order = Order(
            order_id=order_id,
            client_order_id=client_order_id,
            symbol=symbol,
            order_type=order_type,
            action=action,
            volume=volume,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            status=order_status_initial, # Use determined initial status
            created_at=current_time,
            comment=comment,
            magic_number=magic_number
        )
        if expiration_time:
            order.platform_specific_details['expiration_time'] = expiration_time.isoformat() if expiration_time else None
        
        if slippage_points is not None:
            self.logger.info(f"Slippage points ({slippage_points}) provided but not actively used in paper trading fill logic.")
        
        self.all_orders[order_id] = order # Archive all orders

        if order.order_type == OrderType.MARKET:
            # Market orders are now added to a pending list and processed in next_bar
            self.market_orders_pending_fill_next_bar.append(order_id)
            self.logger.info(f"Market order {order_id} ({action.name} {volume} {symbol}) accepted, pending fill on next bar's open.")
        
        elif order.order_type in [OrderType.LIMIT, OrderType.STOP]:
            # This case is handled by setting order_status_initial to OPEN if price is valid
            # and the early return for invalid price.
            if order.status == OrderStatus.OPEN: # Ensure it was not rejected
                 self.pending_orders[order_id] = order
                 self.logger.info(f"Pending order {order_id} ({order.order_type.name} {action.name} {volume} {symbol} at {order.price}) is now OPEN.")
        
        elif order.status == OrderStatus.REJECTED: # Already logged if rejected due to missing price
            pass # Do nothing more, it's already in all_orders as REJECTED
        else: # Should not happen if logic above is correct
            self.logger.error(f"Unsupported order type or state for order {order_id}: {order.order_type}, status: {order.status}")
            order.status = OrderStatus.REJECTED # Fallback
        
        return order

    def modify_order(self, order_id: str, new_price: Optional[float] = None, # Changed param names
                     new_stop_loss: Optional[float] = None, new_take_profit: Optional[float] = None) -> Optional[Order]: # Return Optional[Order]
        if not self._is_connected: return None # Return None on failure
        
        order = self.all_orders.get(order_id)
        if not order:
            self.logger.warning(f"Order {order_id} not found for modification.")
            return None
        
        if order_id not in self.pending_orders:
            self.logger.warning(f"Order {order_id} is not pending (status: {order.status.name}), cannot modify.")
            return None

        mod_time = datetime.now(timezone.utc)
        if self.current_bar_index >=0 and self.current_bar_index < len(self.historical_data):
            mod_time = self.historical_data.iloc[self.current_bar_index]['timestamp']
            if mod_time.tzinfo is None: mod_time = mod_time.replace(tzinfo=timezone.utc)


        modified = False
        if new_price is not None and order.order_type in [OrderType.LIMIT, OrderType.STOP]: # Use new_price
            order.price = new_price
            modified = True
        if new_stop_loss is not None: # Use new_stop_loss
            order.stop_loss = new_stop_loss
            modified = True
        if new_take_profit is not None: # Use new_take_profit
            order.take_profit = new_take_profit
            modified = True
        
        if modified:
            order.updated_at = mod_time # Use updated_at from Order model
            self.logger.info(f"Order {order_id} modified. New Price: {order.price}, SL: {order.stop_loss}, TP: {order.take_profit}")
            return order # Return modified order
        else:
            self.logger.info(f"No modifications applied to order {order_id}.")
            return order # Return order even if not modified


    def cancel_order(self, order_id: str) -> Optional[Order]: # Return Optional[Order]
        if not self._is_connected: return None # Return None on failure

        order = self.all_orders.get(order_id)
        if not order:
            self.logger.warning(f"Order {order_id} not found for cancellation.")
            return None

        if order_id in self.pending_orders:
            cancelled_order_instance = self.pending_orders.pop(order_id) # Use different var name
            cancelled_order_instance.status = OrderStatus.CANCELLED
            
            cancel_time = datetime.now(timezone.utc) # Use different var name
            if self.current_bar_index >=0 and self.current_bar_index < len(self.historical_data):
                 cancel_time = self.historical_data.iloc[self.current_bar_index]['timestamp']
                 if cancel_time.tzinfo is None: cancel_time = cancel_time.replace(tzinfo=timezone.utc)
            cancelled_order_instance.closed_time = cancel_time # Use closed_time from Order model
            
            self.logger.info(f"Order {order_id} cancelled.")
            return cancelled_order_instance # Return cancelled order
        else:
            self.logger.warning(f"Order {order_id} is not in pending list (current status: {order.status.name}). Cannot cancel.")
            return None

    def modify_position_sl_tp(self, position_id: str, stop_loss: Optional[float] = None, take_profit: Optional[float] = None) -> Optional[Position]: # Return Optional[Position]
        if not self._is_connected: return None # Return None on failure
        
        position = self.open_positions.get(position_id)
        if not position:
            self.logger.warning(f"Position {position_id} not found for SL/TP modification.")
            return None

        modified = False
        if stop_loss is not None:
            position.stop_loss = stop_loss
            modified = True
        if take_profit is not None:
            position.take_profit = take_profit
            modified = True
        
        if modified:
            self.logger.info(f"Position {position_id} SL/TP modified. New SL: {position.stop_loss}, TP: {position.take_profit}")
            return position # Return modified position
        else:
            self.logger.info(f"No SL/TP modifications applied to position {position_id}.")
            return position


    def _get_instrument_properties_for_pnl(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetches instrument-specific properties required for P&L calculation.
        These should be pre-loaded into self.config.loaded_instrument_details.
        Example keys: 'point', 'digits', 'pip_value_in_account_currency_per_lot', 'contract_size'.
        """
        if hasattr(self.config, 'loaded_instrument_details') and \
           isinstance(self.config.loaded_instrument_details, dict):

            # First try direct symbol lookup
            if symbol in self.config.loaded_instrument_details:
                props = self.config.loaded_instrument_details[symbol]
            else:
                # Try to find by platform_symbol mapping (e.g., EURUSD_FTMO -> EURUSD)
                props = None
                for config_key, config_data in self.config.loaded_instrument_details.items():
                    if isinstance(config_data, dict) and config_data.get('platform_symbol') == symbol:
                        props = config_data
                        self.logger.debug(f"Found instrument config for {symbol} under key {config_key}")
                        break

                if props is None:
                    self.logger.error(f"Symbol {symbol} not found in loaded_instrument_details (neither direct nor by platform_symbol mapping).")
                    return None

            # Ensure necessary keys are present
            required_keys = ['digits', 'pip_value_in_account_currency_per_lot', 'contract_size']
            # Calculate point from digits if not present
            if 'point' not in props:
                digits = props.get('digits', 5)
                props['point'] = 10 ** (-digits)

            if all(key in props for key in required_keys):
                return props
            else:
                self.logger.error(f"Missing one or more required P&L properties for symbol {symbol} in loaded_instrument_details. Required: {required_keys}")
                return None
        else:
            self.logger.error(f"loaded_instrument_details not found in config for P&L calculation.")
            return None

    def _calculate_pnl(self, symbol: str, open_price: float, close_price: float, volume: float, position_action: OrderAction) -> float:
        instrument_props = self._get_instrument_properties_for_pnl(symbol)
        if not instrument_props:
            self.logger.warning(f"Could not get instrument properties for {symbol}. P&L calculation will be inaccurate (0.0).")
            # Fallback to a very simple calculation or return 0, as accuracy is key.
            # For now, returning 0.0 if props are missing, to highlight the issue.
            # A more robust fallback might use self.point and self.lot_size if available,
            # but that would revert to the less accurate old method.
            # price_diff_simple = (close_price - open_price) if position_action == OrderAction.BUY else (open_price - close_price)
            # return price_diff_simple * self.lot_size * volume # Old simplified P&L
            return 0.0

        point_size = instrument_props['point']
        digits = instrument_props['digits']
        pip_value_per_lot = instrument_props['pip_value_in_account_currency_per_lot']
        # contract_size = instrument_props['contract_size'] # Not directly used if pip_value_per_lot is available

        price_diff_points_val = 0.0
        if position_action == OrderAction.BUY:
            price_diff_points_val = close_price - open_price
        elif position_action == OrderAction.SELL:
            price_diff_points_val = open_price - close_price

        # Determine the size of a standard pip based on digits
        # (e.g., 0.0001 for EURUSD (5 digits), 0.01 for USDJPY (3 digits))
        std_pip_increment = 0.0
        if digits <= 3:  # Typically JPY pairs (e.g., USDJPY with 2 or 3 digits, pip is 0.01)
            std_pip_increment = 0.01
        else:  # Typically non-JPY pairs (e.g., EURUSD with 4 or 5 digits, pip is 0.0001)
            std_pip_increment = 0.0001
        
        if std_pip_increment == 0.0: # Should ideally not happen with valid digits
            self.logger.error(f"Could not determine standard pip increment for P&L calc for {symbol} with {digits} digits. P&L will be 0.0.")
            return 0.0

        if point_size == 0.0: # Avoid division by zero
             self.logger.error(f"Point size for {symbol} is 0. Cannot calculate P&L. P&L will be 0.0.")
             return 0.0

        # Calculate price difference in terms of standard pips
        price_diff_in_std_pips = price_diff_points_val / std_pip_increment
        
        # PnL = (Price Difference in Standard Pips) * (Value of 1 Standard Pip per Lot in Account Currency) * (Volume in Lots)
        pnl = price_diff_in_std_pips * pip_value_per_lot * volume
        
        self.logger.debug(f"P&L Calc for {symbol}: Action={position_action.name}, Vol={volume}, Open={open_price}, Close={close_price}")
        self.logger.debug(f"  PriceDiffPoints={price_diff_points_val}, StdPipIncrement={std_pip_increment}, PriceDiffStdPips={price_diff_in_std_pips}")
        self.logger.debug(f"  PipValuePerLot={pip_value_per_lot}, Calculated PnL={pnl:.2f}")
        
        return pnl

    def close_position(self, position_id: str, volume_to_close: Optional[float] = None,
                     price: Optional[float] = None, comment: Optional[str] = None) -> Optional[Order]:
        if not self._is_connected: return None
        if self.current_bar_index < 0 or self.current_bar_index >= len(self.historical_data):
            self.logger.error("No current market data to close position.")
            return None

        position = self.open_positions.get(position_id)
        if not position:
            self.logger.warning(f"Position {position_id} not found for closing.")
            return None
            
        close_volume = volume_to_close if volume_to_close is not None and 0 < volume_to_close <= position.volume else position.volume
        close_price = price if price is not None else self.historical_data.iloc[self.current_bar_index]['close']
        current_time = self.historical_data.iloc[self.current_bar_index]['timestamp']
        if current_time.tzinfo is None: current_time = current_time.replace(tzinfo=timezone.utc)


        realized_pnl = self._calculate_pnl(position.symbol, position.open_price, close_price, close_volume, position.action)
        self.balance += realized_pnl
        
        fill_action = OrderAction.SELL if position.action == OrderAction.BUY else OrderAction.BUY
        
        closing_order_id = self._generate_order_id()
        closing_order = Order(
            order_id=closing_order_id,
            symbol=position.symbol,
            order_type=OrderType.MARKET,
            action=fill_action,
            volume=close_volume,
            price=close_price, # This is the entry price for the closing order, effectively the close price of position
            status=OrderStatus.FILLED,
            created_at=current_time,
            filled_time=current_time, # Time the closing order was filled
            filled_price=close_price, # Price the closing order was filled at
            closed_time=current_time, # Also marks the "closure" of this order itself
            comment=comment if comment else f"Close pos {position_id}",
            pnl=realized_pnl # Store P&L on the closing order itself
        )
        # Store P&L in platform_specific_details as requested
        closing_order.platform_specific_details['pnl'] = realized_pnl
        
        self.all_orders[closing_order_id] = closing_order

        closing_trade_fill = TradeFill(
            fill_id=f"tf_close_{position_id}_{closing_order_id}",
            order_id=closing_order_id, # Use synthetic closing order's ID
            position_id=position_id,
            symbol=position.symbol,
            action=fill_action,
            volume=close_volume,
            price=close_price,
            timestamp=current_time,
            commission=0.0,
            profit_realized_on_fill=realized_pnl # Use profit_realized_on_fill
        )
        self.trade_history.append(closing_trade_fill)

        self.logger.info(f"Position {position_id} ({position.action.name} {close_volume} {position.symbol}) closed at {close_price}. P&L: {realized_pnl:.2f}. New Balance: {self.balance:.2f}") # Use position.action.name

        original_opening_order_id = position.orders_associated[0] if position.orders_associated else None # Use orders_associated

        if close_volume < position.volume:
            position.volume -= close_volume
            self.logger.info(f"Position {position_id} partially closed. Remaining volume: {position.volume:.2f}")
        else:
            del self.open_positions[position_id]
            # Update the original opening order(s) associated with this position
            if original_opening_order_id and original_opening_order_id in self.all_orders:
                opening_order = self.all_orders[original_opening_order_id]
                opening_order.closed_time = current_time
                # Optionally, if the opening order should also reflect P&L from its closure:
                # opening_order.pnl = realized_pnl # This might be complex if partial closes happened before
                # opening_order.platform_specific_details['closed_pnl'] = realized_pnl
            self.logger.info(f"Position {position_id} fully closed.")
        
        # Update equity after balance change (floating PnL for this position is now 0)
        self._update_equity() # Recalculate equity based on new balance and remaining floating PnLs

        return closing_order

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        if not self._is_connected: return []
        
        open_orders_list = list(self.pending_orders.values())
        if symbol:
            return [o for o in open_orders_list if o.symbol == symbol]
        return open_orders_list

    def get_open_positions(self, symbol: Optional[str] = None) -> List[Position]:
        if not self._is_connected: return []
        
        open_positions_list = list(self.open_positions.values())
        if symbol:
            return [p for p in open_positions_list if p.symbol == symbol]
        return open_positions_list

    def get_order(self, order_id: str) -> Optional[Order]: # PlatformInterface calls this get_order_status
        if not self._is_connected: return None
        return self.all_orders.get(order_id)

    # Alias for interface compliance if needed, or rename this method
    def get_order_status(self, order_id: str) -> Optional[Order]:
        return self.get_order(order_id)


    def get_position(self, position_id: str) -> Optional[Position]:
        if not self._is_connected: return None
        return self.open_positions.get(position_id)

    def get_current_backtest_timestamp(self) -> Optional[pd.Timestamp]:
        """
        Returns the current timestamp being processed in the backtest.
        Used by MarketDataManager to provide progressive data windows.
        """
        if not self._is_connected or self.current_bar_index < 0:
            return None

        if self.current_bar_index >= len(self.historical_data):
            return None

        current_bar = self.historical_data.iloc[self.current_bar_index]
        current_timestamp = current_bar['timestamp']

        # Ensure timestamp is pandas Timestamp with timezone
        if not isinstance(current_timestamp, pd.Timestamp):
            current_timestamp = pd.Timestamp(current_timestamp)
        if current_timestamp.tz is None:
            current_timestamp = current_timestamp.tz_localize('UTC')

        return current_timestamp

    def _synchronize_timeframe_indices(self, current_timestamp: datetime) -> None:
        """
        Synchronize all timeframe indices to the current timestamp.

        This ensures that when accessing data for different timeframes,
        we only get data up to the current point in time across all timeframes.

        Args:
            current_timestamp: Current timestamp from primary timeframe
        """
        for timeframe, data in self.timeframe_datasets.items():
            if data.empty:
                continue

            # Find the latest bar index for this timeframe that is <= current_timestamp
            # This simulates real-time progression where higher timeframes may not have
            # new bars at every primary timeframe tick

            # Filter data up to current timestamp
            valid_data = data[data['timestamp'] <= current_timestamp]

            if not valid_data.empty:
                # Set index to the last valid bar
                new_index = len(valid_data) - 1
                self.current_bar_indices[timeframe] = new_index

                self.logger.debug(f"Timeframe {timeframe.name} synchronized to index {new_index} "
                                f"(timestamp: {valid_data.iloc[-1]['timestamp']})")
            else:
                # No data available yet for this timeframe
                self.current_bar_indices[timeframe] = -1

    def next_bar(self) -> bool:
        if not self._is_connected:
            self.logger.error("Not connected. Cannot advance to next bar.")
            return False

        # Advance primary timeframe (historical_data)
        self.current_bar_index += 1
        if self.current_bar_index >= len(self.historical_data):
            self.logger.info("End of historical data reached.")
            return False

        # Update multi-timeframe indices based on current timestamp
        current_bar_series = self.historical_data.iloc[self.current_bar_index]
        current_time = current_bar_series['timestamp']
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)

        # Synchronize all timeframe indices to current timestamp
        self._synchronize_timeframe_indices(current_time)

        current_price_open = current_bar_series['open']
        current_price_low = current_bar_series['low']
        current_price_high = current_bar_series['high']
        current_price_close = current_bar_series['close']


        self.logger.debug(f"Processing Bar {self.current_bar_index}: Time={current_time}, O={current_price_open}, H={current_price_high}, L={current_price_low}, C={current_price_close}")

        # --- 0. Emit tick data for this bar ---
        # Create tick data from current bar's close price and emit it to subscribers
        tick_data = TickData(
            timestamp=current_time,
            bid=current_price_close,  # Simplification: bid/ask spread is zero
            ask=current_price_close,
            symbol=self.symbol
        )
        self._on_tick(tick_data)  # Notify MarketDataManager and other tick subscribers

        # --- 1. Process Market Orders Pending Fill from Previous Iteration ---
        if self.market_orders_pending_fill_next_bar:
            orders_to_fill_this_bar = list(self.market_orders_pending_fill_next_bar) # Copy before modifying list
            self.market_orders_pending_fill_next_bar.clear()

            for order_id_to_fill in orders_to_fill_this_bar:
                order = self.all_orders.get(order_id_to_fill)
                if not order or order.status != OrderStatus.PENDING_OPEN:
                    self.logger.warning(f"Market order {order_id_to_fill} not found or not in PENDING_OPEN state. Skipping fill.")
                    continue

                # Fill market order at the current bar's OPEN price
                fill_price = current_price_open
                order.filled_price = fill_price
                order.filled_time = current_time # Use current bar's timestamp
                order.status = OrderStatus.FILLED
                
                position_id = self._generate_position_id()
                new_position = Position(
                    position_id=position_id,
                    orders_associated=[order.order_id],
                    symbol=order.symbol,
                    action=order.action,
                    volume=order.volume,
                    open_price=fill_price,
                    open_time=current_time,
                    stop_loss=order.stop_loss,
                    take_profit=order.take_profit,
                    comment=order.comment,
                    profit=0.0
                )
                self.open_positions[position_id] = new_position
                
                trade_fill = TradeFill(
                    fill_id=f"tf_market_open_{order.order_id}",
                    order_id=order.order_id,
                    position_id=position_id,
                    symbol=order.symbol,
                    action=order.action,
                    volume=order.volume,
                    price=fill_price,
                    timestamp=current_time,
                    commission=0.0, # Simplified
                    profit_realized_on_fill=0.0
                )
                self.trade_history.append(trade_fill)
                self.logger.info(f"Market order {order.order_id} ({order.action.name} {order.volume} {order.symbol}) filled at {fill_price} (bar open). Position {position_id} opened.")

        # --- 1. Update Floating P/L and Equity for Open Positions ---
        self._update_equity(current_price_close=current_price_close)


        # --- 2. Check SL/TP Triggers for Open Positions ---
        # Iterate over a copy of items in case positions are closed and removed from dict
        for pos_id, pos in list(self.open_positions.items()):
            # Ensure position still exists, might have been closed by a previous trigger in same bar (less likely here)
            if pos_id not in self.open_positions:
                continue

            triggered_close_price = None
            close_reason = ""

            if pos.action == OrderAction.BUY: # Use pos.action
                if pos.stop_loss is not None and current_price_low <= pos.stop_loss:
                    triggered_close_price = pos.stop_loss
                    close_reason = "Stop Loss"
                elif pos.take_profit is not None and current_price_high >= pos.take_profit:
                    triggered_close_price = pos.take_profit
                    close_reason = "Take Profit"
            elif pos.action == OrderAction.SELL: # Use pos.action
                if pos.stop_loss is not None and current_price_high >= pos.stop_loss:
                    triggered_close_price = pos.stop_loss
                    close_reason = "Stop Loss"
                elif pos.take_profit is not None and current_price_low <= pos.take_profit:
                    triggered_close_price = pos.take_profit
                    close_reason = "Take Profit"

            if triggered_close_price is not None:
                self.logger.info(f"Position {pos_id} ({pos.action.name}) {close_reason} triggered at {triggered_close_price} (Bar L/H: {current_price_low}/{current_price_high}).") # Use pos.action.name
                closing_order_result = self.close_position(pos_id, volume_to_close=pos.volume, price=triggered_close_price, comment=f"{close_reason} triggered") # Add comment
                if not closing_order_result:
                     self.logger.error(f"Failed to close position {pos_id} for {close_reason} trigger.")


        # --- 3. Check Pending Order Triggers ---
        # Iterate over a copy of items in case orders are filled and removed from dict
        for order_id, order in list(self.pending_orders.items()):
            # Ensure order still exists and is pending
            if order_id not in self.pending_orders:
                continue
            
            actual_fill_price = None # This will be order.price as per new requirement

            if order.action == OrderAction.BUY:
                if order.order_type == OrderType.LIMIT and current_price_low <= order.price:
                    actual_fill_price = order.price # Fill at order.price
                    self.logger.info(f"Pending BUY LIMIT order {order_id} triggered at {actual_fill_price} (Bar Low: {current_price_low})")
                elif order.order_type == OrderType.STOP and current_price_high >= order.price:
                    actual_fill_price = order.price # Fill at order.price
                    self.logger.info(f"Pending BUY STOP order {order_id} triggered at {actual_fill_price} (Bar High: {current_price_high})")
            
            elif order.action == OrderAction.SELL:
                if order.order_type == OrderType.LIMIT and current_price_high >= order.price:
                    actual_fill_price = order.price # Fill at order.price
                    self.logger.info(f"Pending SELL LIMIT order {order_id} triggered at {actual_fill_price} (Bar High: {current_price_high})")
                elif order.order_type == OrderType.STOP and current_price_low <= order.price:
                    actual_fill_price = order.price # Fill at order.price
                    self.logger.info(f"Pending SELL STOP order {order_id} triggered at {actual_fill_price} (Bar Low: {current_price_low})")

            if actual_fill_price is not None:
                # Update the order in all_orders
                # 'order' is already the instance from self.pending_orders, which is also in self.all_orders
                order.filled_price = actual_fill_price
                order.filled_time = current_time
                order.status = OrderStatus.FILLED
                
                # Remove from pending_orders dict
                del self.pending_orders[order_id]

                position_id = self._generate_position_id()
                new_position = Position(
                    position_id=position_id,
                    orders_associated=[order_id],
                    symbol=order.symbol,
                    action=order.action,
                    volume=order.volume,
                    open_price=actual_fill_price,
                    open_time=current_time,
                    stop_loss=order.stop_loss,
                    take_profit=order.take_profit,
                    comment=order.comment,
                    profit=0.0
                )
                self.open_positions[position_id] = new_position
                
                trade_fill = TradeFill(
                    fill_id=f"tf_pending_fill_{order_id}",
                    order_id=order_id,
                    position_id=position_id,
                    symbol=order.symbol,
                    action=order.action,
                    volume=order.volume,
                    price=actual_fill_price,
                    timestamp=current_time,
                    commission=0.0, # Simplified
                    profit_realized_on_fill=0.0
                )
                self.trade_history.append(trade_fill)
                self.logger.info(f"Pending order {order_id} ({order.order_type.name} {order.action.name}) filled at {actual_fill_price}. Position {position_id} opened.")
        
        # --- 4. Log Equity History ---
        self.equity_history.append((current_time, self.equity))
        self.logger.debug(f"Appended to equity history: ({current_time}, {self.equity:.2f}). Length: {len(self.equity_history)}")

        return True

    def _update_equity(self, current_price_close: Optional[float] = None):
        """Helper to recalculate total floating P&L and update equity."""
        total_floating_pnl = 0.0
        if current_price_close is not None: # Only calculate floating PnL if current price is given
            for pos in self.open_positions.values():
                # P&L calculation needs symbol
                pos.profit = self._calculate_pnl(pos.symbol, pos.open_price, current_price_close, pos.volume, pos.action)
                total_floating_pnl += pos.profit if pos.profit is not None else 0.0
        
        self.equity = self.balance + total_floating_pnl
        self.logger.debug(f"Equity updated: {self.equity:.2f} (Balance: {self.balance:.2f}, Floating PnL: {total_floating_pnl:.2f})")


    # --- Other PlatformInterface methods (placeholders or simple implementations) ---
    def get_server_time(self) -> Optional[datetime]:
        if not self._is_connected: return datetime.now(timezone.utc)
        if self.current_bar_index >= 0 and self.current_bar_index < len(self.historical_data):
            server_t = self.historical_data.iloc[self.current_bar_index]['timestamp']
            if server_t.tzinfo is None: server_t = server_t.replace(tzinfo=timezone.utc) # Corrected
            return server_t
        return datetime.now(timezone.utc)

    def get_trade_history(self, symbol: Optional[str] = None, # Added symbol
                          start_time: Optional[datetime] = None, # Changed from_time
                          end_time: Optional[datetime] = None,   # Changed to_time
                          count: Optional[int] = None) -> List[TradeFill]: # Added count, removed order/pos ids
        if not self._is_connected: return []
        
        history_to_return = list(self.trade_history)

        if symbol: # Added symbol filter
            history_to_return = [tf for tf in history_to_return if tf.symbol == symbol]
        if start_time:
            if start_time.tzinfo is None: start_time = start_time.replace(tzinfo=timezone.utc) # Corrected
            history_to_return = [tf for tf in history_to_return if tf.timestamp >= start_time]
        if end_time:
            if end_time.tzinfo is None: end_time = end_time.replace(tzinfo=timezone.utc) # Corrected
            history_to_return = [tf for tf in history_to_return if tf.timestamp <= end_time]
        
        if count is not None and count > 0: # Added count logic
            history_to_return.sort(key=lambda tf: tf.timestamp, reverse=True)
            history_to_return = history_to_return[:count]
            history_to_return.sort(key=lambda tf: tf.timestamp)
            
        return history_to_return

    def close_all_positions(self, symbol: Optional[str] = None) -> bool: # Interface doesn't define this, but it's useful
        if not self._is_connected: return False
        if self.current_bar_index < 0:
            self.logger.warning("Cannot close all positions, no current bar data.")
            return False
            
        closed_any = False
        for pos_id in list(self.open_positions.keys()):
            pos = self.open_positions.get(pos_id)
            if pos and (symbol is None or pos.symbol == symbol):
                self.logger.info(f"Closing position {pos_id} as part of close_all_positions.")
                closing_order = self.close_position(pos_id, comment="Close all positions") # Add comment
                if closing_order and closing_order.status == OrderStatus.FILLED:
                    closed_any = True
        return closed_any

    def cancel_all_orders(self, symbol: Optional[str] = None) -> bool:
        if not self._is_connected: return False
        cancelled_any = False
        # Iterate over a copy of keys as cancel_order modifies self.pending_orders
        for order_id in list(self.pending_orders.keys()):
            order = self.pending_orders.get(order_id) # Re-fetch
            if order and (symbol is None or order.symbol == symbol):
                self.logger.info(f"Cancelling order {order_id} as part of cancel_all_orders.")
                cancelled_order_result = self.cancel_order(order_id) # Use different var name
                if cancelled_order_result and cancelled_order_result.status == OrderStatus.CANCELLED:
                    cancelled_any = True
        return cancelled_any

    # --- Methods required by PlatformInterface but not fully implemented/used by paper trader ---
    def get_all_tradable_symbols(self) -> List[SymbolInfo]:
        self.logger.warning("get_all_tradable_symbols not fully implemented for PaperTradingAdapter. Returning info for primary symbol.")
        info = self.get_symbol_info(self.symbol)
        return [info] if info else []

    def subscribe_ticks(self, symbol: str, callback: Any) -> bool: # Type Any for callback from base
        self.logger.info(f"PaperTradingAdapter: Tick subscription for {symbol} noted. Data is driven by next_bar().")
        # Store the callback using the base class method
        super().register_tick_subscriber(symbol, callback)
        return True

    def unsubscribe_ticks(self, symbol: str, callback: Optional[Any] = None) -> bool:
        self.logger.info(f"PaperTradingAdapter: Tick unsubscription for {symbol} noted.")
        if callback:
            super().unregister_tick_subscriber(symbol, callback)
        else:
            # Remove all callbacks for this symbol
            if symbol in self.tick_subscribers:
                del self.tick_subscribers[symbol]
        return True

    def subscribe_bars(self, symbol: str, timeframe: Timeframe, callback: Any) -> bool:
        self.logger.info(f"PaperTradingAdapter: Bar subscription for {symbol}/{timeframe.name} noted. Data is driven by next_bar().")
        return True

    def unsubscribe_bars(self, symbol: str, timeframe: Timeframe, callback: Optional[Any] = None) -> bool:
        self.logger.info(f"PaperTradingAdapter: Bar unsubscription for {symbol}/{timeframe.name} noted.")
        return True
    
    def get_order_history(
        self,
        symbol: Optional[str] = None,
        start_time: Optional[datetime] = None, # UTC-aware
        end_time: Optional[datetime] = None,   # UTC-aware
        count: Optional[int] = None
    ) -> List[Order]:
        orders_to_return = list(self.all_orders.values())
        if symbol:
            orders_to_return = [o for o in orders_to_return if o.symbol == symbol]
        if start_time:
            if start_time.tzinfo is None: start_time = start_time.replace(tzinfo=timezone.utc)
            orders_to_return = [o for o in orders_to_return if o.created_at >= start_time] # Using created_at
        if end_time:
            if end_time.tzinfo is None: end_time = end_time.replace(tzinfo=timezone.utc)
            orders_to_return = [o for o in orders_to_return if o.created_at <= end_time]
        
        # Sort by creation time to make 'count' meaningful for latest/earliest
        orders_to_return.sort(key=lambda o: o.created_at, reverse=True) # Latest first
        if count is not None and count > 0:
            orders_to_return = orders_to_return[:count]
            orders_to_return.sort(key=lambda o: o.created_at) # Optional: re-sort to chronological
        
        return orders_to_return

  
