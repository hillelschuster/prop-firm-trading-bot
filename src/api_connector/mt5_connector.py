# prop_firm_trading_bot/adapters/mt5_adapter.py

import MetaTrader5 as mt5
import numpy as np # Added import
from datetime import datetime, timezone, timedelta
import time
import pytz # For timezone handling with MT5
import uuid # For generating client_order_ids if not provided
import threading # For polling loop

from src.api_connector.base_connector import PlatformInterface
from src.core.enums import OrderType, OrderAction, OrderStatus, PositionStatus, Timeframe
from src.core.models import (
    Order, Position, AccountInfo, OHLCVData, TickData, SymbolInfo, TradeFill, MarketEvent
)
# Assuming config_loader and custom_logger are accessible or config/logger are passed directly
# from src.utils.config_loader import AppConfig # Passed in constructor
# from src.utils.custom_logger import setup_logging # Logger passed in constructor

from typing import List, Optional, Callable, Any, Dict

class MT5Adapter(PlatformInterface):
    """MetaTrader 5 Platform Adapter."""

    def __init__(self, config, logger): # config is AppConfig
        super().__init__(config.platform.mt5, logger)
        self.platform_config = config.platform.mt5
        self.bot_config = config.bot_settings
        self.credentials = config.Config.platform_credentials # Fetched by config_loader
        self._is_connected = False
        self.mt5_timeframe_map = {
            Timeframe.M1: mt5.TIMEFRAME_M1,
            Timeframe.M5: mt5.TIMEFRAME_M5,
            Timeframe.M15: mt5.TIMEFRAME_M15,
            Timeframe.M30: mt5.TIMEFRAME_M30,
            Timeframe.H1: mt5.TIMEFRAME_H1,
            Timeframe.H4: mt5.TIMEFRAME_H4,
            Timeframe.D1: mt5.TIMEFRAME_D1,
            Timeframe.W1: mt5.TIMEFRAME_W1,
            Timeframe.MN1: mt5.TIMEFRAME_MN1,
        }
        self._subscribed_tick_symbols = set()
        self._subscribed_bar_symbols_tf = {} # Dict[str, Dict[Timeframe, datetime]] symbol -> timeframe -> last_bar_time_processed

        self._polling_active = False
        self._polling_thread: Optional[threading.Thread] = None
        self._polling_interval_seconds = 5 # How often to poll for data (configurable or adaptive)
        
        self._last_tick_data: Dict[str, TickData] = {} # Store last tick to only push updates
        self._last_ohlcv_data: Dict[str, Dict[Timeframe, OHLCVData]] = {} # Store last bar data

        self.ftmo_timezone = pytz.timezone(self.bot_config.ftmo_server_timezone) # [cite: 1] config.yaml

    def _map_mt5_order_type_to_common(self, mt5_order_type: int) -> Optional[OrderType]:
        mapping = {
            mt5.ORDER_TYPE_BUY: OrderType.MARKET, # Assuming direct market execution for simple BUY
            mt5.ORDER_TYPE_SELL: OrderType.MARKET, # Assuming direct market execution for simple SELL
            mt5.ORDER_TYPE_BUY_LIMIT: OrderType.LIMIT,
            mt5.ORDER_TYPE_SELL_LIMIT: OrderType.LIMIT,
            mt5.ORDER_TYPE_BUY_STOP: OrderType.STOP,
            mt5.ORDER_TYPE_SELL_STOP: OrderType.STOP,
            # mt5.ORDER_TYPE_BUY_STOP_LIMIT: OrderType.STOP_LIMIT, # If supporting
            # mt5.ORDER_TYPE_SELL_STOP_LIMIT: OrderType.STOP_LIMIT, # If supporting
        }
        return mapping.get(mt5_order_type)

    def _map_common_order_type_to_mt5(self, common_order_type: OrderType, action: OrderAction) -> Optional[int]:
        if common_order_type == OrderType.MARKET:
            return mt5.ORDER_TYPE_BUY if action == OrderAction.BUY else mt5.ORDER_TYPE_SELL
        elif common_order_type == OrderType.LIMIT:
            return mt5.ORDER_TYPE_BUY_LIMIT if action == OrderAction.BUY else mt5.ORDER_TYPE_SELL_LIMIT
        elif common_order_type == OrderType.STOP:
            return mt5.ORDER_TYPE_BUY_STOP if action == OrderAction.BUY else mt5.ORDER_TYPE_SELL_STOP
        # Add other mappings if OrderType enum is expanded
        return None

    def _map_mt5_order_state_to_common_status(self, mt5_order_state: int) -> OrderStatus:
        # Ref: https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_state
        mapping = {
            mt5.ORDER_STATE_STARTED: OrderStatus.PENDING_OPEN, # Just accepted by the server
            mt5.ORDER_STATE_PLACED: OrderStatus.OPEN,      # Pending order placed in order book
            mt5.ORDER_STATE_CANCELED: OrderStatus.CANCELLED,
            mt5.ORDER_STATE_PARTIAL: OrderStatus.PARTIALLY_FILLED, # Partially filled
            mt5.ORDER_STATE_FILLED: OrderStatus.FILLED,       # Fully filled
            mt5.ORDER_STATE_REJECTED: OrderStatus.REJECTED,
            mt5.ORDER_STATE_EXPIRED: OrderStatus.EXPIRED,
            # mt5.ORDER_STATE_REQUEST_ADD: OrderStatus.NEW, # Not a final state
            # mt5.ORDER_STATE_REQUEST_MODIFY: OrderStatus.PENDING_OPEN, # Not a final state
            # mt5.ORDER_STATE_REQUEST_CANCEL: OrderStatus.PENDING_CANCEL # Not a final state
        }
        return mapping.get(mt5_order_state, OrderStatus.ERROR) # Default to ERROR for unknown states

    def _map_mt5_position_type_to_common_action(self, mt5_position_type: int) -> Optional[OrderAction]:
        # Ref: https://www.mql5.com/en/docs/constants/tradingconstants/positionproperties#enum_position_type
        if mt5_position_type == mt5.POSITION_TYPE_BUY:
            return OrderAction.BUY
        elif mt5_position_type == mt5.POSITION_TYPE_SELL:
            return OrderAction.SELL
        return None

    def _convert_mt5_order_to_common_order(self, mt5_order: Any) -> Optional[Order]:
        if not mt5_order:
            return None
        
        common_order_type = self._map_mt5_order_type_to_common(mt5_order.type)
        if not common_order_type:
            self.logger.warning(f"Unsupported MT5 order type {mt5_order.type} for order ticket {mt5_order.ticket}")
            return None # Or handle as an error/unknown type

        action = OrderAction.BUY if mt5_order.type in [mt5.ORDER_TYPE_BUY, mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_BUY_STOP] else OrderAction.SELL

        return Order(
            order_id=str(mt5_order.ticket),
            symbol=mt5_order.symbol,
            order_type=common_order_type,
            action=action,
            volume=mt5_order.volume_current, # volume_initial for original volume
            price=mt5_order.price_open if common_order_type != OrderType.MARKET else mt5_order.price_current, # price_open for pending, price_current for market if filled
            stop_loss=mt5_order.sl if mt5_order.sl > 0 else None,
            take_profit=mt5_order.tp if mt5_order.tp > 0 else None,
            status=self._map_mt5_order_state_to_common_status(mt5_order.state),
            created_at=datetime.fromtimestamp(mt5_order.time_setup, tz=timezone.utc), # time_setup is placement time
            updated_at=datetime.fromtimestamp(mt5_order.time_update, tz=timezone.utc) if mt5_order.time_update > 0 else None,
            filled_price=mt5_order.price_current if mt5_order.state == mt5.ORDER_STATE_FILLED or mt5_order.state == mt5.ORDER_STATE_PARTIAL else None, # Price of execution
            filled_volume=mt5_order.volume_current if mt5_order.state == mt5.ORDER_STATE_FILLED or mt5_order.state == mt5.ORDER_STATE_PARTIAL else 0.0,
            comment=mt5_order.comment,
            platform_specific_details={
                "magic": mt5_order.magic,
                "time_done_msc": mt5_order.time_done_msc,
                "type_filling": mt5_order.type_filling,
                "state": mt5_order.state,
                "reason": getattr(mt5_order, 'reason', None) # If available
            }
        )

    def _convert_mt5_position_to_common_position(self, mt5_pos: Any) -> Optional[Position]:
        if not mt5_pos:
            return None
        action = self._map_mt5_position_type_to_common_action(mt5_pos.type)
        if not action:
            self.logger.warning(f"Unsupported MT5 position type {mt5_pos.type} for position ticket {mt5_pos.ticket}")
            return None

        return Position(
            position_id=str(mt5_pos.ticket),
            symbol=mt5_pos.symbol,
            action=action,
            volume=mt5_pos.volume,
            open_price=mt5_pos.price_open,
            current_price=mt5_pos.price_current,
            stop_loss=mt5_pos.sl if mt5_pos.sl > 0 else None,
            take_profit=mt5_pos.tp if mt5_pos.tp > 0 else None,
            open_time=datetime.fromtimestamp(mt5_pos.time, tz=timezone.utc),
            # close_time is not part of MT5 position object directly
            commission=mt5_pos.commission,
            swap=mt5_pos.swap,
            profit=mt5_pos.profit,
            comment=mt5_pos.comment,
            status=PositionStatus.OPEN, # MT5 positions_get only returns open positions
            platform_specific_details={
                "magic": mt5_pos.magic,
                "identifier": mt5_pos.identifier, # Link to order that opened it
                "time_msc": mt5_pos.time_msc,
                "time_update_msc": mt5_pos.time_update_msc,
            }
        )

    def _convert_mt5_symbol_info_to_common_symbol_info(self, mt5_sym_info: Any) -> Optional[SymbolInfo]:
        if not mt5_sym_info:
            return None
        return SymbolInfo(
            name=mt5_sym_info.name,
            description=mt5_sym_info.description,
            digits=mt5_sym_info.digits,
            point=mt5_sym_info.point,
            min_volume=mt5_sym_info.volume_min,
            max_volume=mt5_sym_info.volume_max,
            volume_step=mt5_sym_info.volume_step,
            contract_size=mt5_sym_info.trade_contract_size,
            currency_base=mt5_sym_info.currency_base,
            currency_profit=mt5_sym_info.currency_profit,
            currency_margin=mt5_sym_info.currency_margin,
            trade_allowed=mt5_sym_info.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL,
            platform_specific_details={
                "spread": mt5_sym_info.spread,
                "bid": mt5_sym_info.bid,
                "ask": mt5_sym_info.ask,
                "session_deals": mt5_sym_info.session_deals, # Number of deals in current session
                "trade_mode": mt5_sym_info.trade_mode,
                "tick_value": mt5_sym_info.trade_tick_value,
                "tick_size": mt5_sym_info.trade_tick_size,
            }
        )

    def _convert_mt5_account_info_to_common_account_info(self, mt5_acc_info: Any) -> Optional[AccountInfo]:
        if not mt5_acc_info:
            return None
        return AccountInfo(
            account_id=str(mt5_acc_info.login),
            balance=mt5_acc_info.balance,
            equity=mt5_acc_info.equity,
            margin=mt5_acc_info.margin,
            margin_free=mt5_acc_info.margin_free,
            margin_level_pct=mt5_acc_info.margin_level if mt5_acc_info.margin_level > 0 else None,
            currency=mt5_acc_info.currency,
            server_time=datetime.fromtimestamp(mt5.terminal_info().time_server, tz=self.ftmo_timezone), # Server time from terminal_info
            platform_specific_details={
                "name": mt5_acc_info.name,
                "server": mt5_acc_info.server,
                "leverage": mt5_acc_info.leverage,
                "trade_mode": mt5_acc_info.trade_mode # ENUM_ACCOUNT_TRADE_MODE
            }
        )
        
    def _convert_mt5_tick_to_common_tick(self, symbol: str, mt5_tick: Any) -> Optional[TickData]:
        if not mt5_tick:
            return None
        return TickData(
            timestamp=datetime.fromtimestamp(mt5_tick.time, tz=timezone.utc).astimezone(self.ftmo_timezone),
            symbol=symbol,
            bid=mt5_tick.bid,
            ask=mt5_tick.ask,
            last=mt5_tick.last if mt5_tick.last > 0 else None,
            volume=mt5_tick.volume if hasattr(mt5_tick, 'volume') and mt5_tick.volume > 0 else None # Tick volume for exchange instruments
        )

    def _convert_mt5_rates_to_common_ohlcv(self, symbol: str, timeframe: Timeframe, mt5_rates: np.ndarray) -> List[OHLCVData]:
        common_rates = []
        if mt5_rates is None:
            return common_rates
        for rate in mt5_rates:
            # MT5 rate is a tuple or structured array: (time, open, high, low, close, tick_volume, spread, real_volume)
            common_rates.append(OHLCVData(
                timestamp=datetime.fromtimestamp(rate['time'], tz=timezone.utc).astimezone(self.ftmo_timezone),
                symbol=symbol,
                timeframe=timeframe,
                open=rate['open'],
                high=rate['high'],
                low=rate['low'],
                close=rate['close'],
                volume=rate['tick_volume'] # Or rate['real_volume'] if available and preferred
            ))
        return common_rates
    
    def _poll_data_and_events(self):
        """Internal polling loop to fetch data and check for events."""
        self.logger.info("MT5Adapter polling thread started.")
        while self._polling_active:
            if not self._is_connected:
                time.sleep(self._polling_interval_seconds) # Wait before retrying connection
                if not self.connect(): # Attempt to reconnect
                    self.logger.warning("Polling: Reconnect failed. Will retry later.")
                    self._on_error("Polling: Reconnect failed in polling loop", None)
                continue

            # Poll for subscribed ticks
            for symbol in list(self._subscribed_tick_symbols): # List to allow modification during iteration if needed
                tick = mt5.symbol_info_tick(symbol)
                if tick:
                    common_tick = self._convert_mt5_tick_to_common_tick(symbol, tick)
                    if common_tick:
                        last_known_tick = self._last_tick_data.get(symbol)
                        if not last_known_tick or common_tick.timestamp > last_known_tick.timestamp or \
                           common_tick.bid != last_known_tick.bid or common_tick.ask != last_known_tick.ask:
                            self._last_tick_data[symbol] = common_tick
                            self._on_tick(common_tick)
                else:
                    err_code, err_desc = mt5.last_error()
                    self.logger.warning(f"Polling: Failed to get tick for {symbol}. Error {err_code}: {err_desc}")
                    self._on_error(f"Polling: Tick fetch error for {symbol} ({err_code})", None)


            # Poll for subscribed bars
            for symbol, tf_map in list(self._subscribed_bar_symbols_tf.items()):
                for timeframe_enum, last_bar_time in list(tf_map.items()):
                    mt5_tf = self.mt5_timeframe_map.get(timeframe_enum)
                    if not mt5_tf:
                        continue
                    
                    # Fetch the last 2 bars to check if a new one has formed
                    rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, 2)
                    if rates is not None and len(rates) > 0:
                        latest_bar_mt5 = rates[-1]
                        latest_bar_time_utc = datetime.fromtimestamp(latest_bar_mt5['time'], tz=timezone.utc)
                        
                        # Compare with last processed bar time for this symbol/tf
                        if latest_bar_time_utc > last_bar_time:
                            self._subscribed_bar_symbols_tf[symbol][timeframe_enum] = latest_bar_time_utc
                            common_bar = OHLCVData(
                                timestamp=latest_bar_time_utc.astimezone(self.ftmo_timezone),
                                symbol=symbol,
                                timeframe=timeframe_enum,
                                open=latest_bar_mt5['open'],
                                high=latest_bar_mt5['high'],
                                low=latest_bar_mt5['low'],
                                close=latest_bar_mt5['close'],
                                volume=latest_bar_mt5['tick_volume']
                            )
                            self._last_ohlcv_data.setdefault(symbol, {})[timeframe_enum] = common_bar
                            self._on_bar(common_bar)
                    else:
                        err_code, err_desc = mt5.last_error()
                        self.logger.warning(f"Polling: Failed to get rates for {symbol}/{timeframe_enum.name}. Error {err_code}: {err_desc}")
                        self._on_error(f"Polling: Bar fetch error for {symbol}/{timeframe_enum.name} ({err_code})", None)
            
            # Poll for order/position updates (MT5 doesn't push these to Python directly)
            # This is a simplified check. A more robust system would track order/position states.
            # For now, let's assume updates are checked less frequently or driven by actions.
            # We could periodically call get_open_orders() and get_open_positions() and compare.

            time.sleep(self._polling_interval_seconds)
        self.logger.info("MT5Adapter polling thread stopped.")

    def connect(self) -> bool:
        if self._is_connected:
            self.logger.info("MT5Adapter already connected.")
            return True
        
        path = self.platform_config.path if self.platform_config else None
        login = int(self.credentials.get('mt5_account', 0))
        password = self.credentials.get('mt5_password', "")
        server = self.credentials.get('mt5_server', "")
        timeout = self.platform_config.timeout_ms if self.platform_config else 10000

        self.logger.info(f"Attempting to initialize MT5: Path='{path}', Login='{login}', Server='{server}'")
        
        init_params = {}
        if path: init_params['path'] = path
        if login: init_params['login'] = login
        if password: init_params['password'] = password
        if server: init_params['server'] = server
        init_params['timeout'] = timeout
        # init_params['portable'] = False # Optional: if MT5 is in portable mode

        if not mt5.initialize(**init_params):
            err_code, err_desc = mt5.last_error()
            self.logger.error(f"MT5 initialize() failed. Error {err_code}: {err_desc}")
            self._is_connected = False
            self._on_error(f"MT5 Connection Failed ({err_code})", None)
            return False

        # Check login status
        if not mt5.terminal_info().connected or not mt5.account_info():
             err_code, err_desc = mt5.last_error()
             self.logger.error(f"MT5 connected to terminal but not logged into trade account or no account info. Error {err_code}: {err_desc}")
             mt5.shutdown()
             self._is_connected = False
             self._on_error(f"MT5 Login/Account Info Failed ({err_code})", None)
             return False

        self.logger.info(f"MT5 initialized and logged in successfully to account {mt5.account_info().login} on server {mt5.account_info().server}.")
        self._is_connected = True
        
        # Start polling thread if there are subscriptions
        if (self._subscribed_tick_symbols or self._subscribed_bar_symbols_tf) and not self._polling_active:
            self._polling_active = True
            self._polling_thread = threading.Thread(target=self._poll_data_and_events, daemon=True)
            self._polling_thread.start()
        return True

    def disconnect(self) -> None:
        self.logger.info("Disconnecting MT5Adapter...")
        self._polling_active = False # Signal polling thread to stop
        if self._polling_thread and self._polling_thread.is_alive():
            self.logger.info("Waiting for polling thread to terminate...")
            self._polling_thread.join(timeout=self._polling_interval_seconds * 2) # Wait for it
            if self._polling_thread.is_alive():
                 self.logger.warning("Polling thread did not terminate cleanly.")
        
        if self._is_connected:
            mt5.shutdown()
            self._is_connected = False
            self.logger.info("MT5 connection shut down.")
        else:
            self.logger.info("MT5Adapter was not connected.")

    def is_connected(self) -> bool:
        # Check actual terminal and account connection status beyond our flag
        if not self._is_connected:
            return False
        term_info = mt5.terminal_info()
        acc_info = mt5.account_info()
        if not term_info or not term_info.connected or not acc_info or acc_info.login == 0:
            self.logger.warning("MT5 connection check failed: terminal or account not properly connected.")
            self._is_connected = False # Update our state
            self._on_error("MT5 connection lost (terminal/account status)", None)
        return self._is_connected

    def get_account_info(self) -> Optional[AccountInfo]:
        if not self.is_connected():
            self.logger.error("Cannot get account info: Not connected to MT5.")
            return None
        mt5_acc_info = mt5.account_info()
        if mt5_acc_info:
            common_acc_info = self._convert_mt5_account_info_to_common_account_info(mt5_acc_info)
            self._on_account_update(common_acc_info) # Dispatch event
            return common_acc_info
        else:
            err_code, err_desc = mt5.last_error()
            self.logger.error(f"Failed to retrieve MT5 account info. Error {err_code}: {err_desc}")
            self._on_error(f"MT5 Get Account Info Failed ({err_code})", None)
            return None

    def get_symbol_info(self, symbol: str) -> Optional[SymbolInfo]:
        if not self.is_connected():
            self.logger.error(f"Cannot get symbol info for {symbol}: Not connected.")
            return None
        
        # Ensure symbol is selected/visible
        selected = mt5.symbol_select(symbol, True)
        if not selected:
            # Try to select again, sometimes it needs a moment
            time.sleep(0.1)
            selected = mt5.symbol_select(symbol,True)
            if not selected:
                self.logger.warning(f"Symbol {symbol} could not be selected in MarketWatch. Info might be partial or unavailable.")
                # We can still try to get info, MT5 might return it if symbol exists on server
        
        mt5_sym_info = mt5.symbol_info(symbol)
        if mt5_sym_info:
            return self._convert_mt5_symbol_info_to_common_symbol_info(mt5_sym_info)
        else:
            err_code, err_desc = mt5.last_error()
            self.logger.error(f"Failed to retrieve symbol info for {symbol}. Error {err_code}: {err_desc}")
            self._on_error(f"MT5 Get Symbol Info Failed for {symbol} ({err_code})", None)
            return None

    def get_all_tradable_symbols(self) -> List[SymbolInfo]:
        if not self.is_connected():
            self.logger.error("Cannot get tradable symbols: Not connected.")
            return []
        
        symbols_raw = mt5.symbols_get() # Gets all symbols available on server
        tradable_symbols_info = []
        if symbols_raw:
            for s_raw in symbols_raw:
                # Check if it's usable/tradable by trying to get more info, or if it's in our config.assets
                # For simplicity, we'll convert all that we get info for.
                # A production system might filter based on config.assets.
                if mt5.symbol_select(s_raw.name, True): # Ensure it's in MarketWatch
                    s_info = mt5.symbol_info(s_raw.name)
                    if s_info and s_info.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL:
                        common_s_info = self._convert_mt5_symbol_info_to_common_symbol_info(s_info)
                        if common_s_info:
                             tradable_symbols_info.append(common_s_info)
        else:
            err_code, err_desc = mt5.last_error()
            self.logger.error(f"Failed to retrieve symbols from server. Error {err_code}: {err_desc}")
            self._on_error(f"MT5 Get All Symbols Failed ({err_code})", None)
        return tradable_symbols_info

    def get_latest_tick(self, symbol: str) -> Optional[TickData]:
        if not self.is_connected():
            self.logger.error(f"Cannot get latest tick for {symbol}: Not connected.")
            return None
        tick = mt5.symbol_info_tick(symbol)
        if tick:
            return self._convert_mt5_tick_to_common_tick(symbol, tick)
        else:
            err_code, err_desc = mt5.last_error()
            self.logger.warning(f"Failed to get latest tick for {symbol}. Error {err_code}: {err_desc}")
            return None # Don't trigger global error for every failed tick pull if polling

    def get_historical_ohlcv(
        self,
        symbol: str,
        timeframe: Timeframe,
        start_time: Optional[datetime] = None, # UTC-aware datetime
        end_time: Optional[datetime] = None,   # UTC-aware datetime
        count: Optional[int] = None
    ) -> List[OHLCVData]:
        if not self.is_connected():
            self.logger.error(f"Cannot get historical OHLCV for {symbol}: Not connected.")
            return []

        mt5_tf = self.mt5_timeframe_map.get(timeframe)
        if not mt5_tf:
            self.logger.error(f"Unsupported timeframe {timeframe.name} for MT5.")
            self._on_error(f"MT5 Get Historical OHLCV: Unsupported timeframe {timeframe.name}", None)
            return []

        rates = None
        try:
            if start_time and end_time:
                # MT5 copy_rates_range expects naive datetimes in local time of PC where terminal runs, or UTC if server is UTC.
                # Best to convert to UTC epoch seconds.
                start_ts = int(start_time.timestamp())
                end_ts = int(end_time.timestamp())
                rates = mt5.copy_rates_range(symbol, mt5_tf, start_ts, end_ts)
            elif start_time and count:
                start_ts = int(start_time.timestamp())
                rates = mt5.copy_rates_from(symbol, mt5_tf, start_ts, count)
            elif count: # Most recent 'count' bars
                rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, count)
            else: # Default: get a reasonable number of recent bars, e.g., 100
                self.logger.warning(f"get_historical_ohlcv for {symbol}/{timeframe.name} called with no count or date range, fetching 100 bars.")
                rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, 100)
        except Exception as e:
            self.logger.error(f"Exception during mt5.copy_rates call for {symbol}/{timeframe.name}: {e}", exc_info=True)
            self._on_error(f"MT5 Get Historical OHLCV Exception for {symbol}/{timeframe.name}", e)
            return []
            
        if rates is None or len(rates) == 0:
            err_code, err_desc = mt5.last_error()
            # Don't log as error if it's just no data for range, could be normal
            self.logger.info(f"No historical data returned for {symbol}/{timeframe.name} for the specified criteria. MT5 Error (if any): {err_code}: {err_desc}")
            return []
        
        return self._convert_mt5_rates_to_common_ohlcv(symbol, timeframe, rates)

    def subscribe_ticks(self, symbol: str, callback: Callable[[TickData], None]) -> bool:
        if not self.is_connected():
            self.logger.error(f"Cannot subscribe to ticks for {symbol}: Not connected.")
            return False
        
        super().register_tick_subscriber(symbol, callback) # Store in PlatformInterface
        self._subscribed_tick_symbols.add(symbol)
        
        # Ensure symbol is selected for ticks
        if not mt5.symbol_select(symbol, True):
            self.logger.warning(f"Could not select symbol {symbol} for tick subscription.")
            # Proceeding anyway, polling will try.
            
        self.logger.info(f"Tick subscription request for {symbol} registered. Polling will provide data.")
        
        # If polling isn't active, start it
        if not self._polling_active:
            self._polling_active = True
            self._polling_thread = threading.Thread(target=self._poll_data_and_events, daemon=True)
            self._polling_thread.start()
        return True

    def unsubscribe_ticks(self, symbol: str, callback: Optional[Callable[[TickData], None]] = None) -> bool:
        if callback:
            super().unregister_tick_subscriber(symbol, callback)
        else: # Unsubscribe all for this symbol
            if symbol in self.tick_subscribers:
                for cb in list(self.tick_subscribers[symbol]): # list() for safe removal
                    super().unregister_tick_subscriber(symbol, cb)
        
        if symbol in self.tick_subscribers and not self.tick_subscribers[symbol]:
            self._subscribed_tick_symbols.discard(symbol)
            self.logger.info(f"All tick subscriptions for {symbol} removed.")
            # MT5 doesn't have an explicit "unsubscribe"; polling just won't fetch if not in _subscribed_tick_symbols
        return True

    def subscribe_bars(self, symbol: str, timeframe: Timeframe, callback: Callable[[OHLCVData], None]) -> bool:
        if not self.is_connected():
            self.logger.error(f"Cannot subscribe to bars for {symbol}/{timeframe.name}: Not connected.")
            return False

        super().register_bar_subscriber(symbol, timeframe, callback)
        self._subscribed_bar_symbols_tf.setdefault(symbol, {})[timeframe] = datetime.fromtimestamp(0, tz=timezone.utc) # Initialize last bar time

        if not mt5.symbol_select(symbol, True):
             self.logger.warning(f"Could not select symbol {symbol} for bar subscription.")

        self.logger.info(f"Bar subscription request for {symbol}/{timeframe.name} registered. Polling will provide data.")

        if not self._polling_active:
            self._polling_active = True
            self._polling_thread = threading.Thread(target=self._poll_data_and_events, daemon=True)
            self._polling_thread.start()
        return True

    def unsubscribe_bars(self, symbol: str, timeframe: Timeframe, callback: Optional[Callable[[OHLCVData], None]] = None) -> bool:
        if callback:
            super().unregister_bar_subscriber(symbol, timeframe, callback)
        else: # Unsubscribe all for this symbol/tf
            if symbol in self.bar_subscribers and timeframe in self.bar_subscribers[symbol]:
                for cb in list(self.bar_subscribers[symbol][timeframe]):
                     super().unregister_bar_subscriber(symbol, timeframe, cb)
        
        if symbol in self.bar_subscribers and timeframe in self.bar_subscribers[symbol] and not self.bar_subscribers[symbol][timeframe]:
            del self._subscribed_bar_symbols_tf[symbol][timeframe]
            if not self._subscribed_bar_symbols_tf[symbol]:
                del self._subscribed_bar_symbols_tf[symbol]
            self.logger.info(f"All bar subscriptions for {symbol}/{timeframe.name} removed.")
        return True

    def place_order(
        self,
        symbol: str,
        order_type: OrderType,
        action: OrderAction,
        volume: float,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        client_order_id: Optional[str] = None, # Not directly used by MT5 order_send, put in comment
        slippage_points: Optional[int] = None, 
        comment: Optional[str] = None,
        expiration_time: Optional[datetime] = None # MT5 uses ENUM_ORDER_TYPE_TIME
    ) -> Optional[Order]:
        if not self.is_connected():
            self.logger.error(f"Cannot place order for {symbol}: Not connected.")
            self._on_error("MT5 Place Order: Not Connected", None)
            return None

        mt5_order_type = self._map_common_order_type_to_mt5(order_type, action)
        if mt5_order_type is None:
            self.logger.error(f"Unsupported common order type '{order_type.name}' for MT5.")
            self._on_error(f"MT5 Place Order: Unsupported order type '{order_type.name}'", None)
            return None
        
        symbol_info_raw = mt5.symbol_info(symbol)
        if not symbol_info_raw:
            self.logger.error(f"Could not get symbol info for {symbol} before placing order.")
            self._on_error(f"MT5 Place Order: Symbol info not found for {symbol}", None)
            return None

        request_price = price
        if order_type == OrderType.MARKET:
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                self.logger.error(f"Could not get current tick for market order on {symbol}.")
                self._on_error(f"MT5 Place Order: Market order tick fetch failed for {symbol}", None)
                return None
            request_price = tick.ask if action == OrderAction.BUY else tick.bid
        
        # Volume check and adjustment based on symbol's lot step
        volume = round(volume / symbol_info_raw.volume_step) * symbol_info_raw.volume_step
        volume = max(symbol_info_raw.volume_min, volume)
        volume = min(symbol_info_raw.volume_max, volume)
        if volume < symbol_info_raw.volume_min: # After adjustment, if it's still too low (e.g. calculated risk is tiny)
            self.logger.warning(f"Adjusted order volume {volume} for {symbol} is below min_volume {symbol_info_raw.volume_min}. Order may fail or be adjusted by broker.")
            # Some brokers might accept it and round up, others reject. Smallest possible is min_volume
            if volume == 0 and symbol_info_raw.volume_min > 0 : # if initial calc was so small it rounded to 0
                 self.logger.info(f"Calculated volume was effectively zero for {symbol}, attempting with min_volume {symbol_info_raw.volume_min}")
                 volume = symbol_info_raw.volume_min
            elif volume == 0 and symbol_info_raw.volume_min == 0: # Should not happen for valid symbols
                 self.logger.error(f"Cannot place order for {symbol} with zero volume and zero min_volume.")
                 return None


        effective_comment = comment or ""
        if client_order_id:
            effective_comment = f"coid:{client_order_id}|{effective_comment}"
            effective_comment = effective_comment[:31] # MT5 comment length limit for request struct (actually 27 for some versions?)

        request = {
            "action": mt5.TRADE_ACTION_DEAL if order_type == OrderType.MARKET else mt5.TRADE_ACTION_PENDING,
            "symbol": symbol,
            "volume": float(volume),
            "type": mt5_order_type,
            "price": float(request_price) if request_price is not None else 0.0,
            "sl": float(stop_loss) if stop_loss is not None else 0.0,
            "tp": float(take_profit) if take_profit is not None else 0.0,
            "deviation": slippage_points if slippage_points is not None else (self.platform_config.slippage_default_points if hasattr(self.platform_config, 'slippage_default_points') else 20), # Default slippage
            "magic": self.platform_config.magic_number if hasattr(self.platform_config, 'magic_number') else (self.bot_config.magic_number_default if hasattr(self.bot_config, 'magic_number_default') else 12345), # from config
            "comment": effective_comment,
            "type_time": mt5.ORDER_TIME_GTC, # Default, add logic for expiration if needed
            "type_filling": mt5.ORDER_FILLING_IOC, # Or FILLING_FOK or FILLING_RETURN as per firm/strategy
        }
        
        if expiration_time: # For pending orders
            request["type_time"] = mt5.ORDER_TIME_SPECIFIED
            request["expiration"] = int(expiration_time.timestamp())


        self.logger.info(f"Placing order request for {symbol}: {request}")
        result = mt5.order_send(request)

        if result is None:
            err_code, err_desc = mt5.last_error()
            self.logger.error(f"Order send call failed for {symbol} (returned None). MT5 Error {err_code}: {err_desc}. Request: {request}")
            self._on_error(f"MT5 Place Order Failed ({err_code}) for {symbol}", None)
            return None # Immediate failure

        if result.retcode != mt5.TRADE_RETCODE_DONE and result.retcode != mt5.TRADE_RETCODE_PLACED: # PLACED for pending
            self.logger.error(f"Order send for {symbol} not successful. Retcode: {result.retcode}, Comment: {result.comment}. Request: {result.request}")
            self._on_error(f"MT5 Place Order Unsuccessful ({result.retcode}) for {symbol}", None)
            # Create a rejected order object
            failed_order = Order(
                order_id=str(uuid.uuid4()), # Placeholder ID for rejected internal tracking
                client_order_id=client_order_id,
                symbol=symbol, order_type=order_type, action=action, volume=volume,
                price=price, stop_loss=stop_loss, take_profit=take_profit,
                status=OrderStatus.REJECTED, created_at=datetime.now(timezone.utc),
                comment=f"MT5 Reject: {result.comment} (retcode {result.retcode})"
            )
            self._on_order_update(failed_order)
            return failed_order # Return a rejected order object

        self.logger.info(f"Order for {symbol} sent successfully. Ticket: {result.order}, Deal: {result.deal}, Retcode: {result.retcode}, Result Comment: {result.comment}")
        
        # Fetch the order details to return a common Order object
        # order_send for market order creates a deal and a position directly.
        # For pending order, it creates an order.
        time.sleep(0.2) # Give server a moment to process
        
        final_order_obj = None
        if result.order > 0 : # If an order ticket was returned (usually for pending, or sometimes filled market)
            mt5_ord_info = mt5.history_orders_get(ticket=result.order)
            if mt5_ord_info and len(mt5_ord_info) > 0:
                final_order_obj = self._convert_mt5_order_to_common_order(mt5_ord_info[0])
            else: # Fallback to active orders if not in history yet
                mt5_ord_info_active = mt5.orders_get(ticket=result.order)
                if mt5_ord_info_active and len(mt5_ord_info_active) > 0:
                     final_order_obj = self._convert_mt5_order_to_common_order(mt5_ord_info_active[0])

        if final_order_obj:
            self._on_order_update(final_order_obj)
            return final_order_obj
        else:
            # For market orders that result in immediate deal, we might not have an "Order" object in MT5 sense
            # but rather a "Position" and a "Deal". Let's construct a synthetic filled order.
            self.logger.info(f"Order for {symbol} likely resulted in immediate deal {result.deal}. Constructing synthetic Order object.")
            
            # Try to get deal info
            deal_info_list = mt5.history_deals_get(ticket=result.deal) if result.deal > 0 else None
            filled_px = request_price # Best guess if deal not found
            filled_vol = volume
            deal_time = datetime.now(timezone.utc)
            deal_order_id_from_deal = str(uuid.uuid4()) # Synthetic if no order ticket

            if deal_info_list and len(deal_info_list) > 0:
                deal_info = deal_info_list[0]
                filled_px = deal_info.price
                filled_vol = deal_info.volume
                deal_time = datetime.fromtimestamp(deal_info.time, tz=timezone.utc)
                deal_order_id_from_deal = str(deal_info.order) # Actual order that created this deal

            synthetic_order = Order(
                order_id=deal_order_id_from_deal if result.order == 0 else str(result.order), # Use deal's order if no main order ticket
                client_order_id=client_order_id,
                symbol=symbol, order_type=order_type, action=action, volume=filled_vol,
                price=filled_px, stop_loss=stop_loss, take_profit=take_profit,
                status=OrderStatus.FILLED, created_at=deal_time, updated_at=deal_time,
                filled_price=filled_px, filled_volume=filled_vol,
                comment=f"Market Executed (Deal: {result.deal}) {effective_comment}".strip(),
                commission=deal_info.commission if deal_info_list and len(deal_info_list) > 0 else None,
                swap=None, # Market open usually no swap
                platform_specific_details={"deal_ticket": result.deal, "retcode": result.retcode}
            )
            self._on_order_update(synthetic_order)
            return synthetic_order


    def modify_order(
        self,
        order_id: str, 
        new_price: Optional[float] = None,
        new_stop_loss: Optional[float] = None,
        new_take_profit: Optional[float] = None,
        new_volume: Optional[float] = None, # MT5 typically doesn't allow modifying volume of pending
        new_expiration_time: Optional[datetime] = None
    ) -> Optional[Order]:
        if not self.is_connected():
            self.logger.error(f"Cannot modify order {order_id}: Not connected.")
            self._on_error("MT5 Modify Order: Not Connected", None)
            return None

        order_ticket = int(order_id)
        # Fetch existing order to get other details
        existing_order_raw = mt5.orders_get(ticket=order_ticket)
        if not existing_order_raw or len(existing_order_raw) == 0:
            self.logger.error(f"Cannot modify order {order_id}: Order not found in active orders.")
            self._on_error(f"MT5 Modify Order: Order {order_id} not found", None)
            return None
        
        existing_order = existing_order_raw[0]

        request = {
            "action": mt5.TRADE_ACTION_MODIFY,
            "order": order_ticket,
            "symbol": existing_order.symbol, # Must provide symbol
            "price": float(new_price) if new_price is not None else existing_order.price_open,
            "sl": float(new_stop_loss) if new_stop_loss is not None else existing_order.sl,
            "tp": float(new_take_profit) if new_take_profit is not None else existing_order.tp,
            "type_time": existing_order.type_time, # Preserve original time type unless new expiration
            "expiration": existing_order.time_expiration, # Preserve original expiration
            # Volume modification is tricky for pending orders, usually not allowed by brokers for placed ones.
            # For MT5, TRADE_ACTION_MODIFY works on pending orders.
            # "volume": float(new_volume) if new_volume is not None else existing_order.volume_initial, # This might be rejected
        }
        if new_volume is not None and new_volume != existing_order.volume_initial:
             self.logger.warning(f"Attempting to modify volume for order {order_id} from {existing_order.volume_initial} to {new_volume}. This may not be supported.")
             # MT5 docs say volume cannot be changed by TRADE_ACTION_MODIFY. Would need cancel & new.
             # For simplicity, we are not including volume modification here for TRADE_ACTION_MODIFY.

        if new_expiration_time:
            request["type_time"] = mt5.ORDER_TIME_SPECIFIED
            request["expiration"] = int(new_expiration_time.timestamp())
        
        # Ensure SL/TP are zero if not provided and were zero
        if request["sl"] is None: request["sl"] = 0.0
        if request["tp"] is None: request["tp"] = 0.0

        self.logger.info(f"Modifying order {order_id} with request: {request}")
        result = mt5.order_send(request)

        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            err_code, err_desc = mt5.last_error() if result is None else (result.retcode, result.comment)
            self.logger.error(f"Failed to modify order {order_id}. MT5 Error/RetCode {err_code}: {err_desc}. Request: {request if result is None else result.request}")
            self._on_error(f"MT5 Modify Order Failed ({err_code}) for {order_id}", None)
            return self.get_order_status(order_id) # Return current status if modify failed

        self.logger.info(f"Order {order_id} modified successfully. Ticket: {result.order}, Retcode: {result.retcode}")
        
        time.sleep(0.2) # Allow server to process
        updated_order_info = mt5.orders_get(ticket=result.order)
        if updated_order_info and len(updated_order_info) > 0:
            common_order = self._convert_mt5_order_to_common_order(updated_order_info[0])
            self._on_order_update(common_order)
            return common_order
        return None


    def cancel_order(self, order_id: str) -> Optional[Order]:
        if not self.is_connected():
            self.logger.error(f"Cannot cancel order {order_id}: Not connected.")
            self._on_error("MT5 Cancel Order: Not Connected", None)
            return None

        order_ticket = int(order_id)
        existing_order_raw = mt5.orders_get(ticket=order_ticket)
        if not existing_order_raw or len(existing_order_raw) == 0:
            self.logger.warning(f"Cannot cancel order {order_id}: Order not found in active orders. It might have been filled or already cancelled.")
            # Check history
            hist_order_raw = mt5.history_orders_get(ticket=order_ticket)
            if hist_order_raw and len(hist_order_raw) > 0:
                return self._convert_mt5_order_to_common_order(hist_order_raw[0])
            return None


        request = {
            "action": mt5.TRADE_ACTION_REMOVE, # For pending orders
            "order": order_ticket,
            # "symbol": existing_order_raw[0].symbol # Symbol might be needed for some brokers
        }
        self.logger.info(f"Cancelling order {order_id} with request: {request}")
        result = mt5.order_send(request)

        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            err_code, err_desc = mt5.last_error() if result is None else (result.retcode, result.comment)
            self.logger.error(f"Failed to cancel order {order_id}. MT5 Error/RetCode {err_code}: {err_desc}. Request: {request if result is None else result.request}")
            self._on_error(f"MT5 Cancel Order Failed ({err_code}) for {order_id}", None)
            return self.get_order_status(order_id)

        self.logger.info(f"Order {order_id} cancelled successfully. Ticket: {result.order}, Retcode: {result.retcode}")
        time.sleep(0.2)
        
        # Order should now be in history
        hist_order_raw = mt5.history_orders_get(ticket=order_ticket)
        if hist_order_raw and len(hist_order_raw) > 0:
            common_order = self._convert_mt5_order_to_common_order(hist_order_raw[0])
            self._on_order_update(common_order)
            return common_order
        
        self.logger.warning(f"Could not confirm cancellation status for order {order_id} from history after cancellation.")
        return None # Fallback

    def get_order_status(self, order_id: str) -> Optional[Order]:
        if not self.is_connected():
            self.logger.error(f"Cannot get order status for {order_id}: Not connected.")
            return None
        
        order_ticket = int(order_id)
        # Check active orders first
        active_orders = mt5.orders_get(ticket=order_ticket)
        if active_orders and len(active_orders) > 0:
            return self._convert_mt5_order_to_common_order(active_orders[0])
        
        # If not active, check history
        historical_orders = mt5.history_orders_get(ticket=order_ticket)
        if historical_orders and len(historical_orders) > 0:
            return self._convert_mt5_order_to_common_order(historical_orders[0])
            
        self.logger.warning(f"Order {order_id} not found in active or historical orders.")
        return None

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        if not self.is_connected():
            self.logger.error("Cannot get open orders: Not connected.")
            return []
        
        mt5_orders_raw = mt5.orders_get(symbol=symbol) if symbol else mt5.orders_get()
        common_orders = []
        if mt5_orders_raw:
            for mt5_ord in mt5_orders_raw:
                common_ord = self._convert_mt5_order_to_common_order(mt5_ord)
                if common_ord:
                    common_orders.append(common_ord)
        return common_orders

    def get_order_history(
        self,
        symbol: Optional[str] = None,
        start_time: Optional[datetime] = None, # UTC-aware
        end_time: Optional[datetime] = None,   # UTC-aware
        count: Optional[int] = None # Not directly supported by history_orders_get for date range
    ) -> List[Order]:
        if not self.is_connected():
            self.logger.error("Cannot get order history: Not connected.")
            return []

        orders_list = []
        if start_time and end_time:
            start_ts = int(start_time.timestamp())
            end_ts = int(end_time.timestamp())
            # If symbol is provided, MT5's history_orders_get doesn't filter by symbol for a date range.
            # We'd have to fetch all for the date range and then filter locally if symbol is specified.
            all_hist_orders = mt5.history_orders_get(date_from=start_ts, date_to=end_ts)
            if all_hist_orders:
                for mt5_ord in all_hist_orders:
                    if symbol and mt5_ord.symbol != symbol:
                        continue
                    common_ord = self._convert_mt5_order_to_common_order(mt5_ord)
                    if common_ord:
                        orders_list.append(common_ord)
        elif count and not symbol: # Get last N orders across all symbols (MT5 doesn't have a direct count for history_orders_get)
                                   # This needs a more complex fetch and sort if strictly "last N" is needed.
                                   # For simplicity, let's fetch for a recent wide period.
            self.logger.warning("get_order_history with 'count' without date range is approximated by fetching recent history.")
            recent_start_time = datetime.now(timezone.utc) - timedelta(days=30) # Fetch last 30 days
            start_ts = int(recent_start_time.timestamp())
            end_ts = int(datetime.now(timezone.utc).timestamp())
            all_hist_orders = mt5.history_orders_get(date_from=start_ts, date_to=end_ts)
            if all_hist_orders:
                for mt5_ord in reversed(all_hist_orders): # Iterate newest first
                    common_ord = self._convert_mt5_order_to_common_order(mt5_ord)
                    if common_ord:
                        orders_list.append(common_ord)
                    if len(orders_list) >= count:
                        break
        elif symbol: # Get all history for a specific symbol
            all_hist_orders_for_symbol = mt5.history_orders_get(date_from=0, date_to=int(time.time())) # All time
            if all_hist_orders_for_symbol:
                for mt5_ord in all_hist_orders_for_symbol:
                    if mt5_ord.symbol == symbol:
                        common_ord = self._convert_mt5_order_to_common_order(mt5_ord)
                        if common_ord:
                            orders_list.append(common_ord)
        else: # Fetch all history (can be large!)
             self.logger.warning("Fetching all order history. This can be resource intensive.")
             all_hist_orders = mt5.history_orders_get(date_from=0, date_to=int(time.time()))
             if all_hist_orders:
                for mt5_ord in all_hist_orders:
                    common_ord = self._convert_mt5_order_to_common_order(mt5_ord)
                    if common_ord:
                        orders_list.append(common_ord)
        return orders_list


    def get_open_positions(self, symbol: Optional[str] = None) -> List[Position]:
        if not self.is_connected():
            self.logger.error("Cannot get open positions: Not connected.")
            return []
        
        mt5_positions_raw = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
        common_positions = []
        if mt5_positions_raw:
            for mt5_pos in mt5_positions_raw:
                common_pos = self._convert_mt5_position_to_common_position(mt5_pos)
                if common_pos:
                    # Augment with associated orders if possible (simplified here)
                    # common_pos.orders_associated.append(str(mt5_pos.identifier))
                    common_positions.append(common_pos)
                    self._on_position_update(common_pos) # Dispatch event
        return common_positions

    def get_position(self, position_id: str) -> Optional[Position]:
        if not self.is_connected():
            self.logger.error(f"Cannot get position {position_id}: Not connected.")
            return None
        
        pos_ticket = int(position_id)
        mt5_pos_raw = mt5.positions_get(ticket=pos_ticket)
        if mt5_pos_raw and len(mt5_pos_raw) > 0:
            return self._convert_mt5_position_to_common_position(mt5_pos_raw[0])
        
        self.logger.warning(f"Position {position_id} not found in open positions.")
        return None
        
    def close_position(
        self,
        position_id: str,
        volume: Optional[float] = None, # Amount to close
        price: Optional[float] = None,  # For closing with a limit/stop, not typical for market close
        comment: Optional[str] = None
    ) -> Optional[Order]: # Returns the closing order details
        if not self.is_connected():
            self.logger.error(f"Cannot close position {position_id}: Not connected.")
            self._on_error("MT5 Close Position: Not Connected", None)
            return None

        pos_ticket = int(position_id)
        mt5_pos_info_list = mt5.positions_get(ticket=pos_ticket)

        if not mt5_pos_info_list or len(mt5_pos_info_list) == 0:
            self.logger.error(f"Position {position_id} not found or already closed.")
            self._on_error(f"MT5 Close Position: Position {position_id} not found", None)
            return None
        
        mt5_pos_info = mt5_pos_info_list[0]
        
        close_volume = volume if volume is not None and volume <= mt5_pos_info.volume else mt5_pos_info.volume
        if close_volume <= 0:
            self.logger.error(f"Invalid volume {close_volume} for closing position {position_id}.")
            return None

        # Determine opposite action for closing
        close_action = mt5.ORDER_TYPE_SELL if mt5_pos_info.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
        
        current_tick = mt5.symbol_info_tick(mt5_pos_info.symbol)
        if not current_tick:
            self.logger.error(f"Could not get current tick for closing position {position_id} on {mt5_pos_info.symbol}.")
            self._on_error(f"MT5 Close Position: Tick fetch failed for {mt5_pos_info.symbol}", None)
            return None
        
        close_price = current_tick.bid if close_action == mt5.ORDER_TYPE_SELL else current_tick.ask
        if price is not None: # If specific closing price is provided (e.g. for a TP/SL like mechanism)
            close_price = price # This makes it like a limit order close, less common for direct close

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": pos_ticket, # Specify position ticket to close
            "symbol": mt5_pos_info.symbol,
            "volume": float(close_volume),
            "type": close_action,
            "price": float(close_price),
            "deviation": self.platform_config.slippage_default_points if hasattr(self.platform_config, 'slippage_default_points') else 20,
            "magic": mt5_pos_info.magic, # Use original magic for closure if possible, or bot's magic
            "comment": comment or f"Close Pos {position_id}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        self.logger.info(f"Closing position {position_id} for {mt5_pos_info.symbol} with request: {request}")
        result = mt5.order_send(request)

        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            err_code, err_desc = mt5.last_error() if result is None else (result.retcode, result.comment)
            self.logger.error(f"Failed to close position {position_id}. MT5 Error/RetCode {err_code}: {err_desc}. Request: {request if result is None else result.request}")
            self._on_error(f"MT5 Close Position Failed ({err_code}) for {position_id}", None)
            return None # Closing order failed

        self.logger.info(f"Position {position_id} close request sent successfully. Deal: {result.deal}, Retcode: {result.retcode}")
        time.sleep(0.2) # Allow server to process
        
        # Construct a synthetic closing Order object based on the deal
        deal_info_list = mt5.history_deals_get(ticket=result.deal) if result.deal > 0 else None
        if deal_info_list and len(deal_info_list) > 0:
            deal_info = deal_info_list[0]
            closing_order_action = OrderAction.SELL if mt5_pos_info.type == mt5.POSITION_TYPE_BUY else OrderAction.BUY
            
            closing_order = Order(
                order_id=str(deal_info.order), # The order that executed this deal
                client_order_id=None, # This was a position closure
                symbol=deal_info.symbol,
                order_type=OrderType.MARKET, # Closure is a market execution
                action=closing_order_action,
                volume=deal_info.volume,
                price=deal_info.price, # Fill price
                status=OrderStatus.FILLED,
                created_at=datetime.fromtimestamp(deal_info.time, tz=timezone.utc).astimezone(self.ftmo_timezone),
                updated_at=datetime.fromtimestamp(deal_info.time, tz=timezone.utc).astimezone(self.ftmo_timezone),
                filled_price=deal_info.price,
                filled_volume=deal_info.volume,
                commission=deal_info.commission,
                swap=deal_info.swap, # Swap might be associated with deal
                comment=f"Close Position {position_id} (Deal: {result.deal}) {comment or ''}".strip(),
                platform_specific_details={"deal_ticket": result.deal, "position_id_closed": position_id}
            )
            self._on_order_update(closing_order) # Notify of the closing order
            # Also, the position itself would be updated to CLOSED, need to fetch its final state from history_deals
            # This part is more complex if we want to emit a PositionUpdate to CLOSED status.
            # For now, just returning the closing order.
            return closing_order
        return None


    def modify_position_sl_tp(
        self,
        position_id: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Optional[Position]:
        if not self.is_connected():
            self.logger.error(f"Cannot modify SL/TP for position {position_id}: Not connected.")
            self._on_error("MT5 Modify Position SL/TP: Not Connected", None)
            return None

        pos_ticket = int(position_id)
        mt5_pos_info_list = mt5.positions_get(ticket=pos_ticket)

        if not mt5_pos_info_list or len(mt5_pos_info_list) == 0:
            self.logger.error(f"Position {position_id} not found for SL/TP modification.")
            self._on_error(f"MT5 Modify Position SL/TP: Position {position_id} not found", None)
            return None
        
        mt5_pos_info = mt5_pos_info_list[0]

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": pos_ticket,
            "symbol": mt5_pos_info.symbol, # Important: must provide symbol for TRADE_ACTION_SLTP
            "sl": float(stop_loss) if stop_loss is not None else 0.0, # Use 0.0 to remove SL/TP
            "tp": float(take_profit) if take_profit is not None else 0.0,
        }
        # If SL/TP is None, we want to preserve existing SL/TP if one exists.
        # If they are explicitly set to 0 by user, then they will be removed.
        # If None, use existing values from position.
        if stop_loss is None and mt5_pos_info.sl > 0 : request["sl"] = mt5_pos_info.sl
        if take_profit is None and mt5_pos_info.tp > 0 : request["tp"] = mt5_pos_info.tp


        self.logger.info(f"Modifying SL/TP for position {position_id} ({mt5_pos_info.symbol}) with request: {request}")
        result = mt5.order_send(request)

        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            err_code, err_desc = mt5.last_error() if result is None else (result.retcode, result.comment)
            self.logger.error(f"Failed to modify SL/TP for position {position_id}. MT5 Error/RetCode {err_code}: {err_desc}. Request: {request if result is None else result.request}")
            self._on_error(f"MT5 Modify Position SL/TP Failed ({err_code}) for {position_id}", None)
            return self.get_position(position_id) # Return current state

        self.logger.info(f"SL/TP for position {position_id} modified successfully. Request ID (if any): {result.order}, Retcode: {result.retcode}") # .order often 0 for SLTP action
        time.sleep(0.2)
        
        updated_pos_info_list = mt5.positions_get(ticket=pos_ticket)
        if updated_pos_info_list and len(updated_pos_info_list) > 0:
            common_pos = self._convert_mt5_position_to_common_position(updated_pos_info_list[0])
            self._on_position_update(common_pos) # Dispatch event
            return common_pos
        return None

    def get_trade_history(
        self,
        symbol: Optional[str] = None,
        start_time: Optional[datetime] = None, # UTC-aware
        end_time: Optional[datetime] = None,   # UTC-aware
        count: Optional[int] = None # MT5 history_deals_get doesn't directly use count with date range
    ) -> List[TradeFill]:
        if not self.is_connected():
            self.logger.error("Cannot get trade history (deals): Not connected.")
            return []

        deals_list = []
        start_ts = int(start_time.timestamp()) if start_time else 0
        end_ts = int(end_time.timestamp()) if end_time else int(time.time())

        # MT5 history_deals_get works by date range. If symbol is specified, we filter after fetching.
        # If count is specified, we fetch for the date range (or wide range) and take the last 'count'.
        
        all_deals_raw = mt5.history_deals_get(date_from=start_ts, date_to=end_ts)
        
        if not all_deals_raw:
            err_code, err_desc = mt5.last_error()
            if err_code != 0: # 0 might mean success but no deals in range
                 self.logger.warning(f"Failed to get deals. MT5 Error {err_code}: {err_desc}")
            return []

        processed_deals = []
        for mt5_deal in all_deals_raw:
            if symbol and mt5_deal.symbol != symbol:
                continue
            
            action = OrderAction.BUY if mt5_deal.type == mt5.DEAL_TYPE_BUY else \
                     (OrderAction.SELL if mt5_deal.type == mt5.DEAL_TYPE_SELL else None)
            if not action: # Skip balance/credit operations etc.
                # DEAL_ENTRY_IN, DEAL_ENTRY_OUT, DEAL_ENTRY_INOUT are typical trade entries/exits
                # Check entry type: DEAL_ENTRY_IN (buy), DEAL_ENTRY_OUT (sell)
                # DEAL_TYPE_BUY, DEAL_TYPE_SELL are actual trade operations
                if mt5_deal.entry == mt5.DEAL_ENTRY_IN: action = OrderAction.BUY # Or determine by deal type
                elif mt5_deal.entry == mt5.DEAL_ENTRY_OUT: action = OrderAction.SELL
                else: continue # Not a buy/sell trade deal

            common_deal = TradeFill(
                fill_id=str(mt5_deal.ticket),
                order_id=str(mt5_deal.order),
                position_id=str(mt5_deal.position_id) if mt5_deal.position_id > 0 else None,
                timestamp=datetime.fromtimestamp(mt5_deal.time, tz=timezone.utc).astimezone(self.ftmo_timezone),
                symbol=mt5_deal.symbol,
                action=action,
                volume=mt5_deal.volume,
                price=mt5_deal.price,
                commission=mt5_deal.commission,
                fee=mt5_deal.fee, # Usually zero for forex
                platform_specific_details={
                    "swap": mt5_deal.swap,
                    "profit": mt5_deal.profit,
                    "magic": mt5_deal.magic,
                    "entry_type": mt5_deal.entry, # IN, OUT, INOUT
                    "deal_type": mt5_deal.type # BUY, SELL, BALANCE, CREDIT
                }
            )
            processed_deals.append(common_deal)
        
        # Sort by time (descending for "last N") and apply count if needed
        processed_deals.sort(key=lambda d: d.timestamp, reverse=True)
        
        if count is not None and len(processed_deals) > count:
            return processed_deals[:count]
        return processed_deals


    def get_server_time(self) -> Optional[datetime]:
        if not self.is_connected():
            self.logger.warning("Cannot get server time: Not connected.")
            return None
        
        term_info = mt5.terminal_info()
        if term_info and term_info.time_server > 0:
            # time_server is a Unix timestamp (UTC)
            return datetime.fromtimestamp(term_info.time_server, tz=timezone.utc).astimezone(self.ftmo_timezone)
        else:
            err_code, err_desc = mt5.last_error()
            self.logger.error(f"Failed to get server time from terminal_info. Error {err_code}: {err_desc}")
            return None

    def set_initial_bar_timestamp(self, symbol: str, timeframe: Timeframe, timestamp: datetime) -> None:
        """Sets the last known bar timestamp for a symbol/timeframe, typically after historical fetch."""
        if symbol in self._subscribed_bar_symbols_tf and timeframe in self._subscribed_bar_symbols_tf[symbol]:
            # Ensure timestamp is UTC for internal storage consistency if adapter's polling logic expects UTC
            utc_timestamp = timestamp.astimezone(timezone.utc) if timestamp.tzinfo else timestamp.replace(tzinfo=timezone.utc)
            self._subscribed_bar_symbols_tf[symbol][timeframe] = utc_timestamp
            self.logger.debug(f"MT5Adapter: Initial bar timestamp for {symbol}/{timeframe.name} set to {utc_timestamp} by MarketDataManager.")
        else:
            self.logger.warning(f"MT5Adapter: Attempted to set initial bar timestamp for non-subscribed {symbol}/{timeframe.name}.")



  
