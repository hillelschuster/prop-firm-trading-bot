# prop_firm_trading_bot/src/api_connector/base_connector.py

from abc import ABC, abstractmethod
from typing import List, Optional, Callable, Any, Dict
from datetime import datetime

# Imports from the core package
from prop_firm_trading_bot.src.core.enums import OrderType, OrderAction, Timeframe
from prop_firm_trading_bot.src.core.models import (
    Order, Position, AccountInfo, OHLCVData, TickData, SymbolInfo, TradeFill, MarketEvent
)

# Define callback types for asynchronous data or event notifications
TickCallback = Callable[[TickData], None]
BarCallback = Callable[[OHLCVData], None] # For newly closed bars
OrderUpdateCallback = Callable[[Order], None]
PositionUpdateCallback = Callable[[Position], None]
AccountUpdateCallback = Callable[[AccountInfo], None]
ErrorCallback = Callable[[str, Optional[Exception]], None] # For connection errors, API errors etc.
MarketEventCallback = Callable[[MarketEvent], None] # For news, or other market events


class PlatformInterface(ABC):
    """
    Abstract Base Class defining the interface for all trading platform adapters.
    This ensures that the core application logic can interact with different
    platforms (MT5, cTrader, paper trading) in a consistent way.
    """

    def __init__(self, config, logger): # config is typically the AppConfig.platform specific part
        self.config = config # Specific platform config (e.g., AppConfig.platform.mt5)
        self.logger = logger
        # Initialize subscriber dictionaries carefully
        self.tick_subscribers: Dict[str, List[TickCallback]] = {} # symbol -> list of callbacks
        self.bar_subscribers: Dict[str, Dict[Timeframe, List[BarCallback]]] = {} # symbol -> timeframe -> list of callbacks
        self.order_update_callbacks: List[OrderUpdateCallback] = []
        self.position_update_callbacks: List[PositionUpdateCallback] = []
        self.account_update_callbacks: List[AccountUpdateCallback] = []
        self.error_callbacks: List[ErrorCallback] = []
        self.market_event_callbacks: List[MarketEventCallback] = []


    # --- Connection Management ---
    @abstractmethod
    def connect(self) -> bool:
        """Establishes connection to the trading platform."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Closes connection to the trading platform."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Checks if the connection is active."""
        pass

    # --- Account Information ---
    @abstractmethod
    def get_account_info(self) -> Optional[AccountInfo]:
        """Retrieves current account details (balance, equity, margin, etc.)."""
        pass

    # --- Symbol & Market Data ---
    @abstractmethod
    def get_symbol_info(self, symbol: str) -> Optional[SymbolInfo]:
        """Retrieves detailed information about a specific trading symbol."""
        pass

    @abstractmethod
    def get_all_tradable_symbols(self) -> List[SymbolInfo]:
        """Retrieves a list of all symbols tradable on the platform, ideally filtered by config."""
        pass
    
    @abstractmethod
    def get_latest_tick(self, symbol: str) -> Optional[TickData]:
        """Fetches the most recent tick data for a symbol."""
        pass

    @abstractmethod
    def get_historical_ohlcv(
        self,
        symbol: str,
        timeframe: Timeframe,
        start_time: Optional[datetime] = None, # Inclusive, UTC-aware
        end_time: Optional[datetime] = None,   # Inclusive, UTC-aware
        count: Optional[int] = None
    ) -> List[OHLCVData]:
        """
        Fetches historical OHLCV data.
        Behavior depends on which parameters are provided (start_time, end_time, count).
        Timestamps in returned OHLCVData should be timezone-aware (preferably UTC or localized to a consistent TZ).
        """
        pass

    # --- Streaming Data Subscription ---
    @abstractmethod
    def subscribe_ticks(self, symbol: str, callback: TickCallback) -> bool:
        """Subscribes to real-time tick data for a symbol.
        The adapter is responsible for invoking the callback when new tick data arrives.
        """
        pass

    @abstractmethod
    def unsubscribe_ticks(self, symbol: str, callback: Optional[TickCallback] = None) -> bool:
        """
        Unsubscribes from tick data.
        If callback is provided, removes only that specific callback.
        If callback is None, removes all callbacks for that symbol.
        """
        pass

    @abstractmethod
    def subscribe_bars(self, symbol: str, timeframe: Timeframe, callback: BarCallback) -> bool:
        """Subscribes to real-time bar data (notifications for newly closed bars).
        The adapter is responsible for invoking the callback when a new bar closes.
        """
        pass

    @abstractmethod
    def unsubscribe_bars(self, symbol: str, timeframe: Timeframe, callback: Optional[BarCallback] = None) -> bool:
        """
        Unsubscribes from bar data.
        If callback is provided, removes only that specific callback.
        If callback is None, removes all callbacks for that symbol and timeframe.
        """
        pass

    # --- Order Management ---
    @abstractmethod
    def place_order(
        self,
        symbol: str,
        order_type: OrderType,
        action: OrderAction,
        volume: float,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        client_order_id: Optional[str] = None, # Optional bot-generated ID
        slippage_points: Optional[int] = None, # Permissible slippage in points, if applicable
        comment: Optional[str] = None,
        expiration_time: Optional[datetime] = None # For pending orders, UTC-aware
    ) -> Optional[Order]: # Returns the initial state of the order (e.g. PENDING_OPEN or REJECTED), or None if immediate critical failure
        """Places a new trade order."""
        pass

    @abstractmethod
    def modify_order(
        self,
        order_id: str, # Platform's order ID
        new_price: Optional[float] = None, # For pending orders
        new_stop_loss: Optional[float] = None,
        new_take_profit: Optional[float] = None,
        # Modifying volume or expiration might require cancel & new order on some platforms
        # new_volume: Optional[float] = None, 
        # new_expiration_time: Optional[datetime] = None 
    ) -> Optional[Order]: # Returns updated order status or None if error
        """Modifies an existing pending order (typically price, SL, TP)."""
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> Optional[Order]: # Returns updated order status or None if error
        """Cancels an existing pending order."""
        pass

    @abstractmethod
    def get_order_status(self, order_id: str) -> Optional[Order]:
        """Retrieves the current status of a specific order by its platform ID."""
        pass

    @abstractmethod
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Retrieves all currently active (pending) orders, optionally filtered by symbol."""
        pass
    
    @abstractmethod
    def get_order_history(
        self,
        symbol: Optional[str] = None,
        start_time: Optional[datetime] = None, # UTC-aware
        end_time: Optional[datetime] = None,   # UTC-aware
        count: Optional[int] = None
    ) -> List[Order]:
        """Retrieves historical orders (filled, cancelled, rejected, expired)."""
        pass


    # --- Position Management ---
    @abstractmethod
    def get_open_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """Retrieves all currently open positions, optionally filtered by symbol."""
        pass

    @abstractmethod
    def get_position(self, position_id: str) -> Optional[Position]:
        """Retrieves details for a specific open position by its platform ID."""
        pass
    
    @abstractmethod
    def close_position(
        self,
        position_id: str, # Platform's position ID
        volume_to_close: Optional[float] = None, # Amount to close, if None or >= position.volume, closes entire position
        price: Optional[float] = None, # For market close, this is typically ignored. For closing with a limit/stop, this would be the price.
        comment: Optional[str] = None
    ) -> Optional[Order]: # Returns details of the closing order, or None if failure
        """Closes an open position or part of it, typically via a market order."""
        pass

    @abstractmethod
    def modify_position_sl_tp(
        self,
        position_id: str, # Platform's position ID
        stop_loss: Optional[float] = None, # New SL price; 0 or None might mean remove, platform-dependent
        take_profit: Optional[float] = None # New TP price; 0 or None might mean remove, platform-dependent
    ) -> Optional[Position]: # Returns updated position or None if error
        """Modifies the Stop Loss and/or Take Profit for an open position."""
        pass

    @abstractmethod
    def get_trade_history( # Represents executed fills or deals
        self,
        symbol: Optional[str] = None,
        start_time: Optional[datetime] = None, # UTC-aware
        end_time: Optional[datetime] = None,   # UTC-aware
        count: Optional[int] = None
    ) -> List[TradeFill]:
        """Retrieves historical trade fills/deals."""
        pass

    # --- Server Time ---
    @abstractmethod
    def get_server_time(self) -> Optional[datetime]: # Returns timezone-aware datetime (preferably UTC)
        """Gets the current server time from the trading platform."""
        pass

    # --- Callback Registration Methods (implemented in base class for convenience) ---
    def register_tick_subscriber(self, symbol: str, callback: TickCallback):
        self.tick_subscribers.setdefault(symbol, []).append(callback)
        # self.logger.debug(f"Registered tick subscriber for {symbol}: {getattr(callback, '__name__', repr(callback))}")

    def unregister_tick_subscriber(self, symbol: str, callback: TickCallback):
        if symbol in self.tick_subscribers and callback in self.tick_subscribers[symbol]:
            self.tick_subscribers[symbol].remove(callback)
            # self.logger.debug(f"Unregistered tick subscriber for {symbol}: {getattr(callback, '__name__', repr(callback))}")
            if not self.tick_subscribers[symbol]: # If no more subscribers for this symbol
                del self.tick_subscribers[symbol]
                # The concrete adapter's unsubscribe_ticks should handle actual platform unsubscription if needed.

    def register_bar_subscriber(self, symbol: str, timeframe: Timeframe, callback: BarCallback):
        self.bar_subscribers.setdefault(symbol, {}).setdefault(timeframe, []).append(callback)
        # self.logger.debug(f"Registered bar subscriber for {symbol}/{timeframe.name}: {getattr(callback, '__name__', repr(callback))}")

    def unregister_bar_subscriber(self, symbol: str, timeframe: Timeframe, callback: BarCallback):
        if symbol in self.bar_subscribers and \
           timeframe in self.bar_subscribers.get(symbol, {}) and \
           callback in self.bar_subscribers[symbol].get(timeframe, []):
            self.bar_subscribers[symbol][timeframe].remove(callback)
            # self.logger.debug(f"Unregistered bar subscriber for {symbol}/{timeframe.name}: {getattr(callback, '__name__', repr(callback))}")
            if not self.bar_subscribers[symbol][timeframe]:
                del self.bar_subscribers[symbol][timeframe]
            if not self.bar_subscribers[symbol]:
                del self.bar_subscribers[symbol]

    def register_order_update_callback(self, callback: OrderUpdateCallback):
        if callback not in self.order_update_callbacks:
            self.order_update_callbacks.append(callback)

    def register_position_update_callback(self, callback: PositionUpdateCallback):
        if callback not in self.position_update_callbacks:
            self.position_update_callbacks.append(callback)
            
    def register_account_update_callback(self, callback: AccountUpdateCallback):
        if callback not in self.account_update_callbacks:
            self.account_update_callbacks.append(callback)

    def register_error_callback(self, callback: ErrorCallback):
        if callback not in self.error_callbacks:
            self.error_callbacks.append(callback)

    def register_market_event_callback(self, callback: MarketEventCallback):
        if callback not in self.market_event_callbacks:
            self.market_event_callbacks.append(callback)

    # --- Internal Helper Methods for Adapters to dispatch events to subscribers ---
    # These are called by the concrete adapter implementations (e.g., MT5Adapter, CTraderAdapter)
    # when they receive data or events from the platform.

    def _on_tick(self, tick_data: TickData):
        """Helper for adapter to dispatch new tick data to all relevant subscribers."""
        if tick_data.symbol in self.tick_subscribers:
            for callback in self.tick_subscribers[tick_data.symbol]:
                try:
                    callback(tick_data)
                except Exception as e:
                    self.logger.error(f"Error in tick callback for {tick_data.symbol}: {e}", exc_info=True)
                    self._on_error(f"Tick callback error for {tick_data.symbol} via {getattr(callback, '__name__', repr(callback))}", e)

    def _on_bar(self, bar_data: OHLCVData):
        """Helper for adapter to dispatch new bar data to all relevant subscribers."""
        if bar_data.symbol in self.bar_subscribers and \
           bar_data.timeframe in self.bar_subscribers.get(bar_data.symbol, {}):
            for callback in self.bar_subscribers[bar_data.symbol][bar_data.timeframe]:
                try:
                    callback(bar_data)
                except Exception as e:
                    self.logger.error(f"Error in bar callback for {bar_data.symbol}/{bar_data.timeframe.name}: {e}", exc_info=True)
                    self._on_error(f"Bar callback error for {bar_data.symbol}/{bar_data.timeframe.name} via {getattr(callback, '__name__', repr(callback))}", e)

    def _on_order_update(self, order_data: Order):
        """Helper for adapter to dispatch order updates."""
        for callback in self.order_update_callbacks:
            try:
                callback(order_data)
            except Exception as e:
                self.logger.error(f"Error in order update callback for order {order_data.order_id}: {e}", exc_info=True)
                self._on_error(f"Order update callback error for {order_data.order_id} via {getattr(callback, '__name__', repr(callback))}", e)

    def _on_position_update(self, position_data: Position):
        """Helper for adapter to dispatch position updates."""
        for callback in self.position_update_callbacks:
            try:
                callback(position_data)
            except Exception as e:
                self.logger.error(f"Error in position update callback for position {position_data.position_id}: {e}", exc_info=True)
                self._on_error(f"Position update callback error for {position_data.position_id} via {getattr(callback, '__name__', repr(callback))}", e)
                
    def _on_account_update(self, account_data: AccountInfo):
        """Helper for adapter to dispatch account updates."""
        for callback in self.account_update_callbacks:
            try:
                callback(account_data)
            except Exception as e:
                self.logger.error(f"Error in account update callback: {e}", exc_info=True)
                self._on_error(f"Account update callback error via {getattr(callback, '__name__', repr(callback))}", e)

    def _on_error(self, message: str, exception: Optional[Exception] = None):
        """Helper for adapter to dispatch platform or connection errors."""
        # Log it here primarily if the adapter itself wants to use this for internal errors.
        # The adapter should also log errors directly when they occur.
        # This dispatch is for notifying external subscribers.
        self.logger.debug(f"Dispatching error to subscribers: {message}")
        for callback in self.error_callbacks:
            try:
                callback(message, exception)
            except Exception as e_cb: # Error in the error callback itself
                self.logger.critical(f"CRITICAL: Error in error_callback itself ({getattr(callback, '__name__', repr(callback))}): {e_cb}", exc_info=True)
    
    def _on_market_event(self, market_event: MarketEvent):
        """Helper for adapter to dispatch market events (e.g., news from NewsFilter via Orchestrator)."""
        self.logger.debug(f"Dispatching market event to subscribers: {market_event.event_type}")
        for callback in self.market_event_callbacks:
            try:
                callback(market_event)
            except Exception as e:
                self.logger.error(f"Error in market event callback for {market_event.event_type}: {e}", exc_info=True)
                self._on_error(f"Market event callback error for {market_event.event_type} via {getattr(callback, '__name__', repr(callback))}", e)
