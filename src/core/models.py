# prop_firm_trading_bot/src/core/models.py

from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

# Assuming enums.py is in the same 'core' package or the import path is adjusted
# If enums.py is in the same directory (core), this import is fine.
from .enums import OrderType, OrderAction, OrderStatus, PositionStatus, Timeframe, PositionType

class TickData(BaseModel):
    timestamp: datetime
    symbol: str
    bid: float
    ask: float
    last: Optional[float] = None # For instruments where last traded price is relevant
    volume: Optional[float] = None # Tick volume

class OHLCVData(BaseModel):
    timestamp: datetime # Typically the start time of the bar
    symbol: str
    timeframe: Timeframe
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None

    class Config:
        pass

class Order(BaseModel):
    order_id: str # Platform-specific order ID
    client_order_id: Optional[str] = None # Custom ID generated by the bot, if used
    symbol: str
    order_type: OrderType
    action: OrderAction # Buy or Sell
    volume: float # In lots
    price: Optional[float] = None # Entry price for LIMIT/STOP orders, or fill price for MARKET
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    status: OrderStatus
    created_at: datetime # Timestamp when the order was created/sent
    updated_at: Optional[datetime] = None # Timestamp of the last update
    filled_price: Optional[float] = None # Average filled price if order is filled/partially_filled
    filled_volume: Optional[float] = None # Filled volume if order is filled/partially_filled
    filled_time: Optional[datetime] = None # Timestamp when the order was filled
    closed_time: Optional[datetime] = None # Timestamp when the order was closed (cancelled, or position closed)
    commission: Optional[float] = None
    swap: Optional[float] = None # Usually associated with positions, but some platforms might show on order/deal
    pnl: Optional[float] = None # Profit or loss if the order resulted in a closed trade or is a closing order itself
    comment: Optional[str] = None
    magic_number: Optional[int] = None # Custom magic number for the order
    platform_specific_details: Dict[str, Any] = Field(default_factory=dict) # For any extra fields from platform

    class Config:
        pass

class Position(BaseModel):
    position_id: str # Platform-specific position ID
    symbol: str
    action: OrderAction # Buy or Sell (direction of the position)
    volume: float # Current volume in lots
    open_price: float # Average open price
    current_price: Optional[float] = None # Last known market price for P/L calculation, needs to be updated
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    open_time: datetime
    close_time: Optional[datetime] = None
    commission: Optional[float] = None # Accumulated commission for this position
    swap: Optional[float] = None # Accumulated swap for this position
    profit: Optional[float] = None # Realized (if closed) or unrealized (if open) P/L
    comment: Optional[str] = None
    status: PositionStatus = PositionStatus.OPEN # Default, can be updated
    orders_associated: List[str] = Field(default_factory=list) # List of order_ids that opened/modified this position
    platform_specific_details: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        pass

class AccountInfo(BaseModel):
    account_id: str
    balance: float
    equity: float
    margin: float # Margin used
    margin_free: float
    margin_level_pct: Optional[float] = None # Margin Level in percentage
    currency: str
    server_time: Optional[datetime] = None # Current server time, timezone-aware
    platform_specific_details: Dict[str, Any] = Field(default_factory=dict) # e.g., leverage, account type

    class Config:
        pass

class SymbolInfo(BaseModel):
    name: str # Platform-specific symbol name (e.g., "EURUSD", "US30.cash")
    description: Optional[str] = None
    digits: int # Number of decimal places for price (e.g., 5 for EURUSD, 2 for XAUUSD)
    point: float # Value of one point (smallest price change, e.g., 0.00001 for EURUSD)
    min_volume_lots: float
    max_volume_lots: float
    volume_step_lots: float
    contract_size: float # Units per lot (e.g., 100000 for standard Forex lot)
    currency_base: str
    currency_profit: str # Currency in which profit is calculated
    currency_margin: str # Currency for margin calculation
    trade_allowed: bool = True
    # Platform/Broker specific values needed for precise risk calculation
    trade_tick_value: Optional[float] = None # Value of one tick for one lot in deposit currency (MT5 specific)
    trade_tick_size: Optional[float] = None  # Size of one tick (MT5 specific)
    # Custom fields
    is_restricted_for_news: bool = False # Can be dynamically updated by NewsFilter or config
    news_target_currencies: List[str] = Field(default_factory=list) # From instruments_ftmo.json
    pip_value_in_account_currency_per_lot: Optional[float] = None # For convenience, if pre-calculated from instruments_ftmo.json
    platform_specific_details: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        pass

class TradeFill(BaseModel): # Represents an executed deal/trade
    fill_id: str # Platform-specific deal/fill ID
    order_id: str # The order ID that generated this fill
    position_id: Optional[str] = None # The position ID this fill contributes to/closes
    timestamp: datetime # Time of execution, timezone-aware
    symbol: str
    action: OrderAction # BUY or SELL
    volume: float # Filled volume for this specific deal
    price: float # Fill price for this specific deal
    commission: Optional[float] = None
    fee: Optional[float] = None # Additional fees if any
    swap: Optional[float] = None # Swap applied on this deal if applicable (rare for fills, more for positions)
    profit_realized_on_fill: Optional[float] = None # Profit realized by this specific fill (e.g. for partial close)
    platform_specific_details: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        pass

class MarketEvent(BaseModel): # For news or other discrete market-impacting events
    timestamp: datetime # Time of the event or when it's expected/relevant, timezone-aware
    event_type: str # e.g., "NEWS_HIGH_IMPACT", "NEWS_MEDIUM_IMPACT", "MARKET_OPEN", "MARKET_CLOSE", "HOLIDAY"
    description: str
    symbols_affected: Optional[List[str]] = None # Specific symbols directly affected
    currencies_affected: Optional[List[str]] = None # Currencies affected
    source: Optional[str] = None # e.g., "Finnhub", "ForexFactoryJSON", "InternalScheduler"
    metadata: Dict[str, Any] = Field(default_factory=dict) # For any additional event-specific data

    class Config:
        pass


  
