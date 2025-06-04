# prop_firm_trading_bot/src/core/enums.py

from enum import Enum

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    # Prop firms might restrict STOP_LIMIT or other advanced types
    # For FTMO, standard order types are generally fine.
    # STOP_LIMIT = "STOP_LIMIT"
    # MARKET_IF_TOUCHED = "MARKET_IF_TOUCHED"

class OrderAction(Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderStatus(Enum):
    NEW = "NEW" # For client-side representation before sending
    PENDING_OPEN = "PENDING_OPEN" # Submitted to broker, not yet live/active
    OPEN = "OPEN" # Actively working order (e.g., Limit order waiting for fill, or a pending stop not yet triggered)
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    ERROR = "ERROR" # Error in processing or unknown state
    PENDING_CANCEL = "PENDING_CANCEL" # Request to cancel sent to broker

class PositionStatus(Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    # PENDING_CLOSE = "PENDING_CLOSE" # If async close is a state, not typically used for MT5 direct positions

class Timeframe(Enum):
    TICK = "TICK"
    M1 = "M1"
    M5 = "M5"
    M15 = "M15"
    M30 = "M30"
    H1 = "H1"
    H4 = "H4"
    D1 = "D1"
    W1 = "W1"
    MN1 = "MN1"

    def to_mt5_timeframe_string(self) -> str:
        """Converts enum to MetaTrader 5 TIMEFRAME string representation name."""
        # This returns the string name of the MT5 constant.
        # The MT5Adapter will use getattr(mt5, returned_string_name) to get the actual mt5 object.
        mapping = {
            Timeframe.M1: "TIMEFRAME_M1",
            Timeframe.M5: "TIMEFRAME_M5",
            Timeframe.M15: "TIMEFRAME_M15",
            Timeframe.M30: "TIMEFRAME_M30",
            Timeframe.H1: "TIMEFRAME_H1",
            Timeframe.H4: "TIMEFRAME_H4",
            Timeframe.D1: "TIMEFRAME_D1",
            Timeframe.W1: "TIMEFRAME_W1",
            Timeframe.MN1: "TIMEFRAME_MN1",
        }
        if self in mapping:
            return mapping[self]
        raise ValueError(f"Unsupported timeframe for MT5 string conversion: {self.name}")

    def to_ctrader_timeframe(self) -> str:
        """Converts enum to cTrader timeframe string."""
        # cTrader uses string representations like "m1", "h1", "d1"
        # Ref: ProtoOATrendbarPeriod enum in cTrader Open API for exact values.
        mapping = {
            Timeframe.M1: "m1", # ProtoOATrendbarPeriod.M1
            Timeframe.M5: "m5", # ProtoOATrendbarPeriod.M5
            Timeframe.M15: "m15",# ProtoOATrendbarPeriod.M15
            Timeframe.M30: "m30",# ProtoOATrendbarPeriod.M30
            Timeframe.H1: "h1",  # ProtoOATrendbarPeriod.H1
            Timeframe.H4: "h4",  # ProtoOATrendbarPeriod.H4
            Timeframe.D1: "d1",  # ProtoOATrendbarPeriod.D1
            Timeframe.W1: "w1",  # ProtoOATrendbarPeriod.W1
            Timeframe.MN1: "mn1",# ProtoOATrendbarPeriod.MN1
        }
        if self in mapping:
            return mapping[self]
        raise ValueError(f"Unsupported timeframe for cTrader: {self.name}")

    def to_seconds(self) -> int:
        """Converts timeframe to seconds. Returns 0 for TICK."""
        if self == Timeframe.TICK: return 0
        if self == Timeframe.M1: return 60
        if self == Timeframe.M5: return 5 * 60
        if self == Timeframe.M15: return 15 * 60
        if self == Timeframe.M30: return 30 * 60
        if self == Timeframe.H1: return 60 * 60
        if self == Timeframe.H4: return 4 * 60 * 60
        if self == Timeframe.D1: return 24 * 60 * 60
        if self == Timeframe.W1: return 7 * 24 * 60 * 60
        if self == Timeframe.MN1: return 30 * 24 * 60 * 60 # Approximate, can vary
        raise ValueError(f"Unknown timeframe for seconds conversion: {self.name}")

class BrokerType(Enum):
    METATRADER5 = "MetaTrader5"
    CTRADER = "cTrader"
    PAPER = "Paper" # For a paper trading only adapter
    # Add other brokers if needed

class StrategySignal(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD" # No action or continue holding existing position
    CLOSE_LONG = "CLOSE_LONG" # Signal to close an existing long position
    CLOSE_SHORT = "CLOSE_SHORT" # Signal to close an existing short position
    MODIFY_SLTP = "MODIFY_SLTP" # Signal to modify Stop Loss or Take Profit of an existing position
    NO_SIGNAL = "NO_SIGNAL" # Explicitly no trade signal

class PlatformType(Enum):
    METATRADER5 = "MetaTrader5"
    CTRADER = "cTrader"
    # Add more as needed
