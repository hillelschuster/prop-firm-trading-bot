# prop_firm_trading_bot/src/strategies/base_strategy.py

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, TYPE_CHECKING
import pandas as pd
import logging

from prop_firm_trading_bot.src.core.enums import StrategySignal, Timeframe
from prop_firm_trading_bot.src.core.models import OHLCVData, TickData, Order, Position, TradeFill, MarketEvent

if TYPE_CHECKING:
    from prop_firm_trading_bot.src.data_handler.market_data_manager import MarketDataManager
    from prop_firm_trading_bot.src.api_connector.base_connector import PlatformInterface
    from prop_firm_trading_bot.src.config_manager import AppConfig


class BaseStrategy(ABC):
    """
    Abstract Base Class for all trading strategies.
    Each strategy must implement methods to initialize indicators, generate trading signals,
    and potentially manage open positions.
    """

    def __init__(self,
                 strategy_params: Dict[str, Any],
                 config: 'AppConfig', # Full app config for broader context if needed
                 platform_adapter: 'PlatformInterface',
                 market_data_manager: 'MarketDataManager',
                 logger: logging.Logger,
                 asset_profile_key: str # e.g., "EURUSD_MeanReversion_M15"
                ):
        self.strategy_params = strategy_params # Specific parameters for this strategy instance
        self.config = config
        self.platform_adapter = platform_adapter
        self.market_data_manager = market_data_manager
        self.logger = logger
        self.asset_profile_key = asset_profile_key # Key for this asset-strategy combination

        # Derived from asset_profile in main_config, resolved by Orchestrator
        self.symbol = self.config.asset_strategy_profiles[asset_profile_key].symbol
        # Ensure Timeframe is correctly accessed from the enum using the string value
        timeframe_str = self.strategy_params.get("timeframe", "H1").upper()
        try:
            self.timeframe = Timeframe[timeframe_str]
        except KeyError:
            self.logger.error(f"Invalid timeframe string '{timeframe_str}' in strategy_params for {self.asset_profile_key}. Defaulting to H1.")
            self.timeframe = Timeframe.H1


        self.logger.info(f"Strategy '{self.__class__.__name__}' initialized for {self.symbol} on {self.timeframe.name} with params: {strategy_params}")
        self._initialize_indicators()

    @abstractmethod
    def _initialize_indicators(self) -> None:
        """
        Initialize or confirm necessary indicators for the strategy.
        This might involve ensuring the MarketDataManager is configured to calculate them.
        """
        # Example: self.market_data_manager.ensure_indicator(self.symbol, self.timeframe, "SMA", {"length": self.strategy_params.get("sma_period")})
        # For now, MarketDataManager calculates a fixed set or is told by Orchestrator based on all strategies.
        # This method can verify if required data columns exist in the DataFrame from MarketDataManager.
        self.logger.debug(f"[{self.symbol}/{self.timeframe.name}] BaseStrategy: _initialize_indicators called (implement in child).")
        pass

    @abstractmethod
    def generate_signal(self) -> Optional[Dict[str, Any]]:
        """
        Generates a trading signal based on the current market data and strategy logic.

        Returns:
            Optional[Dict[str, Any]]: A dictionary representing the trade signal, e.g.,
            {
                "signal": StrategySignal.BUY, # or StrategySignal.SELL, StrategySignal.CLOSE_LONG, etc.
                "price": Optional[float],     # Suggested entry/exit price (for limit orders or reference)
                "sl_pips": Optional[float],   # Stop loss in pips from entry (if applicable)
                "tp_pips": Optional[float],   # Take profit in pips from entry (if applicable)
                "volume_pct_of_max": Optional[float], # Suggested volume as % of max allowed by risk (e.g. 1.0 for full risk)
                "comment": Optional[str]
            }
            Returns None if no signal is generated.
        """
        pass

    def manage_open_position(self, position: Position, latest_bar: Optional[OHLCVData] = None, latest_tick: Optional[TickData] = None) -> Optional[Dict[str, Any]]:
        """
        Optional method for strategies that require active management of open positions,
        such as trailing stops, partial closes, or time-based exits not covered by SL/TP.

        Args:
            position (Position): The current open position object.
            latest_bar (Optional[OHLCVData]): The latest closed bar data for the position's symbol/timeframe.
            latest_tick (Optional[TickData]): The latest tick data for the position's symbol.

        Returns:
            Optional[Dict[str, Any]]: An action dictionary if management is needed, e.g.,
            {
                "signal": StrategySignal.CLOSE_LONG, # or StrategySignal.CLOSE_SHORT
                "position_id": position.position_id,
                "volume_to_close_pct": Optional[float], # Percentage of position to close (e.g., 0.5 for 50%, 1.0 for full)
                "price": Optional[float], # Price for limit close if applicable
                "comment": Optional[str]
            }
            or for SL/TP modification:
            {
                "signal": StrategySignal.MODIFY_SLTP, # A new signal type might be needed if not closing
                "position_id": position.position_id,
                "new_stop_loss": Optional[float],
                "new_take_profit": Optional[float],
                "comment": Optional[str]
            }
            Returns None if no management action is required by the strategy at this moment.
        """
        # Default implementation: no active management beyond initial SL/TP set by RiskController.
        # Subclasses can override this for more complex exit/management logic.
        self.logger.debug(f"[{self.symbol}/{self.timeframe.name}] BaseStrategy: manage_open_position called for {position.position_id} (default: no action).")
        return None

    def get_required_data(self) -> Dict[str, Any]:
        """
        Allows the strategy to specify its data requirements (symbols, timeframes, indicators).
        The Orchestrator or MarketDataManager can use this.
        """
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "indicators": self.strategy_params.get("indicators_needed", []) # Example
        }

    def on_order_update(self, order: Order):
        """
        Optional callback for the strategy to react to its own order updates.
        """
        self.logger.debug(f"[{self.symbol}/{self.timeframe.name}] Strategy received order update: {order.order_id} status {order.status.name}")
        pass

    def on_fill(self, fill: TradeFill):
        """
        Optional callback for the strategy to react to its own trade fills.
        """
        self.logger.info(f"[{self.symbol}/{self.timeframe.name}] Strategy received fill: {fill.action.name} {fill.volume} @ {fill.price} for order {fill.order_id}")
        pass

    def on_market_event(self, market_event: MarketEvent):
        """
        Optional callback for the strategy to react to broader market events (e.g., news filtered by NewsFilter).
        """
        self.logger.debug(f"[{self.symbol}/{self.timeframe.name}] Strategy received market event: {market_event.event_type}")
        # Example: A strategy might choose to flatten its position if a certain critical market event occurs.
        # if market_event.event_type == "CRITICAL_NEWS_UNEXPECTED" and self.symbol in market_event.symbols_affected:
        #     self.logger.warning(f"Strategy for {self.symbol} reacting to critical market event: {market_event.description}")
        #     # Logic to signal a close might be triggered here.
        pass
