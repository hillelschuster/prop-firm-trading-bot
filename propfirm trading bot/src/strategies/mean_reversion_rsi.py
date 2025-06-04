# prop_firm_trading_bot/src/strategies/mean_reversion_rsi.py

import pandas as pd
import logging
from typing import Dict, Any, Optional, TYPE_CHECKING

from prop_firm_trading_bot.src.strategies.base_strategy import BaseStrategy
from prop_firm_trading_bot.src.core.enums import StrategySignal, OrderAction, Timeframe
from prop_firm_trading_bot.src.core.models import OHLCVData, TickData, Order, Position # Order, Position not used in this example, but good for reference

if TYPE_CHECKING:
    from prop_firm_trading_bot.src.data_handler.market_data_manager import MarketDataManager # Corrected path
    from prop_firm_trading_bot.src.api_connector.base_connector import PlatformInterface
    from prop_firm_trading_bot.src.config_manager import AppConfig


class MeanReversionRSI(BaseStrategy):
    """
    A mean-reversion strategy using the Relative Strength Index (RSI)
    to identify overbought/oversold conditions.
    - Enters long when RSI crosses below an oversold level (e.g., 30) if trend filter allows.
    - Enters short when RSI crosses above an overbought level (e.g., 70) if trend filter allows.
    - Exits are based on RSI moving back to a neutral zone or hitting SL/TP.
    - Optionally uses a longer-term SMA as a trend filter.
    """

    def __init__(self,
                 strategy_params: Dict[str, Any],
                 config: 'AppConfig',
                 platform_adapter: 'PlatformInterface',
                 market_data_manager: 'MarketDataManager',
                 logger: logging.Logger,
                 asset_profile_key: str):
        super().__init__(strategy_params, config, platform_adapter, market_data_manager, logger, asset_profile_key)

        self.rsi_period: int = self.strategy_params.get("rsi_period", 14)
        self.oversold_level: float = self.strategy_params.get("rsi_oversold", 30.0)
        self.overbought_level: float = self.strategy_params.get("rsi_overbought", 70.0)
        self.trend_filter_ma_period: Optional[int] = self.strategy_params.get("trend_filter_ma_period") # e.g., 200
        
        self.stop_loss_atr_period: int = self.strategy_params.get("stop_loss_atr_period", 14)
        self.stop_loss_atr_multiplier: float = self.strategy_params.get("stop_loss_atr_multiplier", 1.5)
        self.take_profit_atr_multiplier: float = self.strategy_params.get("take_profit_atr_multiplier", 2.0) # For a fixed RR
        
        self.exit_rsi_neutral_low: Optional[float] = self.strategy_params.get("exit_rsi_neutral_low", 45.0) # e.g., exit long if RSI crosses above 45
        self.exit_rsi_neutral_high: Optional[float] = self.strategy_params.get("exit_rsi_neutral_high", 55.0) # e.g., exit short if RSI crosses below 55

        self.max_position_age_bars: Optional[int] = self.strategy_params.get("max_position_age_bars")

    def _initialize_indicators(self) -> None:
        """
        Relies on MarketDataManager to provide:
        - RSI_rsi_period (e.g., RSI_14)
        - SMA_trend_filter_ma_period (e.g., SMA_200), if trend_filter_ma_period is set
        - ATR_stop_loss_atr_period (e.g., ATR_14)
        """
        self.logger.debug(f"[{self.symbol}/{self.timeframe.name}] MeanReversionRSI: Initializing. Expecting MarketDataManager to provide RSI, optional SMA trend filter, and ATR.")
        # As with TrendFollowingSMA, MarketDataManager should be configured for these.

    def generate_signal(self) -> Optional[Dict[str, Any]]:
        market_data_df = self.market_data_manager.get_market_data(self.symbol, self.timeframe)

        min_data_length = self.rsi_period + 1
        if self.trend_filter_ma_period:
            min_data_length = max(min_data_length, self.trend_filter_ma_period + 1)
        if self.stop_loss_atr_period:
             min_data_length = max(min_data_length, self.stop_loss_atr_period + 1)


        if market_data_df is None or market_data_df.empty or len(market_data_df) < min_data_length:
            self.logger.debug(f"[{self.symbol}/{self.timeframe.name}] Not enough data for RSI strategy (need {min_data_length}, got {len(market_data_df) if market_data_df is not None else 0}).")
            return None

        rsi_col = f'RSI_{self.rsi_period}'
        atr_col = f'ATR_{self.stop_loss_atr_period}'
        trend_ma_col = f'SMA_trend_{self.trend_filter_ma_period}' if self.trend_filter_ma_period else None

        required_cols = [rsi_col, atr_col, 'close', 'high', 'low']
        if trend_ma_col:
            required_cols.append(trend_ma_col)
        
        if not all(col in market_data_df.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in market_data_df.columns]
            self.logger.warning(f"[{self.symbol}/{self.timeframe.name}] Missing required columns in market data: {missing_cols}. Available: {market_data_df.columns.tolist()}")
            return None

        last_row = market_data_df.iloc[-1]
        prev_row = market_data_df.iloc[-2] # Need previous row for crossover detection of RSI levels

        if pd.isna(last_row[rsi_col]) or pd.isna(prev_row[rsi_col]) or pd.isna(last_row[atr_col]):
            self.logger.debug(f"[{self.symbol}/{self.timeframe.name}] RSI or ATR values are NaN. Not enough data for signal.")
            return None
        if trend_ma_col and pd.isna(last_row[trend_ma_col]):
            self.logger.debug(f"[{self.symbol}/{self.timeframe.name}] Trend MA value is NaN. Not enough data for signal.")
            return None

        current_positions = self.platform_adapter.get_open_positions(symbol=self.symbol)
        active_long_position = next((p for p in current_positions if p.action == OrderAction.BUY), None)
        active_short_position = next((p for p in current_positions if p.action == OrderAction.SELL), None)

        current_tick = self.market_data_manager.get_latest_tick_data(self.symbol)
        if not current_tick:
            self.logger.warning(f"[{self.symbol}/{self.timeframe.name}] Could not get current tick for price reference.")
            return None
            
        symbol_info = self.market_data_manager.platform_adapter.get_symbol_info(self.symbol)
        if not symbol_info:
            self.logger.error(f"[{self.symbol}/{self.timeframe.name}] Could not get symbol info for rounding prices.")
            return None

        signal_details = None
        atr_value = last_row[atr_col]
        current_rsi = last_row[rsi_col]
        previous_rsi = prev_row[rsi_col]

        # Trend Filter Logic
        trend_allows_long = True
        trend_allows_short = True
        if trend_ma_col:
            current_close = last_row['close']
            trend_ma_value = last_row[trend_ma_col]
            if current_close < trend_ma_value: # Price below long-term MA suggests downtrend
                trend_allows_long = False
            if current_close > trend_ma_value: # Price above long-term MA suggests uptrend
                trend_allows_short = False

        # --- Entry Signals ---
        # Buy Signal: RSI crosses up from oversold, and trend filter allows long
        if previous_rsi < self.oversold_level and current_rsi >= self.oversold_level:
            if not active_long_position and trend_allows_long:
                entry_price = current_tick.ask
                sl_distance = atr_value * self.stop_loss_atr_multiplier
                tp_distance = sl_distance * self.take_profit_atr_multiplier
                stop_loss_price = entry_price - sl_distance
                take_profit_price = entry_price + tp_distance
                
                signal_details = {
                    "signal": StrategySignal.BUY,
                    "price": entry_price,
                    "stop_loss_price": round(stop_loss_price, symbol_info.digits),
                    "take_profit_price": round(take_profit_price, symbol_info.digits),
                    "comment": f"RSI Buy ({current_rsi:.2f} crossed {self.oversold_level})"
                }
                self.logger.info(f"[{self.symbol}/{self.timeframe.name}] {signal_details['comment']} at ~{entry_price}")

        # Sell Signal: RSI crosses down from overbought, and trend filter allows short
        elif previous_rsi > self.overbought_level and current_rsi <= self.overbought_level:
            if not active_short_position and trend_allows_short:
                entry_price = current_tick.bid
                sl_distance = atr_value * self.stop_loss_atr_multiplier
                tp_distance = sl_distance * self.take_profit_atr_multiplier
                stop_loss_price = entry_price + sl_distance
                take_profit_price = entry_price - tp_distance

                signal_details = {
                    "signal": StrategySignal.SELL,
                    "price": entry_price,
                    "stop_loss_price": round(stop_loss_price, symbol_info.digits),
                    "take_profit_price": round(take_profit_price, symbol_info.digits),
                    "comment": f"RSI Sell ({current_rsi:.2f} crossed {self.overbought_level})"
                }
                self.logger.info(f"[{self.symbol}/{self.timeframe.name}] {signal_details['comment']} at ~{entry_price}")
        
        # --- Exit Signals (based on RSI returning to neutral, if no other signal generated yet) ---
        if not signal_details: # Only check these if no entry signal was generated
            if active_long_position and self.exit_rsi_neutral_high is not None and \
               previous_rsi < self.exit_rsi_neutral_high and current_rsi >= self.exit_rsi_neutral_high:
                signal_details = {
                    "signal": StrategySignal.CLOSE_LONG,
                    "position_id": active_long_position.position_id,
                    "price": current_tick.bid,
                    "comment": f"RSI Close Long (RSI crossed neutral {self.exit_rsi_neutral_high})"
                }
                self.logger.info(f"[{self.symbol}/{self.timeframe.name}] {signal_details['comment']} at ~{current_tick.bid}")

            elif active_short_position and self.exit_rsi_neutral_low is not None and \
                 previous_rsi > self.exit_rsi_neutral_low and current_rsi <= self.exit_rsi_neutral_low:
                signal_details = {
                    "signal": StrategySignal.CLOSE_SHORT,
                    "position_id": active_short_position.position_id,
                    "price": current_tick.ask,
                    "comment": f"RSI Close Short (RSI crossed neutral {self.exit_rsi_neutral_low})"
                }
                self.logger.info(f"[{self.symbol}/{self.timeframe.name}] {signal_details['comment']} at ~{current_tick.ask}")

        return signal_details

    def manage_open_position(self, position: Position, latest_bar: Optional[OHLCVData] = None, latest_tick: Optional[TickData] = None) -> Optional[Dict[str, Any]]:
        """Manages open position, e.g., for max position age."""
        if not latest_tick or not latest_bar: # Need latest_bar for its timestamp
            return None

        # Check for max position age exit
        if self.max_position_age_bars:
            # This requires bar data with timestamps and knowing when the position was opened.
            # Position.open_time is datetime. We need to count bars since then.
            # This is a simplified check based on total seconds vs. bar duration.
            # A more robust one would look at bar index in dataframe if available.
            time_open_seconds = (latest_bar.timestamp - position.open_time).total_seconds()
            bar_duration_seconds = self.timeframe.to_seconds()
            if bar_duration_seconds > 0: # Avoid division by zero for TICK timeframe if it were used
                bars_open = time_open_seconds / bar_duration_seconds
                if bars_open >= self.max_position_age_bars:
                    self.logger.info(f"[{self.symbol}/{self.timeframe.name}] Position {position.position_id} "
                                     f"exceeded max age of {self.max_position_age_bars} bars ({bars_open:.1f} bars). Signaling close.")
                    close_price = latest_tick.bid if position.action == OrderAction.BUY else latest_tick.ask
                    close_signal = StrategySignal.CLOSE_LONG if position.action == OrderAction.BUY else StrategySignal.CLOSE_SHORT
                    return {
                        "signal": close_signal,
                        "position_id": position.position_id,
                        "price": close_price,
                        "comment": f"Max position age ({self.max_position_age_bars} bars) reached."
                    }
        return None
