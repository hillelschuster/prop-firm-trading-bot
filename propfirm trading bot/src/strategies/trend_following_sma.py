# prop_firm_trading_bot/src/strategies/trend_following_sma.py

import pandas as pd
import logging
from typing import Dict, Any, Optional, TYPE_CHECKING

from prop_firm_trading_bot.src.strategies.base_strategy import BaseStrategy
from prop_firm_trading_bot.src.core.enums import StrategySignal, OrderAction, Timeframe
from prop_firm_trading_bot.src.core.models import OHLCVData, TickData, Order, Position

if TYPE_CHECKING:
    from prop_firm_trading_bot.src.data_handler.market_data_manager import MarketDataManager # Corrected import path
    from prop_firm_trading_bot.src.api_connector.base_connector import PlatformInterface
    from prop_firm_trading_bot.src.config_manager import AppConfig


class TrendFollowingSMA(BaseStrategy):
    """
    A trend-following strategy based on Simple Moving Average (SMA) crossovers.
    - Enters long when fast SMA crosses above slow SMA.
    - Exits long (or enters short if configured) when fast SMA crosses below slow SMA.
    - Uses ATR for stop-loss placement.
    - Aims for a minimum reward/risk ratio for take-profit.
    """

    def __init__(self,
                 strategy_params: Dict[str, Any],
                 config: 'AppConfig',
                 platform_adapter: 'PlatformInterface',
                 market_data_manager: 'MarketDataManager',
                 logger: logging.Logger,
                 asset_profile_key: str):
        super().__init__(strategy_params, config, platform_adapter, market_data_manager, logger, asset_profile_key)
        
        self.fast_sma_period: int = self.strategy_params.get("fast_sma_period", 10)
        self.slow_sma_period: int = self.strategy_params.get("slow_sma_period", 50)
        self.atr_period_for_sl: int = self.strategy_params.get("atr_period_for_sl", 14)
        self.atr_multiplier_for_sl: float = self.strategy_params.get("atr_multiplier_for_sl", 2.0)
        self.min_reward_risk_ratio: float = self.strategy_params.get("min_reward_risk_ratio", 1.5)
        
        self.use_trailing_stop: bool = self.strategy_params.get("use_trailing_stop", False)
        self.trailing_stop_atr_period: int = self.strategy_params.get("trailing_stop_atr_period", 14)
        self.trailing_stop_atr_multiplier: float = self.strategy_params.get("trailing_stop_atr_multiplier", 2.0)
        
        self.max_position_age_bars: Optional[int] = self.strategy_params.get("max_position_age_bars")


    def _initialize_indicators(self) -> None:
        """
        Ensures necessary indicators (SMAs, ATR) are calculated by MarketDataManager
        or that the strategy knows how to calculate them if needed.
        For this example, we rely on MarketDataManager to provide columns like:
        - SMA_fast_sma_period (e.g., SMA_10)
        - SMA_slow_sma_period (e.g., SMA_50)
        - ATR_atr_period_for_sl (e.g., ATR_14)
        """
        self.logger.debug(f"[{self.symbol}/{self.timeframe.name}] TrendFollowingSMA: Initializing. Expecting MarketDataManager to provide SMA and ATR.")
        # The MarketDataManager should be configured (based on all strategy needs)
        # to calculate these indicators. This method can serve as a check or
        # a place to request specific calculations if MDM supports dynamic indicator additions.
        # For now, we assume MDM is pre-configured via main config's strategy definitions.
        
        # Example: Check if MarketDataManager has fetched initial data for this strategy's timeframe
        # This might be done by the orchestrator before calling generate_signal for the first time.
        # self.market_data_manager.fetch_initial_history(self.symbol, self.timeframe, count=self.slow_sma_period + self.atr_period_for_sl + 5)


    def generate_signal(self) -> Optional[Dict[str, Any]]:
        """
        Generates a trading signal based on SMA crossover and ATR for SL/TP.
        """
        market_data_df = self.market_data_manager.get_market_data(self.symbol, self.timeframe)

        if market_data_df is None or market_data_df.empty or len(market_data_df) < self.slow_sma_period + 1:
            self.logger.debug(f"[{self.symbol}/{self.timeframe.name}] Not enough data for SMA crossover strategy (need {self.slow_sma_period + 1}, got {len(market_data_df) if market_data_df is not None else 0}).")
            return None

        # Ensure required indicator columns are present (MarketDataManager should have added them)
        fast_sma_col = f'SMA_{self.fast_sma_period}'
        slow_sma_col = f'SMA_{self.slow_sma_period}'
        atr_col = f'ATR_{self.atr_period_for_sl}'

        required_cols = [fast_sma_col, slow_sma_col, atr_col, 'close', 'high', 'low']
        # Check if 'close', 'high', 'low' are in index or columns if they are not indicators
        if not all(col in market_data_df.columns for col in [fast_sma_col, slow_sma_col, atr_col]):
             # Check for OHLC columns separately as they are fundamental
            if not all(col in market_data_df.columns for col in ['close', 'high', 'low']):
                self.logger.warning(f"[{self.symbol}/{self.timeframe.name}] Missing fundamental OHLC columns in market data. Available: {market_data_df.columns.tolist()}")
                return None
            self.logger.warning(f"[{self.symbol}/{self.timeframe.name}] Missing required indicator columns in market data: Expected {required_cols}. Available: {market_data_df.columns.tolist()}")
            return None
        
        # Get the last two rows for crossover detection and current ATR
        last_row = market_data_df.iloc[-1]
        prev_row = market_data_df.iloc[-2]

        if pd.isna(last_row[fast_sma_col]) or pd.isna(last_row[slow_sma_col]) or \
           pd.isna(prev_row[fast_sma_col]) or pd.isna(prev_row[slow_sma_col]) or \
           pd.isna(last_row[atr_col]):
            self.logger.debug(f"[{self.symbol}/{self.timeframe.name}] SMA or ATR values are NaN on last/prev row. Not enough data for signal.")
            return None

        current_positions = self.platform_adapter.get_open_positions(symbol=self.symbol)
        active_long_position = next((p for p in current_positions if p.action == OrderAction.BUY), None)
        active_short_position = next((p for p in current_positions if p.action == OrderAction.SELL), None)

        current_tick = self.market_data_manager.get_latest_tick_data(self.symbol)
        if not current_tick:
            self.logger.warning(f"[{self.symbol}/{self.timeframe.name}] Could not get current tick for price reference.")
            return None
        
        symbol_info = self.market_data_manager.platform_adapter.get_symbol_info(self.symbol) # Get symbol info for digits
        if not symbol_info:
            self.logger.error(f"[{self.symbol}/{self.timeframe.name}] Could not get symbol info for rounding prices.")
            return None


        signal_details = None
        atr_value = last_row[atr_col]
        
        # Bullish Crossover: Fast SMA crosses above Slow SMA
        if prev_row[fast_sma_col] <= prev_row[slow_sma_col] and \
           last_row[fast_sma_col] > last_row[slow_sma_col]:
            if active_short_position: # If currently short, signal to close short
                signal_details = {
                    "signal": StrategySignal.CLOSE_SHORT,
                    "position_id": active_short_position.position_id,
                    "price": current_tick.ask, # Price to buy back to cover short
                    "comment": f"SMA Crossover: Close Short {self.symbol}"
                }
                self.logger.info(f"[{self.symbol}/{self.timeframe.name}] {signal_details['comment']} at ~{current_tick.ask}")
            elif not active_long_position: # If no active long, signal to buy
                entry_price = current_tick.ask
                stop_loss_price = entry_price - (atr_value * self.atr_multiplier_for_sl)
                sl_distance = atr_value * self.atr_multiplier_for_sl
                tp_distance = sl_distance * self.min_reward_risk_ratio
                
                signal_details = {
                    "signal": StrategySignal.BUY,
                    "price": entry_price, 
                    "stop_loss_price": round(stop_loss_price, symbol_info.digits),
                    "take_profit_price": round(entry_price + tp_distance, symbol_info.digits),
                    "comment": f"SMA Crossover: Buy {self.symbol}"
                }
                self.logger.info(f"[{self.symbol}/{self.timeframe.name}] {signal_details['comment']} at ~{entry_price}, SL ~{signal_details['stop_loss_price']:.{symbol_info.digits}f}, TP ~{signal_details['take_profit_price']:.{symbol_info.digits}f}")

        # Bearish Crossover: Fast SMA crosses below Slow SMA
        elif prev_row[fast_sma_col] >= prev_row[slow_sma_col] and \
             last_row[fast_sma_col] < last_row[slow_sma_col]:
            if active_long_position: # If currently long, signal to close long
                signal_details = {
                    "signal": StrategySignal.CLOSE_LONG,
                    "position_id": active_long_position.position_id,
                    "price": current_tick.bid, 
                    "comment": f"SMA Crossover: Close Long {self.symbol}"
                }
                self.logger.info(f"[{self.symbol}/{self.timeframe.name}] {signal_details['comment']} at ~{current_tick.bid}")
            elif not active_short_position: 
                entry_price = current_tick.bid
                stop_loss_price = entry_price + (atr_value * self.atr_multiplier_for_sl)
                sl_distance = atr_value * self.atr_multiplier_for_sl
                tp_distance = sl_distance * self.min_reward_risk_ratio

                signal_details = {
                    "signal": StrategySignal.SELL,
                    "price": entry_price, 
                    "stop_loss_price": round(stop_loss_price, symbol_info.digits),
                    "take_profit_price": round(entry_price - tp_distance, symbol_info.digits),
                    "comment": f"SMA Crossover: Sell {self.symbol}"
                }
                self.logger.info(f"[{self.symbol}/{self.timeframe.name}] {signal_details['comment']} at ~{entry_price}, SL ~{signal_details['stop_loss_price']:.{symbol_info.digits}f}, TP ~{signal_details['take_profit_price']:.{symbol_info.digits}f}")
        
        return signal_details


    def manage_open_position(self, position: Position, latest_bar: Optional[OHLCVData] = None, latest_tick: Optional[TickData] = None) -> Optional[Dict[str, Any]]:
        if not latest_tick: # Need current price for trailing stop
            self.logger.warning(f"[{self.symbol}/{self.timeframe.name}] No latest_tick data for manage_open_position.")
            return None

        symbol_info = self.market_data_manager.platform_adapter.get_symbol_info(self.symbol)
        if not symbol_info:
            self.logger.error(f"[{self.symbol}/{self.timeframe.name}] Could not get symbol info for rounding prices in manage_open_position.")
            return None

        # ATR Trailing Stop Logic
        if self.use_trailing_stop:
            market_data_df = self.market_data_manager.get_market_data(self.symbol, self.timeframe) # Use strategy's primary timeframe for ATR
            if market_data_df is None or market_data_df.empty:
                self.logger.warning(f"[{self.symbol}/{self.timeframe.name}] No market data for trailing stop ATR calculation.")
                return None # Cannot calculate TS without data
                
            ts_atr_col = f'ATR_{self.trailing_stop_atr_period}'
            if ts_atr_col not in market_data_df.columns or pd.isna(market_data_df[ts_atr_col].iloc[-1]):
                self.logger.warning(f"[{self.symbol}/{self.timeframe.name}] Trailing stop ATR column '{ts_atr_col}' not found or NaN.")
                return None # Cannot calculate TS

            current_atr_for_ts = market_data_df[ts_atr_col].iloc[-1]
            new_stop_loss = None
            current_sl = position.stop_loss
            
            # Determine comparison price based on tick data
            comparison_price = latest_tick.bid if position.action == OrderAction.BUY else latest_tick.ask

            if position.action == OrderAction.BUY:
                potential_new_sl = comparison_price - (current_atr_for_ts * self.trailing_stop_atr_multiplier)
                if potential_new_sl > (current_sl or float('-inf')): # Only trail up
                    # Ensure SL is not beyond open price unless some profit is locked
                    # This simple version just trails if it's higher than current SL.
                    # More complex logic could ensure it's also above open_price + buffer.
                    new_stop_loss = round(potential_new_sl, symbol_info.digits)
            
            elif position.action == OrderAction.SELL:
                potential_new_sl = comparison_price + (current_atr_for_ts * self.trailing_stop_atr_multiplier)
                if potential_new_sl < (current_sl or float('inf')): # Only trail down
                    new_stop_loss = round(potential_new_sl, symbol_info.digits)

            if new_stop_loss is not None and new_stop_loss != current_sl:
                self.logger.info(f"[{self.symbol}/{self.timeframe.name}] Trailing SL for position {position.position_id}. Current SL: {current_sl}, New SL: {new_stop_loss:.{symbol_info.digits}f}")
                return {
                    "signal": StrategySignal.MODIFY_SLTP,
                    "position_id": position.position_id,
                    "new_stop_loss": new_stop_loss,
                    "new_take_profit": position.take_profit 
                }
            
        # Max Position Age Exit Logic
        if self.max_position_age_bars and latest_bar:
            # This requires bar data with timestamps and knowing when the position was opened.
            # Position.open_time is datetime. We need to count bars since then.
            # This is a simplified check. A robust one would look at bar index in dataframe.
            # Assuming latest_bar.timestamp is the close time of the most recent bar.
            # And position.open_time is when the position was initiated.
            
            # Get the DataFrame for the strategy's timeframe
            df = self.market_data_manager.get_market_data(self.symbol, self.timeframe)
            if df is not None and not df.empty and 'timestamp' in df.index.names:
                # Find the index of the bar at or just after the position open time
                try:
                    # Ensure position.open_time is timezone-aware and matches df.index timezone
                    open_time_aware = position.open_time.astimezone(df.index.tz) if df.index.tz else position.open_time.replace(tzinfo=None)
                    
                    open_bar_index = df.index.get_indexer([open_time_aware], method='bfill')[0] # Get index of bar at or after open
                    current_bar_index = df.index.get_loc(latest_bar.timestamp.astimezone(df.index.tz) if df.index.tz else latest_bar.timestamp.replace(tzinfo=None))
                    
                    bars_open = current_bar_index - open_bar_index
                    
                    if bars_open >= self.max_position_age_bars:
                        self.logger.info(f"[{self.symbol}/{self.timeframe.name}] Position {position.position_id} (opened {bars_open} bars ago) exceeded max age of {self.max_position_age_bars} bars. Signaling close.")
                        close_price = latest_tick.bid if position.action == OrderAction.BUY else latest_tick.ask
                        close_signal = StrategySignal.CLOSE_LONG if position.action == OrderAction.BUY else StrategySignal.CLOSE_SHORT
                        return {
                            "signal": close_signal,
                            "position_id": position.position_id,
                            "price": close_price,
                            "comment": f"Max position age ({self.max_position_age_bars} bars) reached."
                        }
                except Exception as e:
                    self.logger.error(f"[{self.symbol}/{self.timeframe.name}] Error calculating bars_open for max_position_age: {e}", exc_info=True)

        return None
