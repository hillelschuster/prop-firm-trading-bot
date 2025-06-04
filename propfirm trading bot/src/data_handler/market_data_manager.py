# prop_firm_trading_bot/src/data_handler/market_data_manager.py

import pandas as pd
import pandas_ta as ta # For technical indicators
from typing import Dict, List, Optional, TYPE_CHECKING, Callable, Any
from datetime import datetime, timedelta, timezone
import logging
import time

from prop_firm_trading_bot.src.core.enums import Timeframe
from prop_firm_trading_bot.src.core.models import OHLCVData, TickData
# Assuming PlatformInterface is correctly in base_connector
from prop_firm_trading_bot.src.api_connector.base_connector import PlatformInterface 

if TYPE_CHECKING:
    from prop_firm_trading_bot.src.config_manager import AppConfig, AssetStrategyProfile

# Define callback types for internal data updates if needed by other modules
ProcessedBarCallback = Callable[[str, Timeframe, pd.DataFrame], None] # symbol, timeframe, dataframe_with_indicators
ProcessedTickCallback = Callable[[str, TickData], None] # symbol, tickdata

class MarketDataManager:
    def __init__(self, config: 'AppConfig', platform_adapter: PlatformInterface, logger: logging.Logger):
        self.config = config
        self.platform_adapter = platform_adapter
        self.logger = logger

        self.ohlcv_data: Dict[str, Dict[Timeframe, pd.DataFrame]] = {} # symbol -> timeframe -> DataFrame
        self.latest_ticks: Dict[str, TickData] = {}
        
        self._active_subscriptions: Dict[str, Dict[str, Any]] = {"ticks": {}, "bars": {}}

        # Indicator configurations will now be resolved dynamically per asset
        # profile based on strategy parameter files loaded by ConfigManager.
        self._initialize_data_stores_and_subscriptions()

    def _initialize_indicator_configs(self):
        """Prepares indicator configurations from strategy parameters."""
        for profile_key, profile_details in self.config.asset_strategy_profiles.items():
            if not profile_details.enabled:
                continue
            
            strategy_params_key = profile_details.strategy_params_key
            # Assuming strategy parameters are loaded into a central accessible place by Orchestrator
            # or we need to load them from their respective JSON files here.
            # For simplicity, let's assume strategy_config is accessible via self.config.strategy_parameter_sets
            # This part needs to be wired correctly with how strategy parameters are loaded and accessed.
            # For now, we'll mock this or assume a flat structure in main_config.
            # The config structure has `strategy_definitions` and `asset_strategy_profiles` referencing param keys.
            # The actual parameters for a profile would be loaded from the strategy_*.json files.
            # This manager would need access to those resolved parameters.
            
            # Let's assume a temporary way to get strategy params based on our config structure:
            # This logic needs to be robust: find the strategy definition, then load its specific parameters file.
            # This is simplified for now.
            # In a real setup, Orchestrator might pass resolved strategy param dicts.

            # Placeholder: This logic needs to be more robust to fetch actual strategy parameters
            # from the strategy JSON files based on strategy_params_key.
            # For now, we'll just create an empty dict.
            # Example from config: self.config.strategies (this was the old name, now strategy_definitions)
            # And strategy_*.json files hold params.
            # This MarketDataManager would typically be given the *specific parameters* for indicators
            # it needs to calculate for each symbol/timeframe, derived from the strategy configs.
            
            # Example: If a strategy for EURUSD_M15 needs RSI(14) and SMA(50)
            # symbol_tf_key = f"{profile_details.symbol}_{profile_details.strategy_parameters.get('timeframe', 'M15')}"
            # self.indicator_configs[symbol_tf_key] = {
            #     "rsi": {"length": profile_details.strategy_parameters.get('rsi_period', 14)},
            #     "sma_slow": {"length": profile_details.strategy_parameters.get('slow_sma_period', 50)}
            # }
            # This is highly dependent on how strategy parameters are structured and accessed.
            # For now, we'll make it flexible to add indicators dynamically or upon data request.
            pass


    def _initialize_data_stores_and_subscriptions(self):
        """Initializes data structures and subscribes to data streams based on active profiles."""
        self.logger.info("MarketDataManager: Initializing data stores and subscriptions based on active asset_strategy_profiles...")

        for profile_key, profile in self.config.asset_strategy_profiles.items():
            if not profile.enabled:
                self.logger.debug(f"Skipping disabled asset profile: {profile_key}")
                continue

            symbol = profile.symbol
            self.ohlcv_data.setdefault(symbol, {})

            strategy_specific_params = {}
            if hasattr(self.config, 'loaded_strategy_parameters') and isinstance(self.config.loaded_strategy_parameters, dict):
                strategy_param_set = self.config.loaded_strategy_parameters.get(profile.strategy_params_key, {})
                strategy_specific_params = strategy_param_set.get('parameters', {})

            timeframe_str = strategy_specific_params.get("timeframe", "H1").upper()
            try:
                strategy_timeframe = Timeframe[timeframe_str]
            except KeyError:
                self.logger.error(f"Invalid timeframe string '{timeframe_str}' for strategy profile '{profile_key}'. Defaulting to H1.")
                strategy_timeframe = Timeframe.H1

            self.logger.info(f"Profile '{profile_key}' for symbol {symbol} uses timeframe {strategy_timeframe.name}.")

            if strategy_timeframe not in self.ohlcv_data[symbol]:
                self.ohlcv_data[symbol][strategy_timeframe] = pd.DataFrame()

            if symbol not in self._active_subscriptions["ticks"]:
                if self.platform_adapter.subscribe_ticks(symbol, self._on_tick_data_received):
                    self.logger.info(f"MarketDataManager: Subscribed to ticks for {symbol}")
                    self._active_subscriptions["ticks"][symbol] = True
                else:
                    self.logger.error(f"MarketDataManager: Failed to subscribe to ticks for {symbol}")

            is_bar_subscribed = self._active_subscriptions["bars"].get(symbol, {}).get(strategy_timeframe, False)
            if not is_bar_subscribed:
                if self.platform_adapter.subscribe_bars(symbol, strategy_timeframe, self._on_bar_data_received):
                    self.logger.info(f"MarketDataManager: Subscribed to {strategy_timeframe.name} bars for {symbol} (Profile: {profile_key})")
                    self._active_subscriptions["bars"].setdefault(symbol, {})[strategy_timeframe] = True
                    if hasattr(self.platform_adapter, '_subscribed_bar_symbols_tf'):
                        self.platform_adapter._subscribed_bar_symbols_tf.setdefault(symbol, {})[strategy_timeframe] = datetime.fromtimestamp(0, tz=timezone.utc)
                else:
                    self.logger.error(f"MarketDataManager: Failed to subscribe to {strategy_timeframe.name} bars for {symbol} (Profile: {profile_key})")

            self.fetch_initial_history(symbol, strategy_timeframe, count=200)


    def _on_tick_data_received(self, tick_data: TickData):
        """Callback for when new tick data arrives from the platform adapter."""
        self.logger.debug(f"Tick received for {tick_data.symbol}: Bid={tick_data.bid}, Ask={tick_data.ask}")
        self.latest_ticks[tick_data.symbol] = tick_data
        # Here, you could also trigger an event or notify strategies that need tick data.

    def _on_bar_data_received(self, bar_data: OHLCVData):
        """Callback for when new bar data (closed bar) arrives from the platform adapter."""
        symbol = bar_data.symbol
        tf = bar_data.timeframe
        self.logger.debug(f"New {tf.name} bar received for {symbol} at {bar_data.timestamp}")

        df = self.ohlcv_data.setdefault(symbol, {}).setdefault(tf, pd.DataFrame())
        
        new_bar_df = pd.DataFrame([{
            "timestamp": bar_data.timestamp, # Already localized by adapter
            "open": bar_data.open,
            "high": bar_data.high,
            "low": bar_data.low,
            "close": bar_data.close,
            "volume": bar_data.volume
        }]).set_index("timestamp")

        # Append new bar, ensuring no duplicates and sorted index
        if not new_bar_df.index.isin(df.index).any():
            df = pd.concat([df, new_bar_df])
            df.sort_index(inplace=True)
            # Limit stored history size if necessary (e.g., keep last 1000 bars)
            max_bars = getattr(self.config.bot_settings, 'max_historical_bars_per_tf', 1000)  # type: ignore
            if len(df) > max_bars: # type: ignore
                df = df.iloc[-max_bars:] # type: ignore
            
            # Recalculate indicators for this symbol/timeframe
            df_with_indicators = self._calculate_and_store_indicators(symbol, tf, df.copy()) # Use copy
            self.ohlcv_data[symbol][tf] = df_with_indicators
            self.logger.debug(f"Updated {tf.name} data for {symbol} with new bar and indicators. Shape: {df_with_indicators.shape}")
        else:
            self.logger.debug(f"Bar at {bar_data.timestamp} for {symbol}/{tf.name} already exists. Updating last row.")
            # Update last row if timestamp matches (though new bar should be a new timestamp)
            df.loc[new_bar_df.index[0]] = new_bar_df.iloc[0]
            df_with_indicators = self._calculate_and_store_indicators(symbol, tf, df.copy())
            self.ohlcv_data[symbol][tf] = df_with_indicators


    def fetch_initial_history(self, symbol: str, timeframe: Timeframe, count: int = 200) -> bool:
        """Fetches initial historical data for a symbol and timeframe."""
        self.logger.info(f"Fetching initial {count} bars for {symbol}/{timeframe.name}...")
        try:
            historical_data = self.platform_adapter.get_historical_ohlcv(symbol, timeframe, count=count)
            if historical_data:
                # Convert List[OHLCVData] to DataFrame
                data_list = [{
                    "timestamp": bar.timestamp, # Already localized by adapter
                    "open": bar.open, "high": bar.high, "low": bar.low,
                    "close": bar.close, "volume": bar.volume
                } for bar in historical_data]
                
                df = pd.DataFrame(data_list).set_index("timestamp")
                df.sort_index(inplace=True) # Ensure chronological order

                # Calculate indicators and store
                df_with_indicators = self._calculate_and_store_indicators(symbol, timeframe, df)
                self.ohlcv_data.setdefault(symbol, {})[timeframe] = df_with_indicators
                
                self.logger.info(f"Successfully fetched and processed {len(df_with_indicators)} initial bars for {symbol}/{timeframe.name}.")
                
                # Update the last bar time for the polling mechanism in the adapter if it relies on it
                if self.platform_adapter._subscribed_bar_symbols_tf.get(symbol, {}).get(timeframe) is not None and not df.empty: # type: ignore
                     self.platform_adapter._subscribed_bar_symbols_tf[symbol][timeframe] = df.index[-1].tz_convert('UTC') # type: ignore

                return True
            else:
                self.logger.warning(f"No initial historical data returned for {symbol}/{timeframe.name}.")
                self.ohlcv_data.setdefault(symbol, {})[timeframe] = pd.DataFrame() # Ensure empty DF exists
                return False
        except Exception as e:
            self.logger.error(f"Error fetching initial history for {symbol}/{timeframe.name}: {e}", exc_info=True)
            self.ohlcv_data.setdefault(symbol, {})[timeframe] = pd.DataFrame()
            return False

    def _get_indicator_params_for_profile(self, asset_profile_key: str) -> Dict[str, Any]:
        """Retrieve indicator parameters for the given asset profile."""
        profile_details = self.config.asset_strategy_profiles.get(asset_profile_key)
        if not profile_details:
            self.logger.warning(f"No asset profile found for key: {asset_profile_key}")
            return {}

        strategy_params_key = profile_details.strategy_params_key

        if hasattr(self.config, 'loaded_strategy_parameters') and isinstance(self.config.loaded_strategy_parameters, dict):
            strategy_specific_params_set = self.config.loaded_strategy_parameters.get(strategy_params_key)
            if strategy_specific_params_set and 'parameters' in strategy_specific_params_set:
                self.logger.debug(
                    f"Found parameters for '{strategy_params_key}': {strategy_specific_params_set['parameters']}"
                )
                return strategy_specific_params_set['parameters']
            else:
                self.logger.warning(
                    f"No detailed parameters found for strategy_params_key '{strategy_params_key}' in loaded_strategy_parameters."
                )
                return {}
        else:
            self.logger.warning(
                "'self.config.loaded_strategy_parameters' not found or not a dict. Cannot fetch strategy-specific indicator params."
            )
            return {}

    def _calculate_and_store_indicators(self, symbol: str, timeframe: Timeframe, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates and appends configured indicators to the DataFrame."""
        if df.empty:
            return df

        # Get indicator parameters specific to the strategy for this symbol/timeframe
        # This part needs to be properly wired with the strategy configuration.
        # For now, we use a placeholder or a predefined set.
        
        # Determine which asset profile this symbol/timeframe belongs to
        asset_profile_key_for_symbol = None
        for pk, pv in self.config.asset_strategy_profiles.items():
            if pv.symbol == symbol and pv.enabled:
                asset_profile_key_for_symbol = pk
                break

        if not asset_profile_key_for_symbol:
            self.logger.debug(f"No active asset_profile for {symbol} found to determine indicator parameters. Skipping indicator calculation.")
            return df

        temp_indicator_params = self._get_indicator_params_for_profile(asset_profile_key_for_symbol)
        self.logger.debug(f"Calculating indicators for {symbol}/{timeframe.name} using resolved params: {temp_indicator_params}")

        try:
            if "sma_fast" in temp_indicator_params and "length" in temp_indicator_params["sma_fast"]:
                df.ta.sma(length=temp_indicator_params["sma_fast"]["length"], append=True, col_names=(f'SMA_{temp_indicator_params["sma_fast"]["length"]}',))
            if "sma_slow" in temp_indicator_params and "length" in temp_indicator_params["sma_slow"]:
                df.ta.sma(length=temp_indicator_params["sma_slow"]["length"], append=True, col_names=(f'SMA_{temp_indicator_params["sma_slow"]["length"]}',))
            if "sma_trend" in temp_indicator_params and "length" in temp_indicator_params["sma_trend"]: # For trend filter
                df.ta.sma(length=temp_indicator_params["sma_trend"]["length"], append=True, col_names=(f'SMA_trend_{temp_indicator_params["sma_trend"]["length"]}',))
            
            if "rsi" in temp_indicator_params and "length" in temp_indicator_params["rsi"]:
                df.ta.rsi(length=temp_indicator_params["rsi"]["length"], append=True, col_names=(f'RSI_{temp_indicator_params["rsi"]["length"]}',))
            
            if "bollinger" in temp_indicator_params and "length" in temp_indicator_params["bollinger"] and "std" in temp_indicator_params["bollinger"]:
                df.ta.bbands(length=temp_indicator_params["bollinger"]["length"], std=temp_indicator_params["bollinger"]["std"], append=True) # Appends BBL_len_std, BBM_len_std, BBU_len_std, BBB_len_std, BBP_len_std
            
            if "atr" in temp_indicator_params and "length" in temp_indicator_params["atr"]:
                df.ta.atr(length=temp_indicator_params["atr"]["length"], append=True, col_names=(f'ATR_{temp_indicator_params["atr"]["length"]}',)) # Appends ATR_length

            # Example: MACD
            # if "macd" in self.indicator_configs.get(f"{symbol}_{timeframe.name}", {}):
            #     macd_params = self.indicator_configs[f"{symbol}_{timeframe.name}"]["macd"]
            #     df.ta.macd(fast=macd_params.get('fast', 12), slow=macd_params.get('slow', 26), signal=macd_params.get('signal', 9), append=True)
        except Exception as e:
            self.logger.error(f"Error calculating indicators for {symbol}/{timeframe.name}: {e}", exc_info=True)
        
        return df

    def get_market_data(self, symbol: str, timeframe: Timeframe) -> Optional[pd.DataFrame]:
        """Returns the DataFrame with OHLCV and calculated indicators for a symbol/timeframe."""
        return self.ohlcv_data.get(symbol, {}).get(timeframe)

    def get_latest_tick_data(self, symbol: str) -> Optional[TickData]:
        """Returns the latest known tick data for a symbol."""
        return self.latest_ticks.get(symbol)

    def get_current_price(self, symbol: str, side: str = "mid") -> Optional[float]:
        """Returns the current bid, ask, or mid price for a symbol from the latest tick."""
        tick = self.latest_ticks.get(symbol)
        if not tick:
            # Fallback: try to get from platform directly if no tick received recently
            tick_from_platform = self.platform_adapter.get_latest_tick(symbol)
            if not tick_from_platform:
                self.logger.warning(f"No current price available for {symbol}")
                return None
            tick = tick_from_platform # Use this for price
            self.latest_ticks[symbol] = tick # Update cache

        if side == "bid":
            return tick.bid
        elif side == "ask":
            return tick.ask
        elif side == "mid":
            return (tick.bid + tick.ask) / 2
        else:
            self.logger.warning(f"Invalid price side '{side}' requested for {symbol}.")
            return None

    def ensure_data_subscription(self, symbol: str, timeframe: Timeframe):
        """Ensures that bar data subscription for the given symbol/timeframe is active."""
        if symbol not in self._active_subscriptions["bars"] or \
           timeframe not in self._active_subscriptions["bars"].get(symbol, {}):
            if self.platform_adapter.subscribe_bars(symbol, timeframe, self._on_bar_data_received):
                self.logger.info(f"MarketDataManager: Dynamically subscribed to {timeframe.name} bars for {symbol}")
                self._active_subscriptions["bars"].setdefault(symbol, {})[timeframe] = True
                # Initialize last bar time to ensure polling picks up new bars
                self.platform_adapter._subscribed_bar_symbols_tf.setdefault(symbol, {})[timeframe] = datetime.fromtimestamp(0, tz=timezone.utc) # type: ignore
                 # Fetch some initial history after subscribing to populate indicators
                if not self.fetch_initial_history(symbol, timeframe, count=200): # count should be enough for indicators
                    self.logger.warning(f"Failed to fetch initial history for dynamically subscribed {symbol}/{timeframe.name}")
            else:
                self.logger.error(f"MarketDataManager: Failed to dynamically subscribe to {timeframe.name} bars for {symbol}")

    def stop_all_subscriptions(self):
        self.logger.info("MarketDataManager: Stopping all data subscriptions...")
        for symbol in list(self._active_subscriptions["ticks"].keys()):
            self.platform_adapter.unsubscribe_ticks(symbol) # Unsubscribe all callbacks
            # self._active_subscriptions["ticks"].pop(symbol, None) # Should be handled by platform_interface or adapter
        
        for symbol, tf_map in list(self._active_subscriptions["bars"].items()):
            for tf in list(tf_map.keys()):
                self.platform_adapter.unsubscribe_bars(symbol, tf)
                # self._active_subscriptions["bars"][symbol].pop(tf, None)
        self.logger.info("MarketDataManager: All data subscriptions stopped.")
