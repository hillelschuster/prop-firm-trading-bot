# prop_firm_trading_bot/src/data_handler/market_data_manager.py

import pandas as pd
import pandas_ta as ta # For technical indicators
from typing import Dict, List, Optional, TYPE_CHECKING, Callable, Any
from datetime import datetime, timedelta, timezone
import logging
import time

from src.core.enums import Timeframe
from src.core.models import OHLCVData, TickData
# Assuming PlatformInterface is correctly in base_connector
from src.api_connector.base_connector import PlatformInterface

if TYPE_CHECKING:
    from src.config_manager import AppConfig, AssetStrategyProfile

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

    def _initialize_data_stores_and_subscriptions(self):
        """
        Initializes data structures and subscribes to data streams for all active strategies.
        This method iterates through enabled asset profiles and ensures that the necessary
        market data subscriptions (ticks and bars) are active.
        """
        self.logger.info("Initializing data stores and subscriptions for active strategies...")

        for profile_key, profile in self.config.asset_strategy_profiles.items():
            if not profile.enabled:
                self.logger.debug(f"Skipping disabled asset profile: {profile_key}")
                continue

            symbol = profile.symbol
            self.ohlcv_data.setdefault(symbol, {})

            # Determine the timeframe required by the strategy for this profile
            strategy_params = self._get_strategy_params(profile.strategy_params_key)
            timeframe_str = strategy_params.get("timeframe", "H1").upper()
            try:
                strategy_timeframe = getattr(Timeframe, timeframe_str)
            except AttributeError:
                self.logger.error(f"Invalid timeframe '{timeframe_str}' for profile '{profile_key}'. Defaulting to H1.")
                strategy_timeframe = Timeframe.H1

            self.logger.info(f"Ensuring data for profile '{profile_key}' ({symbol} @ {strategy_timeframe.name}).")
            self.ensure_data_subscription(symbol, strategy_timeframe)

    def _get_strategy_params(self, params_key: str) -> Dict[str, Any]:
        """Helper to safely retrieve strategy parameters from the loaded config."""
        if hasattr(self.config, 'loaded_strategy_parameters'):
            param_set = self.config.loaded_strategy_parameters.get(params_key)
            if param_set and hasattr(param_set, 'parameters'):
                return param_set.parameters
        return {}

    def _on_tick_data_received(self, tick_data: TickData):
        """Callback for when new tick data arrives from the platform adapter."""
        self.logger.debug(f"Tick received for {tick_data.symbol}: Bid={tick_data.bid}, Ask={tick_data.ask}")
        self.latest_ticks[tick_data.symbol] = tick_data

    def _on_bar_data_received(self, bar_data: OHLCVData):
        """
        Callback for when a new closed bar arrives from the platform adapter.
        It appends the new bar to the historical data, ensures the data store
        doesn't exceed the maximum configured size, and recalculates indicators.
        """
        symbol, tf = bar_data.symbol, bar_data.timeframe
        self.logger.debug(f"New {tf.name} bar received for {symbol} at {bar_data.timestamp}")

        df = self.ohlcv_data.setdefault(symbol, {}).setdefault(tf, pd.DataFrame())
        
        new_bar_df = pd.DataFrame([bar_data.dict()]).set_index("timestamp")
        # Ensure proper DatetimeIndex
        if not isinstance(new_bar_df.index, pd.DatetimeIndex):
            new_bar_df.index = pd.to_datetime(new_bar_df.index, utc=True)

        if not new_bar_df.index.isin(df.index).any():
            df = pd.concat([df, new_bar_df])
            df.sort_index(inplace=True)
            
            max_bars = self.config.bot_settings.max_historical_bars_per_tf
            if len(df) > max_bars:
                df = df.iloc[-max_bars:]
            
            df_with_indicators = self._calculate_and_store_indicators(symbol, tf, df.copy())
            self.ohlcv_data[symbol][tf] = df_with_indicators
            self.logger.debug(f"Updated {tf.name} data for {symbol}. Shape: {df_with_indicators.shape}")
        else:
            # This case handles situations where an existing bar's data is updated.
            self.logger.debug(f"Bar at {bar_data.timestamp} for {symbol}/{tf.name} already exists. Updating.")
            df.loc[new_bar_df.index[0]] = new_bar_df.iloc[0]
            df_with_indicators = self._calculate_and_store_indicators(symbol, tf, df.copy())
            self.ohlcv_data[symbol][tf] = df_with_indicators

    def fetch_initial_history(self, symbol: str, timeframe: Timeframe, count: int = 200) -> bool:
        """
        Fetches an initial set of historical OHLCV data for a given symbol and timeframe,
        calculates indicators, and stores the resulting DataFrame.
        """
        self.logger.info(f"Fetching initial {count} bars for {symbol}/{timeframe.name}...")
        try:
            historical_data = self.platform_adapter.get_historical_ohlcv(symbol, timeframe, count=count)
            if not historical_data:
                self.logger.warning(f"No initial historical data returned for {symbol}/{timeframe.name}.")
                self.ohlcv_data.setdefault(symbol, {})[timeframe] = pd.DataFrame()
                return False

            df = pd.DataFrame([bar.dict() for bar in historical_data]).set_index("timestamp")
            # Ensure proper DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, utc=True)
            df.sort_index(inplace=True)

            df_with_indicators = self._calculate_and_store_indicators(symbol, timeframe, df)
            self.ohlcv_data.setdefault(symbol, {})[timeframe] = df_with_indicators
            
            self.logger.info(f"Successfully processed {len(df_with_indicators)} initial bars for {symbol}/{timeframe.name}.")
            
            if not df.empty:
                last_bar_timestamp = df.index[-1]
                if isinstance(last_bar_timestamp, pd.Timestamp):
                    self.platform_adapter.set_initial_bar_timestamp(symbol, timeframe, last_bar_timestamp.to_pydatetime())

            return True
        except Exception as e:
            self.logger.error(f"Error fetching initial history for {symbol}/{timeframe.name}: {e}", exc_info=True)
            self.ohlcv_data.setdefault(symbol, {})[timeframe] = pd.DataFrame()
            return False

    def _find_asset_profile_keys(self, symbol: str, timeframe: Timeframe) -> List[str]:
        """Finds all active asset profile keys for a given symbol and timeframe."""
        matching_keys = []
        for key, profile in self.config.asset_strategy_profiles.items():
            if not (profile.enabled and profile.symbol == symbol):
                continue
            
            params = self._get_strategy_params(profile.strategy_params_key)
            profile_timeframe_str = params.get("timeframe", "H1").upper()
            if profile_timeframe_str == timeframe.name:
                matching_keys.append(key)
        return matching_keys

    def _extract_indicator_config_for_profile(self, asset_profile_key: str) -> Dict[str, Any]:
        """Extracts and transforms indicator parameters from a strategy's configuration."""
        profile_details = self.config.asset_strategy_profiles.get(asset_profile_key)
        if not profile_details:
            return {}

        params_set = self.config.loaded_strategy_parameters.get(profile_details.strategy_params_key)
        if not (params_set and hasattr(params_set, 'parameters') and hasattr(params_set, 'strategy_definition_key')):
            return {}
        
        return self._extract_indicator_requirements(
            params_set.strategy_definition_key,
            params_set.parameters
        )

    def _extract_indicator_requirements(self, strategy_definition_key: str, strategy_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts indicator requirements from strategy parameters based on the strategy type.
        This acts as a mapping between strategy parameter names and pandas-ta function arguments.
        """
        indicator_config = {}
        try:
            if "SMA" in strategy_definition_key or "TrendFollowing" in strategy_definition_key:
                if "fast_sma_period" in strategy_params:
                    indicator_config["sma_fast"] = {"length": strategy_params["fast_sma_period"]}
                if "slow_sma_period" in strategy_params:
                    indicator_config["sma_slow"] = {"length": strategy_params["slow_sma_period"]}
                if "atr_period_for_sl" in strategy_params:
                    indicator_config["atr"] = {"length": strategy_params["atr_period_for_sl"]}

            elif "RSI" in strategy_definition_key or "MeanReversion" in strategy_definition_key:
                if "rsi_period" in strategy_params:
                    indicator_config["rsi"] = {"length": strategy_params["rsi_period"]}
                if "bollinger_period" in strategy_params and "bollinger_std_dev" in strategy_params:
                    indicator_config["bollinger"] = {"length": strategy_params["bollinger_period"], "std": strategy_params["bollinger_std_dev"]}
                if "trend_filter_ma_period" in strategy_params:
                    indicator_config["sma_trend"] = {"length": strategy_params["trend_filter_ma_period"]}
                if "stop_loss_atr_period" in strategy_params:
                    indicator_config["atr"] = {"length": strategy_params["stop_loss_atr_period"]}
        except Exception as e:
            self.logger.error(f"Error extracting indicator requirements for {strategy_definition_key}: {e}", exc_info=True)
        return indicator_config

    def _aggregate_indicator_configs(self, profile_keys: List[str]) -> Dict[str, Any]:
        """Aggregates indicator configurations from multiple strategy profiles, avoiding duplicates."""
        aggregated_config = {}
        for key in profile_keys:
            profile_config = self._extract_indicator_config_for_profile(key)
            for name, config in profile_config.items():
                if name in aggregated_config and aggregated_config[name] != config:
                    self.logger.warning(f"Conflicting configs for indicator '{name}'. Using existing: {aggregated_config[name]}")
                else:
                    aggregated_config[name] = config
        return aggregated_config

    def _apply_indicators_to_dataframe(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Applies a set of indicator calculations to a given DataFrame."""
        try:
            if "sma_fast" in config:
                df.ta.sma(length=config["sma_fast"]["length"], append=True, col_names=(f'SMA_{config["sma_fast"]["length"]}',))
            if "sma_slow" in config:
                df.ta.sma(length=config["sma_slow"]["length"], append=True, col_names=(f'SMA_{config["sma_slow"]["length"]}',))
            if "sma_trend" in config:
                df.ta.sma(length=config["sma_trend"]["length"], append=True, col_names=(f'SMA_trend_{config["sma_trend"]["length"]}',))
            if "rsi" in config:
                df.ta.rsi(length=config["rsi"]["length"], append=True, col_names=(f'RSI_{config["rsi"]["length"]}',))
            if "bollinger" in config:
                df.ta.bbands(length=config["bollinger"]["length"], std=config["bollinger"]["std"], append=True)
            if "atr" in config:
                df.ta.atr(length=config["atr"]["length"], append=True, col_names=(f'ATR_{config["atr"]["length"]}',))
        except Exception as e:
            self.logger.error(f"Error applying indicators to DataFrame: {e}", exc_info=True)
        return df

    def _calculate_and_store_indicators(self, symbol: str, timeframe: Timeframe, df: pd.DataFrame) -> pd.DataFrame:
        """
        Orchestrates the calculation of all required indicators for a given symbol and timeframe
        by finding relevant strategies, aggregating their indicator needs, and applying them.
        """
        if df.empty:
            return df

        # 1. Find all strategy profiles for this symbol/timeframe
        matching_profile_keys = self._find_asset_profile_keys(symbol, timeframe)
        if not matching_profile_keys:
            self.logger.warning(f"No active strategy profile for {symbol}/{timeframe.name}. Skipping indicator calculation.")
            return df

        # 2. Aggregate all unique indicator configurations from these profiles
        aggregated_config = self._aggregate_indicator_configs(matching_profile_keys)
        self.logger.info(f"Calculating aggregated indicators for {symbol}/{timeframe.name}: {aggregated_config}")

        # 3. Apply the final set of indicators to the DataFrame
        return self._apply_indicators_to_dataframe(df, aggregated_config)

    def _is_backtesting_mode(self) -> bool:
        """Detects if the bot is in backtesting mode."""
        return hasattr(self.platform_adapter, 'get_timeframe_data')

    def _fetch_data_from_backtest_adapter(self, symbol: str, timeframe: Timeframe, up_to_timestamp: Optional[pd.Timestamp] = None) -> Optional[pd.DataFrame]:
        """Fetches historical data directly from the paper trading adapter during a backtest."""
        try:
            if not hasattr(self.platform_adapter, 'get_timeframe_data'):
                return None
            
            # The get_timeframe_data method is expected to handle the timestamp filtering.
            data = self.platform_adapter.get_timeframe_data(timeframe, up_to_timestamp.to_pydatetime() if up_to_timestamp else None) # type: ignore

            if data is not None and not data.empty:
                self.logger.info(f"Retrieved {len(data)} bars for {symbol}/{timeframe.name} from backtesting adapter.")
                return data
        except Exception as e:
            self.logger.error(f"Error getting backtesting data for {symbol}/{timeframe.name}: {e}", exc_info=True)
        return None

    def get_market_data(self, symbol: str, timeframe: Timeframe, up_to_timestamp: Optional[pd.Timestamp] = None) -> Optional[pd.DataFrame]:
        """
        Returns the DataFrame with OHLCV and calculated indicators for a symbol/timeframe.
        In backtesting, it ensures data is filtered to prevent look-ahead bias.
        """
        full_data = self.ohlcv_data.get(symbol, {}).get(timeframe)

        if full_data is None and self._is_backtesting_mode():
            full_data = self._fetch_data_from_backtest_adapter(symbol, timeframe, up_to_timestamp)
            if full_data is not None and not full_data.empty:
                full_data = self._calculate_and_store_indicators(symbol, timeframe, full_data.copy())
                self.ohlcv_data.setdefault(symbol, {})[timeframe] = full_data

        if full_data is None:
            return None

        if up_to_timestamp is None:
            return full_data

        # Filter data for backtesting to prevent look-ahead bias
        try:
            filter_ts = pd.to_datetime(up_to_timestamp, utc=True)

            # Ensure the DataFrame index is a proper DatetimeIndex
            if not isinstance(full_data.index, pd.DatetimeIndex):
                full_data.index = pd.to_datetime(full_data.index, utc=True)

            filtered_df = full_data[full_data.index <= filter_ts].copy()
            return filtered_df if isinstance(filtered_df, pd.DataFrame) else None
        except Exception as e:
            self.logger.error(f"Error filtering market data for {symbol}/{timeframe.name} up to {up_to_timestamp}: {e}", exc_info=True)
            return None

    def get_latest_tick_data(self, symbol: str) -> Optional[TickData]:
        """Returns the latest known tick data for a symbol."""
        return self.latest_ticks.get(symbol)

    def get_current_price(self, symbol: str, side: str = "mid") -> Optional[float]:
        """
        Returns the current bid, ask, or mid price for a symbol.
        It uses the latest tick data and falls back to a direct platform query if needed.
        """
        tick = self.latest_ticks.get(symbol)
        if not tick:
            tick = self.platform_adapter.get_latest_tick(symbol)
            if not tick:
                self.logger.warning(f"No current price available for {symbol}")
                return None
            self.latest_ticks[symbol] = tick

        if side == "bid": return tick.bid
        if side == "ask": return tick.ask
        if side == "mid": return (tick.bid + tick.ask) / 2
        
        self.logger.warning(f"Invalid price side '{side}' requested for {symbol}.")
        return None

    def ensure_data_subscription(self, symbol: str, timeframe: Timeframe):
        """
        Ensures that tick and bar data subscriptions for a given symbol and timeframe are active.
        If not, it subscribes and fetches initial historical data.
        """
        # Ensure tick subscription is active for the symbol
        if symbol not in self._active_subscriptions["ticks"]:
            if self.platform_adapter.subscribe_ticks(symbol, self._on_tick_data_received):
                self.logger.info(f"Dynamically subscribed to ticks for {symbol}")
                self._active_subscriptions["ticks"][symbol] = True
            else:
                self.logger.error(f"Failed to dynamically subscribe to ticks for {symbol}")

        # Ensure bar subscription is active for the symbol and timeframe
        if timeframe not in self._active_subscriptions["bars"].get(symbol, {}):
            if self.platform_adapter.subscribe_bars(symbol, timeframe, self._on_bar_data_received):
                self.logger.info(f"Dynamically subscribed to {timeframe.name} bars for {symbol}")
                self._active_subscriptions["bars"].setdefault(symbol, {})[timeframe] = True
                self.fetch_initial_history(symbol, timeframe, count=self.config.bot_settings.max_historical_bars_per_tf)
            else:
                self.logger.error(f"Failed to dynamically subscribe to {timeframe.name} bars for {symbol}")

    def stop_all_subscriptions(self):
        """Unsubscribes from all active market data streams."""
        self.logger.info("Stopping all data subscriptions...")
        for symbol in list(self._active_subscriptions["ticks"].keys()):
            self.platform_adapter.unsubscribe_ticks(symbol)
        
        for symbol, tf_map in list(self._active_subscriptions["bars"].items()):
            for tf in list(tf_map.keys()):
                self.platform_adapter.unsubscribe_bars(symbol, tf)
        self.logger.info("All data subscriptions stopped.")


  
