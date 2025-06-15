# This is the tests/unit/test_strategy_sma.py file.
import pytest
from unittest import mock
import pandas as pd
from datetime import datetime, timezone, timedelta
import logging

# Imports from your project
from prop_firm_trading_bot.src.strategies.trend_following_sma import TrendFollowingSMA
from prop_firm_trading_bot.src.core.enums import StrategySignal, OrderAction, Timeframe
from prop_firm_trading_bot.src.core.models import OHLCVData, TickData, Position, SymbolInfo
from prop_firm_trading_bot.src.config_manager import AppConfig, AssetStrategyProfile, StrategyParameterSet
# Assuming PlatformInterface is imported if needed for type hinting mocks
from prop_firm_trading_bot.src.api_connector.base_connector import PlatformInterface
from prop_firm_trading_bot.src.data_handler.market_data_manager import MarketDataManager


# --- Fixtures ---

@pytest.fixture
def mock_logger(mocker):
    return mocker.MagicMock(spec=logging.Logger)

@pytest.fixture
def mock_platform_adapter(mocker):
    adapter = mocker.MagicMock(spec=PlatformInterface)
    # Default mock for symbol_info needed for rounding
    mock_sym_info = mocker.MagicMock(spec=SymbolInfo)
    mock_sym_info.digits = 5 # Example for EURUSD
    mock_sym_info.point = 0.00001 # Example for EURUSD
    adapter.get_symbol_info.return_value = mock_sym_info
    return adapter

@pytest.fixture
def mock_market_data_manager(mocker):
    manager = mocker.MagicMock(spec=MarketDataManager)
    # Note: get_instrument_properties is no longer used by TrendFollowingSMA for point/digits for SL/TP.
    # That logic now uses platform_adapter.get_symbol_info().
    # This mock might be relevant if other parts of MarketDataManager are used by the strategy.
    return manager

@pytest.fixture
def sma_strategy_params_eurusd_h1():
    # Based on your uploaded strategy_sma_eurusd_h1.json
    return {
        "timeframe": "H1",
        "fast_sma_period": 20,
        "slow_sma_period": 50,
        "atr_period_for_sl": 14,
        "atr_multiplier_for_sl": 2.0,
        "min_reward_risk_ratio": 1.5,
        "use_trailing_stop": True,
        "trailing_stop_atr_period": 14, # Assuming same as SL ATR for simplicity here
        "trailing_stop_atr_multiplier": 1.5,
        "max_position_age_bars": 100
    }

@pytest.fixture
def mock_app_config_for_sma_strategy(mocker, sma_strategy_params_eurusd_h1):
    config = mocker.MagicMock(spec=AppConfig)
    
    # Mock structure for asset_strategy_profiles
    config.asset_strategy_profiles = {
        "EURUSD_SMA_H1_Profile": AssetStrategyProfile(
            symbol="EURUSD",
            enabled=True,
            instrument_details_key="EURUSD_FTMO", # Not directly used by strategy, but good for completeness
            strategy_params_key="strategy_sma_eurusd_h1" # Key to link to loaded_strategy_parameters
        )
    }
    # Mock loaded_strategy_parameters
    config.loaded_strategy_parameters = {
        "strategy_sma_eurusd_h1": StrategyParameterSet(
            description="Trend Following SMA Crossover for EURUSD H1",
            strategy_definition_key="TrendFollowing_SMA_Cross", # Not directly used by strategy instance once params are passed
            parameters=sma_strategy_params_eurusd_h1
        )
    }
    # Mock bot_settings if strategy uses it (e.g., for timezone, though less common for pure strategy logic)
    config.bot_settings.ftmo_server_timezone = "Europe/Prague"
    return config

@pytest.fixture
def trend_following_sma_strategy(
    mock_app_config_for_sma_strategy, 
    mock_platform_adapter, 
    mock_market_data_manager, 
    mock_logger,
    sma_strategy_params_eurusd_h1 # Directly use the params for strategy init
    ):
    # The strategy gets its direct params, not the full AppConfig for its core logic usually.
    # It needs AppConfig for symbol/timeframe resolution primarily via asset_profile_key
    strategy = TrendFollowingSMA(
        strategy_params=sma_strategy_params_eurusd_h1,
        config=mock_app_config_for_sma_strategy, # Full config for context
        platform_adapter=mock_platform_adapter,
        market_data_manager=mock_market_data_manager,
        logger=mock_logger,
        asset_profile_key="EURUSD_SMA_H1_Profile"
    )
    return strategy

# --- Helper Function to Create Sample Market Data ---

def create_sample_market_df(fast_sma_period=20, slow_sma_period=50, atr_period=14, rows=60):
    base_time = datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    data = {
        'open': [1.1000 + i*0.0001 for i in range(rows)],
        'high': [1.1010 + i*0.0001 for i in range(rows)],
        'low': [1.0990 + i*0.0001 for i in range(rows)],
        'close': [1.1005 + i*0.0001 for i in range(rows)],
        'volume': [100 + i for i in range(rows)]
    }
    index = [base_time + timedelta(hours=i) for i in range(rows)] # H1 timeframe
    df = pd.DataFrame(data, index=pd.Index(index, name="timestamp"))
    
    # Add mock indicators - strategy relies on these being present
    df[f'SMA_{fast_sma_period}'] = df['close'].rolling(window=fast_sma_period).mean()
    df[f'SMA_{slow_sma_period}'] = df['close'].rolling(window=slow_sma_period).mean()
    # ATR calculation is more complex, using a simple mock value for ATR column
    df[f'ATR_{atr_period}'] = 0.00100 # Mock ATR value
    return df

# --- Test Cases for generate_signal ---

class TestTrendFollowingSMAGenerateSignal:

    def test_generate_signal_not_enough_data(self, trend_following_sma_strategy):
        # Pass empty DataFrame directly to generate_signal
        signal = trend_following_sma_strategy.generate_signal(
            market_data_df=pd.DataFrame(),
            active_position=None,
            latest_tick=None
        )
        assert signal is None
        trend_following_sma_strategy.logger.debug.assert_any_call(
            f"[{trend_following_sma_strategy.symbol}/{trend_following_sma_strategy.timeframe.name}] Not enough data for SMA crossover strategy (need {trend_following_sma_strategy.slow_sma_period + 1}, got 0)."
        )

    def test_generate_signal_missing_indicator_columns(self, trend_following_sma_strategy):
        df = create_sample_market_df(rows=trend_following_sma_strategy.slow_sma_period + 5)
        # Remove a required column
        df_missing_sma = df.drop(columns=[f'SMA_{trend_following_sma_strategy.fast_sma_period}'])
        
        mock_tick = TickData(timestamp=datetime.now(timezone.utc), symbol="EURUSD", bid=1.09090, ask=1.09110)

        signal = trend_following_sma_strategy.generate_signal(
            market_data_df=df_missing_sma,
            active_position=None,
            latest_tick=mock_tick
        )
        assert signal is None
        trend_following_sma_strategy.logger.warning.assert_any_call(
            f"[{trend_following_sma_strategy.symbol}/{trend_following_sma_strategy.timeframe.name}] Missing required indicator columns in market data: Expected {['SMA_20', 'SMA_50', 'ATR_14', 'close', 'high', 'low']}. Available: {df_missing_sma.columns.tolist()}"
        )

    def test_generate_signal_missing_ohlc_columns(self, trend_following_sma_strategy):
        df = create_sample_market_df(rows=trend_following_sma_strategy.slow_sma_period + 5)
        # Remove a fundamental OHLC column
        df_missing_ohlc = df.drop(columns=['close'])
        
        mock_tick = TickData(timestamp=datetime.now(timezone.utc), symbol="EURUSD", bid=1.09090, ask=1.09110)

        signal = trend_following_sma_strategy.generate_signal(
            market_data_df=df_missing_ohlc,
            active_position=None,
            latest_tick=mock_tick
        )
        assert signal is None
        # This specific log might change based on the order of checks in strategy
        # For now, assuming it hits the indicator check first if 'close' is also used by indicators.
        # The strategy code first checks indicators, then OHLC if indicators are missing.
        # If 'close' is missing, it will be caught by the indicator check if indicators depend on it,
        # or by the explicit OHLC check.
        # Based on current strategy: it checks for indicator columns first. If an indicator (e.g. SMA) needs 'close' and 'close' is missing,
        # the indicator calculation would fail earlier or the column wouldn't exist.
        # The strategy's check is: `if not all(col in market_data_df.columns for col in [fast_sma_col, slow_sma_col, atr_col]):`
        # then `if not all(col in market_data_df.columns for col in ['close', 'high', 'low']):`
        # So, if SMA_X is missing (because 'close' was missing for its calculation), it logs missing indicators.
        # If SMA_X columns are present (e.g. pre-calculated externally but 'close' is missing from df for current logic),
        # then it would log missing OHLC.
        # For this test, let's assume indicator columns are present but 'close' is missing from the df passed to generate_signal.
        # The create_sample_market_df adds SMA columns based on 'close'. If 'close' is dropped *after* SMA creation,
        # the SMA columns would exist.

        df_with_indicators_no_close = create_sample_market_df(rows=trend_following_sma_strategy.slow_sma_period + 5)
        df_with_indicators_no_close = df_with_indicators_no_close.drop(columns=['close'])


        signal_ohlc_missing = trend_following_sma_strategy.generate_signal(
            market_data_df=df_with_indicators_no_close, # SMAs exist, but 'close' is gone
            active_position=None,
            latest_tick=mock_tick
        )
        assert signal_ohlc_missing is None
        trend_following_sma_strategy.logger.warning.assert_any_call(
             f"[{trend_following_sma_strategy.symbol}/{trend_following_sma_strategy.timeframe.name}] Missing fundamental OHLC columns in market data. Available: {df_with_indicators_no_close.columns.tolist()}"
        )


    def test_generate_signal_nan_in_indicators(self, trend_following_sma_strategy, sma_strategy_params_eurusd_h1):
        fast_p = sma_strategy_params_eurusd_h1['fast_sma_period']
        slow_p = sma_strategy_params_eurusd_h1['slow_sma_period']
        atr_p = sma_strategy_params_eurusd_h1['atr_period_for_sl']
        
        df = create_sample_market_df(fast_sma_period=fast_p, slow_sma_period=slow_p, atr_period=atr_p, rows=slow_p + 5)
        # Set NaN for a required indicator on the last row
        df.loc[df.index[-1], f'SMA_{fast_p}'] = pd.NA
        
        mock_tick = TickData(timestamp=datetime.now(timezone.utc), symbol="EURUSD", bid=1.09090, ask=1.09110)

        signal = trend_following_sma_strategy.generate_signal(
            market_data_df=df,
            active_position=None,
            latest_tick=mock_tick
        )
        assert signal is None
        trend_following_sma_strategy.logger.debug.assert_any_call(
            f"[{trend_following_sma_strategy.symbol}/{trend_following_sma_strategy.timeframe.name}] SMA or ATR values are NaN on last/prev row. Not enough data for signal."
        )

    def test_generate_signal_latest_tick_none(self, trend_following_sma_strategy, sma_strategy_params_eurusd_h1):
        fast_p = sma_strategy_params_eurusd_h1['fast_sma_period']
        slow_p = sma_strategy_params_eurusd_h1['slow_sma_period']
        atr_p = sma_strategy_params_eurusd_h1['atr_period_for_sl']
        df = create_sample_market_df(fast_sma_period=fast_p, slow_sma_period=slow_p, atr_period=atr_p, rows=slow_p + 5)
        # Ensure crossover condition to attempt signal generation
        df.loc[df.index[-2], f'SMA_{fast_p}'] = 1.0900
        df.loc[df.index[-2], f'SMA_{slow_p}'] = 1.0905
        df.loc[df.index[-1], f'SMA_{fast_p}'] = 1.0910
        df.loc[df.index[-1], f'SMA_{slow_p}'] = 1.0908

        signal = trend_following_sma_strategy.generate_signal(
            market_data_df=df,
            active_position=None,
            latest_tick=None # Key condition for this test
        )
        assert signal is None
        trend_following_sma_strategy.logger.warning.assert_any_call(
            f"[{trend_following_sma_strategy.symbol}/{trend_following_sma_strategy.timeframe.name}] Could not get current tick (latest_tick) for price reference."
        )

    def test_generate_signal_symbol_info_none(self, trend_following_sma_strategy, sma_strategy_params_eurusd_h1, mock_platform_adapter):
        fast_p = sma_strategy_params_eurusd_h1['fast_sma_period']
        slow_p = sma_strategy_params_eurusd_h1['slow_sma_period']
        atr_p = sma_strategy_params_eurusd_h1['atr_period_for_sl']
        df = create_sample_market_df(fast_sma_period=fast_p, slow_sma_period=slow_p, atr_period=atr_p, rows=slow_p + 5)
        df.loc[df.index[-2], f'SMA_{fast_p}'] = 1.0900; df.loc[df.index[-2], f'SMA_{slow_p}'] = 1.0905
        df.loc[df.index[-1], f'SMA_{fast_p}'] = 1.0910; df.loc[df.index[-1], f'SMA_{slow_p}'] = 1.0908
        
        mock_tick = TickData(timestamp=datetime.now(timezone.utc), symbol="EURUSD", bid=1.09090, ask=1.09110)
        mock_platform_adapter.get_symbol_info.return_value = None # Key condition

        signal = trend_following_sma_strategy.generate_signal(
            market_data_df=df, active_position=None, latest_tick=mock_tick
        )
        assert signal is None
        trend_following_sma_strategy.logger.error.assert_any_call(
            f"[{trend_following_sma_strategy.symbol}/{trend_following_sma_strategy.timeframe.name}] Could not get symbol info for rounding prices."
        )
        
    # This test is removed as its functionality (testing missing symbol info for SL/TP)
    # is covered by test_generate_signal_symbol_info_none, and the strategy
    # now uses platform_adapter.get_symbol_info() not market_data_manager.get_instrument_properties()
    # for point/digits.
    # def test_generate_signal_instrument_properties_none(...)

    def test_generate_signal_instrument_properties_invalid_point(self, trend_following_sma_strategy, sma_strategy_params_eurusd_h1, mock_platform_adapter):
        fast_p = sma_strategy_params_eurusd_h1['fast_sma_period']
        slow_p = sma_strategy_params_eurusd_h1['slow_sma_period']
        atr_p = sma_strategy_params_eurusd_h1['atr_period_for_sl']
        df = create_sample_market_df(fast_sma_period=fast_p, slow_sma_period=slow_p, atr_period=atr_p, rows=slow_p + 5)
        df.loc[df.index[-2], f'SMA_{fast_p}'] = 1.0900; df.loc[df.index[-2], f'SMA_{slow_p}'] = 1.0905
        df.loc[df.index[-1], f'SMA_{fast_p}'] = 1.0910; df.loc[df.index[-1], f'SMA_{slow_p}'] = 1.0908
        
        mock_tick = TickData(timestamp=datetime.now(timezone.utc), symbol="EURUSD", bid=1.09090, ask=1.09110)
        
        # Mock platform_adapter to return symbol_info with an invalid point
        mock_invalid_sym_info = mock.MagicMock(spec=SymbolInfo)
        mock_invalid_sym_info.digits = 5
        mock_invalid_sym_info.point = 0  # Invalid point
        mock_platform_adapter.get_symbol_info.return_value = mock_invalid_sym_info
        # Re-initialize strategy or update its symbol_info if it's cached in __init__
        # For this test, we assume generate_signal re-fetches or uses the updated mock from platform_adapter.
        # The TrendFollowingSMA strategy fetches symbol_info in __init__ and stores it as self.symbol_info.
        # So, to test this path correctly, the strategy instance itself needs to have bad symbol_info.
        # This can be done by re-initializing the strategy with the mock_platform_adapter already configured to return bad info,
        # or by directly setting trend_following_sma_strategy.symbol_info to the bad mock.
        # For simplicity here, we'll assume the test setup implies the strategy will use this bad info.
        # A more robust test would re-initialize the strategy fixture.
        # However, the strategy's _calculate_sl_tp_pips directly uses self.platform_adapter.get_symbol_info()
        # so this direct mock of platform_adapter should be fine.

        signal = trend_following_sma_strategy.generate_signal(
            market_data_df=df, active_position=None, latest_tick=mock_tick
        )
        assert signal is None
        trend_following_sma_strategy.logger.error.assert_any_call(
            f"[{trend_following_sma_strategy.symbol}/{trend_following_sma_strategy.timeframe.name}] Invalid 'point' value (0) for {trend_following_sma_strategy.symbol} (from SymbolInfo). Must be a positive number. Cannot calculate sl_pips."
        )


    def test_generate_signal_bullish_crossover_no_active_position(self, trend_following_sma_strategy, sma_strategy_params_eurusd_h1):
        fast_p = sma_strategy_params_eurusd_h1['fast_sma_period']
        slow_p = sma_strategy_params_eurusd_h1['slow_sma_period']
        atr_p = sma_strategy_params_eurusd_h1['atr_period_for_sl']
        atr_mult = sma_strategy_params_eurusd_h1['atr_multiplier_for_sl']
        rr_ratio = sma_strategy_params_eurusd_h1['min_reward_risk_ratio']

        df = create_sample_market_df(fast_sma_period=fast_p, slow_sma_period=slow_p, atr_period=atr_p)
        # Simulate bullish crossover
        df.loc[df.index[-2], f'SMA_{fast_p}'] = 1.0900
        df.loc[df.index[-2], f'SMA_{slow_p}'] = 1.0905 # Fast below slow
        df.loc[df.index[-1], f'SMA_{fast_p}'] = 1.0910 # Fast crosses above slow
        df.loc[df.index[-1], f'SMA_{slow_p}'] = 1.0908
        df.loc[df.index[-1], f'ATR_{atr_p}'] = 0.00200 # Example ATR

        # No need to mock market_data_manager.get_market_data or platform_adapter.get_open_positions for this call
        # No need to mock market_data_manager.get_latest_tick_data for this call
        
        mock_tick = TickData(timestamp=datetime.now(timezone.utc), symbol="EURUSD", bid=1.09090, ask=1.09110)

        signal = trend_following_sma_strategy.generate_signal(
            market_data_df=df,
            active_position=None, # No active positions
            latest_tick=mock_tick
        )

        assert signal is not None
        assert signal['signal'] == StrategySignal.BUY
        assert signal['price'] == mock_tick.ask # Entry at ask for buy
        expected_sl_distance = df[f'ATR_{atr_p}'].iloc[-1] * atr_mult
        expected_sl = round(mock_tick.ask - expected_sl_distance, 5)
        expected_tp = round(mock_tick.ask + (expected_sl_distance * rr_ratio), 5)
        
        # Calculate expected sl_pips
        point_value = trend_following_sma_strategy.platform_adapter.get_symbol_info(trend_following_sma_strategy.symbol).point
        expected_sl_pips = expected_sl_distance / point_value

        assert signal['stop_loss_price'] == expected_sl
        assert signal['take_profit_price'] == expected_tp
        assert signal['sl_pips'] == pytest.approx(expected_sl_pips)
        assert "SMA Crossover: Buy EURUSD" in signal['comment']
        trend_following_sma_strategy.logger.info.assert_called()

    def test_generate_signal_bearish_crossover_new_sell_no_active_position(self, trend_following_sma_strategy, sma_strategy_params_eurusd_h1):
        fast_p = sma_strategy_params_eurusd_h1['fast_sma_period']
        slow_p = sma_strategy_params_eurusd_h1['slow_sma_period']
        atr_p = sma_strategy_params_eurusd_h1['atr_period_for_sl']
        atr_mult = sma_strategy_params_eurusd_h1['atr_multiplier_for_sl']
        rr_ratio = sma_strategy_params_eurusd_h1['min_reward_risk_ratio']

        df = create_sample_market_df(fast_sma_period=fast_p, slow_sma_period=slow_p, atr_period=atr_p)
        # Simulate bearish crossover
        df.loc[df.index[-2], f'SMA_{fast_p}'] = 1.0905  # Fast above slow
        df.loc[df.index[-2], f'SMA_{slow_p}'] = 1.0900
        df.loc[df.index[-1], f'SMA_{fast_p}'] = 1.0895  # Fast crosses below slow
        df.loc[df.index[-1], f'SMA_{slow_p}'] = 1.0898
        df.loc[df.index[-1], f'ATR_{atr_p}'] = 0.00250 # Example ATR

        mock_tick = TickData(timestamp=datetime.now(timezone.utc), symbol="EURUSD", bid=1.08940, ask=1.08960)

        signal = trend_following_sma_strategy.generate_signal(
            market_data_df=df,
            active_position=None, # No active positions
            latest_tick=mock_tick
        )

        assert signal is not None
        assert signal['signal'] == StrategySignal.SELL
        assert signal['price'] == mock_tick.bid # Entry at bid for sell
        expected_sl_distance = df[f'ATR_{atr_p}'].iloc[-1] * atr_mult
        expected_sl = round(mock_tick.bid + expected_sl_distance, 5)
        expected_tp = round(mock_tick.bid - (expected_sl_distance * rr_ratio), 5)
        
        point_value = trend_following_sma_strategy.platform_adapter.get_symbol_info(trend_following_sma_strategy.symbol).point
        expected_sl_pips = expected_sl_distance / point_value
        
        assert signal['stop_loss_price'] == expected_sl
        assert signal['take_profit_price'] == expected_tp
        assert signal['sl_pips'] == pytest.approx(expected_sl_pips)
        assert "SMA Crossover: Sell EURUSD" in signal['comment']
        trend_following_sma_strategy.logger.info.assert_called()

    def test_generate_signal_no_crossover_fast_above_slow(self, trend_following_sma_strategy, sma_strategy_params_eurusd_h1):
        fast_p = sma_strategy_params_eurusd_h1['fast_sma_period']
        slow_p = sma_strategy_params_eurusd_h1['slow_sma_period']
        atr_p = sma_strategy_params_eurusd_h1['atr_period_for_sl']
        df = create_sample_market_df(fast_sma_period=fast_p, slow_sma_period=slow_p, atr_period=atr_p, rows=slow_p + 5)
        # Fast consistently above slow
        df.loc[df.index[-2], f'SMA_{fast_p}'] = 1.0910
        df.loc[df.index[-2], f'SMA_{slow_p}'] = 1.0900
        df.loc[df.index[-1], f'SMA_{fast_p}'] = 1.0915
        df.loc[df.index[-1], f'SMA_{slow_p}'] = 1.0905
        mock_tick = TickData(timestamp=datetime.now(timezone.utc), symbol="EURUSD", bid=1.09140, ask=1.09160)
        signal = trend_following_sma_strategy.generate_signal(df, None, mock_tick)
        assert signal is None

    def test_generate_signal_no_crossover_fast_below_slow(self, trend_following_sma_strategy, sma_strategy_params_eurusd_h1):
        fast_p = sma_strategy_params_eurusd_h1['fast_sma_period']
        slow_p = sma_strategy_params_eurusd_h1['slow_sma_period']
        atr_p = sma_strategy_params_eurusd_h1['atr_period_for_sl']
        df = create_sample_market_df(fast_sma_period=fast_p, slow_sma_period=slow_p, atr_period=atr_p, rows=slow_p + 5)
        # Fast consistently below slow
        df.loc[df.index[-2], f'SMA_{fast_p}'] = 1.0900
        df.loc[df.index[-2], f'SMA_{slow_p}'] = 1.0910
        df.loc[df.index[-1], f'SMA_{fast_p}'] = 1.0895
        df.loc[df.index[-1], f'SMA_{slow_p}'] = 1.0905
        mock_tick = TickData(timestamp=datetime.now(timezone.utc), symbol="EURUSD", bid=1.08940, ask=1.08960)
        signal = trend_following_sma_strategy.generate_signal(df, None, mock_tick)
        assert signal is None

    def test_generate_signal_bullish_crossover_close_short(self, trend_following_sma_strategy, sma_strategy_params_eurusd_h1):
        fast_p = sma_strategy_params_eurusd_h1['fast_sma_period']
        slow_p = sma_strategy_params_eurusd_h1['slow_sma_period']
        atr_p = sma_strategy_params_eurusd_h1['atr_period_for_sl']
        df = create_sample_market_df(fast_sma_period=fast_p, slow_sma_period=slow_p, atr_period=atr_p, rows=slow_p + 5)
        # Simulate bullish crossover
        df.loc[df.index[-2], f'SMA_{fast_p}'] = 1.0900
        df.loc[df.index[-2], f'SMA_{slow_p}'] = 1.0905 # Fast below slow
        df.loc[df.index[-1], f'SMA_{fast_p}'] = 1.0910 # Fast crosses above slow
        df.loc[df.index[-1], f'SMA_{slow_p}'] = 1.0908
        
        mock_short_pos = Position(position_id="S1", symbol="EURUSD", action=OrderAction.SELL, volume=0.1, open_price=1.0950, open_time=datetime.now(timezone.utc))
        mock_tick = TickData(timestamp=datetime.now(timezone.utc), symbol="EURUSD", bid=1.09090, ask=1.09110)

        signal = trend_following_sma_strategy.generate_signal(
            market_data_df=df,
            active_position=mock_short_pos,
            latest_tick=mock_tick
        )
        assert signal is not None
        assert signal['signal'] == StrategySignal.CLOSE_SHORT
        assert signal['position_id'] == "S1"
        assert signal['price'] == mock_tick.ask # Close short at ask
        assert "SMA Crossover: Close Short EURUSD" in signal['comment']

    def test_generate_signal_bullish_crossover_already_long_no_signal(self, trend_following_sma_strategy, sma_strategy_params_eurusd_h1):
        fast_p = sma_strategy_params_eurusd_h1['fast_sma_period']
        slow_p = sma_strategy_params_eurusd_h1['slow_sma_period']
        atr_p = sma_strategy_params_eurusd_h1['atr_period_for_sl']
        df = create_sample_market_df(fast_sma_period=fast_p, slow_sma_period=slow_p, atr_period=atr_p, rows=slow_p + 5)
        # Simulate bullish crossover
        df.loc[df.index[-2], f'SMA_{fast_p}'] = 1.0900
        df.loc[df.index[-2], f'SMA_{slow_p}'] = 1.0905
        df.loc[df.index[-1], f'SMA_{fast_p}'] = 1.0910
        df.loc[df.index[-1], f'SMA_{slow_p}'] = 1.0908
        
        mock_long_pos = Position(position_id="L1", symbol="EURUSD", action=OrderAction.BUY, volume=0.1, open_price=1.0850, open_time=datetime.now(timezone.utc))
        mock_tick = TickData(timestamp=datetime.now(timezone.utc), symbol="EURUSD", bid=1.09090, ask=1.09110)

        signal = trend_following_sma_strategy.generate_signal(
            market_data_df=df,
            active_position=mock_long_pos,
            latest_tick=mock_tick
        )
        assert signal is None # Already long, no new buy, no close signal from this crossover type

    def test_generate_signal_bearish_crossover_close_long(self, trend_following_sma_strategy, sma_strategy_params_eurusd_h1):
        fast_p = sma_strategy_params_eurusd_h1['fast_sma_period']
        slow_p = sma_strategy_params_eurusd_h1['slow_sma_period']
        atr_p = sma_strategy_params_eurusd_h1['atr_period_for_sl']

        df = create_sample_market_df(fast_sma_period=fast_p, slow_sma_period=slow_p, atr_period=atr_p)
        # Simulate bearish crossover
        df.loc[df.index[-2], f'SMA_{fast_p}'] = 1.0905
        df.loc[df.index[-2], f'SMA_{slow_p}'] = 1.0900 # Fast above slow
        df.loc[df.index[-1], f'SMA_{fast_p}'] = 1.0895 # Fast crosses below slow
        df.loc[df.index[-1], f'SMA_{slow_p}'] = 1.0898
        
        # Simulate active long position
        mock_long_pos = Position(position_id="L1", symbol="EURUSD", action=OrderAction.BUY, volume=0.1, open_price=1.0800, open_time=datetime.now(timezone.utc))
        
        mock_tick = TickData(timestamp=datetime.now(timezone.utc), symbol="EURUSD", bid=1.08940, ask=1.08960)

        signal = trend_following_sma_strategy.generate_signal(
            market_data_df=df,
            active_position=mock_long_pos,
            latest_tick=mock_tick
        )

        assert signal is not None
        assert signal['signal'] == StrategySignal.CLOSE_LONG
        assert signal['position_id'] == "L1"
        assert signal['price'] == mock_tick.bid # Close long at bid
        assert "SMA Crossover: Close Long EURUSD" in signal['comment']

    def test_generate_signal_bearish_crossover_already_short_no_signal(self, trend_following_sma_strategy, sma_strategy_params_eurusd_h1):
        fast_p = sma_strategy_params_eurusd_h1['fast_sma_period']
        slow_p = sma_strategy_params_eurusd_h1['slow_sma_period']
        atr_p = sma_strategy_params_eurusd_h1['atr_period_for_sl']
        df = create_sample_market_df(fast_sma_period=fast_p, slow_sma_period=slow_p, atr_period=atr_p, rows=slow_p + 5)
        # Simulate bearish crossover
        df.loc[df.index[-2], f'SMA_{fast_p}'] = 1.0905
        df.loc[df.index[-2], f'SMA_{slow_p}'] = 1.0900
        df.loc[df.index[-1], f'SMA_{fast_p}'] = 1.0895
        df.loc[df.index[-1], f'SMA_{slow_p}'] = 1.0898
        
        mock_short_pos = Position(position_id="S1", symbol="EURUSD", action=OrderAction.SELL, volume=0.1, open_price=1.0950, open_time=datetime.now(timezone.utc))
        mock_tick = TickData(timestamp=datetime.now(timezone.utc), symbol="EURUSD", bid=1.08940, ask=1.08960)

        signal = trend_following_sma_strategy.generate_signal(
            market_data_df=df,
            active_position=mock_short_pos,
            latest_tick=mock_tick
        )
        assert signal is None # Already short, no new sell, no close signal from this crossover type

    def test_generate_signal_ranging_market_no_clear_crossover(self, trend_following_sma_strategy, sma_strategy_params_eurusd_h1):
        fast_p = sma_strategy_params_eurusd_h1['fast_sma_period']
        slow_p = sma_strategy_params_eurusd_h1['slow_sma_period']
        atr_p = sma_strategy_params_eurusd_h1['atr_period_for_sl']
        df = create_sample_market_df(fast_sma_period=fast_p, slow_sma_period=slow_p, atr_period=atr_p, rows=slow_p + 5)
        
        # Scenario 1: SMAs are very close, fast slightly above slow on both bars (no crossover)
        df.loc[df.index[-2], f'SMA_{fast_p}'] = 1.0901
        df.loc[df.index[-2], f'SMA_{slow_p}'] = 1.0900
        df.loc[df.index[-1], f'SMA_{fast_p}'] = 1.0902
        df.loc[df.index[-1], f'SMA_{slow_p}'] = 1.0901
        mock_tick = TickData(timestamp=datetime.now(timezone.utc), symbol="EURUSD", bid=1.09015, ask=1.09025)
        signal = trend_following_sma_strategy.generate_signal(df, None, mock_tick)
        assert signal is None

        # Scenario 2: SMAs cross then cross back (whipsaw, but last state is no crossover from prev)
        # prev: fast < slow, last: fast > slow (bullish) -> then if next bar fast < slow again
        # For this test, we only look at prev and last.
        # Let's make prev: fast > slow, last: fast < slow (bearish crossover)
        # Then immediately after, prev: fast < slow, last: fast > slow (bullish crossover)
        # The test should focus on a single call to generate_signal.
        # Let's set prev: fast > slow, last: fast slightly > slow (no cross from below)
        df.loc[df.index[-2], f'SMA_{fast_p}'] = 1.0905
        df.loc[df.index[-2], f'SMA_{slow_p}'] = 1.0900
        df.loc[df.index[-1], f'SMA_{fast_p}'] = 1.0906 # Still above, no cross from below
        df.loc[df.index[-1], f'SMA_{slow_p}'] = 1.0900
        signal = trend_following_sma_strategy.generate_signal(df, None, mock_tick)
        assert signal is None

        # Scenario 3: SMAs are equal on previous bar, then fast moves slightly above (bullish crossover)
        # This IS a crossover by the logic `prev_row[fast_sma_col] <= prev_row[slow_sma_col]`
        # So this is not a "no signal" case.

        # Scenario 4: Fast SMA touches Slow SMA from above and bounces (no bearish crossover)
        # prev: fast > slow
        # last: fast == slow (or fast slightly > slow again)
        df.loc[df.index[-2], f'SMA_{fast_p}'] = 1.0905 # Fast above slow
        df.loc[df.index[-2], f'SMA_{slow_p}'] = 1.0900
        df.loc[df.index[-1], f'SMA_{fast_p}'] = 1.0900 # Fast touches slow
        df.loc[df.index[-1], f'SMA_{slow_p}'] = 1.0900
        signal = trend_following_sma_strategy.generate_signal(df, None, mock_tick)
        assert signal is None # No bearish crossover because last_row[fast] is not < last_row[slow]

    def test_generate_signal_bullish_crossover_zero_atr(self, trend_following_sma_strategy, sma_strategy_params_eurusd_h1):
        fast_p = sma_strategy_params_eurusd_h1['fast_sma_period']
        slow_p = sma_strategy_params_eurusd_h1['slow_sma_period']
        atr_p = sma_strategy_params_eurusd_h1['atr_period_for_sl']
        # atr_mult = sma_strategy_params_eurusd_h1['atr_multiplier_for_sl'] # Not directly needed for SL=entry check
        # rr_ratio = sma_strategy_params_eurusd_h1['min_reward_risk_ratio']

        df = create_sample_market_df(fast_sma_period=fast_p, slow_sma_period=slow_p, atr_period=atr_p)
        # Simulate bullish crossover
        df.loc[df.index[-2], f'SMA_{fast_p}'] = 1.0900
        df.loc[df.index[-2], f'SMA_{slow_p}'] = 1.0905
        df.loc[df.index[-1], f'SMA_{fast_p}'] = 1.0910
        df.loc[df.index[-1], f'SMA_{slow_p}'] = 1.0908
        df.loc[df.index[-1], f'ATR_{atr_p}'] = 0.00000 # Zero ATR

        mock_tick = TickData(timestamp=datetime.now(timezone.utc), symbol="EURUSD", bid=1.09090, ask=1.09110)
        
        signal = trend_following_sma_strategy.generate_signal(df, None, mock_tick)

        assert signal is not None
        assert signal['signal'] == StrategySignal.BUY
        assert signal['price'] == mock_tick.ask
        assert signal['stop_loss_price'] == mock_tick.ask # SL = entry price due to zero ATR distance
        assert signal['take_profit_price'] == mock_tick.ask # TP = entry price due to zero risk
        
        point_value = trend_following_sma_strategy.platform_adapter.get_symbol_info(trend_following_sma_strategy.symbol).point
        assert point_value > 0 # Precondition for sl_pips calculation
        expected_sl_pips = (df[f'ATR_{atr_p}'].iloc[-1] * sma_strategy_params_eurusd_h1['atr_multiplier_for_sl']) / point_value
        assert signal['sl_pips'] == pytest.approx(expected_sl_pips)
        assert signal['sl_pips'] == 0.0

    def test_generate_signal_bullish_crossover_negative_atr(self, trend_following_sma_strategy, sma_strategy_params_eurusd_h1):
        fast_p = sma_strategy_params_eurusd_h1['fast_sma_period']
        slow_p = sma_strategy_params_eurusd_h1['slow_sma_period']
        atr_p = sma_strategy_params_eurusd_h1['atr_period_for_sl']
        atr_mult = sma_strategy_params_eurusd_h1['atr_multiplier_for_sl']
        rr_ratio = sma_strategy_params_eurusd_h1['min_reward_risk_ratio']

        df = create_sample_market_df(fast_sma_period=fast_p, slow_sma_period=slow_p, atr_period=atr_p)
        # Simulate bullish crossover
        df.loc[df.index[-2], f'SMA_{fast_p}'] = 1.0900
        df.loc[df.index[-2], f'SMA_{slow_p}'] = 1.0905
        df.loc[df.index[-1], f'SMA_{fast_p}'] = 1.0910
        df.loc[df.index[-1], f'SMA_{slow_p}'] = 1.0908
        negative_atr_val = -0.00100
        df.loc[df.index[-1], f'ATR_{atr_p}'] = negative_atr_val

        mock_tick = TickData(timestamp=datetime.now(timezone.utc), symbol="EURUSD", bid=1.09090, ask=1.09110)
        
        signal = trend_following_sma_strategy.generate_signal(df, None, mock_tick)

        assert signal is not None
        assert signal['signal'] == StrategySignal.BUY
        assert signal['price'] == mock_tick.ask
        
        expected_sl_distance = negative_atr_val * atr_mult # This will be negative
        expected_sl = round(mock_tick.ask - expected_sl_distance, 5) # Subtracting a negative makes it addition
        expected_tp = round(mock_tick.ask + (expected_sl_distance * rr_ratio), 5) # Adding a negative makes it subtraction
        
        assert signal['stop_loss_price'] == expected_sl
        assert signal['stop_loss_price'] > mock_tick.ask # SL is on the "wrong" side (above entry for BUY)
        assert signal['take_profit_price'] == expected_tp
        assert signal['take_profit_price'] < mock_tick.ask # TP is on the "wrong" side (below entry for BUY if RR is positive)

        point_value = trend_following_sma_strategy.platform_adapter.get_symbol_info(trend_following_sma_strategy.symbol).point
        expected_sl_pips = expected_sl_distance / point_value
        assert signal['sl_pips'] == pytest.approx(expected_sl_pips)
        assert signal['sl_pips'] < 0.0

    def test_generate_signal_bullish_crossover_nan_ask_price_in_tick(self, trend_following_sma_strategy, sma_strategy_params_eurusd_h1):
        fast_p = sma_strategy_params_eurusd_h1['fast_sma_period']
        slow_p = sma_strategy_params_eurusd_h1['slow_sma_period']
        atr_p = sma_strategy_params_eurusd_h1['atr_period_for_sl']
        atr_mult = sma_strategy_params_eurusd_h1['atr_multiplier_for_sl']
        # rr_ratio = sma_strategy_params_eurusd_h1['min_reward_risk_ratio'] # Not directly used for NaN check

        df = create_sample_market_df(fast_sma_period=fast_p, slow_sma_period=slow_p, atr_period=atr_p)
        # Simulate bullish crossover
        df.loc[df.index[-2], f'SMA_{fast_p}'] = 1.0900
        df.loc[df.index[-2], f'SMA_{slow_p}'] = 1.0905
        df.loc[df.index[-1], f'SMA_{fast_p}'] = 1.0910
        df.loc[df.index[-1], f'SMA_{slow_p}'] = 1.0908
        df.loc[df.index[-1], f'ATR_{atr_p}'] = 0.00200 # Valid ATR

        mock_tick_nan_ask = TickData(timestamp=datetime.now(timezone.utc), symbol="EURUSD", bid=1.09090, ask=float('nan'))

        signal = trend_following_sma_strategy.generate_signal(
            market_data_df=df,
            active_position=None,
            latest_tick=mock_tick_nan_ask
        )

        assert signal is not None
        assert signal['signal'] == StrategySignal.BUY
        assert pd.isna(signal['price'])
        assert pd.isna(signal['stop_loss_price'])
        assert pd.isna(signal['take_profit_price'])
        
        # sl_pips should still be calculable as it depends on ATR_val and point, not entry price directly
        point_value = trend_following_sma_strategy.platform_adapter.get_symbol_info(trend_following_sma_strategy.symbol).point
        expected_sl_distance = df[f'ATR_{atr_p}'].iloc[-1] * atr_mult
        expected_sl_pips = expected_sl_distance / point_value
        assert signal['sl_pips'] == pytest.approx(expected_sl_pips)
        # The strategy might log an info about signal generation, but ideally a warning for NaN prices.
        # For now, we test that a signal structure is returned.
        trend_following_sma_strategy.logger.info.assert_any_call(
            f"[{trend_following_sma_strategy.symbol}/{trend_following_sma_strategy.timeframe.name}] BUY signal generated for {trend_following_sma_strategy.symbol} at NaN (Ask). SL: NaN, TP: NaN, SL_pips: {expected_sl_pips:.2f}"
        )


    def test_generate_signal_bearish_crossover_nan_bid_price_in_tick(self, trend_following_sma_strategy, sma_strategy_params_eurusd_h1):
        fast_p = sma_strategy_params_eurusd_h1['fast_sma_period']
        slow_p = sma_strategy_params_eurusd_h1['slow_sma_period']
        atr_p = sma_strategy_params_eurusd_h1['atr_period_for_sl']
        atr_mult = sma_strategy_params_eurusd_h1['atr_multiplier_for_sl']

        df = create_sample_market_df(fast_sma_period=fast_p, slow_sma_period=slow_p, atr_period=atr_p)
        # Simulate bearish crossover
        df.loc[df.index[-2], f'SMA_{fast_p}'] = 1.0905
        df.loc[df.index[-2], f'SMA_{slow_p}'] = 1.0900
        df.loc[df.index[-1], f'SMA_{fast_p}'] = 1.0895
        df.loc[df.index[-1], f'SMA_{slow_p}'] = 1.0898
        df.loc[df.index[-1], f'ATR_{atr_p}'] = 0.00250

        mock_tick_nan_bid = TickData(timestamp=datetime.now(timezone.utc), symbol="EURUSD", bid=float('nan'), ask=1.08960)

        signal = trend_following_sma_strategy.generate_signal(
            market_data_df=df,
            active_position=None,
            latest_tick=mock_tick_nan_bid
        )

        assert signal is not None
        assert signal['signal'] == StrategySignal.SELL
        assert pd.isna(signal['price'])
        assert pd.isna(signal['stop_loss_price'])
        assert pd.isna(signal['take_profit_price'])

        point_value = trend_following_sma_strategy.platform_adapter.get_symbol_info(trend_following_sma_strategy.symbol).point
        expected_sl_distance = df[f'ATR_{atr_p}'].iloc[-1] * atr_mult
        expected_sl_pips = expected_sl_distance / point_value
        assert signal['sl_pips'] == pytest.approx(expected_sl_pips)
        trend_following_sma_strategy.logger.info.assert_any_call(
             f"[{trend_following_sma_strategy.symbol}/{trend_following_sma_strategy.timeframe.name}] SELL signal generated for {trend_following_sma_strategy.symbol} at NaN (Bid). SL: NaN, TP: NaN, SL_pips: {expected_sl_pips:.2f}"
        )

class TestTrendFollowingSMAManagePosition:

    def test_manage_position_trailing_stop_buy(self, trend_following_sma_strategy, sma_strategy_params_eurusd_h1):
        if not sma_strategy_params_eurusd_h1['use_trailing_stop']:
            pytest.skip("Trailing stop not enabled for this test configuration")

        atr_p_ts = sma_strategy_params_eurusd_h1['trailing_stop_atr_period']
        atr_mult_ts = sma_strategy_params_eurusd_h1['trailing_stop_atr_multiplier']
        
        df = create_sample_market_df(atr_period=atr_p_ts) # Ensure correct ATR column for TS
        # Use a fixed ATR for trailing stop for simplicity in this test
        df[f'ATR_{atr_p_ts}'] = 0.00150 # Fixed ATR for trailing stop
        trend_following_sma_strategy.market_data_manager.get_market_data.return_value = df
        
        current_long_pos = Position(
            position_id="LTS1", symbol="EURUSD", action=OrderAction.BUY, 
            volume=0.1, open_price=1.08000, open_time=datetime.now(timezone.utc) - timedelta(hours=5),
            stop_loss=1.07500 # Initial SL
        )
        mock_tick_higher = TickData(timestamp=datetime.now(timezone.utc), symbol="EURUSD", bid=1.09000, ask=1.09020) # Price moved up

        action = trend_following_sma_strategy.manage_open_position(current_long_pos, latest_tick=mock_tick_higher)
        
        assert action is not None
        assert action['signal'] == StrategySignal.MODIFY_SLTP
        assert action['position_id'] == "LTS1"
        expected_new_sl = round(mock_tick_higher.bid - (df[f'ATR_{atr_p_ts}'].iloc[-1] * atr_mult_ts), 5)
        assert action['new_stop_loss'] == expected_new_sl
        assert action['new_stop_loss'] > current_long_pos.stop_loss # Should have trailed up

    def test_manage_position_max_age_exit(self, trend_following_sma_strategy, sma_strategy_params_eurusd_h1):
        max_age = sma_strategy_params_eurusd_h1['max_position_age_bars']
        if not max_age:
            pytest.skip("Max position age not configured")

        df = create_sample_market_df(rows=max_age + 10) # Ensure enough bars
        trend_following_sma_strategy.market_data_manager.get_market_data.return_value = df
        
        # Position opened 'max_age' bars ago relative to the latest bar in df
        # df.index is Timestamp-based, use its properties
        # position_open_time_utc = df.index[5] # Example open time (6th bar) # This line is not used, can be removed
        # Ensure latest_bar's timestamp implies position is older than max_age
        # latest_bar's timestamp should be at least df.index[5 + max_age]
        
        # Simulate a position opened long enough ago
        # The manage_open_position in strategy compares latest_bar.timestamp with position.open_time
        # using df.index.get_indexer and df.index.get_loc
        
        # Let's make position open time correspond to the 5th bar's timestamp in our mock df
        pos_open_dt = df.index[5] # df.index elements are already datetime objects

        current_long_pos = Position(
            position_id="L_AGE_1", symbol="EURUSD", action=OrderAction.BUY,
            volume=0.1, open_price=1.0800, open_time=pos_open_dt
        )
        
        # Simulate the latest bar being the one that makes the position "too old"
        # The latest_bar passed to manage_open_position should be the (5 + max_age)-th bar or later
        latest_bar_data_for_manage = OHLCVData(
            timestamp=df.index[5 + max_age], # Timestamp of the (5+max_age)-th bar
            symbol="EURUSD", timeframe=trend_following_sma_strategy.timeframe,
            open=df['open'].iloc[5 + max_age], high=df['high'].iloc[5 + max_age],
            low=df['low'].iloc[5 + max_age], close=df['close'].iloc[5 + max_age],
            volume=df['volume'].iloc[5 + max_age]
        )

        mock_tick_at_exit = TickData(timestamp=datetime.now(timezone.utc), symbol="EURUSD", bid=1.09500, ask=1.09520)

        action = trend_following_sma_strategy.manage_open_position(
            current_long_pos, 
            latest_bar=latest_bar_data_for_manage, 
            latest_tick=mock_tick_at_exit
        )

        assert action is not None
        assert action['signal'] == StrategySignal.CLOSE_LONG
        assert action['position_id'] == "L_AGE_1"
        assert f"Max position age ({max_age} bars) reached" in action['comment']

    def test_manage_position_trailing_stop_sell(self, trend_following_sma_strategy, sma_strategy_params_eurusd_h1):
        if not sma_strategy_params_eurusd_h1['use_trailing_stop']:
            pytest.skip("Trailing stop not enabled")
        
        atr_p_ts = sma_strategy_params_eurusd_h1['trailing_stop_atr_period']
        atr_mult_ts = sma_strategy_params_eurusd_h1['trailing_stop_atr_multiplier']
        df = create_sample_market_df(atr_period=atr_p_ts)
        df[f'ATR_{atr_p_ts}'] = 0.00150
        trend_following_sma_strategy.market_data_manager.get_market_data.return_value = df
        
        current_short_pos = Position(
            position_id="STS1", symbol="EURUSD", action=OrderAction.SELL,
            volume=0.1, open_price=1.09000, open_time=datetime.now(timezone.utc) - timedelta(hours=5),
            stop_loss=1.09500 # Initial SL
        )
        mock_tick_lower = TickData(timestamp=datetime.now(timezone.utc), symbol="EURUSD", bid=1.08000, ask=1.08020) # Price moved down

        action = trend_following_sma_strategy.manage_open_position(current_short_pos, latest_tick=mock_tick_lower)
        
        assert action is not None
        assert action['signal'] == StrategySignal.MODIFY_SLTP
        expected_new_sl = round(mock_tick_lower.ask + (df[f'ATR_{atr_p_ts}'].iloc[-1] * atr_mult_ts), 5)
        assert action['new_stop_loss'] == expected_new_sl
        assert action['new_stop_loss'] < current_short_pos.stop_loss # Should have trailed down

    def test_manage_position_trailing_stop_no_move_long(self, trend_following_sma_strategy, sma_strategy_params_eurusd_h1):
        if not sma_strategy_params_eurusd_h1['use_trailing_stop']:
            pytest.skip("Trailing stop not enabled")

        atr_p_ts = sma_strategy_params_eurusd_h1['trailing_stop_atr_period']
        atr_mult_ts = sma_strategy_params_eurusd_h1['trailing_stop_atr_multiplier']
        df = create_sample_market_df(atr_period=atr_p_ts)
        df[f'ATR_{atr_p_ts}'] = 0.00150
        trend_following_sma_strategy.market_data_manager.get_market_data.return_value = df
        
        initial_sl = 1.07800
        current_long_pos = Position(
            position_id="LTS_NM1", symbol="EURUSD", action=OrderAction.BUY,
            volume=0.1, open_price=1.08000, open_time=datetime.now(timezone.utc) - timedelta(hours=5),
            stop_loss=initial_sl
        )
        # Price moved slightly, but not enough to trail SL, or moved against
        mock_tick_stagnant = TickData(timestamp=datetime.now(timezone.utc), symbol="EURUSD", bid=1.08000 - (df[f'ATR_{atr_p_ts}'].iloc[-1] * atr_mult_ts) + 0.0001 , ask=1.08020)
        
        action = trend_following_sma_strategy.manage_open_position(current_long_pos, latest_tick=mock_tick_stagnant)
        assert action is None # No SL modification expected

    def test_manage_position_trailing_stop_no_move_short(self, trend_following_sma_strategy, sma_strategy_params_eurusd_h1):
        if not sma_strategy_params_eurusd_h1['use_trailing_stop']:
            pytest.skip("Trailing stop not enabled")

        atr_p_ts = sma_strategy_params_eurusd_h1['trailing_stop_atr_period']
        atr_mult_ts = sma_strategy_params_eurusd_h1['trailing_stop_atr_multiplier']
        df = create_sample_market_df(atr_period=atr_p_ts)
        df[f'ATR_{atr_p_ts}'] = 0.00150
        trend_following_sma_strategy.market_data_manager.get_market_data.return_value = df

        initial_sl = 1.08200
        current_short_pos = Position(
            position_id="STS_NM1", symbol="EURUSD", action=OrderAction.SELL,
            volume=0.1, open_price=1.08000, open_time=datetime.now(timezone.utc) - timedelta(hours=5),
            stop_loss=initial_sl
        )
        # Price moved slightly, but not enough to trail SL, or moved against
        mock_tick_stagnant = TickData(timestamp=datetime.now(timezone.utc), symbol="EURUSD", bid=1.07980, ask=1.08000 + (df[f'ATR_{atr_p_ts}'].iloc[-1] * atr_mult_ts) - 0.0001)

        action = trend_following_sma_strategy.manage_open_position(current_short_pos, latest_tick=mock_tick_stagnant)
        assert action is None # No SL modification expected

    def test_manage_position_trailing_stop_long_price_moves_against(self, trend_following_sma_strategy, sma_strategy_params_eurusd_h1):
        if not sma_strategy_params_eurusd_h1['use_trailing_stop']:
            pytest.skip("Trailing stop not enabled")

        atr_p_ts = sma_strategy_params_eurusd_h1['trailing_stop_atr_period']
        # atr_mult_ts = sma_strategy_params_eurusd_h1['trailing_stop_atr_multiplier'] # Not directly used in assertion logic here
        df = create_sample_market_df(atr_period=atr_p_ts)
        df[f'ATR_{atr_p_ts}'] = 0.00150
        trend_following_sma_strategy.market_data_manager.get_market_data.return_value = df
        
        initial_sl = 1.07500
        current_long_pos = Position(
            position_id="LTS_PMA", symbol="EURUSD", action=OrderAction.BUY,
            volume=0.1, open_price=1.08000, open_time=datetime.now(timezone.utc) - timedelta(hours=5),
            stop_loss=initial_sl
        )
        # Price moves down, but not enough to hit SL. Potential new SL would be even lower.
        mock_tick_lower = TickData(timestamp=datetime.now(timezone.utc), symbol="EURUSD", bid=1.07900, ask=1.07920)
        
        action = trend_following_sma_strategy.manage_open_position(current_long_pos, latest_tick=mock_tick_lower)
        assert action is None # SL should not move if price moves against the position or new SL is worse

    def test_manage_position_trailing_stop_short_price_moves_against(self, trend_following_sma_strategy, sma_strategy_params_eurusd_h1):
        if not sma_strategy_params_eurusd_h1['use_trailing_stop']:
            pytest.skip("Trailing stop not enabled")

        atr_p_ts = sma_strategy_params_eurusd_h1['trailing_stop_atr_period']
        # atr_mult_ts = sma_strategy_params_eurusd_h1['trailing_stop_atr_multiplier']
        df = create_sample_market_df(atr_period=atr_p_ts)
        df[f'ATR_{atr_p_ts}'] = 0.00150
        trend_following_sma_strategy.market_data_manager.get_market_data.return_value = df

        initial_sl = 1.08500
        current_short_pos = Position(
            position_id="STS_PMA", symbol="EURUSD", action=OrderAction.SELL,
            volume=0.1, open_price=1.08000, open_time=datetime.now(timezone.utc) - timedelta(hours=5),
            stop_loss=initial_sl
        )
        # Price moves up, but not enough to hit SL. Potential new SL would be even higher.
        mock_tick_higher = TickData(timestamp=datetime.now(timezone.utc), symbol="EURUSD", bid=1.08080, ask=1.08100)
        
        action = trend_following_sma_strategy.manage_open_position(current_short_pos, latest_tick=mock_tick_higher)
        assert action is None # SL should not move

    def test_manage_position_trailing_stop_long_initial_sl_none(self, trend_following_sma_strategy, sma_strategy_params_eurusd_h1):
        if not sma_strategy_params_eurusd_h1['use_trailing_stop']:
            pytest.skip("Trailing stop not enabled")

        atr_p_ts = sma_strategy_params_eurusd_h1['trailing_stop_atr_period']
        atr_mult_ts = sma_strategy_params_eurusd_h1['trailing_stop_atr_multiplier']
        df = create_sample_market_df(atr_period=atr_p_ts)
        df[f'ATR_{atr_p_ts}'] = 0.00150
        trend_following_sma_strategy.market_data_manager.get_market_data.return_value = df
        
        current_long_pos = Position(
            position_id="LTS_ISL0", symbol="EURUSD", action=OrderAction.BUY,
            volume=0.1, open_price=1.08000, open_time=datetime.now(timezone.utc) - timedelta(hours=5),
            stop_loss=None # Initial SL is None
        )
        mock_tick_higher = TickData(timestamp=datetime.now(timezone.utc), symbol="EURUSD", bid=1.09000, ask=1.09020) # Price moved up

        action = trend_following_sma_strategy.manage_open_position(current_long_pos, latest_tick=mock_tick_higher)
        
        assert action is not None
        assert action['signal'] == StrategySignal.MODIFY_SLTP
        expected_new_sl = round(mock_tick_higher.bid - (df[f'ATR_{atr_p_ts}'].iloc[-1] * atr_mult_ts), 5)
        assert action['new_stop_loss'] == expected_new_sl
        assert action['new_stop_loss'] > current_long_pos.open_price # Assuming SL trails into profit or at least above open

    def test_manage_position_trailing_stop_short_initial_sl_none(self, trend_following_sma_strategy, sma_strategy_params_eurusd_h1):
        if not sma_strategy_params_eurusd_h1['use_trailing_stop']:
            pytest.skip("Trailing stop not enabled")
        
        atr_p_ts = sma_strategy_params_eurusd_h1['trailing_stop_atr_period']
        atr_mult_ts = sma_strategy_params_eurusd_h1['trailing_stop_atr_multiplier']
        df = create_sample_market_df(atr_period=atr_p_ts)
        df[f'ATR_{atr_p_ts}'] = 0.00150
        trend_following_sma_strategy.market_data_manager.get_market_data.return_value = df
        
        current_short_pos = Position(
            position_id="STS_ISL0", symbol="EURUSD", action=OrderAction.SELL,
            volume=0.1, open_price=1.09000, open_time=datetime.now(timezone.utc) - timedelta(hours=5),
            stop_loss=None # Initial SL is None
        )
        mock_tick_lower = TickData(timestamp=datetime.now(timezone.utc), symbol="EURUSD", bid=1.08000, ask=1.08020) # Price moved down

        action = trend_following_sma_strategy.manage_open_position(current_short_pos, latest_tick=mock_tick_lower)
        
        assert action is not None
        assert action['signal'] == StrategySignal.MODIFY_SLTP
        expected_new_sl = round(mock_tick_lower.ask + (df[f'ATR_{atr_p_ts}'].iloc[-1] * atr_mult_ts), 5)
        assert action['new_stop_loss'] == expected_new_sl
        assert action['new_stop_loss'] < current_short_pos.open_price # Assuming SL trails into profit or at least below open

    def test_manage_position_trailing_stop_disabled(self, trend_following_sma_strategy, sma_strategy_params_eurusd_h1):
        # Temporarily disable trailing stop for this test instance
        trend_following_sma_strategy.use_trailing_stop = False
        
        current_long_pos = Position(
            position_id="LTS_DIS", symbol="EURUSD", action=OrderAction.BUY,
            volume=0.1, open_price=1.08000, open_time=datetime.now(timezone.utc) - timedelta(hours=5),
            stop_loss=1.07500
        )
        mock_tick_higher = TickData(timestamp=datetime.now(timezone.utc), symbol="EURUSD", bid=1.09000, ask=1.09020)

        action = trend_following_sma_strategy.manage_open_position(current_long_pos, latest_tick=mock_tick_higher)
        assert action is None # No action as TS is disabled
        # Restore for other tests if strategy instance is reused (pytest fixtures usually recreate)
        trend_following_sma_strategy.use_trailing_stop = sma_strategy_params_eurusd_h1['use_trailing_stop']


    def test_manage_position_max_age_exit_short(self, trend_following_sma_strategy, sma_strategy_params_eurusd_h1):
        max_age = sma_strategy_params_eurusd_h1['max_position_age_bars']
        if not max_age: pytest.skip("Max position age not configured")

        df = create_sample_market_df(rows=max_age + 10)
        trend_following_sma_strategy.market_data_manager.get_market_data.return_value = df
        pos_open_dt = df.index[5]

        current_short_pos = Position(
            position_id="S_AGE_1", symbol="EURUSD", action=OrderAction.SELL,
            volume=0.1, open_price=1.0900, open_time=pos_open_dt
        )
        latest_bar_data = OHLCVData(
            timestamp=df.index[5 + max_age], symbol="EURUSD", timeframe=trend_following_sma_strategy.timeframe,
            open=df['open'].iloc[5 + max_age], high=df['high'].iloc[5 + max_age],
            low=df['low'].iloc[5 + max_age], close=df['close'].iloc[5 + max_age]
        )
        mock_tick = TickData(timestamp=datetime.now(timezone.utc), symbol="EURUSD", bid=1.07500, ask=1.07520)

        action = trend_following_sma_strategy.manage_open_position(current_short_pos, latest_bar=latest_bar_data, latest_tick=mock_tick)
        assert action is not None
        assert action['signal'] == StrategySignal.CLOSE_SHORT
        assert f"Max position age ({max_age} bars) reached" in action['comment']

    def test_manage_position_max_age_not_reached(self, trend_following_sma_strategy, sma_strategy_params_eurusd_h1):
        max_age = sma_strategy_params_eurusd_h1['max_position_age_bars']
        if not max_age: pytest.skip("Max position age not configured")

        df = create_sample_market_df(rows=max_age + 10)
        trend_following_sma_strategy.market_data_manager.get_market_data.return_value = df
        pos_open_dt = df.index[5]

        current_long_pos = Position(
            position_id="L_AGE_NR", symbol="EURUSD", action=OrderAction.BUY,
            volume=0.1, open_price=1.0800, open_time=pos_open_dt
        )
        # Latest bar is before max_age is reached
        latest_bar_data = OHLCVData(
            timestamp=df.index[5 + max_age - 1], symbol="EURUSD", timeframe=trend_following_sma_strategy.timeframe,
            open=df['open'].iloc[5 + max_age -1], high=df['high'].iloc[5 + max_age -1],
            low=df['low'].iloc[5 + max_age -1], close=df['close'].iloc[5 + max_age-1]
        )
        mock_tick = TickData(timestamp=datetime.now(timezone.utc), symbol="EURUSD", bid=1.08500, ask=1.08520)
        action = trend_following_sma_strategy.manage_open_position(current_long_pos, latest_bar=latest_bar_data, latest_tick=mock_tick)
        assert action is None

    def test_manage_position_max_age_disabled(self, trend_following_sma_strategy, sma_strategy_params_eurusd_h1):
        trend_following_sma_strategy.max_position_age_bars = None # Disable for this test
        
        df = create_sample_market_df(rows=50) # Arbitrary size, won't be used for age calc
        trend_following_sma_strategy.market_data_manager.get_market_data.return_value = df
        pos_open_dt = df.index[5]

        current_long_pos = Position(
            position_id="L_AGE_DIS", symbol="EURUSD", action=OrderAction.BUY,
            volume=0.1, open_price=1.0800, open_time=pos_open_dt
        )
        latest_bar_data = OHLCVData(
            timestamp=df.index[40], # A late bar
            symbol="EURUSD", timeframe=trend_following_sma_strategy.timeframe,
            open=df['open'].iloc[40], high=df['high'].iloc[40],
            low=df['low'].iloc[40], close=df['close'].iloc[40]
        )
        mock_tick = TickData(timestamp=datetime.now(timezone.utc), symbol="EURUSD", bid=1.08500, ask=1.08520)
        action = trend_following_sma_strategy.manage_open_position(current_long_pos, latest_bar=latest_bar_data, latest_tick=mock_tick)
        assert action is None
        # Restore for other tests
        trend_following_sma_strategy.max_position_age_bars = sma_strategy_params_eurusd_h1.get("max_position_age_bars")

    def test_manage_open_position_no_latest_tick(self, trend_following_sma_strategy):
        current_pos = Position(position_id="P1", symbol="EURUSD", action=OrderAction.BUY, volume=0.1, open_price=1.1, open_time=datetime.now(timezone.utc))
        action = trend_following_sma_strategy.manage_open_position(current_pos, latest_tick=None)
        assert action is None
        trend_following_sma_strategy.logger.warning.assert_any_call(
            f"[{trend_following_sma_strategy.symbol}/{trend_following_sma_strategy.timeframe.name}] No latest_tick data for manage_open_position."
        )

    def test_manage_open_position_no_symbol_info_for_ts(self, trend_following_sma_strategy, mock_market_data_manager, sma_strategy_params_eurusd_h1):
        if not sma_strategy_params_eurusd_h1['use_trailing_stop']:
             pytest.skip("Trailing stop not enabled")
        # Setup platform_adapter on market_data_manager for this specific test path
        mock_platform_adapter_on_mdm = mock.MagicMock(spec=PlatformInterface)
        mock_platform_adapter_on_mdm.get_symbol_info.return_value = None
        trend_following_sma_strategy.market_data_manager.platform_adapter = mock_platform_adapter_on_mdm

        current_pos = Position(position_id="P1", symbol="EURUSD", action=OrderAction.BUY, volume=0.1, open_price=1.1, open_time=datetime.now(timezone.utc), stop_loss=1.0)
        mock_tick = TickData(timestamp=datetime.now(timezone.utc), symbol="EURUSD", bid=1.11, ask=1.1102)
        
        action = trend_following_sma_strategy.manage_open_position(current_pos, latest_tick=mock_tick)
        assert action is None
        trend_following_sma_strategy.logger.error.assert_any_call(
            f"[{trend_following_sma_strategy.symbol}/{trend_following_sma_strategy.timeframe.name}] Could not get symbol info for rounding prices in manage_open_position."
        )
        # Important: Reset or ensure mock_market_data_manager is fresh for other tests if it's shared and modified.
        # Pytest fixtures usually handle this by providing fresh instances.

    def test_manage_open_position_ts_missing_atr_column(self, trend_following_sma_strategy, sma_strategy_params_eurusd_h1):
        if not sma_strategy_params_eurusd_h1['use_trailing_stop']:
            pytest.skip("Trailing stop not enabled")

        atr_p_ts = sma_strategy_params_eurusd_h1['trailing_stop_atr_period']
        df_no_atr = create_sample_market_df(atr_period=atr_p_ts)
        df_no_atr = df_no_atr.drop(columns=[f'ATR_{atr_p_ts}']) # Remove ATR column
        trend_following_sma_strategy.market_data_manager.get_market_data.return_value = df_no_atr

        current_pos = Position(position_id="P1", symbol="EURUSD", action=OrderAction.BUY, volume=0.1, open_price=1.1, open_time=datetime.now(timezone.utc), stop_loss=1.0)
        mock_tick = TickData(timestamp=datetime.now(timezone.utc), symbol="EURUSD", bid=1.11, ask=1.1102)
        
        action = trend_following_sma_strategy.manage_open_position(current_pos, latest_tick=mock_tick)
        assert action is None
        trend_following_sma_strategy.logger.warning.assert_any_call(
            f"[{trend_following_sma_strategy.symbol}/{trend_following_sma_strategy.timeframe.name}] Trailing stop ATR column 'ATR_{atr_p_ts}' not found or NaN."
        )

    def test_manage_open_position_ts_nan_atr_value(self, trend_following_sma_strategy, sma_strategy_params_eurusd_h1):
        if not sma_strategy_params_eurusd_h1['use_trailing_stop']:
            pytest.skip("Trailing stop not enabled")

        atr_p_ts = sma_strategy_params_eurusd_h1['trailing_stop_atr_period']
        df_nan_atr = create_sample_market_df(atr_period=atr_p_ts)
        df_nan_atr.loc[df_nan_atr.index[-1], f'ATR_{atr_p_ts}'] = pd.NA # Set ATR to NaN
        trend_following_sma_strategy.market_data_manager.get_market_data.return_value = df_nan_atr

        current_pos = Position(position_id="P1", symbol="EURUSD", action=OrderAction.BUY, volume=0.1, open_price=1.1, open_time=datetime.now(timezone.utc), stop_loss=1.0)
        mock_tick = TickData(timestamp=datetime.now(timezone.utc), symbol="EURUSD", bid=1.11, ask=1.1102)
        
        action = trend_following_sma_strategy.manage_open_position(current_pos, latest_tick=mock_tick)
        assert action is None
        trend_following_sma_strategy.logger.warning.assert_any_call(
            f"[{trend_following_sma_strategy.symbol}/{trend_following_sma_strategy.timeframe.name}] Trailing stop ATR column 'ATR_{atr_p_ts}' not found or NaN."
        )

    def test_manage_open_position_latest_bar_none_with_trailing_stop(self, trend_following_sma_strategy, sma_strategy_params_eurusd_h1):
            if not sma_strategy_params_eurusd_h1['use_trailing_stop']:
                pytest.skip("Trailing stop not enabled")
    
            atr_p_ts = sma_strategy_params_eurusd_h1['trailing_stop_atr_period']
            atr_mult_ts = sma_strategy_params_eurusd_h1['trailing_stop_atr_multiplier']
            df = create_sample_market_df(atr_period=atr_p_ts)
            df[f'ATR_{atr_p_ts}'] = 0.00150
            trend_following_sma_strategy.market_data_manager.get_market_data.return_value = df
            
            current_long_pos = Position(
                position_id="LTS_LB0", symbol="EURUSD", action=OrderAction.BUY,
                volume=0.1, open_price=1.08000, open_time=datetime.now(timezone.utc) - timedelta(hours=5),
                stop_loss=1.07500
            )
            mock_tick_higher = TickData(timestamp=datetime.now(timezone.utc), symbol="EURUSD", bid=1.09000, ask=1.09020)
    
            # latest_bar is None, max_position_age_bars check should be skipped
            # Trailing stop should still work if enabled and data is available
            action = trend_following_sma_strategy.manage_open_position(current_long_pos, latest_bar=None, latest_tick=mock_tick_higher)
            
            assert action is not None # Expecting trailing stop to kick in
            assert action['signal'] == StrategySignal.MODIFY_SLTP
            expected_new_sl = round(mock_tick_higher.bid - (df[f'ATR_{atr_p_ts}'].iloc[-1] * atr_mult_ts), 5)
            assert action['new_stop_loss'] == expected_new_sl
            # Use self.symbol_info from the strategy instance, which is set during __init__
            symbol_info = trend_following_sma_strategy.symbol_info
            assert symbol_info is not None # Precondition for the test
            trend_following_sma_strategy.logger.info.assert_any_call(
                f"[{trend_following_sma_strategy.symbol}/{trend_following_sma_strategy.timeframe.name}] Trailing SL for position {current_long_pos.position_id}. Current SL: {current_long_pos.stop_loss}, New SL: {expected_new_sl:.{symbol_info.digits}f}"
            )
            # Ensure no error/warning about missing latest_bar for age calculation if age calc is conditional on latest_bar
            # (The strategy code already checks `if self.max_position_age_bars and latest_bar:`)
    
    def test_manage_open_position_ts_market_data_df_none(self, trend_following_sma_strategy, sma_strategy_params_eurusd_h1):
            if not sma_strategy_params_eurusd_h1['use_trailing_stop']:
                pytest.skip("Trailing stop not enabled")
    
            trend_following_sma_strategy.market_data_manager.get_market_data.return_value = None # Simulate no data
    
            current_pos = Position(position_id="P_MD0", symbol="EURUSD", action=OrderAction.BUY, volume=0.1, open_price=1.1, open_time=datetime.now(timezone.utc), stop_loss=1.0)
            mock_tick = TickData(timestamp=datetime.now(timezone.utc), symbol="EURUSD", bid=1.11, ask=1.1102)
            
            action = trend_following_sma_strategy.manage_open_position(current_pos, latest_tick=mock_tick)
            assert action is None
            trend_following_sma_strategy.logger.warning.assert_any_call(
                f"[{trend_following_sma_strategy.symbol}/{trend_following_sma_strategy.timeframe.name}] No market data for trailing stop ATR calculation."
            )
    
    def test_manage_position_trailing_stop_long_zero_ts_atr(self, trend_following_sma_strategy, sma_strategy_params_eurusd_h1):
            if not sma_strategy_params_eurusd_h1['use_trailing_stop']:
                pytest.skip("Trailing stop not enabled for this test configuration")
    
            atr_p_ts = sma_strategy_params_eurusd_h1['trailing_stop_atr_period']
            # atr_mult_ts = sma_strategy_params_eurusd_h1['trailing_stop_atr_multiplier'] # Multiplier is irrelevant if ATR is 0
    
            df = create_sample_market_df(atr_period=atr_p_ts)
            df.loc[:, f'ATR_{atr_p_ts}'] = 0.00000 # Zero ATR for trailing stop
            trend_following_sma_strategy.market_data_manager.get_market_data.return_value = df
            
            initial_sl = 1.07500
            current_long_pos = Position(
                position_id="LTS_ZATR1", symbol="EURUSD", action=OrderAction.BUY,
                volume=0.1, open_price=1.08000, open_time=datetime.now(timezone.utc) - timedelta(hours=5),
                stop_loss=initial_sl
            )
            # Price moved up, new SL should be at current bid if ATR is zero
            mock_tick_higher = TickData(timestamp=datetime.now(timezone.utc), symbol="EURUSD", bid=1.09000, ask=1.09020)
    
            action = trend_following_sma_strategy.manage_open_position(current_long_pos, latest_tick=mock_tick_higher)
            
            assert action is not None
            assert action['signal'] == StrategySignal.MODIFY_SLTP
            assert action['position_id'] == "LTS_ZATR1"
            
            # With zero ATR, new_sl = round(latest_tick.bid - (0.0 * atr_mult_ts), digits)
            expected_new_sl = round(mock_tick_higher.bid, trend_following_sma_strategy.symbol_info.digits)
            assert action['new_stop_loss'] == expected_new_sl
            assert action['new_stop_loss'] > initial_sl # Should have trailed up
    
            trend_following_sma_strategy.logger.info.assert_any_call(
                f"[{trend_following_sma_strategy.symbol}/{trend_following_sma_strategy.timeframe.name}] Trailing SL for position {current_long_pos.position_id}. Current SL: {initial_sl}, New SL: {expected_new_sl:.{trend_following_sma_strategy.symbol_info.digits}f}"
            )
    
    def test_manage_position_trailing_stop_short_zero_ts_atr(self, trend_following_sma_strategy, sma_strategy_params_eurusd_h1):
            if not sma_strategy_params_eurusd_h1['use_trailing_stop']:
                pytest.skip("Trailing stop not enabled")
            
            atr_p_ts = sma_strategy_params_eurusd_h1['trailing_stop_atr_period']
            # atr_mult_ts = sma_strategy_params_eurusd_h1['trailing_stop_atr_multiplier']
    
            df = create_sample_market_df(atr_period=atr_p_ts)
            df.loc[:, f'ATR_{atr_p_ts}'] = 0.00000 # Zero ATR
            trend_following_sma_strategy.market_data_manager.get_market_data.return_value = df
            
            initial_sl = 1.09500
            current_short_pos = Position(
                position_id="STS_ZATR1", symbol="EURUSD", action=OrderAction.SELL,
                volume=0.1, open_price=1.09000, open_time=datetime.now(timezone.utc) - timedelta(hours=5),
                stop_loss=initial_sl
            )
            mock_tick_lower = TickData(timestamp=datetime.now(timezone.utc), symbol="EURUSD", bid=1.08000, ask=1.08020) # Price moved down
    
            action = trend_following_sma_strategy.manage_open_position(current_short_pos, latest_tick=mock_tick_lower)
            
            assert action is not None
            assert action['signal'] == StrategySignal.MODIFY_SLTP
            # new_sl = round(latest_tick.ask + (0.0 * atr_mult_ts), digits)
            expected_new_sl = round(mock_tick_lower.ask, trend_following_sma_strategy.symbol_info.digits)
            assert action['new_stop_loss'] == expected_new_sl
            assert action['new_stop_loss'] < initial_sl # Should have trailed down
    
            trend_following_sma_strategy.logger.info.assert_any_call(
                f"[{trend_following_sma_strategy.symbol}/{trend_following_sma_strategy.timeframe.name}] Trailing SL for position {current_short_pos.position_id}. Current SL: {initial_sl}, New SL: {expected_new_sl:.{trend_following_sma_strategy.symbol_info.digits}f}"
            )
    
    def test_manage_position_trailing_stop_long_negative_ts_atr(self, trend_following_sma_strategy, sma_strategy_params_eurusd_h1):
            if not sma_strategy_params_eurusd_h1['use_trailing_stop']:
                pytest.skip("Trailing stop not enabled")
    
            atr_p_ts = sma_strategy_params_eurusd_h1['trailing_stop_atr_period']
            atr_mult_ts = sma_strategy_params_eurusd_h1['trailing_stop_atr_multiplier']
    
            df = create_sample_market_df(atr_period=atr_p_ts)
            negative_atr_val = -0.00100
            df.loc[:, f'ATR_{atr_p_ts}'] = negative_atr_val # Negative ATR
            trend_following_sma_strategy.market_data_manager.get_market_data.return_value = df
            
            initial_sl = 1.07500
            current_long_pos = Position(
                position_id="LTS_NATR1", symbol="EURUSD", action=OrderAction.BUY,
                volume=0.1, open_price=1.08000, open_time=datetime.now(timezone.utc) - timedelta(hours=5),
                stop_loss=initial_sl
            )
            mock_tick_higher = TickData(timestamp=datetime.now(timezone.utc), symbol="EURUSD", bid=1.09000, ask=1.09020)
    
            action = trend_following_sma_strategy.manage_open_position(current_long_pos, latest_tick=mock_tick_higher)
            
            assert action is not None
            assert action['signal'] == StrategySignal.MODIFY_SLTP
            
            ts_atr_distance = negative_atr_val * atr_mult_ts # This will be negative
            # new_sl = round(latest_tick.bid - ts_atr_distance, digits)
            # new_sl = round(latest_tick.bid - (negative_value), digits) = round(latest_tick.bid + abs(negative_value), digits)
            expected_new_sl = round(mock_tick_higher.bid - ts_atr_distance, trend_following_sma_strategy.symbol_info.digits)
            assert action['new_stop_loss'] == expected_new_sl
            # New SL is current_bid + abs(ATR_distance), so it's further away from entry / "worse" SL.
            # However, it should still be > initial_sl if price moved up significantly.
            assert action['new_stop_loss'] > initial_sl
            assert action['new_stop_loss'] > mock_tick_higher.bid # SL is now above current bid for a long
    
            trend_following_sma_strategy.logger.info.assert_any_call(
                f"[{trend_following_sma_strategy.symbol}/{trend_following_sma_strategy.timeframe.name}] Trailing SL for position {current_long_pos.position_id}. Current SL: {initial_sl}, New SL: {expected_new_sl:.{trend_following_sma_strategy.symbol_info.digits}f}"
            )
    
    def test_manage_position_trailing_stop_short_negative_ts_atr(self, trend_following_sma_strategy, sma_strategy_params_eurusd_h1):
            if not sma_strategy_params_eurusd_h1['use_trailing_stop']:
                pytest.skip("Trailing stop not enabled")
    
            atr_p_ts = sma_strategy_params_eurusd_h1['trailing_stop_atr_period']
            atr_mult_ts = sma_strategy_params_eurusd_h1['trailing_stop_atr_multiplier']
    
            df = create_sample_market_df(atr_period=atr_p_ts)
            negative_atr_val = -0.00100
            df.loc[:, f'ATR_{atr_p_ts}'] = negative_atr_val # Negative ATR
            trend_following_sma_strategy.market_data_manager.get_market_data.return_value = df
            
            initial_sl = 1.09500
            current_short_pos = Position(
                position_id="STS_NATR1", symbol="EURUSD", action=OrderAction.SELL,
                volume=0.1, open_price=1.09000, open_time=datetime.now(timezone.utc) - timedelta(hours=5),
                stop_loss=initial_sl
            )
            mock_tick_lower = TickData(timestamp=datetime.now(timezone.utc), symbol="EURUSD", bid=1.08000, ask=1.08020)
    
            action = trend_following_sma_strategy.manage_open_position(current_short_pos, latest_tick=mock_tick_lower)
            
            assert action is not None
            assert action['signal'] == StrategySignal.MODIFY_SLTP
            
            ts_atr_distance = negative_atr_val * atr_mult_ts # Negative
            # new_sl = round(latest_tick.ask + ts_atr_distance, digits)
            # new_sl = round(latest_tick.ask + (negative_value), digits) = round(latest_tick.ask - abs(negative_value), digits)
            expected_new_sl = round(mock_tick_lower.ask + ts_atr_distance, trend_following_sma_strategy.symbol_info.digits)
            assert action['new_stop_loss'] == expected_new_sl
            assert action['new_stop_loss'] < initial_sl
            assert action['new_stop_loss'] < mock_tick_lower.ask # SL is now below current ask for a short
    
            trend_following_sma_strategy.logger.info.assert_any_call(
                f"[{trend_following_sma_strategy.symbol}/{trend_following_sma_strategy.timeframe.name}] Trailing SL for position {current_short_pos.position_id}. Current SL: {initial_sl}, New SL: {expected_new_sl:.{trend_following_sma_strategy.symbol_info.digits}f}"
            )
    
    def test_manage_open_position_ts_market_data_df_empty(self, trend_following_sma_strategy, sma_strategy_params_eurusd_h1):
            if not sma_strategy_params_eurusd_h1['use_trailing_stop']:
                pytest.skip("Trailing stop not enabled")
    
            trend_following_sma_strategy.market_data_manager.get_market_data.return_value = pd.DataFrame() # Simulate empty data
    
            current_pos = Position(position_id="P_MDE", symbol="EURUSD", action=OrderAction.BUY, volume=0.1, open_price=1.1, open_time=datetime.now(timezone.utc), stop_loss=1.0)
            mock_tick = TickData(timestamp=datetime.now(timezone.utc), symbol="EURUSD", bid=1.11, ask=1.1102)
            
            action = trend_following_sma_strategy.manage_open_position(current_pos, latest_tick=mock_tick)
            assert action is None
            trend_following_sma_strategy.logger.warning.assert_any_call(
                f"[{trend_following_sma_strategy.symbol}/{trend_following_sma_strategy.timeframe.name}] No market data for trailing stop ATR calculation."
            )

# salt 2025-06-11T11:27:26
  
