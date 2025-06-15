# This is the tests/unit/test_strategy_rsi.py file.
import pytest
from unittest import mock
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timezone, timedelta
import logging

# Imports from your project
from prop_firm_trading_bot.src.strategies.mean_reversion_rsi import MeanReversionRSI
from prop_firm_trading_bot.src.core.enums import StrategySignal, OrderAction, Timeframe
from prop_firm_trading_bot.src.core.models import OHLCVData, TickData, Position, SymbolInfo
from prop_firm_trading_bot.src.config_manager import AppConfig, AssetStrategyProfile, StrategyParameterSet
from prop_firm_trading_bot.src.api_connector.base_connector import PlatformInterface
from prop_firm_trading_bot.src.data_handler.market_data_manager import MarketDataManager

# --- Fixtures ---

@pytest.fixture
def mock_logger_rsi(mocker): # Renamed to avoid conflict if used in same session as other test's logger
    return mocker.MagicMock(spec=logging.Logger)

@pytest.fixture
def mock_platform_adapter_rsi(mocker): # Renamed
    adapter = mocker.MagicMock(spec=PlatformInterface)
    mock_sym_info = mocker.MagicMock(spec=SymbolInfo)
    mock_sym_info.digits = 5 # Example for GBPUSD
    mock_sym_info.point = 0.00001 # Example for GBPUSD, consistent with digits
    adapter.get_symbol_info.return_value = mock_sym_info
    return adapter

@pytest.fixture
def mock_market_data_manager_rsi(mocker): # Renamed
    manager = mocker.MagicMock(spec=MarketDataManager)
    # Note: get_instrument_properties is no longer used by MeanReversionRSI for point/digits for SL/TP.
    # That logic now uses platform_adapter.get_symbol_info().
    # This mock might be relevant if other parts of MarketDataManager are used by the strategy.
    return manager

@pytest.fixture
def rsi_strategy_params_gbpusd_m15():
    # Based on your uploaded strategy_rsi_gbpusd_m15.json
    return {
        "timeframe": "M15",
        "rsi_period": 14,
        "rsi_oversold": 25,
        "rsi_overbought": 75,
        "bollinger_period": 20, # Bollinger Bands params
        "bollinger_std_dev": 2.0,
        "trend_filter_ma_period": 100,
        "stop_loss_atr_period": 10,
        "stop_loss_atr_multiplier": 1.8,
        "take_profit_atr_period": 10, # Assuming TP uses same ATR period
        "take_profit_atr_multiplier": 2.5,
        "exit_rsi_neutral_low": 45.0, # From MeanReversionRSI __init__
        "exit_rsi_neutral_high": 55.0, # From MeanReversionRSI __init__
        "max_position_age_bars": 48
    }

@pytest.fixture
def mock_app_config_for_rsi_strategy(mocker, rsi_strategy_params_gbpusd_m15, basic_bot_settings_rsi): # Add basic_bot_settings
    config = mocker.MagicMock(spec=AppConfig)
    config.asset_strategy_profiles = {
        "GBPUSD_RSI_M15_Profile": AssetStrategyProfile(
            symbol="GBPUSD",
            enabled=True,
            instrument_details_key="GBPUSD_FTMO",
            strategy_params_key="strategy_rsi_gbpusd_m15"
        )
    }
    config.loaded_strategy_parameters = {
        "strategy_rsi_gbpusd_m15": StrategyParameterSet(
            description="Mean Reversion RSI for GBPUSD M15",
            strategy_definition_key="MeanReversion_RSI_BB",
            parameters=rsi_strategy_params_gbpusd_m15
        )
    }
    config.bot_settings = basic_bot_settings_rsi # Use a BotSettings fixture
    return config

@pytest.fixture # Added this fixture for bot settings
def basic_bot_settings_rsi(mocker): # Added mocker to potentially mock BotSettings if not importable
    # If BotSettings is a simple class/dataclass, direct instantiation is fine.
    # If it's complex or has dependencies, it might need to be mocked.
    # For now, assuming it can be instantiated or is already mocked/available.
    try:
        from prop_firm_trading_bot.src.config_manager import BotSettings # Try to import
        return BotSettings(
            trading_mode="paper",
            main_loop_delay_seconds=1,
            app_name="TestRSINewsBot",
            ftmo_server_timezone="Europe/Prague"
        )
    except ImportError:
        # Fallback to MagicMock if BotSettings cannot be imported (e.g. in isolated test run)
        mock_bs = mocker.MagicMock()
        mock_bs.trading_mode="paper"
        mock_bs.main_loop_delay_seconds=1
        mock_bs.app_name="TestRSINewsBot"
        mock_bs.ftmo_server_timezone="Europe/Prague"
        return mock_bs


@pytest.fixture
def mean_reversion_rsi_strategy(
    mock_app_config_for_rsi_strategy,
    mock_platform_adapter_rsi,
    mock_market_data_manager_rsi,
    mock_logger_rsi,
    rsi_strategy_params_gbpusd_m15
    ):
    strategy = MeanReversionRSI(
        strategy_params=rsi_strategy_params_gbpusd_m15,
        config=mock_app_config_for_rsi_strategy,
        platform_adapter=mock_platform_adapter_rsi,
        market_data_manager=mock_market_data_manager_rsi,
        logger=mock_logger_rsi,
        asset_profile_key="GBPUSD_RSI_M15_Profile"
    )
    return strategy

# --- Helper Function to Create Sample Market Data for RSI ---

def create_sample_rsi_market_df(
    rsi_period=14, bbands_period=20, bbands_std=2.0, trend_ma_period=100, atr_period=10, rows=150):
    base_time = datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    data = {
        'open': [1.2500 + i*0.0001 for i in range(rows)],
        'high': [1.2510 + i*0.0001 for i in range(rows)],
        'low': [1.2490 + i*0.0001 for i in range(rows)],
        'close': [1.2505 + i*0.0001 for i in range(rows)],
        'volume': [100 + i for i in range(rows)]
    }
    index = [base_time + timedelta(minutes=15*i) for i in range(rows)] # M15 timeframe
    df = pd.DataFrame(data, index=pd.Index(index, name="timestamp"))

    # Add mock indicators - strategy relies on these
    # For RSI, actual calculation is complex, mock values or use pandas_ta if simple enough for testing
    # df[f'RSI_{rsi_period}'] = 50.0 # Default neutral RSI
    df.ta.rsi(length=rsi_period, append=True, col_names=(f'RSI_{rsi_period}',))


    # For Bollinger Bands (BBM, BBL, BBU are typical column names from pandas_ta.bbands)
    # We need 'BBM_{bbands_period}_{bbands_std}' etc. The strategy doesn't directly use BB for signals yet.
    # Let's add them based on pandas-ta naming convention for completeness if MeanReversionRSI logic expands
    if bbands_period and bbands_std:
         df.ta.bbands(length=bbands_period, std=bbands_std, append=True)
         # pandas-ta appends: BBL_20_2.0, BBM_20_2.0, BBU_20_2.0, BBB_20_2.0, BBP_20_2.0

    if trend_ma_period:
        df[f'SMA_trend_{trend_ma_period}'] = df['close'].rolling(window=trend_ma_period).mean()
    
    df[f'ATR_{atr_period}'] = 0.00150 # Mock ATR value for SL/TP
    df = df.fillna(method='bfill') # Fill NaNs from rolling calculations for test simplicity
    return df

# --- Test Cases for generate_signal ---

class TestMeanReversionRSIGenerateSignal:

    def test_generate_signal_not_enough_data(self, mean_reversion_rsi_strategy):
        signal = mean_reversion_rsi_strategy.generate_signal(
            market_data_df=pd.DataFrame(),
            active_position=None,
            latest_tick=None
        )
        assert signal is None
        # Add assertion for logger call if specific message is expected
        mean_reversion_rsi_strategy.logger.debug.assert_any_call(
            f"[{mean_reversion_rsi_strategy.symbol}/{mean_reversion_rsi_strategy.timeframe.name}] Not enough data for RSI strategy (need {mean_reversion_rsi_strategy.rsi_period + 1}, got 0)."
        )

    def test_generate_signal_missing_rsi_column(self, mean_reversion_rsi_strategy, rsi_strategy_params_gbpusd_m15):
        rsi_p = rsi_strategy_params_gbpusd_m15['rsi_period']
        df = create_sample_rsi_market_df(rsi_period=rsi_p, rows=rsi_p + 5)
        df_missing = df.drop(columns=[f'RSI_{rsi_p}'])
        mock_tick = TickData(timestamp=datetime.now(timezone.utc), symbol="GBPUSD", bid=1.25000, ask=1.25020)
        signal = mean_reversion_rsi_strategy.generate_signal(df_missing, None, mock_tick)
        assert signal is None
        mean_reversion_rsi_strategy.logger.warning.assert_any_call(
            f"[{mean_reversion_rsi_strategy.symbol}/{mean_reversion_rsi_strategy.timeframe.name}] Missing required columns in market data: {['RSI_14']}. Available: {[col for col in df_missing.columns if col != f'RSI_{rsi_p}']}"
        ) # Actual available columns might vary slightly based on helper

    def test_generate_signal_missing_atr_column(self, mean_reversion_rsi_strategy, rsi_strategy_params_gbpusd_m15):
        atr_p = rsi_strategy_params_gbpusd_m15['stop_loss_atr_period']
        df = create_sample_rsi_market_df(atr_period=atr_p, rows=atr_p + 5)
        df_missing = df.drop(columns=[f'ATR_{atr_p}'])
        mock_tick = TickData(timestamp=datetime.now(timezone.utc), symbol="GBPUSD", bid=1.25000, ask=1.25020)
        signal = mean_reversion_rsi_strategy.generate_signal(df_missing, None, mock_tick)
        assert signal is None
        mean_reversion_rsi_strategy.logger.warning.assert_any_call(
             f"[{mean_reversion_rsi_strategy.symbol}/{mean_reversion_rsi_strategy.timeframe.name}] Missing required columns in market data: {['ATR_10']}. Available: {[col for col in df_missing.columns if col != f'ATR_{atr_p}']}"
        )

    def test_generate_signal_missing_trend_ma_column_if_configured(self, mean_reversion_rsi_strategy, rsi_strategy_params_gbpusd_m15):
        trend_ma_p = rsi_strategy_params_gbpusd_m15['trend_filter_ma_period']
        if not trend_ma_p:
            pytest.skip("Trend MA not configured for this test variation")
        
        df = create_sample_rsi_market_df(trend_ma_period=trend_ma_p, rows=trend_ma_p + 5)
        df_missing = df.drop(columns=[f'SMA_trend_{trend_ma_p}'])
        mock_tick = TickData(timestamp=datetime.now(timezone.utc), symbol="GBPUSD", bid=1.25000, ask=1.25020)
        signal = mean_reversion_rsi_strategy.generate_signal(df_missing, None, mock_tick)
        assert signal is None
        mean_reversion_rsi_strategy.logger.warning.assert_any_call(
            f"[{mean_reversion_rsi_strategy.symbol}/{mean_reversion_rsi_strategy.timeframe.name}] Missing required columns in market data: {['SMA_trend_100']}. Available: {[col for col in df_missing.columns if col != f'SMA_trend_{trend_ma_p}']}"
        )

    def test_generate_signal_nan_in_rsi(self, mean_reversion_rsi_strategy, rsi_strategy_params_gbpusd_m15):
        rsi_p = rsi_strategy_params_gbpusd_m15['rsi_period']
        df = create_sample_rsi_market_df(rsi_period=rsi_p, rows=rsi_p + 5)
        df.loc[df.index[-1], f'RSI_{rsi_p}'] = pd.NA
        mock_tick = TickData(timestamp=datetime.now(timezone.utc), symbol="GBPUSD", bid=1.25000, ask=1.25020)
        signal = mean_reversion_rsi_strategy.generate_signal(df, None, mock_tick)
        assert signal is None
        mean_reversion_rsi_strategy.logger.debug.assert_any_call(
            f"[{mean_reversion_rsi_strategy.symbol}/{mean_reversion_rsi_strategy.timeframe.name}] RSI or ATR values are NaN. Not enough data for signal."
        )

    def test_generate_signal_nan_in_atr(self, mean_reversion_rsi_strategy, rsi_strategy_params_gbpusd_m15):
        rsi_p = rsi_strategy_params_gbpusd_m15['rsi_period']
        atr_p = rsi_strategy_params_gbpusd_m15['stop_loss_atr_period']
        df = create_sample_rsi_market_df(rsi_period=rsi_p, atr_period=atr_p, rows=rsi_p + 5)
        # Ensure RSI is valid for signal generation
        df.loc[df.index[-2], f'RSI_{rsi_p}'] = rsi_strategy_params_gbpusd_m15['rsi_oversold'] - 1
        df.loc[df.index[-1], f'RSI_{rsi_p}'] = rsi_strategy_params_gbpusd_m15['rsi_oversold'] + 1
        df.loc[df.index[-1], f'ATR_{atr_p}'] = pd.NA # Introduce NaN in ATR

        mock_tick = TickData(timestamp=datetime.now(timezone.utc), symbol="GBPUSD", bid=1.25000, ask=1.25020)
        signal = mean_reversion_rsi_strategy.generate_signal(df, None, mock_tick)
        assert signal is None
        mean_reversion_rsi_strategy.logger.debug.assert_any_call(
            f"[{mean_reversion_rsi_strategy.symbol}/{mean_reversion_rsi_strategy.timeframe.name}] RSI or ATR values are NaN. Not enough data for signal."
        )

    def test_generate_signal_nan_in_close_price(self, mean_reversion_rsi_strategy, rsi_strategy_params_gbpusd_m15):
        rsi_p = rsi_strategy_params_gbpusd_m15['rsi_period']
        # Create data, then introduce NaN in close, which should propagate to RSI and Trend MA
        df = create_sample_rsi_market_df(rsi_period=rsi_p, rows=rsi_p + 20) # More rows for MA calc
        df.loc[df.index[-1], 'close'] = pd.NA
        
        # Re-calculate indicators on the DataFrame with NaN close to simulate real scenario
        # pandas_ta might handle NaNs by outputting NaNs for indicators
        df.ta.rsi(length=rsi_p, append=True, col_names=(f'RSI_{rsi_p}',), close=df['close'], dropna=False, fillna=pd.NA) # Explicitly pass close
        if mean_reversion_rsi_strategy.trend_filter_ma_period:
            trend_ma_p = mean_reversion_rsi_strategy.trend_filter_ma_period
            df[f'SMA_trend_{trend_ma_p}'] = df['close'].rolling(window=trend_ma_p).mean()
        
        # ATR might also be affected if it uses close
        atr_p = mean_reversion_rsi_strategy.stop_loss_atr_period
        # For simplicity, assume ATR calculation would also result in NaN or be missing if 'close' is NaN
        # The strategy checks for NaN in ATR column directly.
        # If create_sample_rsi_market_df is robust, it might already put NaN in ATR if close is NaN.
        # Let's ensure ATR is NaN for this test if 'close' is NaN.
        # The helper function `create_sample_rsi_market_df` uses a fixed ATR value, so we need to override it or ensure it becomes NaN.
        # The strategy's primary check is for NaN in the indicator columns themselves.
        # If 'close' being NaN leads to NaN in RSI, that's the path we're testing.

        mock_tick = TickData(timestamp=datetime.now(timezone.utc), symbol="GBPUSD", bid=1.25000, ask=1.25020)
        signal = mean_reversion_rsi_strategy.generate_signal(df, None, mock_tick)
        assert signal is None
        # Expecting a log about NaN in RSI or Trend MA due to NaN close
        # The exact log depends on which check fails first.
        # If RSI becomes NaN:
        mean_reversion_rsi_strategy.logger.debug.assert_any_call(
             f"[{mean_reversion_rsi_strategy.symbol}/{mean_reversion_rsi_strategy.timeframe.name}] RSI or ATR values are NaN. Not enough data for signal."
        )

    def test_generate_signal_nan_in_trend_ma_if_configured(self, mean_reversion_rsi_strategy, rsi_strategy_params_gbpusd_m15):
        trend_ma_p = rsi_strategy_params_gbpusd_m15['trend_filter_ma_period']
        if not trend_ma_p:
            pytest.skip("Trend MA not configured")
        df = create_sample_rsi_market_df(trend_ma_period=trend_ma_p, rows=trend_ma_p + 5)
        df.loc[df.index[-1], f'SMA_trend_{trend_ma_p}'] = pd.NA
        # Ensure RSI and ATR are valid to isolate NaN in Trend MA
        df.loc[df.index[-1], f'RSI_{rsi_strategy_params_gbpusd_m15["rsi_period"]}'] = 50
        df.loc[df.index[-2], f'RSI_{rsi_strategy_params_gbpusd_m15["rsi_period"]}'] = 50
        df.loc[df.index[-1], f'ATR_{rsi_strategy_params_gbpusd_m15["stop_loss_atr_period"]}'] = 0.0010

        mock_tick = TickData(timestamp=datetime.now(timezone.utc), symbol="GBPUSD", bid=1.25000, ask=1.25020)
        signal = mean_reversion_rsi_strategy.generate_signal(df, None, mock_tick)
        assert signal is None
        mean_reversion_rsi_strategy.logger.debug.assert_any_call(
            f"[{mean_reversion_rsi_strategy.symbol}/{mean_reversion_rsi_strategy.timeframe.name}] Trend MA value is NaN. Not enough data for signal."
        )
        
    def test_generate_signal_latest_tick_none(self, mean_reversion_rsi_strategy, rsi_strategy_params_gbpusd_m15):
        rsi_p = rsi_strategy_params_gbpusd_m15['rsi_period']
        df = create_sample_rsi_market_df(rsi_period=rsi_p, rows=rsi_p + 5)
        # Setup for a potential signal
        df.loc[df.index[-2], f'RSI_{rsi_p}'] = rsi_strategy_params_gbpusd_m15['rsi_oversold'] - 1
        df.loc[df.index[-1], f'RSI_{rsi_p}'] = rsi_strategy_params_gbpusd_m15['rsi_oversold'] + 1
        if rsi_strategy_params_gbpusd_m15.get('trend_filter_ma_period'):
            df.loc[df.index[-1], 'close'] = 1.2600
            df.loc[df.index[-1], f"SMA_trend_{rsi_strategy_params_gbpusd_m15['trend_filter_ma_period']}"] = 1.2500

        signal = mean_reversion_rsi_strategy.generate_signal(df, None, None) # latest_tick is None
        assert signal is None
        mean_reversion_rsi_strategy.logger.warning.assert_any_call(
             f"[{mean_reversion_rsi_strategy.symbol}/{mean_reversion_rsi_strategy.timeframe.name}] Could not get current tick (latest_tick) for price reference."
        )

    def test_generate_signal_symbol_info_none(self, mean_reversion_rsi_strategy, rsi_strategy_params_gbpusd_m15, mock_platform_adapter_rsi):
        rsi_p = rsi_strategy_params_gbpusd_m15['rsi_period']
        df = create_sample_rsi_market_df(rsi_period=rsi_p, rows=rsi_p + 5)
        df.loc[df.index[-2], f'RSI_{rsi_p}'] = rsi_strategy_params_gbpusd_m15['rsi_oversold'] - 1
        df.loc[df.index[-1], f'RSI_{rsi_p}'] = rsi_strategy_params_gbpusd_m15['rsi_oversold'] + 1
        if rsi_strategy_params_gbpusd_m15.get('trend_filter_ma_period'):
            df.loc[df.index[-1], 'close'] = 1.2600
            df.loc[df.index[-1], f"SMA_trend_{rsi_strategy_params_gbpusd_m15['trend_filter_ma_period']}"] = 1.2500
        
        mock_tick = TickData(timestamp=datetime.now(timezone.utc), symbol="GBPUSD", bid=1.25000, ask=1.25020)
        mock_platform_adapter_rsi.get_symbol_info.return_value = None

        signal = mean_reversion_rsi_strategy.generate_signal(df, None, mock_tick)
        assert signal is None
        mean_reversion_rsi_strategy.logger.error.assert_any_call(
            f"[{mean_reversion_rsi_strategy.symbol}/{mean_reversion_rsi_strategy.timeframe.name}] Could not get symbol info for rounding prices."
        )
        
    # This test is removed as its functionality (testing missing symbol info for SL/TP)
    # is covered by test_generate_signal_symbol_info_none, and the strategy
    # now uses platform_adapter.get_symbol_info() not market_data_manager.get_instrument_properties()
    # for point/digits. The asserted log message was also from the old path.
    # def test_generate_signal_instrument_properties_none(...)

    def test_generate_signal_instrument_properties_invalid_point(self, mean_reversion_rsi_strategy, rsi_strategy_params_gbpusd_m15, mock_platform_adapter_rsi):
        rsi_p = rsi_strategy_params_gbpusd_m15['rsi_period']
        df = create_sample_rsi_market_df(rsi_period=rsi_p, rows=rsi_p + 5)
        df.loc[df.index[-2], f'RSI_{rsi_p}'] = rsi_strategy_params_gbpusd_m15['rsi_oversold'] - 1
        df.loc[df.index[-1], f'RSI_{rsi_p}'] = rsi_strategy_params_gbpusd_m15['rsi_oversold'] + 1
        if rsi_strategy_params_gbpusd_m15.get('trend_filter_ma_period'):
            df.loc[df.index[-1], 'close'] = 1.2600
            df.loc[df.index[-1], f"SMA_trend_{rsi_strategy_params_gbpusd_m15['trend_filter_ma_period']}"] = 1.2500

        mock_tick = TickData(timestamp=datetime.now(timezone.utc), symbol="GBPUSD", bid=1.25000, ask=1.25020)
        
        # Mock platform_adapter to return symbol_info with an invalid point
        mock_invalid_sym_info = mock.MagicMock(spec=SymbolInfo)
        mock_invalid_sym_info.digits = 5
        mock_invalid_sym_info.point = 0  # Invalid point
        mock_platform_adapter_rsi.get_symbol_info.return_value = mock_invalid_sym_info

        signal = mean_reversion_rsi_strategy.generate_signal(df, None, mock_tick)
        assert signal is None
        mean_reversion_rsi_strategy.logger.error.assert_any_call(
            f"[{mean_reversion_rsi_strategy.symbol}/{mean_reversion_rsi_strategy.timeframe.name}] Invalid 'point' value (0) for {mean_reversion_rsi_strategy.symbol}. Must be a positive number. Cannot calculate sl_pips."
        )

    def test_generate_signal_rsi_oversold_buy_nan_ask_in_tick(self, mean_reversion_rsi_strategy, rsi_strategy_params_gbpusd_m15):
        rsi_p = rsi_strategy_params_gbpusd_m15['rsi_period']
        oversold_lvl = rsi_strategy_params_gbpusd_m15['rsi_oversold']
        trend_ma_p = rsi_strategy_params_gbpusd_m15['trend_filter_ma_period']
        atr_p = rsi_strategy_params_gbpusd_m15['stop_loss_atr_period']
        sl_atr_mult = rsi_strategy_params_gbpusd_m15['stop_loss_atr_multiplier']

        df = create_sample_rsi_market_df(rsi_period=rsi_p, trend_ma_period=trend_ma_p, atr_period=atr_p)
        df.loc[df.index[-2], f'RSI_{rsi_p}'] = oversold_lvl - 1
        df.loc[df.index[-1], f'RSI_{rsi_p}'] = oversold_lvl + 1
        if trend_ma_p:
            df.loc[df.index[-1], 'close'] = 1.2600
            df.loc[df.index[-1], f'SMA_trend_{trend_ma_p}'] = 1.2550
        df.loc[df.index[-1], f'ATR_{atr_p}'] = 0.0020

        mock_tick_nan_ask = TickData(timestamp=datetime.now(timezone.utc), symbol="GBPUSD", bid=1.25990, ask=float('nan'))
        signal = mean_reversion_rsi_strategy.generate_signal(df, None, mock_tick_nan_ask)

        assert signal is not None
        assert signal['signal'] == StrategySignal.BUY
        assert pd.isna(signal['price'])
        assert pd.isna(signal['stop_loss_price'])
        assert pd.isna(signal['take_profit_price'])
        
        point_value = mean_reversion_rsi_strategy.platform_adapter.get_symbol_info(mean_reversion_rsi_strategy.symbol).point
        expected_sl_distance = df[f'ATR_{atr_p}'].iloc[-1] * sl_atr_mult
        expected_sl_pips = expected_sl_distance / point_value
        assert signal['sl_pips'] == pytest.approx(expected_sl_pips) # sl_pips should still be valid
        mean_reversion_rsi_strategy.logger.info.assert_any_call(
            f"[{mean_reversion_rsi_strategy.symbol}/{mean_reversion_rsi_strategy.timeframe.name}] BUY signal generated for GBPUSD at NaN (Ask). SL: NaN, TP: NaN, SL_pips: {expected_sl_pips:.2f}"
        )

    def test_generate_signal_rsi_overbought_sell_nan_bid_in_tick(self, mean_reversion_rsi_strategy, rsi_strategy_params_gbpusd_m15):
        rsi_p = rsi_strategy_params_gbpusd_m15['rsi_period']
        overbought_lvl = rsi_strategy_params_gbpusd_m15['rsi_overbought']
        trend_ma_p = rsi_strategy_params_gbpusd_m15['trend_filter_ma_period']
        atr_p = rsi_strategy_params_gbpusd_m15['stop_loss_atr_period']
        sl_atr_mult = rsi_strategy_params_gbpusd_m15['stop_loss_atr_multiplier']

        df = create_sample_rsi_market_df(rsi_period=rsi_p, trend_ma_period=trend_ma_p, atr_period=atr_p)
        df.loc[df.index[-2], f'RSI_{rsi_p}'] = overbought_lvl + 1
        df.loc[df.index[-1], f'RSI_{rsi_p}'] = overbought_lvl - 1
        if trend_ma_p:
            df.loc[df.index[-1], 'close'] = 1.2400
            df.loc[df.index[-1], f'SMA_trend_{trend_ma_p}'] = 1.2450
        df.loc[df.index[-1], f'ATR_{atr_p}'] = 0.0020

        mock_tick_nan_bid = TickData(timestamp=datetime.now(timezone.utc), symbol="GBPUSD", bid=float('nan'), ask=1.24010)
        signal = mean_reversion_rsi_strategy.generate_signal(df, None, mock_tick_nan_bid)

        assert signal is not None
        assert signal['signal'] == StrategySignal.SELL
        assert pd.isna(signal['price'])
        assert pd.isna(signal['stop_loss_price'])
        assert pd.isna(signal['take_profit_price'])

        point_value = mean_reversion_rsi_strategy.platform_adapter.get_symbol_info(mean_reversion_rsi_strategy.symbol).point
        expected_sl_distance = df[f'ATR_{atr_p}'].iloc[-1] * sl_atr_mult
        expected_sl_pips = expected_sl_distance / point_value
        assert signal['sl_pips'] == pytest.approx(expected_sl_pips) # sl_pips should still be valid
        mean_reversion_rsi_strategy.logger.info.assert_any_call(
            f"[{mean_reversion_rsi_strategy.symbol}/{mean_reversion_rsi_strategy.timeframe.name}] SELL signal generated for GBPUSD at NaN (Bid). SL: NaN, TP: NaN, SL_pips: {expected_sl_pips:.2f}"
        )

    def test_generate_signal_rsi_oversold_buy_with_trend_filter(self, mean_reversion_rsi_strategy, rsi_strategy_params_gbpusd_m15):
        rsi_p = rsi_strategy_params_gbpusd_m15['rsi_period']
        oversold_lvl = rsi_strategy_params_gbpusd_m15['rsi_oversold']
        trend_ma_p = rsi_strategy_params_gbpusd_m15['trend_filter_ma_period']
        atr_p = rsi_strategy_params_gbpusd_m15['stop_loss_atr_period']
        sl_atr_mult = rsi_strategy_params_gbpusd_m15['stop_loss_atr_multiplier']
        tp_atr_mult = rsi_strategy_params_gbpusd_m15['take_profit_atr_multiplier']

        df = create_sample_rsi_market_df(rsi_period=rsi_p, trend_ma_period=trend_ma_p, atr_period=atr_p)
        
        # Simulate RSI crossing up from oversold
        df.loc[df.index[-2], f'RSI_{rsi_p}'] = oversold_lvl - 5  # Below oversold
        df.loc[df.index[-1], f'RSI_{rsi_p}'] = oversold_lvl + 1  # Crossed above oversold
        
        # Simulate trend filter allowing long (close > trend_ma)
        df.loc[df.index[-1], 'close'] = 1.2600
        df.loc[df.index[-1], f'SMA_trend_{trend_ma_p}'] = 1.2550 # Close is above trend MA
        df.loc[df.index[-1], f'ATR_{atr_p}'] = 0.0020 # Example ATR

        mock_tick = TickData(timestamp=datetime.now(timezone.utc), symbol="GBPUSD", bid=1.25990, ask=1.26010)

        signal = mean_reversion_rsi_strategy.generate_signal(
            market_data_df=df,
            active_position=None, # No active positions
            latest_tick=mock_tick
        )

        assert signal is not None
        assert signal['signal'] == StrategySignal.BUY
        assert signal['price'] == mock_tick.ask
        expected_sl_distance = df[f'ATR_{atr_p}'].iloc[-1] * sl_atr_mult
        expected_tp_distance = df[f'ATR_{atr_p}'].iloc[-1] * tp_atr_mult
        expected_sl = round(mock_tick.ask - expected_sl_distance, 5)
        expected_tp = round(mock_tick.ask + expected_tp_distance, 5)
        
        point_value = mean_reversion_rsi_strategy.platform_adapter.get_symbol_info(mean_reversion_rsi_strategy.symbol).point
        expected_sl_pips = expected_sl_distance / point_value
        
        assert signal['stop_loss_price'] == expected_sl
        assert signal['take_profit_price'] == expected_tp
        assert signal['sl_pips'] == pytest.approx(expected_sl_pips)
        assert f"RSI Buy ({df[f'RSI_{rsi_p}'].iloc[-1]:.2f} crossed {oversold_lvl})" in signal['comment']
        mean_reversion_rsi_strategy.logger.info.assert_called()

    def test_generate_signal_rsi_overbought_sell_with_trend_filter(self, mean_reversion_rsi_strategy, rsi_strategy_params_gbpusd_m15):
        rsi_p = rsi_strategy_params_gbpusd_m15['rsi_period']
        overbought_lvl = rsi_strategy_params_gbpusd_m15['rsi_overbought']
        trend_ma_p = rsi_strategy_params_gbpusd_m15['trend_filter_ma_period']
        atr_p = rsi_strategy_params_gbpusd_m15['stop_loss_atr_period']
        sl_atr_mult = rsi_strategy_params_gbpusd_m15['stop_loss_atr_multiplier']
        tp_atr_mult = rsi_strategy_params_gbpusd_m15['take_profit_atr_multiplier']

        df = create_sample_rsi_market_df(rsi_period=rsi_p, trend_ma_period=trend_ma_p, atr_period=atr_p)

        # Simulate RSI crossing down from overbought
        df.loc[df.index[-2], f'RSI_{rsi_p}'] = overbought_lvl + 5 # Above overbought
        df.loc[df.index[-1], f'RSI_{rsi_p}'] = overbought_lvl - 1 # Crossed below overbought

        # Simulate trend filter allowing short (close < trend_ma)
        df.loc[df.index[-1], 'close'] = 1.2400
        df.loc[df.index[-1], f'SMA_trend_{trend_ma_p}'] = 1.2450 # Close is below trend MA
        df.loc[df.index[-1], f'ATR_{atr_p}'] = 0.0020

        mock_tick = TickData(timestamp=datetime.now(timezone.utc), symbol="GBPUSD", bid=1.23990, ask=1.24010)

        signal = mean_reversion_rsi_strategy.generate_signal(
            market_data_df=df,
            active_position=None, # No active positions
            latest_tick=mock_tick
        )

        assert signal is not None
        assert signal['signal'] == StrategySignal.SELL
        assert signal['price'] == mock_tick.bid
        expected_sl_distance = df[f'ATR_{atr_p}'].iloc[-1] * sl_atr_mult
        expected_tp_distance = df[f'ATR_{atr_p}'].iloc[-1] * tp_atr_mult
        expected_sl = round(mock_tick.bid + expected_sl_distance, 5)
        expected_tp = round(mock_tick.bid - expected_tp_distance, 5)

        point_value = mean_reversion_rsi_strategy.platform_adapter.get_symbol_info(mean_reversion_rsi_strategy.symbol).point
        expected_sl_pips = expected_sl_distance / point_value

        assert signal['stop_loss_price'] == expected_sl
        assert signal['take_profit_price'] == expected_tp
        assert signal['sl_pips'] == pytest.approx(expected_sl_pips)
        assert f"RSI Sell ({df[f'RSI_{rsi_p}'].iloc[-1]:.2f} crossed {overbought_lvl})" in signal['comment']

    def test_generate_signal_rsi_exit_long_on_neutral_cross(self, mean_reversion_rsi_strategy, rsi_strategy_params_gbpusd_m15):
        rsi_p = rsi_strategy_params_gbpusd_m15['rsi_period']
        exit_neutral_high = rsi_strategy_params_gbpusd_m15['exit_rsi_neutral_high']
        
        df = create_sample_rsi_market_df(rsi_period=rsi_p)
        # Simulate RSI crossing up to neutral from below
        df.loc[df.index[-2], f'RSI_{rsi_p}'] = exit_neutral_high - 5 
        df.loc[df.index[-1], f'RSI_{rsi_p}'] = exit_neutral_high + 1

        mock_long_pos = Position(position_id="RSI_L1", symbol="GBPUSD", action=OrderAction.BUY, volume=0.1, open_price=1.25000, open_time=datetime.now(timezone.utc))
        mock_tick = TickData(timestamp=datetime.now(timezone.utc), symbol="GBPUSD", bid=1.25500, ask=1.25520)

        signal = mean_reversion_rsi_strategy.generate_signal(
            market_data_df=df,
            active_position=mock_long_pos,
            latest_tick=mock_tick
        )

        assert signal is not None
        assert signal['signal'] == StrategySignal.CLOSE_LONG
        assert signal['position_id'] == "RSI_L1"
        assert signal['price'] == mock_tick.bid
        assert f"RSI Close Long (RSI crossed neutral {exit_neutral_high})" in signal['comment']

    def test_generate_signal_rsi_exit_short_on_neutral_cross(self, mean_reversion_rsi_strategy, rsi_strategy_params_gbpusd_m15):
        rsi_p = rsi_strategy_params_gbpusd_m15['rsi_period']
        exit_neutral_low = rsi_strategy_params_gbpusd_m15['exit_rsi_neutral_low']
        
        df = create_sample_rsi_market_df(rsi_period=rsi_p)
        # Simulate RSI crossing down to neutral from above
        df.loc[df.index[-2], f'RSI_{rsi_p}'] = exit_neutral_low + 5
        df.loc[df.index[-1], f'RSI_{rsi_p}'] = exit_neutral_low - 1
        
        mock_short_pos = Position(position_id="RSI_S1", symbol="GBPUSD", action=OrderAction.SELL, volume=0.1, open_price=1.25000, open_time=datetime.now(timezone.utc))
        mock_tick = TickData(timestamp=datetime.now(timezone.utc), symbol="GBPUSD", bid=1.24500, ask=1.24520)

        signal = mean_reversion_rsi_strategy.generate_signal(df, mock_short_pos, mock_tick)
        
        assert signal is not None
        assert signal['signal'] == StrategySignal.CLOSE_SHORT
        assert signal['position_id'] == "RSI_S1"
        assert signal['price'] == mock_tick.ask # Close short at ask
        assert f"RSI Close Short (RSI crossed neutral {exit_neutral_low})" in signal['comment']

    def test_generate_signal_trend_filter_blocks_buy(self, mean_reversion_rsi_strategy, rsi_strategy_params_gbpusd_m15):
        rsi_p = rsi_strategy_params_gbpusd_m15['rsi_period']
        oversold_lvl = rsi_strategy_params_gbpusd_m15['rsi_oversold']
        trend_ma_p = rsi_strategy_params_gbpusd_m15['trend_filter_ma_period']
        if not trend_ma_p: pytest.skip("Trend filter not configured")

        df = create_sample_rsi_market_df(rsi_period=rsi_p, trend_ma_period=trend_ma_p)
        df.loc[df.index[-2], f'RSI_{rsi_p}'] = oversold_lvl - 1
        df.loc[df.index[-1], f'RSI_{rsi_p}'] = oversold_lvl + 1
        # Trend filter blocks long (close < trend_ma)
        df.loc[df.index[-1], 'close'] = 1.2500
        df.loc[df.index[-1], f'SMA_trend_{trend_ma_p}'] = 1.2550
        mock_tick = TickData(timestamp=datetime.now(timezone.utc), symbol="GBPUSD", bid=1.24990, ask=1.25010)
        signal = mean_reversion_rsi_strategy.generate_signal(df, None, mock_tick)
        assert signal is None

    def test_generate_signal_trend_filter_blocks_sell(self, mean_reversion_rsi_strategy, rsi_strategy_params_gbpusd_m15):
        rsi_p = rsi_strategy_params_gbpusd_m15['rsi_period']
        overbought_lvl = rsi_strategy_params_gbpusd_m15['rsi_overbought']
        trend_ma_p = rsi_strategy_params_gbpusd_m15['trend_filter_ma_period']
        if not trend_ma_p: pytest.skip("Trend filter not configured")

        df = create_sample_rsi_market_df(rsi_period=rsi_p, trend_ma_period=trend_ma_p)
        df.loc[df.index[-2], f'RSI_{rsi_p}'] = overbought_lvl + 1
        df.loc[df.index[-1], f'RSI_{rsi_p}'] = overbought_lvl - 1
        # Trend filter blocks short (close > trend_ma)
        df.loc[df.index[-1], 'close'] = 1.2550
        df.loc[df.index[-1], f'SMA_trend_{trend_ma_p}'] = 1.2500
        mock_tick = TickData(timestamp=datetime.now(timezone.utc), symbol="GBPUSD", bid=1.25490, ask=1.25510)
        signal = mean_reversion_rsi_strategy.generate_signal(df, None, mock_tick)
        assert signal is None

    def test_generate_signal_no_rsi_cross_oversold(self, mean_reversion_rsi_strategy, rsi_strategy_params_gbpusd_m15):
        rsi_p = rsi_strategy_params_gbpusd_m15['rsi_period']
        oversold_lvl = rsi_strategy_params_gbpusd_m15['rsi_oversold']
        df = create_sample_rsi_market_df(rsi_period=rsi_p)
        # RSI stays below oversold
        df.loc[df.index[-2], f'RSI_{rsi_p}'] = oversold_lvl - 5
        df.loc[df.index[-1], f'RSI_{rsi_p}'] = oversold_lvl - 3
        mock_tick = TickData(timestamp=datetime.now(timezone.utc), symbol="GBPUSD", bid=1.25000, ask=1.25020)
        signal = mean_reversion_rsi_strategy.generate_signal(df, None, mock_tick)
        assert signal is None

    def test_generate_signal_no_rsi_cross_overbought(self, mean_reversion_rsi_strategy, rsi_strategy_params_gbpusd_m15):
        rsi_p = rsi_strategy_params_gbpusd_m15['rsi_period']
        overbought_lvl = rsi_strategy_params_gbpusd_m15['rsi_overbought']
        df = create_sample_rsi_market_df(rsi_period=rsi_p)
        # RSI stays above overbought
        df.loc[df.index[-2], f'RSI_{rsi_p}'] = overbought_lvl + 5
        df.loc[df.index[-1], f'RSI_{rsi_p}'] = overbought_lvl + 3
        mock_tick = TickData(timestamp=datetime.now(timezone.utc), symbol="GBPUSD", bid=1.25000, ask=1.25020)
        signal = mean_reversion_rsi_strategy.generate_signal(df, None, mock_tick)
        assert signal is None

    def test_generate_signal_buy_already_long_no_new_signal(self, mean_reversion_rsi_strategy, rsi_strategy_params_gbpusd_m15):
        rsi_p = rsi_strategy_params_gbpusd_m15['rsi_period']
        oversold_lvl = rsi_strategy_params_gbpusd_m15['rsi_oversold']
        df = create_sample_rsi_market_df(rsi_period=rsi_p)
        df.loc[df.index[-2], f'RSI_{rsi_p}'] = oversold_lvl - 1
        df.loc[df.index[-1], f'RSI_{rsi_p}'] = oversold_lvl + 1
        mock_long_pos = Position(position_id="L1", symbol="GBPUSD", action=OrderAction.BUY, volume=0.1, open_price=1.2400, open_time=datetime.now(timezone.utc))
        mock_tick = TickData(timestamp=datetime.now(timezone.utc), symbol="GBPUSD", bid=1.25000, ask=1.25020)
        signal = mean_reversion_rsi_strategy.generate_signal(df, mock_long_pos, mock_tick)
        assert signal is None # Already long, no new buy signal

    def test_generate_signal_sell_already_short_no_new_signal(self, mean_reversion_rsi_strategy, rsi_strategy_params_gbpusd_m15):
        rsi_p = rsi_strategy_params_gbpusd_m15['rsi_period']
        overbought_lvl = rsi_strategy_params_gbpusd_m15['rsi_overbought']
        df = create_sample_rsi_market_df(rsi_period=rsi_p)
        df.loc[df.index[-2], f'RSI_{rsi_p}'] = overbought_lvl + 1
        df.loc[df.index[-1], f'RSI_{rsi_p}'] = overbought_lvl - 1
        mock_short_pos = Position(position_id="S1", symbol="GBPUSD", action=OrderAction.SELL, volume=0.1, open_price=1.2600, open_time=datetime.now(timezone.utc))
        mock_tick = TickData(timestamp=datetime.now(timezone.utc), symbol="GBPUSD", bid=1.25000, ask=1.25020)
        signal = mean_reversion_rsi_strategy.generate_signal(df, mock_short_pos, mock_tick)
        assert signal is None # Already short, no new sell signal

    def test_generate_signal_buy_condition_closes_active_short(self, mean_reversion_rsi_strategy, rsi_strategy_params_gbpusd_m15):
        rsi_p = rsi_strategy_params_gbpusd_m15['rsi_period']
        oversold_lvl = rsi_strategy_params_gbpusd_m15['rsi_oversold']
        trend_ma_p = rsi_strategy_params_gbpusd_m15['trend_filter_ma_period']
        
        df = create_sample_rsi_market_df(rsi_period=rsi_p, trend_ma_period=trend_ma_p)
        df.loc[df.index[-2], f'RSI_{rsi_p}'] = oversold_lvl - 1
        df.loc[df.index[-1], f'RSI_{rsi_p}'] = oversold_lvl + 1 # Buy condition
        if trend_ma_p: # Ensure trend allows buy
            df.loc[df.index[-1], 'close'] = 1.2600
            df.loc[df.index[-1], f'SMA_trend_{trend_ma_p}'] = 1.2550

        mock_short_pos = Position(position_id="S_Active", symbol="GBPUSD", action=OrderAction.SELL, volume=0.1, open_price=1.2650, open_time=datetime.now(timezone.utc))
        mock_tick = TickData(timestamp=datetime.now(timezone.utc), symbol="GBPUSD", bid=1.25990, ask=1.26010)
        
        signal = mean_reversion_rsi_strategy.generate_signal(df, mock_short_pos, mock_tick)
        
        assert signal is not None
        assert signal['signal'] == StrategySignal.CLOSE_SHORT
        assert signal['position_id'] == "S_Active"
        assert signal['price'] == mock_tick.ask # Close short at ask
        assert "RSI Buy signal, closing existing Short" in signal['comment']

    def test_generate_signal_sell_condition_closes_active_long(self, mean_reversion_rsi_strategy, rsi_strategy_params_gbpusd_m15):
        rsi_p = rsi_strategy_params_gbpusd_m15['rsi_period']
        overbought_lvl = rsi_strategy_params_gbpusd_m15['rsi_overbought']
        trend_ma_p = rsi_strategy_params_gbpusd_m15['trend_filter_ma_period']

        df = create_sample_rsi_market_df(rsi_period=rsi_p, trend_ma_period=trend_ma_p)
        df.loc[df.index[-2], f'RSI_{rsi_p}'] = overbought_lvl + 1
        df.loc[df.index[-1], f'RSI_{rsi_p}'] = overbought_lvl - 1 # Sell condition
        if trend_ma_p: # Ensure trend allows sell
            df.loc[df.index[-1], 'close'] = 1.2400
            df.loc[df.index[-1], f'SMA_trend_{trend_ma_p}'] = 1.2450
            
        mock_long_pos = Position(position_id="L_Active", symbol="GBPUSD", action=OrderAction.BUY, volume=0.1, open_price=1.2350, open_time=datetime.now(timezone.utc))
        mock_tick = TickData(timestamp=datetime.now(timezone.utc), symbol="GBPUSD", bid=1.23990, ask=1.24010)

        signal = mean_reversion_rsi_strategy.generate_signal(df, mock_long_pos, mock_tick)

        assert signal is not None
        assert signal['signal'] == StrategySignal.CLOSE_LONG
        assert signal['position_id'] == "L_Active"
        assert signal['price'] == mock_tick.bid # Close long at bid
        assert "RSI Sell signal, closing existing Long" in signal['comment']

    def test_generate_signal_ranging_market_rsi_cross_trend_blocks(self, mean_reversion_rsi_strategy, rsi_strategy_params_gbpusd_m15):
        rsi_p = rsi_strategy_params_gbpusd_m15['rsi_period']
        oversold_lvl = rsi_strategy_params_gbpusd_m15['rsi_oversold']
        overbought_lvl = rsi_strategy_params_gbpusd_m15['rsi_overbought']
        trend_ma_p = rsi_strategy_params_gbpusd_m15['trend_filter_ma_period']
        if not trend_ma_p:
            pytest.skip("Trend filter not configured for this test")

        df = create_sample_rsi_market_df(rsi_period=rsi_p, trend_ma_period=trend_ma_p)
        mock_tick = TickData(timestamp=datetime.now(timezone.utc), symbol="GBPUSD", bid=1.25000, ask=1.25020)

        # Scenario 1: RSI crosses oversold, but trend blocks BUY
        df.loc[df.index[-2], f'RSI_{rsi_p}'] = oversold_lvl - 1
        df.loc[df.index[-1], f'RSI_{rsi_p}'] = oversold_lvl + 1
        df.loc[df.index[-1], 'close'] = 1.2400  # Price below trend MA
        df.loc[df.index[-1], f'SMA_trend_{trend_ma_p}'] = 1.2450
        signal_buy_blocked = mean_reversion_rsi_strategy.generate_signal(df, None, mock_tick)
        assert signal_buy_blocked is None
        mean_reversion_rsi_strategy.logger.debug.assert_any_call(
            f"[{mean_reversion_rsi_strategy.symbol}/{mean_reversion_rsi_strategy.timeframe.name}] RSI Buy condition met, but trend filter blocks long."
        )

        # Scenario 2: RSI crosses overbought, but trend blocks SELL
        df.loc[df.index[-2], f'RSI_{rsi_p}'] = overbought_lvl + 1
        df.loc[df.index[-1], f'RSI_{rsi_p}'] = overbought_lvl - 1
        df.loc[df.index[-1], 'close'] = 1.2500  # Price above trend MA
        df.loc[df.index[-1], f'SMA_trend_{trend_ma_p}'] = 1.2450
        signal_sell_blocked = mean_reversion_rsi_strategy.generate_signal(df, None, mock_tick)
        assert signal_sell_blocked is None
        mean_reversion_rsi_strategy.logger.debug.assert_any_call(
            f"[{mean_reversion_rsi_strategy.symbol}/{mean_reversion_rsi_strategy.timeframe.name}] RSI Sell condition met, but trend filter blocks short."
        )

    def test_generate_signal_rsi_oversold_buy_zero_atr(self, mean_reversion_rsi_strategy, rsi_strategy_params_gbpusd_m15):
        rsi_p = rsi_strategy_params_gbpusd_m15['rsi_period']
        oversold_lvl = rsi_strategy_params_gbpusd_m15['rsi_oversold']
        trend_ma_p = rsi_strategy_params_gbpusd_m15['trend_filter_ma_period']
        atr_p = rsi_strategy_params_gbpusd_m15['stop_loss_atr_period']
        
        df = create_sample_rsi_market_df(rsi_period=rsi_p, trend_ma_period=trend_ma_p, atr_period=atr_p)
        df.loc[df.index[-2], f'RSI_{rsi_p}'] = oversold_lvl - 1
        df.loc[df.index[-1], f'RSI_{rsi_p}'] = oversold_lvl + 1
        if trend_ma_p: # Ensure trend allows buy
            df.loc[df.index[-1], 'close'] = 1.2600
            df.loc[df.index[-1], f'SMA_trend_{trend_ma_p}'] = 1.2550
        df.loc[df.index[-1], f'ATR_{atr_p}'] = 0.00000 # Zero ATR

        mock_tick = TickData(timestamp=datetime.now(timezone.utc), symbol="GBPUSD", bid=1.25990, ask=1.26010)
        signal = mean_reversion_rsi_strategy.generate_signal(df, None, mock_tick)

        assert signal is not None
        assert signal['signal'] == StrategySignal.BUY
        assert signal['price'] == mock_tick.ask
        assert signal['stop_loss_price'] == mock_tick.ask
        assert signal['take_profit_price'] == mock_tick.ask
        assert signal['sl_pips'] == 0.0

    def test_generate_signal_rsi_oversold_buy_negative_atr(self, mean_reversion_rsi_strategy, rsi_strategy_params_gbpusd_m15):
        rsi_p = rsi_strategy_params_gbpusd_m15['rsi_period']
        oversold_lvl = rsi_strategy_params_gbpusd_m15['rsi_oversold']
        trend_ma_p = rsi_strategy_params_gbpusd_m15['trend_filter_ma_period']
        atr_p = rsi_strategy_params_gbpusd_m15['stop_loss_atr_period']
        sl_atr_mult = rsi_strategy_params_gbpusd_m15['stop_loss_atr_multiplier']
        tp_atr_mult = rsi_strategy_params_gbpusd_m15['take_profit_atr_multiplier']

        df = create_sample_rsi_market_df(rsi_period=rsi_p, trend_ma_period=trend_ma_p, atr_period=atr_p)
        df.loc[df.index[-2], f'RSI_{rsi_p}'] = oversold_lvl - 1
        df.loc[df.index[-1], f'RSI_{rsi_p}'] = oversold_lvl + 1
        if trend_ma_p: # Ensure trend allows buy
            df.loc[df.index[-1], 'close'] = 1.2600
            df.loc[df.index[-1], f'SMA_trend_{trend_ma_p}'] = 1.2550
        
        negative_atr_val = -0.00100
        df.loc[df.index[-1], f'ATR_{atr_p}'] = negative_atr_val

        mock_tick = TickData(timestamp=datetime.now(timezone.utc), symbol="GBPUSD", bid=1.25990, ask=1.26010)
        signal = mean_reversion_rsi_strategy.generate_signal(df, None, mock_tick)

        assert signal is not None
        assert signal['signal'] == StrategySignal.BUY
        assert signal['price'] == mock_tick.ask
        
        expected_sl_distance = negative_atr_val * sl_atr_mult
        expected_tp_distance = negative_atr_val * tp_atr_mult # If TP is based on SL distance, it will also be "wrong"
        
        expected_sl = round(mock_tick.ask - expected_sl_distance, 5)
        expected_tp = round(mock_tick.ask + expected_tp_distance, 5) # For RSI, TP is ATR based, so it will also be negative distance

        assert signal['stop_loss_price'] == expected_sl
        assert signal['stop_loss_price'] > mock_tick.ask # SL is on the "wrong" side
        assert signal['take_profit_price'] == expected_tp
        assert signal['take_profit_price'] < mock_tick.ask # TP is on the "wrong" side

        point_value = mean_reversion_rsi_strategy.platform_adapter.get_symbol_info(mean_reversion_rsi_strategy.symbol).point
        expected_sl_pips = expected_sl_distance / point_value
        assert signal['sl_pips'] == pytest.approx(expected_sl_pips)
        assert signal['sl_pips'] < 0.0

class TestMeanReversionRSIManagePosition:

    def test_manage_position_max_age_exit_long_rsi(self, mean_reversion_rsi_strategy, rsi_strategy_params_gbpusd_m15):
        max_age = rsi_strategy_params_gbpusd_m15['max_position_age_bars']
        if not max_age:
            pytest.skip("Max position age not configured for RSI strategy")

        # Strategy uses self.timeframe which is M15
        df = create_sample_rsi_market_df(rows=max_age + 10) # Ensure enough bars for M15
        mean_reversion_rsi_strategy.market_data_manager.get_market_data.return_value = df
        
        # Position opened 'max_age' M15 bars ago
        pos_open_dt = df.index[5] # df.index elements are datetime objects

        current_long_pos = Position(
            position_id="RSI_L_AGE_1", symbol="GBPUSD", action=OrderAction.BUY,
            volume=0.1, open_price=1.2800, open_time=pos_open_dt
        )
        
        # Latest bar is the one that makes the position "too old"
        latest_bar_timestamp_for_manage = df.index[5 + max_age]
        latest_bar_data_for_manage = OHLCVData(
            timestamp=latest_bar_timestamp_for_manage,
            symbol="GBPUSD", timeframe=mean_reversion_rsi_strategy.timeframe, # M15
            open=df['open'].iloc[5 + max_age], high=df['high'].iloc[5 + max_age],
            low=df['low'].iloc[5 + max_age], close=df['close'].iloc[5 + max_age],
            volume=df['volume'].iloc[5 + max_age]
        )
        
        mock_tick_at_exit = TickData(timestamp=datetime.now(timezone.utc), symbol="GBPUSD", bid=1.27500, ask=1.27520)

        # Adjust open_time to ensure max_age is met
        time_diff_needed = timedelta(minutes=15 * max_age) # M15 bars
        current_long_pos.open_time = latest_bar_timestamp_for_manage - time_diff_needed

        action = mean_reversion_rsi_strategy.manage_open_position(
            current_long_pos,
            latest_bar=latest_bar_data_for_manage,
            latest_tick=mock_tick_at_exit
        )

        assert action is not None
        assert action['signal'] == StrategySignal.CLOSE_LONG
        assert action['position_id'] == "RSI_L_AGE_1"
        assert f"Max position age ({max_age} bars) reached" in action['comment']

    def test_manage_position_max_age_exit_short_rsi(self, mean_reversion_rsi_strategy, rsi_strategy_params_gbpusd_m15):
        max_age = rsi_strategy_params_gbpusd_m15['max_position_age_bars']
        if not max_age:
            pytest.skip("Max position age not configured for RSI strategy")

        df = create_sample_rsi_market_df(rows=max_age + 10)
        mean_reversion_rsi_strategy.market_data_manager.get_market_data.return_value = df
        
        pos_open_dt = df.index[5]
        current_short_pos = Position(
            position_id="RSI_S_AGE_1", symbol="GBPUSD", action=OrderAction.SELL,
            volume=0.1, open_price=1.2800, open_time=pos_open_dt
        )
        
        latest_bar_timestamp_for_manage = df.index[5 + max_age]
        latest_bar_data_for_manage = OHLCVData(
            timestamp=latest_bar_timestamp_for_manage,
            symbol="GBPUSD", timeframe=mean_reversion_rsi_strategy.timeframe,
            open=df['open'].iloc[5 + max_age], high=df['high'].iloc[5 + max_age],
            low=df['low'].iloc[5 + max_age], close=df['close'].iloc[5 + max_age],
            volume=df['volume'].iloc[5 + max_age]
        )
        mock_tick_at_exit = TickData(timestamp=datetime.now(timezone.utc), symbol="GBPUSD", bid=1.27500, ask=1.27520)
        
        time_diff_needed = timedelta(minutes=15 * max_age) # M15 bars
        current_short_pos.open_time = latest_bar_timestamp_for_manage - time_diff_needed

        action = mean_reversion_rsi_strategy.manage_open_position(
            current_short_pos,
            latest_bar=latest_bar_data_for_manage,
            latest_tick=mock_tick_at_exit
        )

        assert action is not None
        assert action['signal'] == StrategySignal.CLOSE_SHORT
        assert action['position_id'] == "RSI_S_AGE_1"
        assert f"Max position age ({max_age} bars) reached" in action['comment']

    def test_manage_position_max_age_not_reached_rsi(self, mean_reversion_rsi_strategy, rsi_strategy_params_gbpusd_m15):
        max_age = rsi_strategy_params_gbpusd_m15['max_position_age_bars']
        if not max_age: pytest.skip("Max position age not configured")

        df = create_sample_rsi_market_df(rows=max_age + 10)
        mean_reversion_rsi_strategy.market_data_manager.get_market_data.return_value = df # Not used by RSI manage_pos
        
        latest_bar_ts = df.index[10] # An arbitrary recent bar
        # Position opened recently, less than max_age bars ago
        open_time_for_pos = latest_bar_ts - timedelta(minutes=15 * (max_age - 5))


        current_pos = Position(
            position_id="RSI_L_AGE_NR", symbol="GBPUSD", action=OrderAction.BUY,
            volume=0.1, open_price=1.2800, open_time=open_time_for_pos
        )
        latest_bar_data = OHLCVData(
            timestamp=latest_bar_ts, symbol="GBPUSD", timeframe=mean_reversion_rsi_strategy.timeframe,
            open=1.28, high=1.281, low=1.279, close=1.2805
        )
        mock_tick = TickData(timestamp=datetime.now(timezone.utc), symbol="GBPUSD", bid=1.28040, ask=1.28060)
        
        action = mean_reversion_rsi_strategy.manage_open_position(current_pos, latest_bar=latest_bar_data, latest_tick=mock_tick)
        assert action is None

    def test_manage_position_max_age_disabled_rsi(self, mean_reversion_rsi_strategy, rsi_strategy_params_gbpusd_m15):
        mean_reversion_rsi_strategy.max_position_age_bars = None # Disable for this test
        
        df = create_sample_rsi_market_df(rows=50)
        latest_bar_ts = df.index[40]
        open_time_for_pos = df.index[5]

        current_pos = Position(
            position_id="RSI_L_AGE_DIS", symbol="GBPUSD", action=OrderAction.BUY,
            volume=0.1, open_price=1.2800, open_time=open_time_for_pos
        )
        latest_bar_data = OHLCVData(
            timestamp=latest_bar_ts, symbol="GBPUSD", timeframe=mean_reversion_rsi_strategy.timeframe,
            open=1.28, high=1.281, low=1.279, close=1.2805
        )
        mock_tick = TickData(timestamp=datetime.now(timezone.utc), symbol="GBPUSD", bid=1.28040, ask=1.28060)
        action = mean_reversion_rsi_strategy.manage_open_position(current_pos, latest_bar=latest_bar_data, latest_tick=mock_tick)
        assert action is None
        # Restore for other tests
        mean_reversion_rsi_strategy.max_position_age_bars = rsi_strategy_params_gbpusd_m15.get("max_position_age_bars")

    def test_manage_position_no_latest_tick_or_bar_rsi(self, mean_reversion_rsi_strategy):
        current_pos = Position(position_id="RSI_P1", symbol="GBPUSD", action=OrderAction.BUY, volume=0.1, open_price=1.25, open_time=datetime.now(timezone.utc))
        
        action_no_tick = mean_reversion_rsi_strategy.manage_open_position(current_pos, latest_bar=mock.MagicMock(spec=OHLCVData), latest_tick=None)
        assert action_no_tick is None
        
        action_no_bar = mean_reversion_rsi_strategy.manage_open_position(current_pos, latest_bar=None, latest_tick=mock.MagicMock(spec=TickData))
        assert action_no_bar is None
        
        action_no_both = mean_reversion_rsi_strategy.manage_open_position(current_pos, latest_bar=None, latest_tick=None)
        assert action_no_both is None
# salt 2025-06-11T11:27:26
  
