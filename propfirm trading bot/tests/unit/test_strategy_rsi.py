# This is the tests/unit/test_strategy_rsi.py file.
import pytest
from unittest import mock
import pandas as pd
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
    adapter.get_symbol_info.return_value = mock_sym_info
    return adapter

@pytest.fixture
def mock_market_data_manager_rsi(mocker): # Renamed
    return mocker.MagicMock(spec=MarketDataManager)

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
def basic_bot_settings_rsi():
    return BotSettings(
        trading_mode="paper",
        main_loop_delay_seconds=1,
        app_name="TestRSINewsBot", # Unique app name
        ftmo_server_timezone="Europe/Prague"
    )

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
        mean_reversion_rsi_strategy.market_data_manager.get_market_data.return_value = pd.DataFrame()
        signal = mean_reversion_rsi_strategy.generate_signal()
        assert signal is None
        # Add assertion for logger call if specific message is expected

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

        mean_reversion_rsi_strategy.market_data_manager.get_market_data.return_value = df
        mean_reversion_rsi_strategy.platform_adapter.get_open_positions.return_value = [] # No active positions
        
        mock_tick = TickData(timestamp=datetime.now(timezone.utc), symbol="GBPUSD", bid=1.25990, ask=1.26010)
        mean_reversion_rsi_strategy.market_data_manager.get_latest_tick_data.return_value = mock_tick

        signal = mean_reversion_rsi_strategy.generate_signal()

        assert signal is not None
        assert signal['signal'] == StrategySignal.BUY
        assert signal['price'] == mock_tick.ask
        expected_sl_distance = df[f'ATR_{atr_p}'].iloc[-1] * sl_atr_mult
        expected_tp_distance = df[f'ATR_{atr_p}'].iloc[-1] * tp_atr_mult
        expected_sl = round(mock_tick.ask - expected_sl_distance, 5)
        expected_tp = round(mock_tick.ask + expected_tp_distance, 5)
        assert signal['stop_loss_price'] == expected_sl
        assert signal['take_profit_price'] == expected_tp
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

        mean_reversion_rsi_strategy.market_data_manager.get_market_data.return_value = df
        mean_reversion_rsi_strategy.platform_adapter.get_open_positions.return_value = []

        mock_tick = TickData(timestamp=datetime.now(timezone.utc), symbol="GBPUSD", bid=1.23990, ask=1.24010)
        mean_reversion_rsi_strategy.market_data_manager.get_latest_tick_data.return_value = mock_tick

        signal = mean_reversion_rsi_strategy.generate_signal()

        assert signal is not None
        assert signal['signal'] == StrategySignal.SELL
        assert signal['price'] == mock_tick.bid
        expected_sl_distance = df[f'ATR_{atr_p}'].iloc[-1] * sl_atr_mult
        expected_tp_distance = df[f'ATR_{atr_p}'].iloc[-1] * tp_atr_mult
        expected_sl = round(mock_tick.bid + expected_sl_distance, 5)
        expected_tp = round(mock_tick.bid - expected_tp_distance, 5)
        assert signal['stop_loss_price'] == expected_sl
        assert signal['take_profit_price'] == expected_tp
        assert f"RSI Sell ({df[f'RSI_{rsi_p}'].iloc[-1]:.2f} crossed {overbought_lvl})" in signal['comment']

    def test_generate_signal_rsi_exit_long_on_neutral_cross(self, mean_reversion_rsi_strategy, rsi_strategy_params_gbpusd_m15):
        rsi_p = rsi_strategy_params_gbpusd_m15['rsi_period']
        exit_neutral_high = rsi_strategy_params_gbpusd_m15['exit_rsi_neutral_high']
        
        df = create_sample_rsi_market_df(rsi_period=rsi_p)
        # Simulate RSI crossing up to neutral from below
        df.loc[df.index[-2], f'RSI_{rsi_p}'] = exit_neutral_high - 5 
        df.loc[df.index[-1], f'RSI_{rsi_p}'] = exit_neutral_high + 1 

        mean_reversion_rsi_strategy.market_data_manager.get_market_data.return_value = df
        
        mock_long_pos = Position(position_id="RSI_L1", symbol="GBPUSD", action=OrderAction.BUY, volume=0.1, open_price=1.25000, open_time=datetime.now(timezone.utc))
        mean_reversion_rsi_strategy.platform_adapter.get_open_positions.return_value = [mock_long_pos]
        
        mock_tick = TickData(timestamp=datetime.now(timezone.utc), symbol="GBPUSD", bid=1.25500, ask=1.25520)
        mean_reversion_rsi_strategy.market_data_manager.get_latest_tick_data.return_value = mock_tick

        signal = mean_reversion_rsi_strategy.generate_signal()

        assert signal is not None
        assert signal['signal'] == StrategySignal.CLOSE_LONG
        assert signal['position_id'] == "RSI_L1"
        assert signal['price'] == mock_tick.bid
        assert f"RSI Close Long (RSI crossed neutral {exit_neutral_high})" in signal['comment']

    # Add more generate_signal tests:
    # - Trend filter blocking long/short signals
    # - No signal if RSI conditions not met
    # - NaN values in indicators
    # - RSI exit for short positions

class TestMeanReversionRSIManagePosition:

    def test_manage_position_max_age_exit_rsi(self, mean_reversion_rsi_strategy, rsi_strategy_params_gbpusd_m15):
        max_age = rsi_strategy_params_gbpusd_m15['max_position_age_bars']
        if not max_age:
            pytest.skip("Max position age not configured for RSI strategy")

        # Strategy uses self.timeframe which is M15
        df = create_sample_rsi_market_df(rows=max_age + 10) # Ensure enough bars for M15
        mean_reversion_rsi_strategy.market_data_manager.get_market_data.return_value = df
        
        # Position opened 'max_age' M15 bars ago
        pos_open_dt = df.index[5].to_pydatetime() # Example open time for the 6th bar

        current_short_pos = Position(
            position_id="RSI_S_AGE_1", symbol="GBPUSD", action=OrderAction.SELL,
            volume=0.1, open_price=1.2800, open_time=pos_open_dt
        )
        
        # Latest bar is the one that makes the position "too old"
        latest_bar_timestamp_for_manage = df.index[5 + max_age].to_pydatetime()
        latest_bar_data_for_manage = OHLCVData(
            timestamp=latest_bar_timestamp_for_manage,
            symbol="GBPUSD", timeframe=mean_reversion_rsi_strategy.timeframe, # M15
            open=df['open'].iloc[5 + max_age], high=df['high'].iloc[5 + max_age],
            low=df['low'].iloc[5 + max_age], close=df['close'].iloc[5 + max_age],
            volume=df['volume'].iloc[5 + max_age]
        )
        
        mock_tick_at_exit = TickData(timestamp=datetime.now(timezone.utc), symbol="GBPUSD", bid=1.27500, ask=1.27520)

        # Direct call for testing; Orchestrator would typically handle latest_bar vs position.open_time conversion to bar count.
        # The strategy's manage_open_position itself uses (latest_bar.timestamp - position.open_time).total_seconds()
        # We need to ensure that these timestamps result in bars_open >= max_age
        
        # Re-check the logic for bars_open in the strategy.
        # It does: time_open_seconds = (latest_bar.timestamp - position.open_time).total_seconds()
        # bar_duration_seconds = self.timeframe.to_seconds()
        # bars_open = time_open_seconds / bar_duration_seconds
        # So, the difference between latest_bar.timestamp and position.open_time must be at least max_age * (15*60) seconds.

        # Ensure timestamps reflect this for the test
        time_diff_needed = timedelta(minutes=15 * max_age)
        current_long_pos_adjusted_open_time = Position(
            position_id="RSI_L_AGE_1", symbol="GBPUSD", action=OrderAction.BUY,
            volume=0.1, open_price=1.2800, 
            open_time=latest_bar_timestamp_for_manage - time_diff_needed 
        )


        action = mean_reversion_rsi_strategy.manage_open_position(
            current_long_pos_adjusted_open_time, 
            latest_bar=latest_bar_data_for_manage, 
            latest_tick=mock_tick_at_exit
        )

        assert action is not None
        assert action['signal'] == StrategySignal.CLOSE_LONG # Closing the BUY position
        assert action['position_id'] == "RSI_L_AGE_1"
        assert f"Max position age ({max_age} bars) reached" in action['comment']

    # Add more manage_open_position tests if RSI strategy has other management logic (e.g., dynamic TP based on RSI)