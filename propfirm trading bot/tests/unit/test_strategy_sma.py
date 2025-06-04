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
    adapter.get_symbol_info.return_value = mock_sym_info
    return adapter

@pytest.fixture
def mock_market_data_manager(mocker):
    return mocker.MagicMock(spec=MarketDataManager)

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
        trend_following_sma_strategy.market_data_manager.get_market_data.return_value = pd.DataFrame()
        signal = trend_following_sma_strategy.generate_signal()
        assert signal is None
        trend_following_sma_strategy.logger.debug.assert_any_call(
            f"[{trend_following_sma_strategy.symbol}/{trend_following_sma_strategy.timeframe.name}] Not enough data for SMA crossover strategy (need {trend_following_sma_strategy.slow_sma_period + 1}, got 0)."
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

        trend_following_sma_strategy.market_data_manager.get_market_data.return_value = df
        trend_following_sma_strategy.platform_adapter.get_open_positions.return_value = [] # No active positions
        
        mock_tick = TickData(timestamp=datetime.now(timezone.utc), symbol="EURUSD", bid=1.09090, ask=1.09110)
        trend_following_sma_strategy.market_data_manager.get_latest_tick_data.return_value = mock_tick

        signal = trend_following_sma_strategy.generate_signal()

        assert signal is not None
        assert signal['signal'] == StrategySignal.BUY
        assert signal['price'] == mock_tick.ask # Entry at ask for buy
        expected_sl_distance = df[f'ATR_{atr_p}'].iloc[-1] * atr_mult
        expected_sl = round(mock_tick.ask - expected_sl_distance, 5)
        expected_tp = round(mock_tick.ask + (expected_sl_distance * rr_ratio), 5)
        assert signal['stop_loss_price'] == expected_sl
        assert signal['take_profit_price'] == expected_tp
        assert "SMA Crossover: Buy EURUSD" in signal['comment']
        trend_following_sma_strategy.logger.info.assert_called()

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
        
        trend_following_sma_strategy.market_data_manager.get_market_data.return_value = df
        
        # Simulate active long position
        mock_long_pos = Position(position_id="L1", symbol="EURUSD", action=OrderAction.BUY, volume=0.1, open_price=1.0800, open_time=datetime.now(timezone.utc))
        trend_following_sma_strategy.platform_adapter.get_open_positions.return_value = [mock_long_pos]
        
        mock_tick = TickData(timestamp=datetime.now(timezone.utc), symbol="EURUSD", bid=1.08940, ask=1.08960)
        trend_following_sma_strategy.market_data_manager.get_latest_tick_data.return_value = mock_tick

        signal = trend_following_sma_strategy.generate_signal()

        assert signal is not None
        assert signal['signal'] == StrategySignal.CLOSE_LONG
        assert signal['position_id'] == "L1"
        assert signal['price'] == mock_tick.bid # Close long at bid
        assert "SMA Crossover: Close Long EURUSD" in signal['comment']

    # Add more generate_signal tests:
    # - Bearish crossover for SELL entry (if strategy supports shorting)
    # - Bullish crossover to CLOSE_SHORT (if strategy supports shorting)
    # - No signal if crossover already happened and in position
    # - No signal if no crossover
    # - Handling NaN values in indicators

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
        position_open_time_utc = df.index[5].to_pydatetime() # Example open time (6th bar)
        # Ensure latest_bar's timestamp implies position is older than max_age
        # latest_bar's timestamp should be at least df.index[5 + max_age]
        
        # Simulate a position opened long enough ago
        # The manage_open_position in strategy compares latest_bar.timestamp with position.open_time
        # using df.index.get_indexer and df.index.get_loc
        
        # Let's make position open time correspond to the 5th bar's timestamp in our mock df
        pos_open_dt = df.index[5].to_pydatetime() # Use .to_pydatetime() for direct datetime object

        current_long_pos = Position(
            position_id="L_AGE_1", symbol="EURUSD", action=OrderAction.BUY,
            volume=0.1, open_price=1.0800, open_time=pos_open_dt
        )
        
        # Simulate the latest bar being the one that makes the position "too old"
        # The latest_bar passed to manage_open_position should be the (5 + max_age)-th bar or later
        latest_bar_data_for_manage = OHLCVData(
            timestamp=df.index[5 + max_age].to_pydatetime(), # Timestamp of the (5+max_age)-th bar
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

    # Add more manage_open_position tests:
    # - Trailing stop for SELL position
    # - No trailing stop action if price hasn't moved enough
    # - No action if use_trailing_stop is False
    # - No action if max_position_age_bars is None or not reached
