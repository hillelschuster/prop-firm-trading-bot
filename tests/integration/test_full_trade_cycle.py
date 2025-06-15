# This is the tests/integration/test_full_trade_cycle.py file.
import pytest
import os
import time
import logging
from unittest import mock
import pandas as pd
from datetime import datetime, timezone, timedelta

# Imports from your project
from prop_firm_trading_bot.src.orchestrator import Orchestrator
from prop_firm_trading_bot.src.config_manager import AppConfig, PlatformSettings, AssetStrategyProfile, RiskManagementSettings, LoggingSettings, StateManagementSettings, OperationalComplianceSettings, BotSettings, MT5PlatformSettings, NewsFilterSettings, load_and_validate_config, AssetStrategyProfileConfig, BotSettingsConfig, PlatformConfig, LoggingConfig
from prop_firm_trading_bot.src.core.enums import StrategySignal, OrderAction, OrderStatus, PlatformName
from prop_firm_trading_bot.src.core.models import TickData, OHLCVData, SymbolInfo, AccountInfo
# Import specific strategy for testing if needed, or rely on config loading
from prop_firm_trading_bot.src.strategies.trend_following_sma import TrendFollowingSMA 

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - [%(levelname)s] - %(message)s')


@pytest.fixture(scope="module")
def full_cycle_app_config():
    """Loads config, ensure it points to a DEMO MT5 for this test."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    config_dir = os.path.join(project_root, "config")
    
    # CRITICAL: Ensure main_config.yaml is set for a DEMO MT5 account
    # and relevant environment variables are set (MT5_ACCOUNT, etc.)
    if not all(k in os.environ for k in ["MT5_ACCOUNT", "MT5_PASSWORD", "MT5_SERVER"]):
        pytest.skip("MT5 environment variables not set for full trade cycle integration test.")
        
    try:
        config = load_and_validate_config(config_dir=config_dir, main_config_filename="main_config.yaml")
        if config.platform.name != PlatformName.METATRADER5: # Use Enum here
            pytest.skip("Full trade cycle test requires MetaTrader5 platform configuration.")
        # Ensure at least one strategy is enabled for testing, e.g., SMA Trend Following
        # For this test, we might want a specific config that enables only one simple strategy.
        # You might create a 'test_full_cycle_config.yaml' for this.
        # For now, assume main_config.yaml has a suitable strategy enabled.
        profile_keys = [pk for pk, pv in config.asset_strategy_profiles.items() if pv.enabled]
        if not profile_keys:
            pytest.skip("No enabled strategies in config for full trade cycle test.")
        
        # Select a specific profile for the test if needed
        # For example, ensure 'GBPUSD_TrendFollowing_H1' is active and use its params
        # config.assets_to_trade = ["GBPUSD_TrendFollowing_H1"] # Override for test focus

        return config
    except Exception as e:
        pytest.skip(f"Failed to load configuration for full trade cycle test: {e}")
    return None

@pytest.fixture(scope="module")
def orchestrator_instance_live_demo(full_cycle_app_config, request):
    if not full_cycle_app_config:
        pytest.skip("Skipping orchestrator due to config issue.")
        return None

    test_logger = logging.getLogger("OrchestratorIntegrationTest")
    
    # The Orchestrator will connect to the live demo MT5 based on config
    orchestrator = Orchestrator(config=full_cycle_app_config, main_logger=test_logger)
    
    if not orchestrator.platform_adapter or not orchestrator.platform_adapter.is_connected():
         pytest.fail("Orchestrator failed to connect to MT5 demo for full cycle test.")

    def finalizer():
        logger.info("Shutting down orchestrator after full cycle test.")
        if orchestrator:
            orchestrator.stop() # This should trigger shutdown
            # Give some time for threads to close, platform to disconnect
            if orchestrator._periodic_tasks_thread and orchestrator._periodic_tasks_thread.is_alive():
                orchestrator._periodic_tasks_thread.join(timeout=5)
            if orchestrator.platform_adapter and orchestrator.platform_adapter.is_connected():
                 orchestrator.platform_adapter.disconnect() # Ensure disconnected
    request.addfinalizer(finalizer)
    
    return orchestrator


# --- Test Cases ---
# These tests are complex as they involve timing and real market data on a demo account.
# Controlling the exact conditions to trigger specific strategy signals can be challenging.
# One approach is to run the bot for a short period and observe/assert behavior,
# or to have very simple, easily triggerable strategies in a test-specific config.

@pytest.mark.integration_full_cycle
class TestFullTradeCycle:

    # For this test to be effective, the "GBPUSD_TrendFollowing_H1" (or similar)
    # profile in your main_config.yaml should be enabled, and its strategy
    # (TrendFollowingSMA) should be simple enough to trigger with observable market movements.
    TARGET_PROFILE_KEY = "GBPUSD_TrendFollowing_H1" # Example, match your config
    TARGET_SYMBOL = "GBPUSD" # Must match the symbol in TARGET_PROFILE_KEY

    def test_orchestrator_initializes_and_connects(self, orchestrator_instance_live_demo: Orchestrator):
        assert orchestrator_instance_live_demo is not None
        assert orchestrator_instance_live_demo.is_running is False # Run explicitly
        assert orchestrator_instance_live_demo.platform_adapter.is_connected() is True
        assert len(orchestrator_instance_live_demo.strategies) > 0
        logger.info("Orchestrator initialized and connected to MT5 demo.")

    # @pytest.mark.skip(reason="This test is highly dependent on market conditions and strategy; hard to automate reliably without advanced market simulation.")
    def test_signal_generation_and_potential_execution(self, orchestrator_instance_live_demo: Orchestrator):
        """
        This test attempts to run the orchestrator and see if a signal leads to an order.
        It's hard to guarantee a signal will occur in a short test window with live data.
        Consider:
        1. Using a test-specific strategy with very simple, easily met conditions.
        2. Manually manipulating market (if possible on demo) or waiting for specific setup.
        3. Mocking the MarketDataManager to feed controlled data that guarantees a signal.
           (This blurs line with unit tests but can be useful for testing orchestrator logic).
        """
        if self.TARGET_PROFILE_KEY not in orchestrator_instance_live_demo.strategies:
            pytest.skip(f"Target profile {self.TARGET_PROFILE_KEY} not loaded in orchestrator.")
            return

        orchestrator = orchestrator_instance_live_demo
        
        # Mock OrderExecutionManager.execute_trade_signal to capture calls
        # This allows us to verify if a trade was *attempted* without actually placing it,
        # or to verify the parameters if it is placed.
        mock_execute_signal = mock.MagicMock(return_value=None) # Simulate no order or a dummy order
        orchestrator.order_execution_manager.execute_trade_signal = mock_execute_signal
        
        # Mock RiskController.validate_trade_proposal to control approval
        # For this test, let's assume it approves if a signal comes
        mock_validate_proposal = mock.MagicMock(return_value=(True, "Approved by mock", 0.01)) # is_approved, reason, lot_size
        orchestrator.risk_controller.validate_trade_proposal = mock_validate_proposal

        logger.info(f"Starting orchestrator run loop for a short period to test signal generation for {self.TARGET_SYMBOL}...")
        
        # Run the orchestrator's main loop in a separate thread for a short duration
        # Or, more controllably, extract the core logic of one iteration of orchestrator.run()
        # and call it directly. This is safer for testing.

        # Simplified: Call the core processing logic directly once or twice
        # This assumes the main_loop logic is in a callable method or we replicate its essence
        
        # Let's simulate one cycle of the orchestrator's loop
        # This requires internal access or refactoring orchestrator.run()
        # For now, we'll rely on the idea that the orchestrator has a main_trading_loop method
        # or we directly invoke its core steps.
        
        # Option A: Abstract orchestrator's single loop iteration
        # def _run_single_loop_iteration(self): ... in Orchestrator
        # orchestrator._run_single_loop_iteration()

        # Option B: Manually feed data to the specific strategy to force a signal
        # This is more like a unit test for the strategy *within* an integration context
        strategy_to_test = orchestrator.strategies.get(self.TARGET_PROFILE_KEY)
        if not strategy_to_test:
            pytest.fail(f"Strategy for profile {self.TARGET_PROFILE_KEY} not found in orchestrator.")

        # Create data that WILL trigger a signal for TrendFollowingSMA
        # This part is highly strategy-specific and complex to set up correctly.
        fast_p = strategy_to_test.fast_sma_period
        slow_p = strategy_to_test.slow_sma_period
        df = create_triggering_market_df_for_sma_buy(fast_sma_period=fast_p, slow_sma_period=slow_p, atr_period=strategy_to_test.atr_period_for_sl, rows=slow_p + 5)
        df.loc[df.index[-2], f'SMA_{fast_p}'] = 1.0900
        df.loc[df.index[-2], f'SMA_{slow_p}'] = 1.0905
        df.loc[df.index[-1], f'SMA_{fast_p}'] = 1.0910
        df.loc[df.index[-1], f'SMA_{slow_p}'] = 1.0908
        atr_value_for_signal = 0.00200
        df.loc[df.index[-1], f'ATR_{strategy_to_test.atr_period_for_sl}'] = atr_value_for_signal
        
        # Mock MarketDataManager to return this controlled data
        orchestrator.market_data_manager.get_market_data.return_value = df
        # Mock instrument properties for sl_pips calculation within the strategy
        point_value = 0.00001
        # orchestrator.market_data_manager.get_instrument_properties.return_value = {'point': point_value, 'digits': 5} # Method doesn't exist
        # Mock get_symbol_info on the platform_adapter which strategies now use
        mock_symbol_info_for_strat = SymbolInfo(name=self.TARGET_SYMBOL, digits=5, point=point_value, min_volume_lots=0.01, max_volume_lots=100, volume_step_lots=0.01, contract_size=100000, currency_base="GBP", currency_profit="USD", currency_margin="GBP")
        orchestrator.platform_adapter.get_symbol_info.return_value = mock_symbol_info_for_strat

        mock_tick = TickData(timestamp=datetime.now(timezone.utc), symbol=self.TARGET_SYMBOL, bid=1.09090, ask=1.09110)
        orchestrator.market_data_manager.get_latest_tick_data.return_value = mock_tick
        # Ensure platform adapter also returns symbol info
        mock_sym_info_live = SymbolInfo(name=self.TARGET_SYMBOL, digits=5, point=0.00001, min_volume_lots=0.01, max_volume_lots=10,volume_step_lots=0.01, contract_size=100000,currency_base="GBP",currency_profit="USD",currency_margin="GBP")
        orchestrator.platform_adapter.get_symbol_info.return_value = mock_sym_info_live
        
        # Simulate one iteration of the core logic in orchestrator.run() that processes signals
        # This bypasses the time.sleep and main is_running loop for test control.
        # --- Replicating relevant part of orchestrator's loop ---
        orchestrator.risk_controller._check_and_perform_daily_reset() # Ensure daily state is current
        can_continue, _ = orchestrator.risk_controller.check_all_risk_rules()
        if can_continue:
            # The strategy's generate_signal now needs market_data_df, active_position, latest_tick
            # We've already mocked get_market_data and get_latest_tick_data for the orchestrator's MDM
            # We also need to mock get_open_positions for the orchestrator's platform_adapter
            orchestrator.platform_adapter.get_open_positions.return_value = [] # No active position for new entry

            # Call the orchestrator's internal processing steps for the strategy
            # This is a bit of a simplification of the orchestrator's loop for focused testing
            
            # Simulate orchestrator fetching data for the strategy
            market_data_for_strat = orchestrator.market_data_manager.get_market_data(strategy_to_test.symbol, strategy_to_test.timeframe)
            latest_tick_for_strat = orchestrator.market_data_manager.get_latest_tick_data(strategy_to_test.symbol)
            active_pos_for_strat = next(iter(orchestrator.platform_adapter.get_open_positions(symbol=strategy_to_test.symbol) or []), None)

            trade_signal_details = strategy_to_test.generate_signal(
                market_data_df=market_data_for_strat,
                active_position=active_pos_for_strat,
                latest_tick=latest_tick_for_strat
            )
            
            if trade_signal_details and trade_signal_details.get("signal") not in [StrategySignal.HOLD, StrategySignal.NO_SIGNAL]:
                trade_signal_details.setdefault("symbol", strategy_to_test.symbol)
                if trade_signal_details.get("signal") in [StrategySignal.BUY, StrategySignal.SELL]:
                    # sl_pips is now directly from the strategy signal
                    sl_pips_from_signal = trade_signal_details.get("sl_pips")
                    
                    assert sl_pips_from_signal is not None, "Strategy signal must include sl_pips"
                    assert isinstance(sl_pips_from_signal, (float, int)) and sl_pips_from_signal > 0, "sl_pips must be a positive number"

                    # Verify the sl_pips calculation (optional, but good for sanity)
                    # expected_sl_distance_points = atr_value_for_signal * strategy_to_test.atr_multiplier_for_sl
                    # expected_sl_pips_calc = expected_sl_distance_points / point_value
                    # assert sl_pips_from_signal == pytest.approx(expected_sl_pips_calc)

                    is_approved, reason, lot_size = orchestrator.risk_controller.validate_trade_proposal(
                        symbol=strategy_to_test.symbol,
                        action=OrderAction.BUY if trade_signal_details.get("signal") == StrategySignal.BUY else OrderAction.SELL,
                        strategy_type_name=strategy_to_test.__class__.__name__,
                        stop_loss_pips=sl_pips_from_signal,
                        asset_profile_key=self.TARGET_PROFILE_KEY
                        # price_level=trade_signal_details.get("price") # Parameter not in validate_trade_proposal
                    )
                    if is_approved and lot_size and lot_size > 0:
                        symbol_info_for_exec = orchestrator.platform_adapter.get_symbol_info(strategy_to_test.symbol) 
                        orchestrator.order_execution_manager.execute_trade_signal(
                            trade_signal_details=trade_signal_details, 
                            lot_size=lot_size, 
                            symbol_info=symbol_info_for_exec, 
                            asset_profile_key=self.TARGET_PROFILE_KEY
                        )
        # --- End of replicated loop part ---

        time.sleep(2) # Give a moment for any async ops if they existed or for logs

        if mock_validate_proposal.called:
            logger.info("RiskController.validate_trade_proposal was called.")
            if mock_execute_signal.called:
                logger.info("OrderExecutionManager.execute_trade_signal was called.")
                call_args = mock_execute_signal.call_args[0][0] # First argument is trade_signal_details
                assert call_args['signal'] == StrategySignal.BUY # Based on our forced data
                assert call_args['symbol'] == self.TARGET_SYMBOL
            else:
                logger.warning("execute_trade_signal was NOT called, check risk approval or signal.")
        else:
            logger.warning("validate_trade_proposal was NOT called, check signal generation or pre-risk checks.")
            
        assert mock_validate_proposal.called, "RiskController was not asked to validate a proposal."
        assert mock_execute_signal.called, "OrderExecutionManager was not asked to execute a signal."


    @mock.patch('prop_firm_trading_bot.src.orchestrator.OrderExecutionManager')
    @mock.patch('prop_firm_trading_bot.src.orchestrator.RiskController')
    @mock.patch('prop_firm_trading_bot.src.orchestrator.MarketDataManager')
    @mock.patch('prop_firm_trading_bot.src.orchestrator.MT5Adapter') # Patches the class used by Orchestrator
    @mock.patch('prop_firm_trading_bot.src.orchestrator.StateManager')
    def test_orchestrator_signal_to_execution_pipeline(
        self,
        MockStateManager,
        MockMT5Adapter,
        MockMarketDataManager,
        MockRiskController,
        MockOrderExecutionManager
    ):
        """
        Tests the full pipeline from a mocked signal through risk validation
        to order execution, all with mocked components.
        """
        # 1. Setup Mocks
        mock_platform_adapter = MockMT5Adapter.return_value
        mock_platform_adapter.is_connected.return_value = True
        mock_symbol_eurusd = SymbolInfo(
            name="EURUSD", digits=5, point=0.00001, min_volume_lots=0.01,
            max_volume_lots=100, volume_step_lots=0.01, contract_size=100000,
            currency_base="EUR", currency_profit="USD", currency_margin="EUR"
        )
        mock_platform_adapter.get_symbol_info.return_value = mock_symbol_eurusd
        mock_platform_adapter.get_open_positions.return_value = []

        mock_market_data_manager = MockMarketDataManager.return_value
        mock_market_data_manager.get_symbol_info.return_value = mock_symbol_eurusd # Consistent info

        # Prepare controlled market data to trigger TrendFollowingSMA BUY signal
        # Strategy params from config: fast_sma=10, slow_sma=20, atr_period=14, atr_multiplier=2.0
        # sl_to_tp_ratio defaults to 1.5 in TrendFollowingSMA
        fast_sma_period = 10
        slow_sma_period = 20
        atr_period = 14
        atr_multiplier = 2.0
        sl_to_tp_ratio = 1.5 # Default in TrendFollowingSMA
        
        ohlcv_data = create_triggering_market_df_for_sma_buy(
            fast_sma_period=fast_sma_period,
            slow_sma_period=slow_sma_period,
            atr_period=atr_period,
            rows=slow_sma_period + 5 # Ensure enough data for SMAs
        )
        mock_market_data_manager.get_market_data.return_value = ohlcv_data

        # Latest tick data consistent with the BUY signal (price slightly above last close)
        # The strategy will use the ask price for BUY.
        current_ask_price = ohlcv_data['close'].iloc[-1] + 0.00010
        current_bid_price = ohlcv_data['close'].iloc[-1] - 0.00010
        mock_latest_tick = TickData(
            timestamp=datetime.now(timezone.utc), symbol="EURUSD",
            bid=current_bid_price, ask=current_ask_price
        )
        mock_market_data_manager.get_latest_tick_data.return_value = mock_latest_tick

        # Expected values based on the controlled data and strategy logic
        expected_atr_value = ohlcv_data[f'ATR_{atr_period}'].iloc[-1]
        expected_sl_distance_points = expected_atr_value * atr_multiplier
        expected_sl_pips = expected_sl_distance_points / mock_symbol_eurusd.point
        expected_entry_price = current_ask_price
        
        mock_risk_controller = MockRiskController.return_value
        approved_lot_size = 0.01
        mock_risk_controller.validate_trade_proposal.return_value = (True, "Mock Approved", approved_lot_size)
        mock_risk_controller.check_all_risk_rules.return_value = (True, "Mock risk rules pass")
        mock_risk_controller._check_and_perform_daily_reset.return_value = None

        mock_order_execution_manager = MockOrderExecutionManager.return_value
        mock_order_execution_manager.execute_trade_signal.return_value = {"ticket": 12345, "status": OrderStatus.PLACED}

        MockStateManager.return_value # Just to acknowledge it's mocked

        # 2. Create a minimal AppConfig
        profile_key_under_test = "EURUSD_TrendFollowing_H1_Test"
        minimal_config = AppConfig(
            bot_settings=BotSettings(trading_mode="paper", main_loop_delay_seconds=1, app_name="TestBotPipeline", ftmo_server_timezone="Europe/Prague", magic_number_default=789),
            logging=LoggingSettings(level="INFO", directory="logs", file_name_prefix="test_bot_pipeline"),
            platform=PlatformSettings(name=PlatformName.METATRADER5.value, mt5=MT5PlatformSettings(account_env_var="MT5_ACC_TEST", password_env_var="MT5_PASS_TEST", server_env_var="MT5_SERV_TEST")),
            asset_strategy_profiles={
                profile_key_under_test: AssetStrategyProfile(
                    symbol="EURUSD", instrument_details_key="EURUSD_TEST_KEY", strategy_params_key="EURUSD_SMA_TEST_PARAMS_KEY", enabled=True
                )
            },
            strategy_definitions={ # Need a definition for the strategy used
                "SMA_Test_Def": {"strategy_module": "prop_firm_trading_bot.src.strategies.trend_following_sma", "strategy_class": "TrendFollowingSMA"}
            },
            loaded_strategy_parameters={ # Mock loaded params
                 "EURUSD_SMA_TEST_PARAMS_KEY": {
                     "strategy_definition_key": "SMA_Test_Def",
                     "parameters": {"fast_sma_period": 10, "slow_sma_period": 20, "atr_period_for_sl":14, "atr_multiplier_for_sl":2.0, "timeframe": "H1"}
                 }
            },
            risk_management=RiskManagementSettings(global_max_account_drawdown_pct=0.1, global_daily_drawdown_limit_pct=0.05, default_risk_per_trade_idea_pct=0.01),
            operational_compliance=OperationalComplianceSettings(),
            news_filter=NewsFilterSettings(enabled=False),
            state_management=StateManagementSettings(persistence_file="state/test_bot_pipeline_state.json")
        )

        # 3. Initialize Orchestrator
        test_logger = logging.getLogger("TestOrchestratorPipeline")
        orchestrator = Orchestrator(config=minimal_config, main_logger=test_logger)

        # Ensure the mocked strategy instance is correctly initialized and part of orchestrator.strategies
        strategy_instance = orchestrator.strategies.get(profile_key_under_test)
        assert strategy_instance is not None, f"Strategy for profile {profile_key_under_test} not found."
        assert isinstance(strategy_instance, TrendFollowingSMA), "Strategy is not TrendFollowingSMA"
        # Ensure platform adapter is the mocked one
        assert orchestrator.platform_adapter == mock_platform_adapter

        # 4. Simulate one trading cycle for the profile - Replicating the core logic from orchestrator.run()
        # This is a simplified version of what orchestrator.run() does for a single strategy.
        if orchestrator.risk_controller: orchestrator.risk_controller._check_and_perform_daily_reset()
        can_continue_overall, _ = orchestrator.risk_controller.check_all_risk_rules() if orchestrator.risk_controller else (True, "")

        if can_continue_overall and strategy_instance:
            market_data_for_strat = orchestrator.market_data_manager.get_market_data(strategy_instance.symbol, strategy_instance.timeframe)
            latest_tick_for_strat = orchestrator.market_data_manager.get_latest_tick_data(strategy_instance.symbol)
            active_pos_for_strat = next(iter(orchestrator.platform_adapter.get_open_positions(symbol=strategy_instance.symbol) or []), None)

            trade_signal_details = strategy_instance.generate_signal(
                market_data_df=market_data_for_strat,
                active_position=active_pos_for_strat,
                latest_tick=latest_tick_for_strat
            )
            if trade_signal_details and trade_signal_details.get("signal") not in [StrategySignal.HOLD, StrategySignal.NO_SIGNAL]:
                trade_signal_details.setdefault("symbol", strategy_instance.symbol)
                if trade_signal_details.get("signal") in [StrategySignal.BUY, StrategySignal.SELL]:
                    sl_pips_from_signal = trade_signal_details.get("sl_pips")
                    if sl_pips_from_signal and orchestrator.risk_controller:
                        is_approved, reason, lot_size = orchestrator.risk_controller.validate_trade_proposal(
                            symbol=strategy_instance.symbol,
                            action=OrderAction.BUY if trade_signal_details.get("signal") == StrategySignal.BUY else OrderAction.SELL,
                            strategy_type_name=strategy_instance.__class__.__name__,
                            stop_loss_pips=sl_pips_from_signal,
                            asset_profile_key=profile_key_under_test
                        )
                        if is_approved and lot_size and lot_size > 0 and orchestrator.order_execution_manager:
                            symbol_info_for_exec = orchestrator.platform_adapter.get_symbol_info(strategy_instance.symbol)
                            orchestrator.order_execution_manager.execute_trade_signal(
                                trade_signal_details=trade_signal_details,
                                lot_size=lot_size,
                                symbol_info=symbol_info_for_exec,
                                asset_profile_key=profile_key_under_test
                            )
        # 5. Assertions
        # Assert RiskController.validate_trade_proposal was called
        mock_risk_controller.validate_trade_proposal.assert_called_once()
        call_args_risk_pos, call_kwargs_risk = mock_risk_controller.validate_trade_proposal.call_args

        assert call_kwargs_risk['symbol'] == "EURUSD"
        assert call_kwargs_risk['action'] == OrderAction.BUY # Derived from StrategySignal.BUY
        assert call_kwargs_risk['strategy_type_name'] == TrendFollowingSMA.__name__
        assert call_kwargs_risk['stop_loss_pips'] == pytest.approx(expected_sl_pips)
        assert call_kwargs_risk['asset_profile_key'] == profile_key_under_test
        assert call_kwargs_risk['price_level'] == pytest.approx(expected_entry_price)

        # Assert OrderExecutionManager.execute_trade_signal was called
        mock_order_execution_manager.execute_trade_signal.assert_called_once()
        call_args_oem_pos, call_kwargs_oem = mock_order_execution_manager.execute_trade_signal.call_args
        
        # execute_trade_signal(self, trade_signal_details: dict, lot_size: float, symbol_info: SymbolInfo, asset_profile_key: str)
        captured_signal_details = call_kwargs_oem['trade_signal_details']
        assert captured_signal_details['signal'] == StrategySignal.BUY
        assert captured_signal_details['symbol'] == "EURUSD"
        assert captured_signal_details['price'] == pytest.approx(expected_entry_price)
        assert captured_signal_details['sl_pips'] == pytest.approx(expected_sl_pips)
        
        expected_sl_price = expected_entry_price - expected_sl_distance_points
        assert captured_signal_details['stop_loss_price'] == pytest.approx(expected_sl_price)

        expected_tp_distance_points = expected_sl_distance_points * sl_to_tp_ratio
        expected_tp_price = expected_entry_price + expected_tp_distance_points
        assert captured_signal_details['take_profit_price'] == pytest.approx(expected_tp_price)
        # assert captured_signal_details['strategy_name'] == TrendFollowingSMA.__name__ # strategy_name not in signal_details
        
        assert call_kwargs_oem['lot_size'] == approved_lot_size
        assert call_kwargs_oem['symbol_info'] == mock_symbol_eurusd
        assert call_kwargs_oem['asset_profile_key'] == profile_key_under_test

def test_pipeline_risk_rejection(self, mocker):
        """
        Tests that the full pipeline correctly STOPS when the RiskController rejects a trade.
        This test uses mocks for all external and sub-components to isolate the Orchestrator's logic.
        """
        # 1. Setup mocks for all of the Orchestrator's dependencies
        mock_config_manager = mocker.patch('prop_firm_trading_bot.src.orchestrator.ConfigManager')
        mock_app_config = mocker.MagicMock(spec=AppConfig) # Use AppConfig from your models

        # Populate mock_app_config with just enough detail for this test...
        profile_key = "EURUSD_SMA_REJECT_TEST"
        
        # Mocking the structure AppConfig -> Dict[str, AssetStrategyProfileConfig]
        mock_asset_strategy_profile = mocker.MagicMock(spec=AssetStrategyProfileConfig) # Use your actual model
        mock_asset_strategy_profile.symbol = "EURUSD"
        mock_asset_strategy_profile.enabled = True
        mock_asset_strategy_profile.strategy_name = "TrendFollowingSMA" # Ensure this matches a known strategy key
        mock_asset_strategy_profile.strategy_config_path = "config/strategy_sma_eurusd_h1.json" # Or some valid path/mock

        mock_app_config.asset_strategy_profiles = {
            profile_key: mock_asset_strategy_profile
        }
        
        # Mocking BotSettingsConfig
        mock_bot_settings = mocker.MagicMock(spec=BotSettingsConfig) # Use your actual model
        mock_bot_settings.main_loop_delay_seconds = 0.01 # Faster for test
        mock_bot_settings.max_concurrent_strategies = 1
        mock_app_config.bot_settings = mock_bot_settings

        # Mocking PlatformConfig (assuming it's part of AppConfig or needed by a component)
        mock_platform_config = mocker.MagicMock(spec=PlatformConfig) # Use your actual model
        mock_platform_config.platform_name = "MetaTrader5" # Example
        mock_app_config.platform = mock_platform_config
        
        # Mocking LoggingConfig
        mock_logging_config = mocker.MagicMock(spec=LoggingConfig)
        mock_logging_config.log_level = "INFO"
        mock_app_config.logging = mock_logging_config

        mock_config_manager.return_value.get_config.return_value = mock_app_config
        
        # Patching the actual classes used by Orchestrator
        MockMT5Adapter = mocker.patch('prop_firm_trading_bot.src.orchestrator.MT5Adapter')
        MockMarketDataManager = mocker.patch('prop_firm_trading_bot.src.orchestrator.MarketDataManager')
        MockRiskController = mocker.patch('prop_firm_trading_bot.src.orchestrator.RiskController')
        MockOrderExecutionManager = mocker.patch('prop_firm_trading_bot.src.orchestrator.OrderExecutionManager')
        MockStateManager = mocker.patch('prop_firm_trading_bot.src.orchestrator.StateManager')
        MockTrendFollowingSMA = mocker.patch('prop_firm_trading_bot.src.orchestrator.TrendFollowingSMA') # Assuming this is the strategy

        # 2. Configure mock return values
        mock_platform_adapter_instance = MockMT5Adapter.return_value
        mock_market_data_manager_instance = MockMarketDataManager.return_value
        mock_risk_controller_instance = MockRiskController.return_value
        mock_execution_manager_instance = MockOrderExecutionManager.return_value
        mock_state_manager_instance = MockStateManager.return_value
        mock_strategy_instance = MockTrendFollowingSMA.return_value
        
        # Simulate a valid BUY signal from the strategy
        buy_signal = {
            "signal": StrategySignal.BUY, 
            "symbol": "EURUSD", 
            "price": 1.1000, 
            "stop_loss": 1.0980, 
            "take_profit": 1.1050,
            "sl_pips": 20.0 # Ensure sl_pips is present as per recent changes
        }
        mock_strategy_instance.generate_signal.return_value = buy_signal
        mock_strategy_instance.symbol = "EURUSD" # Ensure the mock strategy instance has a symbol attribute
        mock_strategy_instance.config = mock_asset_strategy_profile # Provide config to strategy instance

        # Mock MarketDataManager to provide necessary data for signal generation (even if generate_signal is directly mocked)
        # This might be needed if Orchestrator calls MDM before strategy.generate_signal
        mock_market_data_manager_instance.get_historical_ohlcv.return_value = pd.DataFrame({
            'timestamp': [pd.Timestamp.now(tz='UTC')], 'open': [1.0], 'high': [1.1], 'low': [0.9], 'close': [1.05], 'volume': [100]
        })
        mock_market_data_manager_instance.get_latest_tick_data.return_value = TickData(symbol="EURUSD", timestamp=pd.Timestamp.now(tz='UTC'), bid=1.0999, ask=1.1001)
        
        # Mock PlatformAdapter for any calls made by Orchestrator or its components during the cycle
        mock_platform_adapter_instance.get_open_positions.return_value = [] # No open positions initially
        mock_platform_adapter_instance.get_open_orders.return_value = []
        mock_platform_adapter_instance.get_account_info.return_value = AccountInfo(balance=10000, equity=10000, currency="USD", server_time=datetime.now(timezone.utc)) # Use your AccountInfo model
        mock_platform_adapter_instance.get_symbol_info.return_value = SymbolInfo(symbol="EURUSD", point=0.00001, digits=5, min_volume_lots=0.01, volume_step_lots=0.01) # Use your SymbolInfo model

        # CRITICAL: Mock the RiskController to REJECT the trade
        # validate_trade_proposal should return: (is_approved: bool, reason: str, trade_params: Optional[TradeParams])
        mock_risk_controller_instance.validate_trade_proposal.return_value = (False, "Daily loss limit reached", None)
        # check_all_risk_rules should return: (can_trade_overall: bool, overall_reason: str)
        mock_risk_controller_instance.check_all_risk_rules.return_value = (True, "OK") # Let it pass the initial global check

        # 3. Initialize Orchestrator
        # The Orchestrator's __init__ will call _initialize_modules, which tries to create real instances.
        # We need to ensure that when these are created, our mocks are used.
        # This is typically handled by patching the class *before* Orchestrator is instantiated.
        
        test_logger = logging.getLogger("TestRiskRejectionPipeline")
        # Patch _create_platform_adapter to return our mock
        with mocker.patch.object(Orchestrator, '_create_platform_adapter', return_value=mock_platform_adapter_instance):
            orchestrator = Orchestrator(config_manager=mock_config_manager.return_value, main_logger=test_logger)
        
        # After init, manually assign other top-level mocked components if _initialize_modules was too complex to fully mock
        # or if Orchestrator creates them internally in a way that's hard to intercept during __init__.
        # However, the class-level patches should ideally cover this.
        # If Orchestrator's __init__ creates these, the patches above should ensure it gets the mocks.
        # For safety, we can re-assign if needed, but it's better if the patches work.
        orchestrator.market_data_manager = mock_market_data_manager_instance
        orchestrator.risk_controller = mock_risk_controller_instance
        orchestrator.order_execution_manager = mock_execution_manager_instance
        orchestrator.state_manager = mock_state_manager_instance
        orchestrator.strategies = {profile_key: mock_strategy_instance} # Ensure strategies are correctly populated
        orchestrator.platform_adapter = mock_platform_adapter_instance # Ensure this is the mock

        orchestrator.is_running = True # To allow one loop run

        # 4. Execute one cycle of the orchestrator's main logic
        # We need to call the method that represents one iteration of the main loop.
        # Assuming it's _run_main_loop_iteration or similar.
        # For this test, let's directly call the sequence of operations we expect.
        
        # Simulate the part of the loop that processes strategies
        orchestrator._process_all_strategy_profiles() # This should internally call _run_trading_cycle_for_strategy

        # 5. Assert the outcome
        mock_strategy_instance.generate_signal.assert_called_once()
        mock_risk_controller_instance.validate_trade_proposal.assert_called_once()
        
        # This is the most important assertion:
        mock_execution_manager_instance.execute_trade_signal.assert_not_called()
        
        logger.info("Successfully verified that a rejected trade proposal does not call the execution manager.")

def create_triggering_market_df_for_sma_buy(fast_sma_period, slow_sma_period, atr_period, rows):
    """
    Creates a DataFrame that will trigger a BUY signal for TrendFollowingSMA:
    - SMA_fast[-1] > SMA_slow[-1]
    - SMA_fast[-2] <= SMA_slow[-2] (crossover)
    """
    base_time = datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc) - timedelta(hours=rows)
    data = {
        'open': [1.1000 + (i*0.00001) for i in range(rows)], # Minimal variation for simplicity
        'high': [1.1005 + (i*0.00001) for i in range(rows)],
        'low': [1.0995 + (i*0.00001) for i in range(rows)],
        'close': [1.1000 + (i*0.00001) for i in range(rows)], # Base close prices
        'volume': [100 + i for i in range(rows)]
    }
    index = [base_time + timedelta(hours=i) for i in range(rows)] # H1 timeframe
    df = pd.DataFrame(data, index=pd.Index(index, name="timestamp"))
    df[f'SMA_{fast_sma_period}'] = df['close'].rolling(window=fast_sma_period).mean()
    df[f'SMA_{slow_sma_period}'] = df['close'].rolling(window=slow_sma_period).mean()
    df[f'ATR_{atr_period}'] = 0.00100
    df = df.fillna(method='bfill').fillna(method='ffill') # Fill NaNs from rolling

    # Manipulate last few rows to ensure SMA crossover for BUY
    # Ensure enough data points exist before trying to access iloc[-3], etc.
    if len(df) > slow_sma_period + 2:
        # State before crossover: fast_sma <= slow_sma
        # We'll adjust close prices to force SMAs, then recompute SMAs
        # This is a simplified way; direct SMA manipulation is easier for test control
        df.loc[df.index[-3], f'SMA_{fast_sma_period}'] = 1.10000
        df.loc[df.index[-3], f'SMA_{slow_sma_period}'] = 1.10000
        
        df.loc[df.index[-2], f'SMA_{fast_sma_period}'] = 1.09900 # fast <= slow
        df.loc[df.index[-2], f'SMA_{slow_sma_period}'] = 1.10000
        
        df.loc[df.index[-1], f'SMA_{fast_sma_period}'] = 1.10100 # fast > slow (BUY signal)
        df.loc[df.index[-1], f'SMA_{slow_sma_period}'] = 1.10000
        df.loc[df.index[-1], f'ATR_{atr_period}'] = 0.00250 # Specific ATR for SL calculation
    return df


  
