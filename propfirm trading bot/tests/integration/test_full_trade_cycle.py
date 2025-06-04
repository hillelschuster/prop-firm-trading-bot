# This is the tests/integration/test_full_trade_cycle.py file.
import pytest
import os
import time
import logging
from unittest import mock

# Imports from your project
from prop_firm_trading_bot.src.orchestrator import Orchestrator
from prop_firm_trading_bot.src.config_manager import AppConfig, load_and_validate_config
from prop_firm_trading_bot.src.core.enums import StrategySignal, OrderAction, OrderStatus
from prop_firm_trading_bot.src.core.models import TickData, OHLCVData # For simulating data
# Import specific strategy for testing if needed, or rely on config loading
from prop_firm_trading_bot.src.strategies.trend_following_sma import TrendFollowingSMA 
import pandas as pd # Added for create_sample_market_df
from datetime import datetime, timezone, timedelta # Added for create_sample_market_df
from prop_firm_trading_bot.src.core.models import SymbolInfo # Added for create_sample_market_df

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
        if config.platform.name != "MetaTrader5":
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
        df = create_sample_market_df(fast_sma_period=fast_p, slow_sma_period=slow_p, atr_period=strategy_to_test.atr_period_for_sl, rows=slow_p + 5)
        df.loc[df.index[-2], f'SMA_{fast_p}'] = 1.0900
        df.loc[df.index[-2], f'SMA_{slow_p}'] = 1.0905
        df.loc[df.index[-1], f'SMA_{fast_p}'] = 1.0910
        df.loc[df.index[-1], f'SMA_{slow_p}'] = 1.0908
        df.loc[df.index[-1], f'ATR_{strategy_to_test.atr_period_for_sl}'] = 0.00200
        
        # Mock MarketDataManager to return this controlled data
        orchestrator.market_data_manager.get_market_data.return_value = df
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
            trade_signal_details = strategy_to_test.generate_signal()
            if trade_signal_details and trade_signal_details.get("signal") not in [StrategySignal.HOLD, StrategySignal.NO_SIGNAL]:
                trade_signal_details.setdefault("symbol", strategy_to_test.symbol)
                if trade_signal_details.get("signal") in [StrategySignal.BUY, StrategySignal.SELL]:
                    sl_pips = 0.00200 / 0.00001 # 20 pips if ATR=0.00200 and point=0.00001 for SL sizing
                    is_approved, reason, lot_size = orchestrator.risk_controller.validate_trade_proposal(
                        symbol=strategy_to_test.symbol,
                        action=OrderAction.BUY if trade_signal_details.get("signal") == StrategySignal.BUY else OrderAction.SELL,
                        strategy_type_name=strategy_to_test.__class__.__name__,
                        stop_loss_pips=sl_pips, # Needs to be calculated based on signal's SL price
                        asset_profile_key=self.TARGET_PROFILE_KEY
                    )
                    if is_approved and lot_size and lot_size > 0:
                        symbol_info_for_exec = orchestrator.market_data_manager.get_symbol_info(strategy_to_test.symbol) # Should use platform_adapter
                        orchestrator.order_execution_manager.execute_trade_signal(
                            trade_signal_details, lot_size, symbol_info_for_exec, self.TARGET_PROFILE_KEY
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

# Helper to create DataFrame for the integration test (can be shared or moved to conftest)
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
    df[f'SMA_{fast_sma_period}'] = df['close'].rolling(window=fast_sma_period).mean()
    df[f'SMA_{slow_sma_period}'] = df['close'].rolling(window=slow_sma_period).mean()
    df[f'ATR_{atr_period}'] = 0.00100 
    df = df.fillna(method='bfill')
    return df
