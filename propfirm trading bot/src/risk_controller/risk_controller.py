# prop_firm_trading_bot/src/risk_controller/risk_controller.py

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List, Tuple, TYPE_CHECKING
import pytz

from prop_firm_trading_bot.src.core.enums import OrderAction, OrderType, StrategySignal
from prop_firm_trading_bot.src.core.models import AccountInfo, SymbolInfo, Order, Position
# Assuming PlatformInterface is correctly in base_connector
from prop_firm_trading_bot.src.api_connector.base_connector import PlatformInterface
from .news_filter import NewsFilter # Sibling import

if TYPE_CHECKING:
    from prop_firm_trading_bot.src.config_manager import AppConfig, RiskManagementSettings, OperationalComplianceSettings, BotSettings
    from prop_firm_trading_bot.src.data_handler.market_data_manager import MarketDataManager


class RiskController:
    """
    Manages all risk and compliance checks before allowing trade execution.
    This is the central gatekeeper for adhering to prop firm rules.
    """
    def __init__(self, 
                 config: 'AppConfig', 
                 platform_adapter: PlatformInterface, 
                 news_filter: NewsFilter,
                 market_data_manager: 'MarketDataManager', # Needed for pip value / symbol info
                 logger: logging.Logger):
                 
        self.config = config
        self.risk_config: 'RiskManagementSettings' = config.risk_management
        self.compliance_config: 'OperationalComplianceSettings' = config.operational_compliance
        self.bot_settings: 'BotSettings' = config.bot_settings # Added type hint
        
        self.platform_adapter = platform_adapter
        self.news_filter = news_filter
        self.market_data_manager = market_data_manager # Store for symbol info access
        self.logger = logger

        self.ftmo_timezone = pytz.timezone(self.bot_settings.ftmo_server_timezone)
        self.utc_timezone = pytz.utc

        self.initial_challenge_balance: float = 0.0
        self.start_of_day_equity: float = 0.0 # Equity at midnight CE(S)T
        self.date_of_last_daily_reset: Optional[datetime.date] = None
        
        self.daily_order_count: int = 0
        self.daily_modification_count: int = 0 # If tracking modifications separately
        self.last_order_timestamp: float = 0.0
        
        self.open_trades_count: int = 0 # Overall bot open trades
        self.open_trades_per_strategy_type: Dict[str, int] = {} # strategy_type_name -> count

        self.trading_halted_due_to_dd: bool = False
        self.emergency_flatten_triggered_today: bool = False

        # Initialize balances upon creation by fetching current account info
        current_account_info = self.platform_adapter.get_account_info()
        if current_account_info:
            self.set_initial_challenge_balances(current_account_info.balance, current_account_info.equity)
            # Perform initial daily reset check
            self._check_and_perform_daily_reset(force_update=True) 
        else:
            self.logger.critical("RiskController init: Failed to get initial account info. Drawdown limits may be incorrect.")
            # Bot should probably not start if this fails. Orchestrator handles this.

    def set_initial_challenge_balances(self, initial_balance: float, current_equity: float):
        """Sets the absolute initial balance for max drawdown and first day's equity."""
        if self.initial_challenge_balance == 0.0: # Set only once
            self.initial_challenge_balance = initial_balance
            self.logger.info(f"Initial challenge balance set to: {self.initial_challenge_balance:.2f}")
        
        # Set start_of_day_equity for the very first time based on current equity if no reset has happened
        if self.start_of_day_equity == 0.0 and self.date_of_last_daily_reset is None:
            self.start_of_day_equity = current_equity
            self.logger.info(f"Initial start-of-day equity (first run) set to: {self.start_of_day_equity:.2f}")


    def _get_current_ftmo_time(self) -> datetime:
        return datetime.now(self.utc_timezone).astimezone(self.ftmo_timezone)

    def _check_and_perform_daily_reset(self, force_update: bool = False) -> None:
        """Checks if it's a new FTMO trading day (midnight CE(S)T) and resets daily limits."""
        current_ftmo_time = self._get_current_ftmo_time()
        current_ftmo_date = current_ftmo_time.date()

        if force_update or self.date_of_last_daily_reset is None or self.date_of_last_daily_reset < current_ftmo_date:
            account_info = self.platform_adapter.get_account_info()
            if account_info:
                self.start_of_day_equity = account_info.equity # FTMO uses equity at midnight CE(S)T
                self.daily_order_count = 0
                self.daily_modification_count = 0
                self.trading_halted_due_to_dd = False # Reset daily halt specific to drawdown
                self.emergency_flatten_triggered_today = False
                self.date_of_last_daily_reset = current_ftmo_date
                self.logger.info(
                    f"Daily limits reset for FTMO date {current_ftmo_date}. "
                    f"New start-of-day equity: {self.start_of_day_equity:.2f}"
                )
                if self.news_filter.news_config.enabled: # Refresh news calendar daily
                    self.news_filter._fetch_economic_calendar_with_retry()
            else:
                self.logger.error("Failed to get account info during daily reset. Daily limits not updated.")
                # Consider implications: bot might operate on stale daily limits.
                # Potentially, set trading_halted if critical info cannot be fetched.

    def check_all_risk_rules(self, current_equity: Optional[float] = None) -> Tuple[bool, str]:
        """
        Performs all critical risk checks: daily reset, drawdowns.
        Returns: (can_continue_trading, reason_if_halted)
        """
        self._check_and_perform_daily_reset()

        if self.trading_halted_due_to_dd: # If already halted by a previous check this cycle
            return False, "Trading halted due to prior drawdown breach today."

        if current_equity is None:
            account_info = self.platform_adapter.get_account_info()
            if not account_info:
                self.logger.error("CRITICAL: Cannot check drawdown limits - failed to get account info.")
                # This is a severe situation, consider a hard stop or specific alert.
                return False, "Failed to get account info for drawdown check."
            current_equity = account_info.equity
        
        # 1. Maximum Overall Loss Check
        max_loss_equity_level = self.initial_challenge_balance * (1.0 - self.risk_config.global_max_account_drawdown_pct)
        if current_equity <= max_loss_equity_level:
            reason = (f"MAX OVERALL LOSS LIMIT BREACHED: Current Equity {current_equity:.2f} "
                      f"<= Limit Level {max_loss_equity_level:.2f} "
                      f"(Initial: {self.initial_challenge_balance:.2f})")
            self.logger.critical(reason)
            self.trading_halted_due_to_dd = True # Permanent halt for this challenge attempt
            return False, reason

        # 2. Maximum Daily Loss Check
        if self.start_of_day_equity == 0.0 and self.initial_challenge_balance > 0.0: 
             self.logger.warning("Start-of-day equity is zero, but initial challenge balance is set. This might be the first check after init.")

        daily_loss_abs_limit = self.start_of_day_equity * self.risk_config.global_daily_drawdown_limit_pct
        current_daily_loss_amount = self.start_of_day_equity - current_equity

        if current_daily_loss_amount >= daily_loss_abs_limit:
            reason = (f"DAILY LOSS LIMIT BREACHED: Current Equity {current_equity:.2f}, "
                      f"Start-of-Day Equity {self.start_of_day_equity:.2f}, "
                      f"Current Loss {current_daily_loss_amount:.2f} >= Limit Amount {daily_loss_abs_limit:.2f}")
            self.logger.critical(reason)
            self.trading_halted_due_to_dd = True # Halt for the rest of the FTMO day
            return False, reason
            
        return True, "All drawdown rules passed."

    def _get_symbol_properties_for_risk(self, symbol: str) -> Optional[Dict[str, Any]]:
        active_profile = None
        for profile_key, profile_value in self.config.asset_strategy_profiles.items():
            if profile_value.symbol == symbol and profile_value.enabled:
                active_profile = profile_value
                break
        
        if not active_profile:
            self.logger.error(f"No active asset_strategy_profile found for symbol {symbol} in config.")
            return None

        instrument_details = self.config.loaded_instrument_details.get(
            active_profile.instrument_details_key, {}
        )

        if not instrument_details:
            self.logger.error(f"Instrument details for key '{active_profile.instrument_details_key}' (for symbol {symbol}) not found in loaded instrument configurations.")
            live_symbol_info = self.platform_adapter.get_symbol_info(symbol)
            if not live_symbol_info:
                self.logger.error(f"Failed to get live symbol info for {symbol} as fallback.")
                return None
            instrument_details = {
                "pip_value_in_account_currency_per_lot": None, 
                "point_value_in_account_currency_per_lot": None, 
                "digits": live_symbol_info.digits,
                "min_volume_lots": live_symbol_info.min_volume_lots,
                "max_volume_lots": live_symbol_info.max_volume_lots,
                "volume_step_lots": live_symbol_info.volume_step_lots,
                "contract_size": live_symbol_info.contract_size,
                "trade_tick_value": live_symbol_info.platform_specific_details.get("trade_tick_value"),
                "trade_tick_size": live_symbol_info.platform_specific_details.get("trade_tick_size"),
            }
            self.logger.warning(f"Using live symbol info for {symbol} due to missing configured details. Pip/Point value may need manual calculation or config.")

        acc_info = self.platform_adapter.get_account_info()
        if not acc_info:
            self.logger.error("Cannot determine pip value: Account info not available.")
            return None

        pip_val = instrument_details.get("pip_value_in_account_currency_per_lot")
        point_val = instrument_details.get("point_value_in_account_currency_per_lot")

        if pip_val is None:
            digits = instrument_details.get("digits")
            tick_value = instrument_details.get("trade_tick_value")
            tick_size = instrument_details.get("trade_tick_size")

            if pip_val is None and point_val is not None and digits is not None:
                if digits in (5, 3):
                    pip_val = point_val * 10
                elif digits == 2:
                    pip_val = point_val * 100
                else:
                    pip_val = point_val
            elif pip_val is None and tick_value is not None and tick_size is not None and digits is not None:
                pip_size = 10 ** (-digits + 1) if digits >= 3 else 10 ** (-digits)
                pip_val = tick_value * (pip_size / tick_size)
            elif pip_val is None and digits is not None:
                pip_size = 10 ** (-digits + 1) if digits >= 3 else 10 ** (-digits)
                contract_size = instrument_details.get("contract_size", 1)
                pip_val = contract_size * pip_size

            if pip_val is not None:
                instrument_details["pip_value_in_account_currency_per_lot"] = pip_val
            else:
                self.logger.error(f"Cannot calculate pip value for {symbol}: insufficient data.")
                return None

        if instrument_details.get("point_value_in_account_currency_per_lot") is None and pip_val is not None:
            if digits is None:
                symbol_info = self.platform_adapter.get_symbol_info(symbol)
                digits = symbol_info.digits if symbol_info else 0
            if digits in (5, 3):
                instrument_details["point_value_in_account_currency_per_lot"] = pip_val / 10
            elif digits == 2:
                instrument_details["point_value_in_account_currency_per_lot"] = pip_val / 100
            else:
                instrument_details["point_value_in_account_currency_per_lot"] = pip_val

        return instrument_details


    def calculate_compliant_position_size(self, 
                                         symbol: str, 
                                         stop_loss_price_distance_pips: float, 
                                         account_equity: float,
                                         asset_profile_key: str) -> float:
        if stop_loss_price_distance_pips <= 0:
            self.logger.error(f"Invalid stop_loss_pips ({stop_loss_price_distance_pips}) for {symbol}. Must be positive.")
            return 0.0

        symbol_props = self._get_symbol_properties_for_risk(symbol)
        if not symbol_props:
            self.logger.error(f"Cannot calculate position size for {symbol}: Missing symbol properties.")
            return 0.0

        asset_profile = self.config.asset_strategy_profiles.get(asset_profile_key)
        risk_pct = self.risk_config.default_risk_per_trade_idea_pct
        if asset_profile and asset_profile.risk_per_trade_idea_pct is not None:
            risk_pct = asset_profile.risk_per_trade_idea_pct
        
        risk_amount_currency = account_equity * risk_pct
        
        pip_value_per_lot = symbol_props.get("pip_value_in_account_currency_per_lot")
        if pip_value_per_lot is None or pip_value_per_lot <= 0:
            self.logger.error(f"Invalid or missing pip_value_per_lot for {symbol} in instrument config.")
            return 0.0
            
        monetary_value_of_stop_loss_per_lot = stop_loss_price_distance_pips * pip_value_per_lot
        if monetary_value_of_stop_loss_per_lot <= 0:
            self.logger.error(f"Monetary value of stop loss per lot is not positive for {symbol} ({monetary_value_of_stop_loss_per_lot}). Check SL pips and pip value.")
            return 0.0
            
        lot_size = risk_amount_currency / monetary_value_of_stop_loss_per_lot

        min_lot = symbol_props.get("min_volume_lots", 0.01)
        max_lot = symbol_props.get("max_volume_lots", 100.0)
        step_lot = symbol_props.get("volume_step_lots", 0.01)

        if step_lot > 0:
            lot_size = round(lot_size / step_lot) * step_lot
        
        lot_size = max(min_lot, lot_size) 
        lot_size = min(max_lot, lot_size) 
        
        # Prop firm max position size per symbol (from asset_strategy_profile, not main_config directly)
        asset_profile_data = self.config.asset_strategy_profiles.get(asset_profile_key)
        asset_max_pos_size = getattr(asset_profile_data, 'max_position_size_lots', None) # Check if exists
        if asset_max_pos_size is not None: # This field is not in current AssetStrategyProfile model
             self.logger.warning("max_position_size_lots not found in AssetStrategyProfile model, consider adding it.")
             # lot_size = min(lot_size, asset_max_pos_size) # If it were present

        if lot_size < min_lot : 
            self.logger.warning(f"Calculated lot size {lot_size:.4f} for {symbol} is effectively zero or below min_lot {min_lot} after constraints. No trade viable at this risk/SL.")
            return 0.0
            
        # Assuming 'digits_lots' is part of symbol_props or use a default like 2
        lot_digits = int(symbol_props.get("digits_lots", 2))
        self.logger.info(f"Calculated position size for {symbol} ({asset_profile_key}): {lot_size:.{lot_digits}f} lots "
                         f"(Equity: {account_equity:.2f}, Risk%: {risk_pct*100:.2f}%, SL Pips: {stop_loss_price_distance_pips}, "
                         f"Pip Value/Lot: {pip_value_per_lot:.2f}, Risk Amount: {risk_amount_currency:.2f})")
        return round(lot_size, lot_digits)

    def _check_order_frequency_limits(self) -> bool:
        current_time = time.time()
        if (current_time - self.last_order_timestamp) < (1.0 / self.compliance_config.max_orders_per_second):
            self.logger.warning(f"Order Throttling: Attempt to place order too soon. Last order at {self.last_order_timestamp}, current {current_time}.")
            return False
        
        if self.daily_order_count >= self.compliance_config.max_total_orders_per_day:
            self.logger.warning(f"Daily Order Limit Reached: {self.daily_order_count} >= {self.compliance_config.max_total_orders_per_day}.")
            return False
        return True

    def _check_concurrent_trade_limits(self, strategy_type_name: str) -> bool:
        if self.open_trades_count >= self.risk_config.max_total_concurrent_trades:
            self.logger.warning(f"Max Total Concurrent Trades limit reached: {self.open_trades_count} >= {self.risk_config.max_total_concurrent_trades}")
            return False
        
        if self.open_trades_per_strategy_type.get(strategy_type_name, 0) >= self.risk_config.max_concurrent_trades_per_strategy_type:
            self.logger.warning(f"Max Concurrent Trades for strategy type '{strategy_type_name}' reached.")
            return False
        return True

    def validate_trade_proposal(self, 
                               symbol: str, 
                               action: OrderAction, 
                               strategy_type_name: str, 
                               stop_loss_pips: float, 
                               asset_profile_key: str
                               ) -> Tuple[bool, str, Optional[float]]:
        self._check_and_perform_daily_reset()

        if self.trading_halted_due_to_dd:
            return False, "Trading halted: Daily or Max Drawdown Limit previously hit.", None

        can_trade_dd, dd_reason = self.check_all_risk_rules()
        if not can_trade_dd:
            return False, dd_reason, None

        if not self.compliance_config.is_swing_account and self.news_filter.news_config.enabled:
            if self.news_filter.is_instrument_restricted(symbol):
                reason = f"News restriction active for {symbol}."
                self.logger.warning(f"Trade REJECTED for {symbol}: {reason}")
                return False, reason, None
        
        if not self._check_order_frequency_limits():
            return False, "Order frequency or daily order limit reached.", None

        if not self._check_concurrent_trade_limits(strategy_type_name):
            return False, "Concurrent trade limits reached.", None

        account_info = self.platform_adapter.get_account_info()
        if not account_info: return False, "Cannot get account info for position sizing.", None
        
        calculated_lot_size = self.calculate_compliant_position_size(symbol, stop_loss_pips, account_info.equity, asset_profile_key)
        if calculated_lot_size <= 0.0:
            return False, f"Calculated position size for {symbol} is zero or invalid.", None

        self.logger.info(f"Trade proposal for {action.name} {symbol} validated. Lot size: {calculated_lot_size:.4f}")
        return True, "Trade proposal approved by RiskController.", calculated_lot_size


    def record_trade_opened(self, strategy_type_name: str):
        self.last_order_timestamp = time.time()
        self.daily_order_count += 1
        self.open_trades_count += 1
        self.open_trades_per_strategy_type[strategy_type_name] = self.open_trades_per_strategy_type.get(strategy_type_name, 0) + 1
        self.logger.info(f"Trade recorded as opened. Daily orders: {self.daily_order_count}. Active trades: {self.open_trades_count}.")

    def record_trade_closed(self, strategy_type_name: str):
        self.open_trades_count = max(0, self.open_trades_count - 1)
        self.open_trades_per_strategy_type[strategy_type_name] = max(0, self.open_trades_per_strategy_type.get(strategy_type_name, 0) - 1)
        self.logger.info(f"Trade recorded as closed. Active trades: {self.open_trades_count}.")
        
    def record_order_modification(self):
        self.daily_modification_count +=1
        self.logger.info(f"Order modification recorded. Daily modifications: {self.daily_modification_count}")


    def should_enforce_weekend_closure(self) -> bool:
        if self.compliance_config.is_swing_account or not self.compliance_config.enforce_weekend_closure:
            return False
        
        current_ftmo_time = self._get_current_ftmo_time()
        if current_ftmo_time.weekday() == 4: # Friday
            # Ensure these attributes exist or provide defaults
            cutoff_hour = getattr(self.compliance_config, 'weekend_closure_hour_ftmo_time', 21)
            cutoff_minute = getattr(self.compliance_config, 'weekend_closure_minute_ftmo_time', 0)
            if current_ftmo_time.hour >= cutoff_hour and current_ftmo_time.minute >= cutoff_minute:
                self.logger.warning(f"Weekend closure period approaching/active at {current_ftmo_time.strftime('%Y-%m-%d %H:%M:%S %Z')}. Signaling to close positions.")
                return True
        return False

    def check_min_trade_duration(self, position_open_time: datetime) -> bool:
        if self.compliance_config.min_trade_duration_seconds <= 0:
            return True 
        
        current_utc_time = datetime.now(timezone.utc)
        duration_seconds = (current_utc_time - position_open_time.astimezone(self.utc_timezone)).total_seconds()
        
        if duration_seconds < self.compliance_config.min_trade_duration_seconds:
            self.logger.debug(f"Position open for {duration_seconds:.1f}s, minimum is {self.compliance_config.min_trade_duration_seconds}s. Too short to close by strategy yet.")
            return False
        return True
        
    def trigger_emergency_flatten_all(self, reason: str) -> None:
        if self.emergency_flatten_triggered_today:
            self.logger.info("Emergency flatten already triggered today. No new action.")
            return

        self.logger.critical(f"EMERGENCY FLATTEN ALL POSITIONS TRIGGERED. Reason: {reason}")
        self.emergency_flatten_triggered_today = True
        self.trading_halted_due_to_dd = True 

        open_positions = self.platform_adapter.get_open_positions()
        if not open_positions:
            self.logger.info("No open positions to flatten.")
            return

        for pos in open_positions:
            self.logger.info(f"Attempting emergency close of position {pos.position_id} ({pos.symbol} {pos.volume} {pos.action.name})")
            is_restricted_now = False
            if not self.compliance_config.is_swing_account and self.news_filter.news_config.enabled:
                if self.news_filter.is_instrument_restricted(pos.symbol):
                    is_restricted_now = True
                    self.logger.warning(f"Emergency close for {pos.symbol} attempted during news restriction. This may violate rules but is for risk mitigation.")
            self.logger.critical(f"SIGNALING EMERGENCY CLOSE for position {pos.position_id} ({pos.symbol}). News Restricted: {is_restricted_now}")
