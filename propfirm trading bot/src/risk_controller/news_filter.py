# prop_firm_trading_bot/src/risk_controller/news_filter.py

import logging
import requests
import json
import time
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, TYPE_CHECKING
import pytz # For timezone handling

if TYPE_CHECKING:
    from prop_firm_trading_bot.src.config_manager import AppConfig, NewsFilterSettings, BotSettings # Added BotSettings

class NewsFilter:
    """
    Manages fetching, parsing, and checking news events to ensure compliance
    with prop firm trading restrictions (e.g., FTMO's +/- 2 minute rule).
    """
    def __init__(self, config: 'AppConfig', logger: logging.Logger):
        self.news_config: 'NewsFilterSettings' = config.news_filter
        self.bot_config: 'BotSettings' = config.bot_settings # For ftmo_server_timezone
        self.logger = logger
        
        self.economic_calendar: List[Dict[str, Any]] = []
        self.last_calendar_fetch_time: Optional[datetime] = None
        # Ensure calendar_fetch_interval_seconds exists in NewsFilterSettings or provide a default
        fetch_interval_seconds = getattr(self.news_config, 'calendar_fetch_interval_seconds', 3600)
        self.calendar_fetch_interval = timedelta(seconds=fetch_interval_seconds)
        
        self.ftmo_timezone = pytz.timezone(self.bot_config.ftmo_server_timezone)
        self.utc_timezone = pytz.utc

        self.currency_to_targeted_instruments: Dict[str, List[str]] = {}
        self._load_instrument_news_mappings(config) 

        if self.news_config.enabled:
            self._fetch_economic_calendar_with_retry()
        else:
            self.logger.info("NewsFilter is disabled in the configuration.")

    def _load_instrument_news_mappings(self, config: 'AppConfig'):
        """
        Populates self.currency_to_targeted_instruments from instrument details
        which should be loaded and accessible via the main AppConfig.
        """
        # This part assumes that the AppConfig object, when passed, has already resolved
        # and loaded the instrument details (e.g., from instruments_ftmo.json).
        # We need a consistent way to access this data.
        # Let's assume `config.resolved_instrument_details` is a dictionary where keys are
        # like "EURUSD_FTMO" (matching instrument_details_key in AssetStrategyProfile)
        # and values are the dictionaries from instruments_ftmo.json.

        # Placeholder: This needs to be connected to how ConfigManager makes instrument data available.
        # For now, we'll try to access a hypothetical attribute.
        # A more robust way would be for ConfigManager to load these into a structured part of AppConfig.
        
        # Attempting to access instrument details if loaded by ConfigManager
        # This is a common pattern: ConfigManager loads all configs and makes them available.
        # We need to define where `instruments_ftmo.json` data lands in `AppConfig`.
        # Let's assume it's `app_config.instrument_data_store` for this example.
        
        instrument_data_store = getattr(config, 'instrument_data_store', None)
        if instrument_data_store and isinstance(instrument_data_store, dict):
            for instrument_key, instrument_details in instrument_data_store.items():
                if isinstance(instrument_details, dict):
                    platform_symbol = instrument_details.get("platform_symbol", instrument_key)
                    target_currencies = instrument_details.get("news_target_currencies", [])
                    if isinstance(target_currencies, list):
                        for currency_code in target_currencies:
                            if isinstance(currency_code, str):
                                self.currency_to_targeted_instruments.setdefault(currency_code.upper(), []).append(platform_symbol)
            if self.currency_to_targeted_instruments:
                 self.logger.info(f"Loaded currency_to_targeted_instruments mapping: {self.currency_to_targeted_instruments}")
            else:
                 self.logger.warning("currency_to_targeted_instruments mapping is empty after attempting to load from instrument data store.")
        else:
            self.logger.warning(
                "Could not load currency_to_targeted_instruments mapping from config.instrument_data_store. "
                "News filter may not identify all targeted instruments correctly. "
                "Ensure instrument details (like from instruments_ftmo.json) are loaded into AppConfig."
            )


    def _get_current_utc_time(self) -> datetime:
        return datetime.now(self.utc_timezone)

    def _fetch_forexfactory_json(self) -> Optional[List[Dict[str, Any]]]:
        url = self.news_config.ff_json_url
        if not url:
            self.logger.error("ForexFactory JSON URL (ff_json_url) not configured.")
            return None
        self.logger.debug(f"Fetching ForexFactory calendar from: {url}")
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            raw_events = response.json()
            
            parsed_calendar = []
            for event in raw_events:
                try:
                    event_title = event.get('title', '')
                    event_country_code = event.get('country', '').upper()
                    event_date_str = event.get('date') 
                    event_impact = event.get('impact', '').lower()

                    if not all([event_title, event_country_code, event_date_str, event_impact]):
                        self.logger.warning(f"Skipping FF event due to missing essential fields: {event_title}")
                        continue
                    
                    event_dt_aware = datetime.fromisoformat(event_date_str)
                    event_dt_utc = event_dt_aware.astimezone(self.utc_timezone)
                                        
                    parsed_calendar.append({
                        'time_utc': event_dt_utc,
                        'currency': event_country_code,
                        'event_name': event_title,
                        'impact': event_impact,
                        'source_provider': 'ForexFactoryJSON'
                    })
                except Exception as e:
                    self.logger.error(f"Error parsing single ForexFactory event '{event.get('title')}': {e}. Event data: {event}", exc_info=True)
            return parsed_calendar
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching ForexFactory calendar from {url}: {e}")
        except json.JSONDecodeError as e:
            self.logger.error(f"Error decoding JSON from ForexFactory calendar at {url}: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error in _fetch_forexfactory_json: {e}", exc_info=True)
        return None

    def _fetch_finnhub_calendar(self) -> Optional[List[Dict[str, Any]]]:
        token = getattr(self.config.Config, 'news_api_key_actual', None) # Access safely
        if not token:
            self.logger.error("Finnhub API key not configured or loaded into AppConfig.Config.news_api_key_actual.")
            return None
            
        base_url = "https://finnhub.io/api/v1/calendar/economic"
        today_str = self._get_current_utc_time().strftime('%Y-%m-%d')
        future_date_str = (self._get_current_utc_time() + timedelta(days=7)).strftime('%Y-%m-%d')
        params = {'token': token, 'from': today_str, 'to': future_date_str}
        self.logger.debug(f"Fetching Finnhub calendar: {base_url} with params (token redacted)")

        try:
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            raw_events = data.get('economicCalendar', [])
            
            parsed_calendar = []
            for event in raw_events:
                try:
                    event_time_str = event.get('time') 
                    if not event_time_str: continue

                    event_dt_utc = datetime.strptime(event_time_str, '%Y-%m-%d %H:%M:%S').replace(tzinfo=self.utc_timezone)
                    
                    parsed_calendar.append({
                        'time_utc': event_dt_utc,
                        'currency': event.get('currency', '').upper(),
                        'event_name': event.get('event', ''),
                        'impact': event.get('impact', '').lower(),
                        'source_provider': 'Finnhub'
                    })
                except Exception as e:
                    self.logger.error(f"Error parsing single Finnhub event '{event.get('event')}': {e}. Event data: {event}", exc_info=True)
            return parsed_calendar
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching Finnhub calendar: {e}")
        except json.JSONDecodeError as e:
            self.logger.error(f"Error decoding JSON from Finnhub: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error in _fetch_finnhub_calendar: {e}", exc_info=True)
        return None

    def _fetch_economic_calendar(self):
        if not self.news_config.enabled:
            self.economic_calendar = []
            return

        provider = self.news_config.api_provider.lower()
        calendar_data: Optional[List[Dict[str, Any]]] = None
        
        if provider == "forexfactoryjson":
            calendar_data = self._fetch_forexfactory_json()
        elif provider == "finnhub":
            calendar_data = self._fetch_finnhub_calendar()
        else:
            self.logger.error(f"Unsupported news API provider configured: {self.news_config.api_provider}")
            self.economic_calendar = []
            return

        if calendar_data is not None:
            self.economic_calendar = calendar_data
            self.last_calendar_fetch_time = self._get_current_utc_time()
            self.logger.info(f"Economic calendar fetched/updated via {provider}. Found {len(self.economic_calendar)} events.")
        else:
            self.logger.warning(f"Failed to fetch economic calendar via {provider}. Using stale data if available (current size: {len(self.economic_calendar)}).")

    def _fetch_economic_calendar_with_retry(self, max_retries=2, delay_seconds=30):
        for attempt in range(max_retries):
            self._fetch_economic_calendar()
            if self.economic_calendar or not self.news_config.enabled : 
                return
            self.logger.warning(f"Calendar fetch attempt {attempt + 1} of {max_retries} failed. Retrying in {delay_seconds}s...")
            time.sleep(delay_seconds)
        self.logger.error(f"Failed to fetch economic calendar after {max_retries} attempts. News filter may be ineffective.")

    def _is_event_high_impact_ftmo_criteria(self, event: Dict[str, Any]) -> bool:
        if not self.news_config.enabled:
            return False

        event_api_impact = str(event.get('impact', '')).lower()
        configured_min_impact = self.news_config.min_impact_to_consider.lower()
        impact_priority = {"low": 1, "medium": 2, "high": 3, "holiday": 0}
        event_priority = impact_priority.get(event_api_impact, 0)
        config_priority = impact_priority.get(configured_min_impact, 3)

        if event_priority >= config_priority:
            self.logger.debug(f"Event '{event.get('event_name')}' matched by API impact '{event_api_impact}' >= configured '{configured_min_impact}'.")
            return True
        
        event_name = str(event.get('event_name', '')).lower()
        for keyword in self.news_config.high_impact_keywords:
            if keyword.lower() in event_name:
                self.logger.debug(f"Event '{event.get('event_name')}' matched by keyword '{keyword}'.")
                return True
        return False

    def get_active_blackout_windows(self, current_time_utc: Optional[datetime] = None) -> List[Dict[str, Any]]:
        if not self.news_config.enabled:
            return []
            
        if current_time_utc is None:
            current_time_utc = self._get_current_utc_time()

        if not self.last_calendar_fetch_time or \
           (current_time_utc - self.last_calendar_fetch_time > self.calendar_fetch_interval):
            self.logger.info("News calendar is stale, attempting refresh before checking blackouts.")
            self._fetch_economic_calendar_with_retry(max_retries=1, delay_seconds=5)

        active_blackouts = []
        if not self.economic_calendar:
            self.logger.warning("Economic calendar is empty. Cannot determine active blackouts.")
            return active_blackouts

        pre_news_delta = timedelta(minutes=self.news_config.pause_minutes_before_news)
        post_news_delta = timedelta(minutes=self.news_config.pause_minutes_after_news)

        for event in self.economic_calendar:
            if not self._is_event_high_impact_ftmo_criteria(event):
                continue
            
            event_dt_utc = event['time_utc']
            if not isinstance(event_dt_utc, datetime):
                self.logger.error(f"Invalid datetime object for event: {event.get('event_name')}")
                continue

            blackout_period_start = event_dt_utc - pre_news_delta
            blackout_period_end = event_dt_utc + post_news_delta

            if blackout_period_start <= current_time_utc < blackout_period_end:
                event_currency = event.get('currency', '').upper()
                targeted_instruments_for_this_event = self.currency_to_targeted_instruments.get(event_currency, [])
                
                if not targeted_instruments_for_this_event:
                    self.logger.debug(f"No specific instruments mapped for news currency {event_currency} for event '{event.get('event_name')}'.")

                for instrument_symbol in targeted_instruments_for_this_event:
                    blackout_info = {
                        'instrument': instrument_symbol,
                        'event_name': event.get('event_name', 'Unknown Event'),
                        'event_currency': event_currency,
                        'event_time_utc': event_dt_utc,
                        'blackout_start_utc': blackout_period_start,
                        'blackout_end_utc': blackout_period_end
                    }
                    active_blackouts.append(blackout_info)
                    self.logger.warning(
                        f"ACTIVE NEWS BLACKOUT: Instrument: {instrument_symbol}, Event: {blackout_info['event_name']} ({event_currency}) "
                        f"at {event_dt_utc.strftime('%Y-%m-%d %H:%M:%S %Z')}. "
                        f"Blackout from {blackout_period_start.strftime('%H:%M:%S %Z')} to {blackout_period_end.strftime('%H:%M:%S %Z')} UTC."
                    )
        return active_blackouts

    def is_instrument_restricted(self, instrument_symbol: str, current_time_utc: Optional[datetime] = None) -> bool:
        if not self.news_config.enabled:
            return False
            
        active_blackouts = self.get_active_blackout_windows(current_time_utc)
        for blackout in active_blackouts:
            if blackout['instrument'] == instrument_symbol:
                return True
        return False
