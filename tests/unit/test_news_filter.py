# This is the tests/unit/test_news_filter.py file.
import pytest
from unittest import mock
from datetime import datetime, timedelta, timezone
import logging
import json # For creating mock API responses
import requests

# Imports from your project
from prop_firm_trading_bot.src.risk_controller.news_filter import NewsFilter
from prop_firm_trading_bot.src.config_manager import AppConfig, NewsFilterSettings, BotSettings
# Assuming you might have a simplified way to represent config for tests,
# or you use a more elaborate fixture from conftest.py

# --- Fixtures ---

@pytest.fixture
def mock_logger(mocker):
    """Fixture for a mock logger."""
    return mocker.MagicMock(spec=logging.Logger)

@pytest.fixture
def basic_bot_settings():
    """Basic BotSettings for tests."""
    return BotSettings(
        trading_mode="paper",
        main_loop_delay_seconds=1,
        app_name="TestNewsBot",
        ftmo_server_timezone="Europe/Prague" # Important for news filter context
    )

@pytest.fixture
def default_news_filter_settings():
    """Default NewsFilterSettings for tests."""
    return NewsFilterSettings(
        enabled=True,
        api_provider="ForexFactoryJSON",
        ff_json_url="https://example.com/ff_calendar.json",
        api_key_env_var=None, # Not needed for FFJSON
        min_impact_to_consider="High",
        pause_minutes_before_news=2,
        pause_minutes_after_news=2,
        high_impact_keywords=["NFP", "FOMC", "Interest Rate"],
        calendar_fetch_interval_seconds=3600
    )

@pytest.fixture
def mock_app_config(basic_bot_settings, default_news_filter_settings, mocker):
    """Fixture for a mock AppConfig tailored for NewsFilter."""
    config = mocker.MagicMock(spec=AppConfig)
    config.news_filter = default_news_filter_settings
    config.bot_settings = basic_bot_settings
    
    # Mock instrument_data_store, which NewsFilter uses for mapping
    # currency to targeted instruments.
    config.instrument_data_store = { # Changed from loaded_instrument_details
        "EURUSD_FTMO": {
            "platform_symbol": "EURUSD",
            "news_target_currencies": ["USD", "EUR"]
        },
        "GBPUSD_FTMO": {
            "platform_symbol": "GBPUSD",
            "news_target_currencies": ["USD", "GBP"]
        },
        "US30_FTMO": { # Example Index
            "platform_symbol": "US30.cash",
            "news_target_currencies": ["USD"]
        },
        "AUDCAD_FTMO": {
            "platform_symbol": "AUDCAD",
            "news_target_currencies": ["AUD", "CAD"]
        },
        "USDJPY_FTMO": { # Added for JPY tests
            "platform_symbol": "USDJPY",
            "news_target_currencies": ["USD", "JPY"]
        }
    }
    # Mock the internal 'Config' attribute used for API keys if needed by other providers
    mock_internal_config = mocker.MagicMock()
    mock_internal_config.news_api_key_actual = "dummy_finnhub_key" # For Finnhub tests
    config.Config = mock_internal_config
    
    return config

@pytest.fixture
def news_filter(mock_app_config, mock_logger):
    """Fixture for a NewsFilter instance."""
    # Temporarily disable fetching during NewsFilter initialization for most tests
    # We will explicitly call fetch methods in dedicated fetch tests
    with mock.patch.object(NewsFilter, '_fetch_economic_calendar_with_retry', return_value=None):
        nf = NewsFilter(config=mock_app_config, logger=mock_logger)
    return nf

# --- Test Cases ---

class TestNewsFilterInitialization:
    def test_initialization_enabled(self, mock_app_config, mock_logger):
        mock_app_config.news_filter.enabled = True
        with mock.patch.object(NewsFilter, '_fetch_economic_calendar_with_retry') as mock_fetch:
            nf = NewsFilter(config=mock_app_config, logger=mock_logger)
            mock_fetch.assert_called_once()
        assert nf.news_config.enabled is True
        assert "USD" in nf.currency_to_targeted_instruments
        assert "EURUSD" in nf.currency_to_targeted_instruments["USD"]
        assert "US30.cash" in nf.currency_to_targeted_instruments["USD"]

    def test_initialization_disabled(self, mock_app_config, mock_logger):
        mock_app_config.news_filter.enabled = False
        with mock.patch.object(NewsFilter, '_fetch_economic_calendar_with_retry') as mock_fetch:
            nf = NewsFilter(config=mock_app_config, logger=mock_logger)
            mock_fetch.assert_not_called()
        assert nf.news_config.enabled is False
        assert nf.economic_calendar == []


class TestNewsFetching:
    @mock.patch('prop_firm_trading_bot.src.risk_controller.news_filter.requests.get')
    def test_fetch_forexfactory_json_success(self, mock_requests_get, news_filter, mock_app_config):
        news_filter.news_config.api_provider = "ForexFactoryJSON"
        mock_response = mock.MagicMock()
        mock_response.status_code = 200
        # Sample data mimicking ForexFactory format
        sample_ff_data = [
            {"title": "Non-Farm Payroll", "country": "USD", "date": "2025-06-06T12:30:00Z", "impact": "High"},
            {"title": "ECB Press Conference", "country": "EUR", "date": "2025-06-06T12:45:00Z", "impact": "Medium"},
            {"title": "CAD Employment Change", "country": "CAD", "date": "2025-06-06T12:30:00Z", "impact": "High"}
        ]
        mock_response.json.return_value = sample_ff_data
        mock_requests_get.return_value = mock_response

        news_filter._fetch_economic_calendar() # Call the specific fetch

        assert len(news_filter.economic_calendar) == 3
        assert news_filter.economic_calendar[0]['event_name'] == "Non-Farm Payroll"
        assert news_filter.economic_calendar[0]['currency'] == "USD"
        assert news_filter.economic_calendar[0]['impact'] == "high" # Lowercased
        assert news_filter.economic_calendar[0]['time_utc'] == datetime(2025, 6, 6, 12, 30, 0, tzinfo=timezone.utc)
        mock_requests_get.assert_called_once_with(mock_app_config.news_filter.ff_json_url, timeout=15)
        news_filter.logger.info.assert_any_call("Economic calendar fetched/updated via ForexFactoryJSON. Found 3 events.")

    @mock.patch('prop_firm_trading_bot.src.risk_controller.news_filter.requests.get')
    def test_fetch_forexfactory_json_request_error(self, mock_requests_get, news_filter, mock_logger):
        news_filter.news_config.api_provider = "ForexFactoryJSON"
        mock_requests_get.side_effect = requests.exceptions.RequestException("Connection error")
        
        news_filter._fetch_economic_calendar()
        
        assert news_filter.economic_calendar == [] # Should remain empty or unchanged
        mock_logger.error.assert_any_call(f"Error fetching ForexFactory calendar from {news_filter.news_config.ff_json_url}: Connection error")

    @mock.patch('prop_firm_trading_bot.src.risk_controller.news_filter.requests.get')
    def test_fetch_forexfactory_json_decode_error(self, mock_requests_get, news_filter, mock_logger):
        news_filter.news_config.api_provider = "ForexFactoryJSON"
        mock_response = mock.MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError("Malformed JSON", "doc", 0)
        mock_requests_get.return_value = mock_response
        
        news_filter._fetch_economic_calendar()
        
        assert news_filter.economic_calendar == []
        mock_logger.error.assert_any_call(f"Error decoding JSON from ForexFactory calendar at {news_filter.news_config.ff_json_url}: Malformed JSON (doc:lineno 1:offset 0)")

    @mock.patch('prop_firm_trading_bot.src.risk_controller.news_filter.requests.get')
    def test_fetch_finnhub_calendar_success(self, mock_requests_get, news_filter, mock_app_config, basic_bot_settings):
        # Setup for Finnhub
        news_filter.news_config.api_provider = "Finnhub"
        news_filter.news_config.api_key_env_var = "FINNHUB_API_KEY" # Should trigger use of mock_app_config.Config.news_api_key_actual
        
        mock_response = mock.MagicMock()
        mock_response.status_code = 200
        sample_finnhub_data = {
            "economicCalendar": [
                {"time": "2025-06-06 14:00:00", "currency": "USD", "event": "Fed Chair Speaks", "impact": "high"},
                {"time": "2025-06-07 08:00:00", "currency": "GBP", "event": "Manufacturing PMI", "impact": "medium"}
            ]
        }
        mock_response.json.return_value = sample_finnhub_data
        mock_requests_get.return_value = mock_response

        news_filter._fetch_economic_calendar()

        assert len(news_filter.economic_calendar) == 2
        assert news_filter.economic_calendar[0]['event_name'] == "Fed Chair Speaks"
        assert news_filter.economic_calendar[0]['currency'] == "USD"
        assert news_filter.economic_calendar[0]['impact'] == "high"
        # Finnhub times are typically UTC if not specified otherwise by API docs
        assert news_filter.economic_calendar[0]['time_utc'] == datetime(2025, 6, 6, 14, 0, 0, tzinfo=timezone.utc)
        
        base_url = "https://finnhub.io/api/v1/calendar/economic"
        # Can't easily check params due to datetime.now() in URL construction, but check base_url
        called_url = mock_requests_get.call_args[0][0]
        assert called_url == base_url
        # Check token was passed
        assert mock_requests_get.call_args[1]['params']['token'] == "dummy_finnhub_key"


    @mock.patch('prop_firm_trading_bot.src.risk_controller.news_filter.requests.get')
    def test_fetch_finnhub_calendar_request_error(self, mock_requests_get, news_filter, mock_logger):
        news_filter.news_config.api_provider = "Finnhub"
        mock_requests_get.side_effect = requests.exceptions.RequestException("Connection failed")

        news_filter._fetch_economic_calendar()

        assert news_filter.economic_calendar == []
        mock_logger.error.assert_any_call("Error fetching Finnhub calendar: Connection failed")

    @mock.patch('prop_firm_trading_bot.src.risk_controller.news_filter.requests.get')
    def test_fetch_finnhub_calendar_decode_error(self, mock_requests_get, news_filter, mock_logger):
        news_filter.news_config.api_provider = "Finnhub"
        mock_response = mock.MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError("Bad JSON", "doc", 0)
        mock_requests_get.return_value = mock_response

        news_filter._fetch_economic_calendar()

        assert news_filter.economic_calendar == []
        mock_logger.error.assert_any_call("Error decoding JSON from Finnhub: Bad JSON (doc:lineno 1:offset 0)")

    def test_fetch_finnhub_calendar_no_api_key(self, news_filter, mock_app_config, mock_logger):
        news_filter.news_config.api_provider = "Finnhub"
        # Simulate API key not being available
        mock_app_config.Config.news_api_key_actual = None
        
        news_filter._fetch_economic_calendar()
        
        assert news_filter.economic_calendar == []
        mock_logger.error.assert_called_with("Finnhub API key not configured or loaded into AppConfig.Config.news_api_key_actual.")

    def test_fetch_unsupported_provider(self, news_filter, mock_logger):
        news_filter.news_config.api_provider = "UnknownProvider"
        news_filter._fetch_economic_calendar()
        assert news_filter.economic_calendar == []
        mock_logger.error.assert_called_with("Unsupported news API provider configured: UnknownProvider")

    @mock.patch.object(NewsFilter, '_fetch_economic_calendar')
    @mock.patch('time.sleep', return_value=None) # To speed up test
    def test_fetch_economic_calendar_with_retry_success_on_retry(self, mock_sleep, mock_fetch_calendar, news_filter, mock_logger):
        # Fail first, then succeed
        # mock_fetch_calendar.side_effect = [None, None] # This was causing issues with the re-assignment below
        
        # Simulate _fetch_economic_calendar populating the calendar on the second attempt
        def fetch_side_effect_impl():
            if mock_fetch_calendar.call_count == 1: # First call, do nothing to simulate failure
                pass
            elif mock_fetch_calendar.call_count == 2: # Second call, succeed
                news_filter.economic_calendar = [{"event": "Test"}]
        mock_fetch_calendar.side_effect = fetch_side_effect_impl

        news_filter.economic_calendar = [] # Ensure it's empty initially
        news_filter._fetch_economic_calendar_with_retry(max_retries=2, delay_seconds=1)
        
        assert mock_fetch_calendar.call_count == 2
        assert len(news_filter.economic_calendar) == 1
        mock_logger.warning.assert_called_once() # One warning for the first failure

    @mock.patch.object(NewsFilter, '_fetch_economic_calendar') # Let side_effect handle behavior
    @mock.patch('time.sleep', return_value=None)
    def test_fetch_economic_calendar_with_retry_all_fails(self, mock_sleep, mock_fetch_calendar, news_filter, mock_logger):
        # Simulate _fetch_economic_calendar always failing to populate
        def always_fail_fetch():
            # news_filter.economic_calendar remains unchanged (or empty)
            pass
        mock_fetch_calendar.side_effect = always_fail_fetch

        news_filter.economic_calendar = []
        news_filter._fetch_economic_calendar_with_retry(max_retries=3, delay_seconds=1)
        
        assert mock_fetch_calendar.call_count == 3
        assert news_filter.economic_calendar == []
        mock_logger.error.assert_called_with("Failed to fetch economic calendar after 3 attempts. News filter may be ineffective.")
        assert mock_logger.warning.call_count == 3 # Warnings for each retry


class TestImpactAndRestrictionLogic:
    def test_is_event_high_impact_ftmo_criteria(self, news_filter, mock_app_config):
        # Test by impact level
        event_high = {'impact': 'High', 'event_name': 'Test High Impact'}
        event_medium = {'impact': 'Medium', 'event_name': 'Test Medium Impact'}
        event_low = {'impact': 'Low', 'event_name': 'Test Low Impact'}
        
        mock_app_config.news_filter.min_impact_to_consider = "High"
        assert news_filter._is_event_high_impact_ftmo_criteria(event_high) is True
        assert news_filter._is_event_high_impact_ftmo_criteria(event_medium) is False
        assert news_filter._is_event_high_impact_ftmo_criteria(event_low) is False

        mock_app_config.news_filter.min_impact_to_consider = "Medium"
        assert news_filter._is_event_high_impact_ftmo_criteria(event_high) is True
        assert news_filter._is_event_high_impact_ftmo_criteria(event_medium) is True
        assert news_filter._is_event_high_impact_ftmo_criteria(event_low) is False

        # Test by keyword
        mock_app_config.news_filter.min_impact_to_consider = "High" # Reset
        mock_app_config.news_filter.high_impact_keywords = ["NFP", "FOMC"]
        event_nfp_low_api = {'impact': 'Low', 'event_name': 'Non-Farm Payroll (NFP)'}
        event_fomc_medium_api = {'impact': 'Medium', 'event_name': 'FOMC Statement'}
        event_other_low_api = {'impact': 'Low', 'event_name': 'Regular Speech'}

        assert news_filter._is_event_high_impact_ftmo_criteria(event_nfp_low_api) is True
        assert news_filter._is_event_high_impact_ftmo_criteria(event_fomc_medium_api) is True # True because FOMC keyword
        assert news_filter._is_event_high_impact_ftmo_criteria(event_other_low_api) is False

    def test_is_instrument_restricted(self, news_filter, mock_app_config, mocker):
        # Setup mock calendar
        event_time_utc = datetime(2025, 6, 6, 12, 30, 0, tzinfo=timezone.utc)
        news_filter.economic_calendar = [
            {'time_utc': event_time_utc, 'currency': 'USD', 'event_name': 'NFP', 'impact': 'High'}
        ]
        news_filter.last_calendar_fetch_time = datetime.now(timezone.utc)

        # Config: +/- 2 minutes window
        mock_app_config.news_filter.pause_minutes_before_news = 2
        mock_app_config.news_filter.pause_minutes_after_news = 2
        
        # Case 1: Current time is well before the event window
        current_time_before = event_time_utc - timedelta(minutes=10)
        assert news_filter.is_instrument_restricted("EURUSD", current_time_before) is False
        assert news_filter.is_instrument_restricted("AUDCAD", current_time_before) is False # Not targeted

        # Case 2: Current time is within the pre-news window
        current_time_in_pre_window = event_time_utc - timedelta(minutes=1)
        assert news_filter.is_instrument_restricted("EURUSD", current_time_in_pre_window) is True
        assert news_filter.is_instrument_restricted("US30.cash", current_time_in_pre_window) is True
        assert news_filter.is_instrument_restricted("AUDCAD", current_time_in_pre_window) is False # Not targeted

        # Case 3: Current time is exactly at news time
        current_time_at_event = event_time_utc
        assert news_filter.is_instrument_restricted("EURUSD", current_time_at_event) is True

        # Case 4: Current time is within the post-news window
        current_time_in_post_window = event_time_utc + timedelta(minutes=1)
        assert news_filter.is_instrument_restricted("EURUSD", current_time_in_post_window) is True

        # Case 5: Current time is just outside the post-news window (e.g. 2min 1sec after)
        current_time_after_window = event_time_utc + timedelta(minutes=2, seconds=1)
        assert news_filter.is_instrument_restricted("EURUSD", current_time_after_window) is False
        
        # Case 6: Current time is much after the window
        current_time_well_after_window = event_time_utc + timedelta(minutes=10)
        assert news_filter.is_instrument_restricted("EURUSD", current_time_well_after_window) is False

        # Case 7: News filter disabled
        mock_app_config.news_filter.enabled = False
        assert news_filter.is_instrument_restricted("EURUSD", current_time_in_pre_window) is False
        mock_app_config.news_filter.enabled = True # Reset for other tests

        # Case 8: Empty calendar
        news_filter.economic_calendar = []
        assert news_filter.is_instrument_restricted("EURUSD", current_time_at_event) is False
        news_filter.logger.warning.assert_any_call("Economic calendar is empty. Cannot determine active blackouts.")
        news_filter.economic_calendar = [ # Reset for other tests if needed
            {'time_utc': datetime(2025, 6, 6, 12, 30, 0, tzinfo=timezone.utc), 'currency': 'USD', 'event_name': 'NFP', 'impact': 'High'}
        ]


    def test_is_instrument_restricted_no_relevant_news(self, news_filter, mock_app_config):
        event_time_utc = datetime(2025, 6, 6, 12, 30, 0, tzinfo=timezone.utc)
        # Calendar has an event, but it's low impact and not a keyword
        news_filter.economic_calendar = [
            {'time_utc': event_time_utc, 'currency': 'USD', 'event_name': 'Regular Speech', 'impact': 'Low'}
        ]
        news_filter.last_calendar_fetch_time = datetime.now(timezone.utc)
        mock_app_config.news_filter.min_impact_to_consider = "High"
        mock_app_config.news_filter.high_impact_keywords = ["NFP"]

        current_time_at_event = event_time_utc
        assert news_filter.is_instrument_restricted("EURUSD", current_time_at_event) is False

    def test_is_instrument_restricted_event_for_non_targeted_currency(self, news_filter, mock_app_config):
        event_time_utc = datetime(2025, 6, 6, 12, 30, 0, tzinfo=timezone.utc)
        # High impact event for NZD, but EURUSD does not target NZD
        news_filter.economic_calendar = [
            {'time_utc': event_time_utc, 'currency': 'NZD', 'event_name': 'NZD Rate Decision', 'impact': 'High'}
        ]
        news_filter.last_calendar_fetch_time = datetime.now(timezone.utc)
        # EURUSD targets USD, EUR. See mock_app_config.
        
        current_time_at_event = event_time_utc
        assert news_filter.is_instrument_restricted("EURUSD", current_time_at_event) is False
        # Ensure AUDCAD (targets AUD, CAD) is also not affected
        assert news_filter.is_instrument_restricted("AUDCAD", current_time_at_event) is False


    def test_is_instrument_restricted_low_medium_impact_ignored(self, news_filter, mock_app_config):
        event_time_utc = datetime(2025, 6, 6, 12, 30, 0, tzinfo=timezone.utc)
        news_filter.economic_calendar = [
            {'time_utc': event_time_utc, 'currency': 'USD', 'event_name': 'Medium Impact Event', 'impact': 'Medium'},
            {'time_utc': event_time_utc + timedelta(hours=1), 'currency': 'EUR', 'event_name': 'Low Impact Event', 'impact': 'Low'}
        ]
        news_filter.last_calendar_fetch_time = datetime.now(timezone.utc)
        mock_app_config.news_filter.min_impact_to_consider = "High"
        
        current_time_at_event = event_time_utc
        assert news_filter.is_instrument_restricted("EURUSD", current_time_at_event) is False
        assert news_filter.is_instrument_restricted("EURUSD", current_time_at_event + timedelta(hours=1)) is False

    @mock.patch('prop_firm_trading_bot.src.risk_controller.news_filter.datetime')
    def test_get_active_blackout_windows_timezone_awareness(self, mock_datetime, news_filter, mock_app_config):
        # Test that timezone conversions are handled correctly if FTMO server time != UTC
        # Current time is 10:29 UTC. Event is 12:30 Prague (CET, UTC+1 typically, or CEST UTC+2)
        # Let's assume CEST (UTC+2) for Prague for this example. So event is 10:30 UTC.
        # Pause window +/- 2 mins. Blackout: 10:28 UTC to 10:32 UTC.
        
        # Mock current UTC time
        current_utc_dt = datetime(2025, 6, 6, 10, 29, 0, tzinfo=timezone.utc) # In blackout
        mock_datetime.now.return_value = current_utc_dt
        
        # Event time in Prague (which is UTC+2 for CEST in summer)
        # The NewsFilter converts event times to UTC internally if they come from API with TZ
        # Or assumes UTC and localizes to FTMO TZ for display, but comparisons are UTC.
        # Let's ensure our calendar event time is UTC for consistent testing
        event_utc = datetime(2025, 6, 6, 10, 30, 0, tzinfo=timezone.utc)

        news_filter.economic_calendar = [
            {'time_utc': event_utc, 'currency': 'USD', 'event_name': 'Mock NFP', 'impact': 'High', 'source_provider': 'Test'}
        ]
        news_filter.last_calendar_fetch_time = current_utc_dt # Prevent re-fetch

        mock_app_config.news_filter.pause_minutes_before_news = 2
        mock_app_config.news_filter.pause_minutes_after_news = 2

        active_blackouts = news_filter.get_active_blackout_windows(current_time_utc=current_utc_dt)
        
        assert len(active_blackouts) > 0
        assert any(bo['instrument'] == "EURUSD" for bo in active_blackouts)
        first_blackout = next(bo for bo in active_blackouts if bo['instrument'] == "EURUSD")
        
        expected_blackout_start_utc = event_utc - timedelta(minutes=2)
        expected_blackout_end_utc = event_utc + timedelta(minutes=2)
        
        assert first_blackout['event_time_utc'] == event_utc
        assert first_blackout['blackout_start_utc'] == expected_blackout_start_utc
        assert first_blackout['blackout_end_utc'] == expected_blackout_end_utc
        news_filter.logger.warning.assert_any_call(mock.ANY) # Check for the ACTIVE NEWS BLACKOUT log


class TestCurrencyInstrumentMapping:
    def test_currency_to_instrument_mapping_loaded_correctly(self, mock_app_config, mock_logger):
        # This test relies on the (corrected) mock_app_config fixture
        # where instrument_data_store is populated.
        nf = NewsFilter(config=mock_app_config, logger=mock_logger)
       
        assert "USD" in nf.currency_to_targeted_instruments
        assert "EURUSD" in nf.currency_to_targeted_instruments["USD"]
        assert "GBPUSD" in nf.currency_to_targeted_instruments["USD"]
        assert "US30.cash" in nf.currency_to_targeted_instruments["USD"]
        assert "USDJPY" in nf.currency_to_targeted_instruments["USD"] # From added USDJPY_FTMO

        assert "EUR" in nf.currency_to_targeted_instruments
        assert "EURUSD" in nf.currency_to_targeted_instruments["EUR"]
       
        assert "GBP" in nf.currency_to_targeted_instruments
        assert "GBPUSD" in nf.currency_to_targeted_instruments["GBP"]

        assert "JPY" in nf.currency_to_targeted_instruments
        assert "USDJPY" in nf.currency_to_targeted_instruments["JPY"]

        assert "AUD" in nf.currency_to_targeted_instruments
        assert "AUDCAD" in nf.currency_to_targeted_instruments["AUD"]
        assert "CAD" in nf.currency_to_targeted_instruments
        assert "AUDCAD" in nf.currency_to_targeted_instruments["CAD"]
       
        mock_logger.info.assert_any_call(f"Loaded currency_to_targeted_instruments mapping: {nf.currency_to_targeted_instruments}")

    def test_instrument_restriction_based_on_event_currency(self, news_filter, mock_app_config):
        event_time_utc = datetime(2025, 6, 6, 12, 30, 0, tzinfo=timezone.utc)
        current_time_in_window = event_time_utc # Exactly at event time for simplicity
       
        mock_app_config.news_filter.pause_minutes_before_news = 2
        mock_app_config.news_filter.pause_minutes_after_news = 2
        news_filter.last_calendar_fetch_time = datetime.now(timezone.utc)

        # Scenario 1: EUR news
        news_filter.economic_calendar = [
            {'time_utc': event_time_utc, 'currency': 'EUR', 'event_name': 'ECB Speech', 'impact': 'High'}
        ]
        assert news_filter.is_instrument_restricted("EURUSD", current_time_in_window) is True
        assert news_filter.is_instrument_restricted("GBPUSD", current_time_in_window) is False # GBPUSD not directly targeted by EUR news
        assert news_filter.is_instrument_restricted("USDJPY", current_time_in_window) is False

        # Scenario 2: USD news
        news_filter.economic_calendar = [
            {'time_utc': event_time_utc, 'currency': 'USD', 'event_name': 'US NFP', 'impact': 'High'}
        ]
        assert news_filter.is_instrument_restricted("EURUSD", current_time_in_window) is True
        assert news_filter.is_instrument_restricted("GBPUSD", current_time_in_window) is True
        assert news_filter.is_instrument_restricted("US30.cash", current_time_in_window) is True
        assert news_filter.is_instrument_restricted("USDJPY", current_time_in_window) is True
        assert news_filter.is_instrument_restricted("AUDCAD", current_time_in_window) is False

        # Scenario 3: JPY news
        news_filter.economic_calendar = [
            {'time_utc': event_time_utc, 'currency': 'JPY', 'event_name': 'BoJ Outlook', 'impact': 'High'}
        ]
        assert news_filter.is_instrument_restricted("USDJPY", current_time_in_window) is True
        assert news_filter.is_instrument_restricted("EURUSD", current_time_in_window) is False


class TestConfigurationEffects:
    @mock.patch.object(NewsFilter, '_fetch_economic_calendar_with_retry')
    def test_calendar_refresh_interval(self, mock_fetch_retry, news_filter, mock_app_config, mock_logger):
        # Setup: Calendar is initially populated, last fetch time is old
        base_time = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        news_filter.economic_calendar = [{"event": "Old Event"}] # type: ignore
        news_filter.last_calendar_fetch_time = base_time
       
        # Set fetch interval to 1 hour (3600s)
        mock_app_config.news_filter.calendar_fetch_interval_seconds = 3600
        news_filter.calendar_fetch_interval = timedelta(seconds=3600) # Re-init interval

        # Current time is just past the refresh interval
        current_time_trigger_refresh = base_time + timedelta(seconds=3601)
       
        # Accessing get_active_blackout_windows should trigger refresh
        news_filter.get_active_blackout_windows(current_time_utc=current_time_trigger_refresh)
        mock_logger.info.assert_any_call("News calendar is stale, attempting refresh before checking blackouts.")
        mock_fetch_retry.assert_called_once_with(max_retries=1, delay_seconds=5)
       
        mock_fetch_retry.reset_mock()
       
        # Current time is within refresh interval, should not trigger
        current_time_no_refresh = base_time + timedelta(seconds=1800)
        news_filter.get_active_blackout_windows(current_time_utc=current_time_no_refresh)
        mock_fetch_retry.assert_not_called()

    def test_news_filter_disabled_globally(self, news_filter, mock_app_config):
        mock_app_config.news_filter.enabled = False
        # Re-initialize or directly set on instance for test
        news_filter.news_config.enabled = False

        event_time_utc = datetime(2025, 6, 6, 12, 30, 0, tzinfo=timezone.utc)
        news_filter.economic_calendar = [
            {'time_utc': event_time_utc, 'currency': 'USD', 'event_name': 'NFP', 'impact': 'High'}
        ]
        news_filter.last_calendar_fetch_time = datetime.now(timezone.utc)
        current_time_in_window = event_time_utc

        assert news_filter.is_instrument_restricted("EURUSD", current_time_in_window) is False
        assert news_filter.get_active_blackout_windows(current_time_in_window) == []
        assert news_filter._is_event_high_impact_ftmo_criteria(news_filter.economic_calendar[0]) is False

    # Tests for minutes_before_event, minutes_after_event_starts, impact_levels_to_consider
    # are already well covered in TestImpactAndRestrictionLogic.test_is_instrument_restricted
    # and TestImpactAndRestrictionLogic.test_is_event_high_impact_ftmo_criteria respectively.
    # Test for high_impact_keywords is also in test_is_event_high_impact_ftmo_criteria.


class TestNewsFilterErrorHandlingDuringRefresh:
    """
    Tests how NewsFilter handles errors when the economic calendar refresh is triggered
    internally by methods like get_active_blackout_windows or is_instrument_restricted.
    """
    CURRENT_TIME_UTC = datetime(2025, 7, 1, 12, 0, 0, tzinfo=timezone.utc)

    @mock.patch('prop_firm_trading_bot.src.risk_controller.news_filter.time.sleep')
    @mock.patch('prop_firm_trading_bot.src.risk_controller.news_filter.requests.get')
    def test_get_active_blackout_windows_handles_connection_error_on_refresh(
        self, mock_requests_get, mock_time_sleep, news_filter, mock_app_config, mock_logger
    ):
        news_filter.news_config.api_provider = "ForexFactoryJSON"
        initial_calendar_state = [{"event_name": "Very Old Event", "time_utc": self.CURRENT_TIME_UTC - timedelta(days=10)}]
        news_filter.economic_calendar = list(initial_calendar_state) # Use a copy
        news_filter.last_calendar_fetch_time = None  # Ensures refresh is triggered

        mock_requests_get.side_effect = requests.exceptions.RequestException("Simulated connection error")

        result = news_filter.get_active_blackout_windows(current_time_utc=self.CURRENT_TIME_UTC)

        assert result == []
        mock_requests_get.assert_called_once_with(mock_app_config.news_filter.ff_json_url, timeout=15)
        mock_time_sleep.assert_called_once_with(5) # delay_seconds=5 for refresh retry logic

        mock_logger.error.assert_any_call(f"Error fetching ForexFactory calendar from {mock_app_config.news_filter.ff_json_url}: Simulated connection error")
        mock_logger.warning.assert_any_call(f"Failed to fetch economic calendar via forexfactoryjson. Using stale data if available (current size: 1).")
        mock_logger.warning.assert_any_call("Calendar fetch attempt 1 of 1 failed. Retrying in 5s...")
        mock_logger.error.assert_any_call("Failed to fetch economic calendar after 1 attempts. News filter may be ineffective.")
        # Depending on whether the stale calendar had relevant events, this log might not appear if it's not empty.
        # If it processes the stale (but non-empty) calendar, it might not log "empty".
        # For this test, the crucial part is it returns [] and logs the fetch failure.
        # If initial_calendar_state was empty, then the "Economic calendar is empty" log would be expected.
        
        assert news_filter.economic_calendar == initial_calendar_state # Calendar should remain stale

    @mock.patch('prop_firm_trading_bot.src.risk_controller.news_filter.time.sleep')
    @mock.patch('prop_firm_trading_bot.src.risk_controller.news_filter.requests.get')
    def test_get_active_blackout_windows_handles_json_decode_error_on_refresh(
        self, mock_requests_get, mock_time_sleep, news_filter, mock_app_config, mock_logger
    ):
        news_filter.news_config.api_provider = "ForexFactoryJSON"
        news_filter.economic_calendar = []
        news_filter.last_calendar_fetch_time = None

        mock_response = mock.MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError("Simulated JSON error", "doc", 0)
        mock_requests_get.return_value = mock_response

        result = news_filter.get_active_blackout_windows(current_time_utc=self.CURRENT_TIME_UTC)

        assert result == []
        mock_requests_get.assert_called_once_with(mock_app_config.news_filter.ff_json_url, timeout=15)
        mock_time_sleep.assert_called_once_with(5)

        mock_logger.error.assert_any_call(f"Error decoding JSON from ForexFactory calendar at {mock_app_config.news_filter.ff_json_url}: Simulated JSON error (doc:lineno 1:offset 0)")
        mock_logger.warning.assert_any_call(f"Failed to fetch economic calendar via forexfactoryjson. Using stale data if available (current size: 0).")
        mock_logger.warning.assert_any_call("Calendar fetch attempt 1 of 1 failed. Retrying in 5s...")
        mock_logger.error.assert_any_call("Failed to fetch economic calendar after 1 attempts. News filter may be ineffective.")
        mock_logger.warning.assert_any_call("Economic calendar is empty. Cannot determine active blackouts.")
        
        assert news_filter.economic_calendar == []

    @mock.patch('prop_firm_trading_bot.src.risk_controller.news_filter.time.sleep')
    @mock.patch('prop_firm_trading_bot.src.risk_controller.news_filter.requests.get')
    def test_is_instrument_restricted_handles_connection_error_on_refresh(
        self, mock_requests_get, mock_time_sleep, news_filter, mock_app_config, mock_logger
    ):
        news_filter.news_config.api_provider = "ForexFactoryJSON"
        initial_calendar_state = [{"event_name": "Old Event", "time_utc": self.CURRENT_TIME_UTC - timedelta(days=1)}]
        news_filter.economic_calendar = list(initial_calendar_state)
        news_filter.last_calendar_fetch_time = None

        mock_requests_get.side_effect = requests.exceptions.RequestException("Simulated connection error")

        restricted = news_filter.is_instrument_restricted("EURUSD", current_time_utc=self.CURRENT_TIME_UTC)

        assert restricted is False
        mock_requests_get.assert_called_once_with(mock_app_config.news_filter.ff_json_url, timeout=15)
        mock_time_sleep.assert_called_once_with(5)
        
        # Logging will be similar to get_active_blackout_windows, check key ones
        mock_logger.error.assert_any_call(f"Error fetching ForexFactory calendar from {mock_app_config.news_filter.ff_json_url}: Simulated connection error")
        mock_logger.error.assert_any_call("Failed to fetch economic calendar after 1 attempts. News filter may be ineffective.")
        
        assert news_filter.economic_calendar == initial_calendar_state

    @mock.patch('prop_firm_trading_bot.src.risk_controller.news_filter.time.sleep')
    @mock.patch('prop_firm_trading_bot.src.risk_controller.news_filter.requests.get')
    def test_is_instrument_restricted_handles_json_decode_error_on_refresh(
        self, mock_requests_get, mock_time_sleep, news_filter, mock_app_config, mock_logger
    ):
        news_filter.news_config.api_provider = "ForexFactoryJSON"
        news_filter.economic_calendar = []
        news_filter.last_calendar_fetch_time = None

        mock_response = mock.MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError("Simulated JSON error", "doc", 0)
        mock_requests_get.return_value = mock_response

        restricted = news_filter.is_instrument_restricted("EURUSD", current_time_utc=self.CURRENT_TIME_UTC)

        assert restricted is False
        mock_requests_get.assert_called_once_with(mock_app_config.news_filter.ff_json_url, timeout=15)
        mock_time_sleep.assert_called_once_with(5)

        mock_logger.error.assert_any_call(f"Error decoding JSON from ForexFactory calendar at {mock_app_config.news_filter.ff_json_url}: Simulated JSON error (doc:lineno 1:offset 0)")
        mock_logger.error.assert_any_call("Failed to fetch economic calendar after 1 attempts. News filter may be ineffective.")
        mock_logger.warning.assert_any_call("Economic calendar is empty. Cannot determine active blackouts.") # Because is_instrument_restricted calls get_active_blackout_windows
        
        assert news_filter.economic_calendar == []


class TestNewsEventScenarios:
    """
    Tests the NewsFilter's core logic for identifying restrictions based on various
    news event scenarios, assuming the calendar has been successfully fetched.
    """
    MOCK_NOW_UTC = datetime(2025, 7, 1, 12, 0, 0, tzinfo=timezone.utc)

    def setup_method(self, method):
        """
        Ensure last_calendar_fetch_time is recent for each test to prevent
        unwanted refresh attempts within is_instrument_restricted.
        """
        # This is a common way to ensure fixtures are 'fresh' or configured per-test
        # if the fixture scope is broader (e.g. session, module).
        # However, our news_filter fixture is function-scoped, so it's fresh.
        # We mainly need to control its internal state like economic_calendar.
        pass

    def test_no_relevant_news_events(self, news_filter, mock_app_config):
        news_filter.economic_calendar = []
        news_filter.last_calendar_fetch_time = self.MOCK_NOW_UTC
        
        assert news_filter.is_instrument_restricted("EURUSD", self.MOCK_NOW_UTC) is False
        assert news_filter.get_active_blackout_windows(self.MOCK_NOW_UTC) == []

    def test_high_impact_event_imminent(self, news_filter, mock_app_config):
        # Event at 12:01:00 UTC, current time 12:00:00 UTC. Pause before = 2 mins.
        event_time = self.MOCK_NOW_UTC + timedelta(minutes=1)
        news_filter.economic_calendar = [
            {'time_utc': event_time, 'currency': 'USD', 'event_name': 'NFP', 'impact': 'High'}
        ]
        news_filter.last_calendar_fetch_time = self.MOCK_NOW_UTC
        mock_app_config.news_filter.pause_minutes_before_news = 2
        mock_app_config.news_filter.pause_minutes_after_news = 2

        assert news_filter.is_instrument_restricted("EURUSD", self.MOCK_NOW_UTC) is True
        active_blackouts = news_filter.get_active_blackout_windows(self.MOCK_NOW_UTC)
        assert len(active_blackouts) > 0
        assert any(bo['instrument'] == "EURUSD" for bo in active_blackouts)

    def test_high_impact_event_occurring(self, news_filter, mock_app_config):
        # Event at 12:00:00 UTC, current time 12:00:00 UTC. Pause after = 2 mins.
        event_time = self.MOCK_NOW_UTC
        news_filter.economic_calendar = [
            {'time_utc': event_time, 'currency': 'USD', 'event_name': 'NFP', 'impact': 'High'}
        ]
        news_filter.last_calendar_fetch_time = self.MOCK_NOW_UTC
        mock_app_config.news_filter.pause_minutes_before_news = 2
        mock_app_config.news_filter.pause_minutes_after_news = 2

        assert news_filter.is_instrument_restricted("EURUSD", self.MOCK_NOW_UTC) is True
        active_blackouts = news_filter.get_active_blackout_windows(self.MOCK_NOW_UTC)
        assert len(active_blackouts) > 0
        eurusd_blackout = next(bo for bo in active_blackouts if bo['instrument'] == "EURUSD")
        assert eurusd_blackout['blackout_start_utc'] == event_time - timedelta(minutes=2)
        assert eurusd_blackout['blackout_end_utc'] == event_time + timedelta(minutes=2)


    def test_high_impact_event_just_after_blackout_window(self, news_filter, mock_app_config):
        # Event at 11:57:00 UTC, current time 12:00:00 UTC. Pause after = 2 mins.
        # Blackout ends 11:57 + 2 mins = 11:59. So at 12:00, it should NOT be restricted.
        event_time = self.MOCK_NOW_UTC - timedelta(minutes=3) # Event was 3 mins ago
        news_filter.economic_calendar = [
            {'time_utc': event_time, 'currency': 'USD', 'event_name': 'NFP', 'impact': 'High'}
        ]
        news_filter.last_calendar_fetch_time = self.MOCK_NOW_UTC
        mock_app_config.news_filter.pause_minutes_before_news = 2
        mock_app_config.news_filter.pause_minutes_after_news = 2 # Blackout ends at event_time + 2min

        assert news_filter.is_instrument_restricted("EURUSD", self.MOCK_NOW_UTC) is False
        assert news_filter.get_active_blackout_windows(self.MOCK_NOW_UTC) == []

    def test_high_impact_event_well_before_blackout_window(self, news_filter, mock_app_config):
        # Event at 12:03:00 UTC, current time 12:00:00 UTC. Pause before = 2 mins.
        # Blackout starts 12:03 - 2 mins = 12:01. So at 12:00, it should NOT be restricted.
        event_time = self.MOCK_NOW_UTC + timedelta(minutes=3) # Event is in 3 mins
        news_filter.economic_calendar = [
            {'time_utc': event_time, 'currency': 'USD', 'event_name': 'NFP', 'impact': 'High'}
        ]
        news_filter.last_calendar_fetch_time = self.MOCK_NOW_UTC
        mock_app_config.news_filter.pause_minutes_before_news = 2
        mock_app_config.news_filter.pause_minutes_after_news = 2

        assert news_filter.is_instrument_restricted("EURUSD", self.MOCK_NOW_UTC) is False
        assert news_filter.get_active_blackout_windows(self.MOCK_NOW_UTC) == []

    def test_news_event_for_non_traded_currency(self, news_filter, mock_app_config):
        # High impact event for NZD, current time within hypothetical window
        event_time = self.MOCK_NOW_UTC
        news_filter.economic_calendar = [
            {'time_utc': event_time, 'currency': 'NZD', 'event_name': 'NZD Rate Decision', 'impact': 'High'}
        ]
        news_filter.last_calendar_fetch_time = self.MOCK_NOW_UTC
        # EURUSD targets USD, EUR. GBPUSD targets USD, GBP. AUDCAD targets AUD, CAD.
        # None of these should be affected by NZD news.
        
        assert news_filter.is_instrument_restricted("EURUSD", self.MOCK_NOW_UTC) is False
        assert news_filter.is_instrument_restricted("GBPUSD", self.MOCK_NOW_UTC) is False
        assert news_filter.is_instrument_restricted("AUDCAD", self.MOCK_NOW_UTC) is False
        assert news_filter.get_active_blackout_windows(self.MOCK_NOW_UTC) == []

    def test_low_impact_event_ignored(self, news_filter, mock_app_config):
        event_time = self.MOCK_NOW_UTC
        news_filter.economic_calendar = [
            {'time_utc': event_time, 'currency': 'USD', 'event_name': 'Low Impact Speech', 'impact': 'Low'}
        ]
        news_filter.last_calendar_fetch_time = self.MOCK_NOW_UTC
        mock_app_config.news_filter.min_impact_to_consider = "High" # Explicitly set for clarity

        assert news_filter.is_instrument_restricted("EURUSD", self.MOCK_NOW_UTC) is False
        assert news_filter.get_active_blackout_windows(self.MOCK_NOW_UTC) == []

    def test_medium_impact_event_ignored_when_high_only(self, news_filter, mock_app_config):
        event_time = self.MOCK_NOW_UTC
        news_filter.economic_calendar = [
            {'time_utc': event_time, 'currency': 'USD', 'event_name': 'Medium Impact Data', 'impact': 'Medium'}
        ]
        news_filter.last_calendar_fetch_time = self.MOCK_NOW_UTC
        mock_app_config.news_filter.min_impact_to_consider = "High"

        assert news_filter.is_instrument_restricted("EURUSD", self.MOCK_NOW_UTC) is False
        assert news_filter.get_active_blackout_windows(self.MOCK_NOW_UTC) == []

    def test_medium_impact_event_considered_when_medium_or_lower(self, news_filter, mock_app_config):
        event_time = self.MOCK_NOW_UTC
        news_filter.economic_calendar = [
            {'time_utc': event_time, 'currency': 'USD', 'event_name': 'Medium Impact Data', 'impact': 'Medium'}
        ]
        news_filter.last_calendar_fetch_time = self.MOCK_NOW_UTC
        mock_app_config.news_filter.min_impact_to_consider = "Medium" # Change config for this test

        assert news_filter.is_instrument_restricted("EURUSD", self.MOCK_NOW_UTC) is True
        active_blackouts = news_filter.get_active_blackout_windows(self.MOCK_NOW_UTC)
        assert len(active_blackouts) > 0
        assert any(bo['instrument'] == "EURUSD" for bo in active_blackouts)
        
        # Reset for other tests if mock_app_config is shared or modified directly
        mock_app_config.news_filter.min_impact_to_consider = "High"


  
