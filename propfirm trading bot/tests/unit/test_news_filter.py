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
    
    # Mock loaded_instrument_details, which NewsFilter uses for mapping
    # currency to targeted instruments.
    config.loaded_instrument_details = {
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
        mock_logger.error.assert_any_call(mock.ANY, exc_info=False) # Check for an error log related to fetching

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
