import pytest
import logging
import os
import sys
from unittest import mock
import importlib
import io # Added for StringIO

# Path to the module to be tested
LOGGING_SERVICE_MODULE = "src.logging_service"

# Mocking the config classes structure for simplicity in tests
class MockBotSettings:
    def __init__(self, app_name="TestApp"):
        self.app_name = app_name

class MockLoggingSettings:
    def __init__(self, level="INFO", directory="test_logs",
                 file_name_prefix="test_log", structured_logging=False,
                 json_log_file_name_prefix="test_json_log", backup_count=1,
                 log_format="%(asctime)s - %(name)s - [%(levelname)s] - %(message)s"):
        self.level = level
        self.directory = directory
        self.file_name_prefix = file_name_prefix
        self.structured_logging = structured_logging
        self.json_log_file_name_prefix = json_log_file_name_prefix
        self.backup_count = backup_count
        self.log_format = log_format

class MockAppConfig:
    def __init__(self, bot_settings=None, logging_settings=None):
        self.bot_settings = bot_settings or MockBotSettings()
        self.logging = logging_settings or MockLoggingSettings()

@pytest.fixture(autouse=True)
def reset_logging_state_and_globals():
    """Ensures logging state and module globals are clean for each test."""
    # Reset logging system
    logging.shutdown()
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
    for filter_obj in list(root_logger.filters):
        root_logger.removeFilter(filter_obj)

    for logger_name in list(logging.Logger.manager.loggerDict.keys()):
        logger = logging.getLogger(logger_name)
        if hasattr(logger, 'handlers'):
            for handler in list(logger.handlers): # Iterate over a copy
                logger.removeHandler(handler)
        if hasattr(logger, 'filters'):
            for filter_obj in list(logger.filters):
                logger.removeFilter(filter_obj)
        # Reset level to default if it was changed
        logger.setLevel(logging.NOTSET)


    # Reset the global app name in the logging_service module
    try:
        # Ensure the module is loaded fresh or its state reset
        if LOGGING_SERVICE_MODULE in sys.modules:
            importlib.reload(sys.modules[LOGGING_SERVICE_MODULE])
        module_to_test = importlib.import_module(LOGGING_SERVICE_MODULE)
        # Set to default value as per logging_service.py
        # Pyright might flag this as an unknown attribute, but it's a module-level global.
        module_to_test._APP_NAME_FOR_LOGGING = "PropFirmAlgoBot"
    except ImportError:
        pass # Module might not be loaded yet or path issue

    yield

    # Post-test cleanup, similar to pre-test
    logging.shutdown()
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
    for filter_obj in list(root_logger.filters):
        root_logger.removeFilter(filter_obj)
    for logger_name in list(logging.Logger.manager.loggerDict.keys()):
        logger = logging.getLogger(logger_name)
        if hasattr(logger, 'handlers'):
            for handler in list(logger.handlers):
                logger.removeHandler(handler)
        if hasattr(logger, 'filters'):
            for filter_obj in list(logger.filters):
                logger.removeFilter(filter_obj)
        logger.setLevel(logging.NOTSET)

    if LOGGING_SERVICE_MODULE in sys.modules:
        module_to_test = importlib.import_module(LOGGING_SERVICE_MODULE)
        # Pyright might flag this as an unknown attribute, but it's a module-level global.
        module_to_test._APP_NAME_FOR_LOGGING = "PropFirmAlgoBot"


@pytest.fixture
def mock_app_config_basic():
    return MockAppConfig(
        bot_settings=MockBotSettings(app_name="MyTestApp"),
        logging_settings=MockLoggingSettings(level="DEBUG", directory="logs/test_run")
    )

@pytest.fixture
def mock_app_config_json():
    return MockAppConfig(
        bot_settings=MockBotSettings(app_name="MyJsonTestApp"),
        logging_settings=MockLoggingSettings(
            level="INFO",
            directory="logs/json_test_run",
            structured_logging=True,
            json_log_file_name_prefix="json_test_log",
            log_format="%(asctime)s - %(app_name)s - %(levelname)s - %(name)s - %(message)s" # Include app_name
        )
    )

def test_setup_logging_no_config(caplog):
    from src.logging_service import setup_logging
    
    logger = setup_logging(config=None, logger_name="NoConfigTestLogger")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "NoConfigTestLogger"
    assert logger.level == logging.INFO # As per basicConfig in the function
    assert len(logger.handlers) >= 1

    logger.info("Info message from no_config logger")
    assert "Info message from no_config logger" in caplog.text
    assert caplog.records[0].levelname == "INFO"
    assert caplog.records[0].name == "NoConfigTestLogger"


@mock.patch(f"{LOGGING_SERVICE_MODULE}.os.path.exists", return_value=True)
@mock.patch(f"{LOGGING_SERVICE_MODULE}.os.makedirs")
@mock.patch(f"{LOGGING_SERVICE_MODULE}.logging.handlers.TimedRotatingFileHandler")
def test_setup_logging_with_basic_app_config(mock_file_handler_cls, mock_makedirs, mock_exists, mock_app_config_basic, caplog):
    from src.logging_service import setup_logging, ContextualFilter
    # Must import _APP_NAME_FOR_LOGGING from the module to check its value after setup_logging
    logging_service_module = importlib.import_module(LOGGING_SERVICE_MODULE)

    config = mock_app_config_basic
    logger = setup_logging(config=config)

    assert logging_service_module._APP_NAME_FOR_LOGGING == config.bot_settings.app_name
    assert logger.name == config.bot_settings.app_name
    assert logger.level == logging.DEBUG

    mock_exists.assert_called_with(config.logging.directory)
    mock_makedirs.assert_not_called()

    assert len(logger.handlers) >= 2 # Console and File
    console_handler = next((h for h in logger.handlers if isinstance(h, logging.StreamHandler) and h.stream == sys.stdout), None)
    assert console_handler is not None
    assert console_handler.level == logging.DEBUG
    assert any(isinstance(f, ContextualFilter) for f in console_handler.filters)
    assert isinstance(console_handler.formatter, logging.Formatter)
    assert console_handler.formatter._fmt == config.logging.log_format

    mock_file_handler_cls.assert_called_once()
    args, kwargs = mock_file_handler_cls.call_args
    expected_log_path = os.path.join(config.logging.directory, f"{config.logging.file_name_prefix}.log")
    assert args[0] == expected_log_path
    assert kwargs['when'] == "midnight"
    assert kwargs['backupCount'] == config.logging.backup_count
    
    file_handler_instance = mock_file_handler_cls.return_value
    assert file_handler_instance.setFormatter.called
    file_formatter = file_handler_instance.setFormatter.call_args[0][0]
    assert isinstance(file_formatter, logging.Formatter)
    assert file_formatter._fmt == config.logging.log_format
    assert any(isinstance(f, ContextualFilter) for f in file_handler_instance.filters)

    assert f"Logging setup complete for '{config.bot_settings.app_name}'" in caplog.text
    assert f"File format for '{os.path.basename(expected_log_path)}': Text" in caplog.text


@mock.patch(f"{LOGGING_SERVICE_MODULE}.os.path.exists", return_value=False)
@mock.patch(f"{LOGGING_SERVICE_MODULE}.os.makedirs")
@mock.patch(f"{LOGGING_SERVICE_MODULE}.logging.handlers.TimedRotatingFileHandler")
def test_setup_logging_creates_log_directory(mock_file_handler_cls, mock_makedirs, mock_exists, mock_app_config_basic):
    from src.logging_service import setup_logging
    config = mock_app_config_basic
    setup_logging(config=config)
    
    mock_exists.assert_called_with(config.logging.directory)
    mock_makedirs.assert_called_once_with(config.logging.directory, exist_ok=True)


@mock.patch(f"{LOGGING_SERVICE_MODULE}._JSON_LOGGER_AVAILABLE", True)
@mock.patch(f"{LOGGING_SERVICE_MODULE}.jsonlogger.JsonFormatter")
@mock.patch(f"{LOGGING_SERVICE_MODULE}.os.path.exists", return_value=True)
@mock.patch(f"{LOGGING_SERVICE_MODULE}.logging.handlers.TimedRotatingFileHandler")
def test_setup_logging_json_enabled_and_available(mock_file_handler_cls, mock_exists, mock_json_formatter_cls, mock_json_available_patch, mock_app_config_json, caplog):
    from src.logging_service import setup_logging
    config = mock_app_config_json
    
    logger = setup_logging(config=config)

    file_handler_instance = mock_file_handler_cls.return_value
    mock_json_formatter_cls.assert_called_once()
    expected_json_format = "%(asctime)s %(levelname)s %(name)s %(module)s %(funcName)s %(lineno)d %(app_name)s %(message)s"
    assert mock_json_formatter_cls.call_args[0][0] == expected_json_format
    
    file_formatter = file_handler_instance.setFormatter.call_args[0][0]
    assert file_formatter == mock_json_formatter_cls.return_value

    expected_log_path = os.path.join(config.logging.directory, f"{config.logging.json_log_file_name_prefix}.json.log")
    assert f"File format for '{os.path.basename(expected_log_path)}': JSON" in caplog.text


@mock.patch(f"{LOGGING_SERVICE_MODULE}._JSON_LOGGER_AVAILABLE", False)
@mock.patch(f"{LOGGING_SERVICE_MODULE}.jsonlogger.JsonFormatter")
@mock.patch(f"{LOGGING_SERVICE_MODULE}.os.path.exists", return_value=True)
@mock.patch(f"{LOGGING_SERVICE_MODULE}.logging.handlers.TimedRotatingFileHandler")
def test_setup_logging_json_enabled_but_unavailable(mock_file_handler_cls, mock_exists, mock_json_formatter_cls, mock_json_unavailable_patch, mock_app_config_json, caplog):
    from src.logging_service import setup_logging
    config = mock_app_config_json
    
    logger = setup_logging(config=config)

    mock_json_formatter_cls.assert_not_called()

    file_handler_instance = mock_file_handler_cls.return_value
    file_formatter = file_handler_instance.setFormatter.call_args[0][0]
    assert isinstance(file_formatter, logging.Formatter)
    assert file_formatter._fmt == config.logging.log_format

    assert "Structured JSON logging was requested, but 'python-json-logger' library is not installed." in caplog.text
    expected_log_path = os.path.join(config.logging.directory, f"{config.logging.file_name_prefix}.log")
    assert f"File format for '{os.path.basename(expected_log_path)}': Text" in caplog.text


@mock.patch(f"{LOGGING_SERVICE_MODULE}.os.path.exists", return_value=True)
@mock.patch(f"{LOGGING_SERVICE_MODULE}.logging.handlers.TimedRotatingFileHandler")
def test_logging_messages_and_contextual_filter(mock_file_handler_cls, mock_exists, mock_app_config_json, caplog):
    # Using mock_app_config_json as its log_format includes %(app_name)s
    from src.logging_service import setup_logging
    logging_service_module = importlib.import_module(LOGGING_SERVICE_MODULE)
    
    config = mock_app_config_json # app_name="MyJsonTestApp"
    logger = setup_logging(config=config)
    
    assert logging_service_module._APP_NAME_FOR_LOGGING == "MyJsonTestApp"

    logger.info("Hello from test app", extra={"custom_field": "value1"})
    
    # Find the "Hello from test app" record
    test_message_record = next((r for r in caplog.records if r.message == "Hello from test app"), None)
    
    assert test_message_record is not None
    assert test_message_record.levelname == "INFO"
    assert test_message_record.name == config.bot_settings.app_name
    
    # ContextualFilter adds 'app_name'. Use getattr for safer access if linter complains.
    assert getattr(test_message_record, 'app_name', None) == config.bot_settings.app_name
    assert getattr(test_message_record, 'custom_field', None) == "value1"

    # Check formatted output from console handler (captured by caplog)
    # Format: "%(asctime)s - %(app_name)s - [%(levelname)s] - %(name)s - %(message)s"
    # Example: "timestamp - MyJsonTestApp - [INFO] - MyJsonTestApp - Hello from test app"
    assert f"{config.bot_settings.app_name} - [INFO] - {config.bot_settings.app_name} - Hello from test app" in caplog.text


@mock.patch(f"{LOGGING_SERVICE_MODULE}.os.path.exists", return_value=True)
@mock.patch(f"{LOGGING_SERVICE_MODULE}.logging.handlers.TimedRotatingFileHandler")
def test_invalid_log_level_defaults_to_info(mock_file_handler_cls, mock_exists, mock_app_config_basic, caplog):
    from src.logging_service import setup_logging
    
    config = mock_app_config_basic
    config.logging.level = "INVALID_LEVEL"
    
    logger = setup_logging(config=config)
    
    assert f"Invalid log level: {config.logging.level}. Defaulting to INFO." in caplog.text
    assert logger.level == logging.INFO


@mock.patch(f"{LOGGING_SERVICE_MODULE}.os.path.exists", return_value=True)
@mock.patch(f"{LOGGING_SERVICE_MODULE}.logging.handlers.TimedRotatingFileHandler")
def test_logger_reconfiguration_clears_old_handlers(mock_trfh, mock_exists, mock_app_config_basic):
    from src.logging_service import setup_logging
    
    config1 = mock_app_config_basic
    logger = setup_logging(config=config1, logger_name="ReusableLogger")
    initial_handler_count = len(logger.handlers) # e.g., 2 (console, file)
    assert initial_handler_count >= 2

    config2 = MockAppConfig(
        bot_settings=MockBotSettings(app_name="ReusableLoggerApp"),
        logging_settings=MockLoggingSettings(level="WARNING", directory="logs/test_run2")
    )
    
    logger_reconfigured = setup_logging(config=config2, logger_name="ReusableLogger") 
    
    assert logger is logger_reconfigured
    assert len(logger_reconfigured.handlers) == initial_handler_count 
    assert mock_trfh.call_count == 2
    assert logger_reconfigured.level == logging.WARNING


def test_contextual_filter_direct():
    from src.logging_service import ContextualFilter
    logging_service_module = importlib.import_module(LOGGING_SERVICE_MODULE)

    test_app_name = "FilterDirectTestApp"
    # Patch the global directly for this specific test of the filter's behavior
    with mock.patch.object(logging_service_module, '_APP_NAME_FOR_LOGGING', test_app_name):
        context_filter = ContextualFilter()
        mock_record = logging.LogRecord(
            name="test_logger", level=logging.INFO, pathname="test.py", lineno=10,
            msg="Test message", args=(), exc_info=None, func="test_func"
        )
        # Simulate attributes that might be expected by formatters or the filter
        mock_record.module = "test_module"
        mock_record.funcName = "test_funcName" # Correct attribute name
        mock_record.lineno = 123

        result = context_filter.filter(mock_record)
        assert result is True
        assert getattr(mock_record, 'app_name', None) == test_app_name


@mock.patch(f"{LOGGING_SERVICE_MODULE}.os.path.exists", return_value=False)
@mock.patch(f"{LOGGING_SERVICE_MODULE}.os.makedirs", side_effect=OSError("Test permission denied"))
@mock.patch(f"{LOGGING_SERVICE_MODULE}.logging.handlers.TimedRotatingFileHandler")
@mock.patch(f"{LOGGING_SERVICE_MODULE}.sys.stderr", new_callable=io.StringIO) # Changed mock.StringIO to io.StringIO
def test_setup_logging_dir_creation_failure_falls_back(mock_stderr, mock_file_handler_cls, mock_makedirs, mock_exists, mock_app_config_basic, caplog):
    from src.logging_service import setup_logging
    config = mock_app_config_basic
    original_log_dir = config.logging.directory
    
    logger = setup_logging(config=config)
    
    mock_makedirs.assert_called_once_with(original_log_dir, exist_ok=True)
    assert f"CRITICAL: Could not create log directory {original_log_dir}: Test permission denied" in mock_stderr.getvalue()
    
    args, kwargs = mock_file_handler_cls.call_args
    expected_fallback_log_path = os.path.join(".", f"{config.logging.file_name_prefix}.log")
    assert args[0] == expected_fallback_log_path
    
    assert f"File format for '{os.path.basename(expected_fallback_log_path)}': Text" in caplog.text