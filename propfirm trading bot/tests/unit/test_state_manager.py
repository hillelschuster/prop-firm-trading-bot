import pytest
import json
import os
import logging
from datetime import datetime, timezone
from unittest import mock

from src.state_management.state_manager import StateManager

# Mocking config classes for StateManager tests
class MockStateManagementSettings:
    def __init__(self, persistence_file="state/bot_state.json", persistence_interval_seconds=300):
        self.persistence_file = persistence_file
        self.persistence_interval_seconds = persistence_interval_seconds

class MockAppConfig:
    def __init__(self, persistence_file="state/bot_state.json", persistence_interval_seconds=300):
        self.state_management = MockStateManagementSettings(persistence_file, persistence_interval_seconds)

@pytest.fixture
def mock_logger():
    return mock.MagicMock(spec=logging.Logger)

@pytest.fixture
def state_manager_instance(tmp_path, mock_logger):
    """Creates a StateManager instance with persistence file in a temporary directory."""
    state_file = tmp_path / "test_state.json"
    config = MockAppConfig(persistence_file=str(state_file))
    manager = StateManager(config=config, logger=mock_logger)
    return manager

@pytest.fixture
def state_manager_instance_no_dir_init(tmp_path, mock_logger):
    """
    Creates a StateManager instance where the state directory does not exist initially,
    to test directory creation logic within save/load methods more directly if needed,
    though __init__ should handle it.
    """
    state_sub_dir = tmp_path / "non_existent_dir"
    state_file = state_sub_dir / "test_state.json"
    config = MockAppConfig(persistence_file=str(state_file))
    # StateManager's __init__ will create state_sub_dir, so this fixture is more
    # about ensuring the path is complex.
    manager = StateManager(config=config, logger=mock_logger)
    return manager


def test_state_manager_initialization(tmp_path, mock_logger):
    state_dir = tmp_path / "custom_state_dir"
    state_file = state_dir / "init_test_state.json"
    
    assert not state_dir.exists() # Ensure dir doesn't exist before init

    config = MockAppConfig(persistence_file=str(state_file))
    manager = StateManager(config=config, logger=mock_logger)

    assert manager.persistence_file_path == str(state_file)
    assert manager.state_config.persistence_interval_seconds == 300 # Default
    assert state_dir.exists() # Check directory was created
    mock_logger.info.assert_any_call(f"Created state directory: {str(state_dir)}")


@mock.patch("src.state_management.state_manager.os.makedirs", side_effect=OSError("Test OS Error"))
def test_state_manager_initialization_dir_creation_error(tmp_path, mock_makedirs_exc, mock_logger):
    state_dir = tmp_path / "error_dir"
    state_file = state_dir / "error_state.json"
    config = MockAppConfig(persistence_file=str(state_file))
    
    StateManager(config=config, logger=mock_logger) # Initialize

    mock_logger.error.assert_called_once_with(
        f"Could not create state directory {str(state_dir)}: Test OS Error. State persistence may fail."
    )


def test_save_and_load_state_success(state_manager_instance, mock_logger):
    manager = state_manager_instance
    original_state = {
        "key1": "value1",
        "timestamp": datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        "nested": {"val": 10, "time": datetime(2023, 1, 2, 10, 30, 0, tzinfo=timezone.utc)}
    }

    save_result = manager.save_state(original_state)
    assert save_result is True
    mock_logger.info.assert_any_call(f"Bot state successfully saved to {manager.persistence_file_path}")

    loaded_state = manager.load_state()
    assert loaded_state is not None
    mock_logger.info.assert_any_call(f"Bot state successfully loaded. Last saved: {datetime.now(timezone.utc).isoformat().split('T')[0]}") # Check for date part
    
    assert loaded_state["key1"] == "value1"
    assert loaded_state["timestamp"] == datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    assert loaded_state["nested"]["val"] == 10
    assert loaded_state["nested"]["time"] == datetime(2023, 1, 2, 10, 30, 0, tzinfo=timezone.utc)

def test_save_state_creates_directory_if_not_exists(tmp_path, mock_logger):
    state_sub_dir = tmp_path / "save_creates_dir"
    state_file = state_sub_dir / "test_state.json"
    
    # Ensure directory does not exist before StateManager is even initialized for this specific test
    # This is slightly artificial as __init__ would create it, but tests save_state's own check.
    if state_sub_dir.exists():
        os.rmdir(state_sub_dir) # Remove if created by a previous test's tmp_path side effect
    
    config = MockAppConfig(persistence_file=str(state_file))
    # Initialize manager *after* ensuring dir doesn't exist, __init__ will create it.
    # To truly test save_state's internal makedirs, we'd need to mock os.path.exists for its check.
    # For simplicity, we rely on __init__ creating it, and save_state should not fail.
    manager = StateManager(config=config, logger=mock_logger)
    
    assert state_sub_dir.exists() # Created by __init__

    # Now, let's simulate the directory being removed *after* init but *before* save
    # This is a more direct test of save_state's makedirs
    os.rmdir(state_sub_dir)
    assert not state_sub_dir.exists()

    test_state = {"data": "content"}
    save_result = manager.save_state(test_state)
    
    assert save_result is True
    assert state_sub_dir.exists() # Check directory was created by save_state
    mock_logger.info.assert_any_call(f"Bot state successfully saved to {manager.persistence_file_path}")


def test_load_state_file_not_found(state_manager_instance, mock_logger):
    manager = state_manager_instance # File path is set, but file doesn't exist yet
    loaded_state = manager.load_state()
    assert loaded_state is None
    mock_logger.info.assert_called_with(f"State file not found at {manager.persistence_file_path}. Starting with a fresh state.")

def test_load_state_io_error_on_open(state_manager_instance, mock_logger):
    manager = state_manager_instance
    # Create the file so os.path.exists is true
    with open(manager.persistence_file_path, "w") as f:
        f.write("dummy content")

    with mock.patch("builtins.open", side_effect=IOError("Test read error")):
        loaded_state = manager.load_state()
    assert loaded_state is None
    mock_logger.error.assert_called_with(
        f"IOError loading state from {manager.persistence_file_path}: Test read error", exc_info=True
    )

def test_load_state_json_decode_error(state_manager_instance, mock_logger):
    manager = state_manager_instance
    with open(manager.persistence_file_path, 'w') as f:
        f.write("this is not valid json")
    
    loaded_state = manager.load_state()
    assert loaded_state is None
    mock_logger.error.assert_called_with(
        f"JSONDecodeError loading state from {manager.persistence_file_path}: Expecting value: line 1 column 1 (char 0). File might be corrupted.",
        exc_info=True
    )

def test_load_state_unexpected_format(state_manager_instance, mock_logger):
    manager = state_manager_instance
    malformed_data = {"some_other_key": "value", "timestamp": datetime.now(timezone.utc).isoformat()}
    with open(manager.persistence_file_path, 'w') as f:
        json.dump(malformed_data, f)

    loaded_state = manager.load_state()
    assert loaded_state is None
    mock_logger.warning.assert_called_with(
        f"Loaded state file {manager.persistence_file_path} has an unexpected format. Discarding."
    )

def test_save_state_io_error(state_manager_instance, mock_logger):
    manager = state_manager_instance
    with mock.patch("builtins.open", side_effect=IOError("Test write error")):
        save_result = manager.save_state({"key": "value"})
    assert save_result is False
    mock_logger.error.assert_called_with(
        f"IOError saving state to {manager.persistence_file_path}: Test write error", exc_info=True
    )

def test_save_state_type_error_serialization(state_manager_instance, mock_logger):
    manager = state_manager_instance
    # A class that is not JSON serializable by default and not handled by _serialize_datetime
    class Unserializable: pass
    unserializable_state = {"data": Unserializable()}

    save_result = manager.save_state(unserializable_state)
    assert save_result is False
    # The exact error message for TypeError can vary slightly based on Python version / json lib
    # We check that a TypeError was logged.
    assert mock_logger.error.call_args[0][0].startswith("TypeError during state serialization to JSON:")
    assert mock_logger.error.call_args[1]['exc_info'] is True


def test_get_persistence_interval(state_manager_instance):
    manager = state_manager_instance
    interval = 350
    manager.state_config.persistence_interval_seconds = interval
    assert manager.get_persistence_interval() == interval

def test_datetime_serialization_and_deserialization_edge_cases(state_manager_instance, mock_logger):
    manager = state_manager_instance
    
    # Test with timezone naive (should ideally not happen if app uses timezone.utc)
    # and with microseconds
    dt_naive = datetime(2023, 5, 5, 10, 0, 0) 
    dt_micros = datetime(2023, 5, 5, 10, 0, 0, 123456, tzinfo=timezone.utc)
    dt_zulu = datetime.fromisoformat("2023-01-01T12:00:00Z") # Parsed as UTC

    state = {
        "naive_time": dt_naive, # _serialize_datetime will make it ISO
        "micro_time": dt_micros,
        "zulu_time": dt_zulu,
        "not_a_time_str": "This is a normal string 2023-01-01T00:00:00Z",
        "looks_like_time_but_invalid": "2023-13-01T00:00:00Z" # Invalid month
    }
    manager.save_state(state)
    loaded = manager.load_state()

    assert loaded is not None
    # Naive datetime becomes aware after fromisoformat if it has no tzinfo in string,
    # but our _serialize_datetime doesn't add 'Z' or offset if naive.
    # json.dump's default for datetime is obj.isoformat().
    # datetime.isoformat() on naive datetime does not include tzinfo.
    # datetime.fromisoformat() on such string results in naive datetime.
    assert isinstance(loaded["naive_time"], datetime)
    assert loaded["naive_time"].tzinfo is None 
    assert loaded["naive_time"] == dt_naive

    assert loaded["micro_time"] == dt_micros
    assert loaded["micro_time"].tzinfo == timezone.utc
    assert loaded["zulu_time"] == dt_zulu
    assert loaded["zulu_time"].tzinfo == timezone.utc
    
    assert loaded["not_a_time_str"] == "This is a normal string 2023-01-01T00:00:00Z"
    assert loaded["looks_like_time_but_invalid"] == "2023-13-01T00:00:00Z" # Stays as string