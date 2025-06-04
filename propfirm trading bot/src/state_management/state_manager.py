# prop_firm_trading_bot/src/state_management/state_manager.py

import json
import os
import logging
from typing import Dict, Any, Optional, TYPE_CHECKING
from datetime import datetime, timezone # Added timezone directly

if TYPE_CHECKING:
    from prop_firm_trading_bot.src.config_manager import AppConfig, StateManagementSettings

class StateManager:
    """
    Manages the persistence and loading of the bot's operational state.
    This allows the bot to resume its operations or maintain context across restarts.
    """

    def __init__(self, config: 'AppConfig', logger: logging.Logger):
        self.state_config: 'StateManagementSettings' = config.state_management
        self.logger = logger
        self.persistence_file_path = self.state_config.persistence_file

        # Ensure the directory for the state file exists
        state_dir = os.path.dirname(self.persistence_file_path)
        if state_dir and not os.path.exists(state_dir):
            try:
                os.makedirs(state_dir, exist_ok=True)
                self.logger.info(f"Created state directory: {state_dir}")
            except OSError as e:
                self.logger.error(f"Could not create state directory {state_dir}: {e}. State persistence may fail.")
                # Depending on desired robustness, could raise an error or operate without persistence.

    def _serialize_datetime(self, obj: Any) -> Any:
        """Custom serializer for datetime objects to ISO format string."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        # Consider raising TypeError for unhandled types to catch issues early
        # raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")
        return obj 

    def _deserialize_datetime_hook(self, dct: Dict[str, Any]) -> Dict[str, Any]:
        """
        Custom object_hook for json.load to convert ISO datetime strings back to datetime objects.
        This hook is applied to each dictionary decoded by json.load.
        """
        for k, v in dct.items():
            if isinstance(v, str):
                try:
                    # A more robust check for ISO format might be needed if other string types are common.
                    # Example: check for 'T' separator and presence of hyphens/colons.
                    if 'T' in v and (v.endswith('Z') or '+' in v or '-' in v[10:]): # Basic ISO check
                        dt_obj = datetime.fromisoformat(v.replace('Z', '+00:00')) # Ensure Z is UTC
                        dct[k] = dt_obj
                except (ValueError, TypeError):
                    pass # Not a datetime string we can parse, leave as is
            # No need to recurse here, json.load applies object_hook to nested dicts.
        return dct

    def save_state(self, current_state: Dict[str, Any]) -> bool:
        """
        Saves the provided state dictionary to the persistence file.
        """
        self.logger.info(f"Attempting to save state to: {self.persistence_file_path}")
        try:
            state_to_save = {
                "last_saved_utc": datetime.now(timezone.utc).isoformat(),
                "bot_state": current_state
            }
            state_dir = os.path.dirname(self.persistence_file_path)
            if state_dir and not os.path.exists(state_dir):
                os.makedirs(state_dir, exist_ok=True)

            with open(self.persistence_file_path, 'w', encoding='utf-8') as f:
                json.dump(state_to_save, f, indent=4, default=self._serialize_datetime)
            self.logger.info(f"Bot state successfully saved to {self.persistence_file_path}")
            return True
        except IOError as e:
            self.logger.error(f"IOError saving state to {self.persistence_file_path}: {e}", exc_info=True)
        except TypeError as e:
            self.logger.error(f"TypeError during state serialization to JSON: {e}. Ensure state is JSON serializable.", exc_info=True)
        except Exception as e:
            self.logger.error(f"Unexpected error saving state: {e}", exc_info=True)
        return False

    def load_state(self) -> Optional[Dict[str, Any]]:
        """
        Loads the bot's state from the persistence file.
        """
        if not os.path.exists(self.persistence_file_path):
            self.logger.info(f"State file not found at {self.persistence_file_path}. Starting with a fresh state.")
            return None

        self.logger.info(f"Attempting to load state from: {self.persistence_file_path}")
        try:
            with open(self.persistence_file_path, 'r', encoding='utf-8') as f:
                loaded_data_wrapper = json.load(f, object_hook=self._deserialize_datetime_hook)
            
            if "bot_state" not in loaded_data_wrapper or "last_saved_utc" not in loaded_data_wrapper:
                self.logger.warning(f"Loaded state file {self.persistence_file_path} has an unexpected format. Discarding.")
                return None

            last_saved_str = loaded_data_wrapper.get("last_saved_utc")
            # last_saved_dt = loaded_data_wrapper.get("last_saved_utc") # This should now be a datetime object
            # self.logger.info(f"Bot state successfully loaded. Last saved: {last_saved_dt.strftime('%Y-%m-%d %H:%M:%S %Z') if isinstance(last_saved_dt, datetime) else last_saved_dt}")
            self.logger.info(f"Bot state successfully loaded. Last saved: {last_saved_str}") # Keep as string for log if already deserialized
            return loaded_data_wrapper.get("bot_state")
            
        except IOError as e:
            self.logger.error(f"IOError loading state from {self.persistence_file_path}: {e}", exc_info=True)
        except json.JSONDecodeError as e:
            self.logger.error(f"JSONDecodeError loading state from {self.persistence_file_path}: {e}. File might be corrupted.", exc_info=True)
        except Exception as e:
            self.logger.error(f"Unexpected error loading state: {e}", exc_info=True)
        return None

    def get_persistence_interval(self) -> int:
        """Returns the configured persistence interval in seconds."""
        return self.state_config.persistence_interval_seconds

if __name__ == '__main__':
    import sys # For test logger output to console
    test_logger = logging.getLogger("StateManagerTest")
    test_logger.setLevel(logging.DEBUG)
    test_console_handler = logging.StreamHandler(sys.stdout)
    test_formatter = logging.Formatter('%(asctime)s - %(name)s - [%(levelname)s] - %(message)s')
    test_console_handler.setFormatter(test_formatter)
    if not test_logger.hasHandlers():
        test_logger.addHandler(test_console_handler)

    class MockStateManagementSettings:
        persistence_file: str = "state/test_bot_state.json"
        persistence_interval_seconds: int = 10

    class MockAppConfig: # Basic mock
        state_management = MockStateManagementSettings()

    mock_config = MockAppConfig()

    state_dir = os.path.dirname(mock_config.state_management.persistence_file)
    if state_dir and not os.path.exists(state_dir): # Ensure state_dir is not empty string
        os.makedirs(state_dir, exist_ok=True)
    
    state_manager = StateManager(config=mock_config, logger=test_logger) # type: ignore

    test_logger.info("--- Testing StateManager ---")
    initial_state = {
        "open_positions": [{"symbol": "EURUSD", "id": "123", "open_time": datetime.now(timezone.utc)}],
        "risk_controller_daily_orders": 5,
        "some_other_metric": 123.45,
        "last_processed_event_time": datetime(2024, 1, 10, 15, 30, 0, tzinfo=timezone.utc),
        "nested_state": {
            "inner_time": datetime(2023, 5, 5, 10, 0, 0, tzinfo=timezone.utc),
            "value": "test"
        }
    }
    save_success = state_manager.save_state(initial_state)
    test_logger.info(f"Save state successful: {save_success}")

    if save_success:
        loaded_state = state_manager.load_state()
        if loaded_state:
            test_logger.info("Loaded state successfully.")
            # test_logger.info(f"Full loaded state: {loaded_state}") # For debugging content

            open_time_val = loaded_state.get("open_positions", [{}])[0].get("open_time")
            if isinstance(open_time_val, datetime):
                 test_logger.info(f"Datetime deserialization for 'open_time' successful: {open_time_val}")
            else:
                 test_logger.error(f"Datetime deserialization for 'open_time' FAILED. Type: {type(open_time_val)}")

            last_event_time_val = loaded_state.get("last_processed_event_time")
            if isinstance(last_event_time_val, datetime):
                 test_logger.info(f"Datetime deserialization for 'last_processed_event_time' successful: {last_event_time_val}")
            else:
                 test_logger.error(f"Datetime deserialization for 'last_processed_event_time' FAILED. Type: {type(last_event_time_val)}")
            
            inner_time_val = loaded_state.get("nested_state", {}).get("inner_time")
            if isinstance(inner_time_val, datetime):
                 test_logger.info(f"Datetime deserialization for 'nested_state.inner_time' successful: {inner_time_val}")
            else:
                 test_logger.error(f"Datetime deserialization for 'nested_state.inner_time' FAILED. Type: {type(inner_time_val)}")


            assert loaded_state.get("risk_controller_daily_orders") == 5, "Daily orders mismatch"
        else:
            test_logger.error("Failed to load state.")

    if os.path.exists(mock_config.state_management.persistence_file):
        os.remove(mock_config.state_management.persistence_file)
    
    test_logger.info("Attempting to load non-existent state file:")
    non_existent_state = state_manager.load_state()
    assert non_existent_state is None, "Should return None for non-existent file"
    test_logger.info("Correctly returned None for non-existent state file.")

    test_logger.info(f"Configured persistence interval: {state_manager.get_persistence_interval()} seconds.")
    test_logger.info("--- Finished testing StateManager ---")