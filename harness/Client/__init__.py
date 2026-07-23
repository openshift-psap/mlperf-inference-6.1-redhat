# Client module for different harness clients

from .base_client import BaseClient
from .loadgen_client import LoadGenClient
from .loadgen_client_offline import LoadGenOfflineClient
from .loadgen_client_server import LoadGenServerClient

__all__ = [
    'BaseClient',
    'LoadGenClient',
    'LoadGenOfflineClient',
    'LoadGenServerClient',
    'create_loadgen_client'
]


def create_loadgen_client(scenario: str, *args, **kwargs) -> LoadGenClient:
    """
    Factory function to create LoadGen client instances.

    Args:
        scenario: "Offline" or "Server"
        *args, **kwargs: Arguments passed to client constructor

    Returns:
        LoadGenClient instance
    """
    scenario_lower = scenario.lower()

    if scenario_lower == "offline":
        return LoadGenOfflineClient(*args, **kwargs)
    elif scenario_lower == "server":
        return LoadGenServerClient(*args, **kwargs)
    else:
        raise ValueError(f"Unknown scenario: {scenario}. Must be 'Offline' or 'Server'")
