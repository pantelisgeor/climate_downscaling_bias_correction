"""
Pytest configuration file for custom command line options.
"""

import pytest
from pathlib import Path
import os


def pytest_addoption(parser):
    """Add custom command line option for netCDF path."""
    parser.addoption(
        "--nc-path",
        action="store",
        default=None,
        help="Path to netCDF file for testing",
    )


@pytest.fixture(scope="session")
def nc_path(request):
    """Get netCDF path from command line or environment variable."""
    # Try command line argument first
    path = request.config.getoption("--nc-path")

    # Fall back to environment variable
    if path is None:
        path = os.environ.get("TEST_NC_PATH")

    # If still None, raise error
    if path is None:
        pytest.skip(
            "No netCDF file specified. Use --nc-path argument or set TEST_NC_PATH environment variable"
        )

    path = Path(path)
    if not path.exists():
        pytest.fail(f"NetCDF file not found: {path}")

    return str(path)
