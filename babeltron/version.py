"""Version information for the Babeltron package."""

import importlib.metadata

# Get version from package metadata
try:
    __version__ = importlib.metadata.version("babeltron")
except importlib.metadata.PackageNotFoundError:
    # If package is not installed, try to get version from pyproject.toml
    try:
        import tomli

        with open("pyproject.toml", "rb") as f:
            pyproject = tomli.load(f)
            __version__ = pyproject["project"]["version"]
    except (FileNotFoundError, KeyError, ImportError):
        __version__ = "dev"
