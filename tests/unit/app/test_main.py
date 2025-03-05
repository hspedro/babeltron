from unittest.mock import patch

from fastapi import FastAPI

from babeltron.app.main import app, __version__


class TestMainApp:
    """Tests for the main app module."""

    def test_app_instance(self):
        """Test that the app is a FastAPI instance."""
        assert isinstance(app, FastAPI)
        assert app.title == "Babeltron Translation API"
        assert app.version == __version__

    @patch("babeltron.app.main.include_routers")
    def test_app_initialization(self, mock_include_routers):
        """Test that the app is initialized correctly."""
        assert app.title == "Babeltron Translation API"
        assert app.version == __version__

        # We can't directly verify that include_routers was called since it happens
        # during import, but we can check that the app has routes
        assert len(app.routes) > 0
