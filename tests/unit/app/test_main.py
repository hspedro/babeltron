from fastapi import FastAPI

from babeltron.app.main import app


class TestMainApp:
    """Tests for the main app module."""

    def test_app_instance(self):
        """Test that the app is a FastAPI instance."""
        assert isinstance(app, FastAPI)
        assert app.title == "Babeltron Translation API"

    def test_app_initialization(self):
        """Test that the app is initialized correctly."""
        assert app.title == "Babeltron Translation API"

        # Check that the app has routes
        assert len(app.routes) > 0
