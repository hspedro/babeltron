import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from fastapi import FastAPI

from babeltron.app.utils import get_model_path, include_routers


class TestUtils:
    @patch.dict(os.environ, {"MODEL_PATH": "/custom/path"})
    def test_get_model_path_from_env(self):
        path = get_model_path()
        assert str(path) == str(Path("/custom/path"))

    @patch.dict(os.environ, {}, clear=True)
    @patch("os.path.exists")
    def test_get_model_path_default(self, mock_exists):
        mock_exists.side_effect = lambda path: path == str(Path("/models"))

        path = get_model_path()
        assert str(Path("/models")) in str(path)

    @patch.dict(os.environ, {}, clear=True)
    @patch("os.path.exists")
    def test_get_model_path_fallback(self, mock_exists):
        mock_exists.return_value = False

        path = get_model_path()
        assert str(Path("/models")) in str(path)

    def test_include_routers(self):
        app = MagicMock(spec=FastAPI)
        include_routers(app)

        assert app.include_router.called
