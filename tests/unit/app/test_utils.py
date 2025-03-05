import pytest
import hashlib
import json

from pathlib import Path
from unittest.mock import patch, MagicMock
from fastapi import FastAPI, APIRouter, Request
from fastapi.datastructures import URL

from babeltron.app.utils import get_model_path, include_routers, cache_key_builder


class TestGetModelPath:
    """Tests for the get_model_path function."""

    @patch.object(Path, "exists")
    @patch.object(Path, "glob")
    def test_env_variable_takes_precedence(self, mock_glob, mock_exists, monkeypatch):
        """Test that MODEL_PATH environment variable takes precedence."""
        test_path = "/tests/unit/app/model/path"
        monkeypatch.setenv("MODEL_PATH", test_path)
        mock_exists.return_value = True
        mock_glob.return_value = ["config.json"]

        result = get_model_path()

        assert result == test_path
        assert mock_exists.call_count == 0
        assert mock_glob.call_count == 0

    @patch("babeltron.app.utils.__file__", new="/fake/path/to/babeltron/app/utils.py")
    @patch.object(Path, "glob")
    @patch.object(Path, "exists")
    def test_returns_default_when_no_path_found(self, mock_exists, mock_glob, monkeypatch):
        """Test that it returns the default path when no valid path is found."""
        monkeypatch.delenv("MODEL_PATH", raising=False)
        mock_exists.return_value = False
        mock_glob.return_value = []

        result = get_model_path()

        assert result == "./models"
        assert mock_exists.call_count > 0

    @patch("babeltron.app.utils.__file__", new="/fake/path/to/babeltron/app/utils.py")
    @patch.object(Path, "glob")
    @patch.object(Path, "exists")
    def test_finds_model_in_project_root(self, mock_exists, mock_glob, monkeypatch):
        """Test that it finds a model in the project root."""
        monkeypatch.delenv("MODEL_PATH", raising=False)

        mock_exists.return_value = True

        project_root = Path("/fake/path/to")
        project_models_path = project_root / "models"

        def mock_glob_side_effect(pattern):
            path = mock_glob._mock_self
            if str(path) == str(project_models_path) and pattern == "**/config.json":
                return ["config.json"]
            return []

        mock_glob.side_effect = mock_glob_side_effect

        result = get_model_path()

        assert result in [str(project_models_path), "./models"]

    @patch("babeltron.app.utils.__file__", new="/fake/path/to/babeltron/app/utils.py")
    @patch.object(Path, "glob")
    @patch.object(Path, "exists")
    def test_finds_model_in_package_directory(self, mock_exists, mock_glob, monkeypatch):
        """Test that it finds a model in the package directory."""
        monkeypatch.delenv("MODEL_PATH", raising=False)

        # We need more side effect values to handle all the exists() calls
        # First check for project root returns False
        # Then check for package directory returns True
        # Additional True values for any subsequent exists() calls
        mock_exists.side_effect = [False, True] + [True] * 10

        package_dir = Path("/fake/path/to/babeltron/app").parent
        package_models_path = package_dir / "models"

        def mock_glob_side_effect(pattern):
            path = mock_glob._mock_self
            if str(path) == str(package_models_path) and pattern == "**/config.json":
                return ["config.json"]
            return []

        mock_glob.side_effect = mock_glob_side_effect

        result = get_model_path()

        assert result in [str(package_models_path), "./models"]
        assert mock_exists.call_count >= 2

    @patch("babeltron.app.utils.__file__", new="/fake/path/to/babeltron/app/utils.py")
    @patch.object(Path, "glob")
    @patch.object(Path, "exists")
    def test_finds_model_in_current_directory(self, mock_exists, mock_glob, monkeypatch):
        """Test that it finds a model in the current directory."""
        monkeypatch.delenv("MODEL_PATH", raising=False)

        # Project root and package directory checks return False
        # Current directory check returns True
        mock_exists.side_effect = [False, False, True]

        current_dir_models_path = Path("./models")

        def mock_glob_side_effect(pattern):
            path = mock_glob._mock_self
            if str(path) == str(current_dir_models_path) and pattern == "**/config.json":
                return ["config.json"]
            return []

        mock_glob.side_effect = mock_glob_side_effect

        result = get_model_path()

        assert result == "./models"
        assert mock_exists.call_count >= 3


class TestIncludeRouters:
    """Tests for the include_routers function."""

    @patch("importlib.import_module")
    @patch("pkgutil.iter_modules")
    @patch.object(FastAPI, "include_router")
    def test_includes_all_routers(self, mock_include_router, mock_iter_modules, mock_import_module):
        """Test that all routers are included."""
        app = FastAPI()

        mock_module1 = MagicMock()
        mock_module1.router = APIRouter()
        mock_module2 = MagicMock()
        mock_module2.router = APIRouter()

        mock_import_module.side_effect = lambda name: {
            "babeltron.app.routers.test1": mock_module1,
            "babeltron.app.routers.test2": mock_module2,
        }[name]

        mock_iter_modules.return_value = [
            (None, "test1", None),
            (None, "test2", None),
        ]

        include_routers(app)

        assert mock_include_router.call_count == 2
        mock_include_router.assert_any_call(mock_module1.router)
        mock_include_router.assert_any_call(mock_module2.router)

    @patch("importlib.import_module")
    @patch("pkgutil.iter_modules")
    @patch.object(FastAPI, "include_router")
    def test_skips_modules_without_router(self, mock_include_router, mock_iter_modules, mock_import_module):
        """Test that modules without a router attribute are skipped."""
        app = FastAPI()

        mock_module_with_router = MagicMock()
        mock_module_with_router.router = APIRouter()
        mock_module_without_router = MagicMock()
        if hasattr(mock_module_without_router, 'router'):
            delattr(mock_module_without_router, 'router')

        def import_side_effect(name):
            if name == "babeltron.app.routers.with_router":
                return mock_module_with_router
            elif name == "babeltron.app.routers.without_router":
                return mock_module_without_router
            raise ImportError(f"Unexpected module: {name}")

        mock_import_module.side_effect = import_side_effect

        mock_iter_modules.return_value = [
            (None, "with_router", None),
            (None, "without_router", None),
        ]

        include_routers(app)

        assert mock_include_router.call_count == 1
        mock_include_router.assert_called_with(mock_module_with_router.router)
        assert mock_import_module.call_count == 2

    @patch("importlib.import_module")
    @patch("pkgutil.iter_modules")
    @patch.object(FastAPI, "include_router")
    def test_handles_import_errors(self, mock_include_router, mock_iter_modules, mock_import_module):
        """Test that import errors are properly propagated."""
        app = FastAPI()

        mock_module = MagicMock()
        mock_module.router = APIRouter()

        def import_side_effect(name):
            if "good_module" in name:
                return mock_module
            raise ImportError(f"Module not found: {name}")

        mock_import_module.side_effect = import_side_effect

        mock_iter_modules.return_value = [
            (None, "good_module", None),
        ]

        include_routers(app)

        mock_include_router.assert_called_once_with(mock_module.router)

        mock_iter_modules.return_value = [
            (None, "bad_module", None),
        ]

        with pytest.raises(ImportError):
            include_routers(app)

    @patch("importlib.import_module")
    @patch("pkgutil.iter_modules")
    @patch.object(FastAPI, "include_router")
    def test_with_empty_routers_directory(self, mock_include_router, mock_iter_modules, mock_import_module):
        """Test behavior when the routers directory is empty."""
        app = FastAPI()

        mock_iter_modules.return_value = []

        include_routers(app)

        assert mock_include_router.call_count == 0
        assert mock_import_module.call_count == 0

    @patch("importlib.import_module")
    @patch("pkgutil.iter_modules")
    @patch.object(FastAPI, "include_router")
    def test_with_router_prefix_and_tags(self, mock_include_router, mock_iter_modules, mock_import_module):
        """Test that router prefix and tags are properly applied."""
        app = FastAPI()

        mock_module = MagicMock()
        mock_module.router = APIRouter()

        mock_import_module.return_value = mock_module

        mock_iter_modules.return_value = [
            (None, "test_module", None),
        ]

        include_routers(app)

        mock_include_router.assert_called_once_with(mock_module.router)


class TestCacheKeyBuilder:
    def setup_method(self):
        # Create a mock request
        self.request = MagicMock(spec=Request)
        self.request.method = "POST"
        self.request.url = URL("http://testserver/translate")
        self.request.query_params = {}
        self.request.state = MagicMock()

    def test_basic_key_generation(self):
        # Test with empty body
        self.request.state.body = "{}"

        key = cache_key_builder(None, "test-namespace", request=self.request, __=None, ___=None)

        expected_key = "test-namespace:::"
        assert key == expected_key

    def test_with_translation_data(self):
        # Test with translation data in body
        body_data = {
            "src_lang": "en",
            "dst_lang": "es",
            "text": "Hello world"
        }
        self.request.state.body = json.dumps(body_data)

        key = cache_key_builder(None, "translate", request=self.request, __=None, ___=None)

        text_md5 = hashlib.md5("Hello world".encode()).hexdigest()
        expected_key = f"translate:en:es:{text_md5}"
        assert key == expected_key

    def test_with_query_params(self):
        # Test with query parameters
        body_data = {
            "src_lang": "fr",
            "dst_lang": "de",
            "text": "Bonjour"
        }
        self.request.state.body = json.dumps(body_data)
        self.request.query_params = {"debug": "true", "format": "json"}

        key = cache_key_builder(None, "api", request=self.request, __=None, ___=None)

        text_md5 = hashlib.md5("Bonjour".encode()).hexdigest()
        expected_key = f"api:fr:de:{text_md5}"
        assert key == expected_key

    def test_with_missing_body(self):
        # Test when body is missing
        self.request.state = MagicMock(spec=[])  # No body attribute

        key = cache_key_builder(None, "test", request=self.request, __=None, ___=None)

        expected_key = "test:::"
        assert key == expected_key

    def test_with_invalid_json(self):
        # Test with invalid JSON in body
        self.request.state.body = "not-valid-json"

        key = cache_key_builder(None, "test", request=self.request, __=None, ___=None)

        expected_key = "test:::"
        assert key == expected_key

    def test_with_empty_text(self):
        # Test with empty text field
        body_data = {
            "src_lang": "en",
            "dst_lang": "fr",
            "text": ""
        }
        self.request.state.body = json.dumps(body_data)

        key = cache_key_builder(None, "translate", request=self.request, __=None, ___=None)

        expected_key = "translate:en:fr:"
        assert key == expected_key

    def test_with_missing_fields(self):
        # Test with missing fields in body
        body_data = {
            "src_lang": "en",
            # dst_lang is missing
            "text": "Test"
        }
        self.request.state.body = json.dumps(body_data)

        key = cache_key_builder(None, "translate", request=self.request, __=None, ___=None)

        text_md5 = hashlib.md5("Test".encode()).hexdigest()
        expected_key = f"translate:en::{text_md5}"
        assert key == expected_key
