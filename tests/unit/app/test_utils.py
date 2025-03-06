import os
from unittest.mock import patch
from pathlib import Path

from babeltron.app.utils import get_model_path, include_routers
from babeltron.app.models.m2m import get_model_path as m2m_get_model_path


def test_get_model_path_env_var():
    with patch.dict(os.environ, {"MODEL_PATH": "/custom/path"}):
        assert get_model_path() == "/custom/path"
        assert m2m_get_model_path() == "/custom/path"


@patch("pathlib.Path.exists")
@patch("pathlib.Path.glob")
def test_get_model_path_default(mock_glob, mock_exists):
    # Mock the path existence and glob results
    mock_exists.return_value = True
    mock_glob.return_value = ["config.json"]

    # Clear the environment variable if it exists
    if "MODEL_PATH" in os.environ:
        del os.environ["MODEL_PATH"]

    # Test both get_model_path functions
    assert get_model_path() in ["/models", str(Path("./models"))]
    assert m2m_get_model_path() in ["/models", str(Path("./models"))]


@patch("importlib.import_module")
@patch("pkgutil.iter_modules")
def test_include_routers(mock_iter_modules, mock_import_module):
    # Mock the router discovery
    mock_iter_modules.return_value = [
        (None, "test_router", False),
        (None, "another_router", False)
    ]

    # Mock the imported modules
    mock_module1 = type('obj', (object,), {'router': 'router1'})
    mock_module2 = type('obj', (object,), {'router': 'router2'})

    # Make import_module return our mock modules
    mock_import_module.side_effect = [mock_module1, mock_module2]

    # Create a mock FastAPI app
    mock_app = type('obj', (object,), {'include_router': lambda x: None})

    # Patch the include_router method to track calls
    mock_app.include_router = lambda x: included_routers.append(x)
    included_routers = []

    # Call the function
    include_routers(mock_app)

    # Verify the routers were included
    assert len(included_routers) == 2
    assert 'router1' in included_routers
    assert 'router2' in included_routers
