from unittest.mock import patch

from babeltron.app.utils import include_routers


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
