import os

import pytest


@pytest.fixture
def change_test_dir(request):
    """
    set pytest environment to have a relative path in the test/test_data folder
    """
    base_path = os.path.join(request.fspath.dirname, "test_data")
    os.chdir(base_path)
    yield
    os.chdir(request.config.invocation_dir)
