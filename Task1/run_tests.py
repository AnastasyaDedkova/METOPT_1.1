import os, sys
os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
import pytest
raise SystemExit(pytest.main(["-q"]))
