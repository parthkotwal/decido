"""
Stub out third-party modules that aren't available in the local test environment.
Modal runs on the cloud; we only test the pure parsing logic here.
"""

import sys
from unittest.mock import MagicMock

# modal is only available inside Modal containers — stub it for local tests
sys.modules.setdefault("modal", MagicMock())

# aiosqlite is installed in the project venv, not system Python
sys.modules.setdefault("aiosqlite", MagicMock())
