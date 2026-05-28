"""Backward compatibility — prefer `from common.taxonomy import ...`."""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from common.taxonomy import *  # noqa: F401, F403
