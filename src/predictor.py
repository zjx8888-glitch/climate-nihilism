"""Backward compatibility — prefer `from common.predictor import ...`."""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from common.predictor import *  # noqa: F401, F403
