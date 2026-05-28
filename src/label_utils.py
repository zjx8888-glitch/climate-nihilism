"""Backward compatibility — prefer `from common.label_utils import ...`."""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from common.label_utils import *  # noqa: F401, F403
from common.paths import (  # noqa: F401
    AUTO_LABELED,
    DATA_PROCESSED,
    FINAL_TRAINING,
    OUTPUTS,
    RECOVERED_LABELED,
    SPLITS_JSON,
)

# Legacy name
PROCESSED = DATA_PROCESSED
