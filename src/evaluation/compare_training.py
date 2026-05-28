#!/usr/bin/env python3
"""
# TODO(Liu): add TF-IDF baseline comparison results here (separate from ClimateBERT outputs)

Legacy comparison script — see outputs/reports/recovered_vs_old_results.md
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def main() -> None:
    print(
        "Deprecated. See outputs/reports/recovered_vs_old_results.md. "
        "TF-IDF comparisons should live under outputs/tfidf/ (TODO Liu)."
    )


if __name__ == "__main__":
    main()
