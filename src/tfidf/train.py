#!/usr/bin/env python3
#!/usr/bin/env python3
"""
TF-IDF baseline training & evaluation (Liu).

# TODO(Liu): improve TF-IDF baseline and move logic from legacy_train_evaluate.py

Outputs (target):
  - outputs/tfidf/tfidf_metrics.json
  - outputs/tfidf/models/
  - outputs/figures/ (TF-IDF plots)

Legacy reference: python src/tfidf/legacy_train_evaluate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# TODO(Liu): implement baseline training and evaluation in this module.


def main() -> None:
    print(
        "TF-IDF training not yet migrated. "
        "Run: python src/tfidf/legacy_train_evaluate.py"
    )


if __name__ == "__main__":
    main()
