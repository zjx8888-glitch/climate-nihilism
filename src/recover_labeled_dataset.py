#!/usr/bin/env python3
"""Shim — use: python src/labeling/recover_labeled_dataset.py"""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).parent / "labeling" / "recover_labeled_dataset.py"), run_name="__main__")
