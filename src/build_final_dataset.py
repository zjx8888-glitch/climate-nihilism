#!/usr/bin/env python3
"""Shim — use: python src/labeling/build_final_dataset.py"""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).parent / "labeling" / "build_final_dataset.py"), run_name="__main__")
