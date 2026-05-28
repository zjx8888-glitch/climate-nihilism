#!/usr/bin/env python3
"""Shim — use: python src/climatebert/train.py  or  PYTHONPATH=src python -m climatebert.train"""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).parent / "climatebert" / "train.py"), run_name="__main__")
