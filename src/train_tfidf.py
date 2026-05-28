#!/usr/bin/env python3
"""Shim — use: python src/tfidf/train.py"""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).parent / "tfidf" / "train.py"), run_name="__main__")
