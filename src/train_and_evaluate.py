#!/usr/bin/env python3
"""Shim — legacy TF-IDF training: python src/tfidf/legacy_train_evaluate.py"""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).parent / "tfidf" / "legacy_train_evaluate.py"), run_name="__main__")
