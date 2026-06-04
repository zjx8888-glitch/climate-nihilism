#!/usr/bin/env python3
"""
Recover and normalize labels in the 2000-comment manual labeling file.

# TODO(Madeleine): improve preprocessing pipeline integration

Excel often shows #NAME? when label cells are interpreted as formulas
(e.g. =Climate denial critique). The underlying CSV text is usually recoverable.

Outputs:
  data/labeled/recovered_labeled_dataset.csv
  data/labeled/broken_label_rows.csv
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from common.label_utils import body_hash
from common.paths import (
    BROKEN_LABEL_ROWS,
    DATA_PROCESSED,
    MANUAL_LABELED,
    MANUAL_LABELED_ALT,
    RECOVERED_LABELED,
    AUTO_LABELED
)
from common.taxonomy import CANONICAL_LABELS, normalize_label

SOURCE_CANDIDATES = [MANUAL_LABELED, MANUAL_LABELED_ALT]
RECOVERED_OUT = RECOVERED_LABELED
BROKEN_OUT = BROKEN_LABEL_ROWS
CLEAN_MANUAL = MANUAL_LABELED

EXCEL_ERROR_PATTERN = re.compile(
    r"^#(NAME\?|REF!|VALUE!|N/A|DIV/0!|NULL!)$",
    re.I,
)


def find_source_file() -> Path:
    for p in SOURCE_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "No manual label CSV found in data/processed/. "
        f"Expected one of: {SOURCE_CANDIDATES}"
    )


def strip_label_wrappers(raw: str) -> str:
    """Remove CSV/Excel quoting artifacts from label text."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return ""
    s = str(raw).strip()
    # Excel formula wrapper: ="Climate denial critique"
    if s.startswith('="') and s.endswith('"'):
        return s[2:-1].strip()
    if s.startswith("="):
        s = s[1:].strip()
    # Repeated double-quotes from CSV
    while len(s) >= 2 and s.startswith('"') and s.endswith('"'):
        s = s[1:-1].strip()
    return s


def is_broken_label(raw: str) -> bool:
    s = strip_label_wrappers(raw)
    if not s or s.lower() == "nan":
        return True
    if EXCEL_ERROR_PATTERN.match(s):
        return True
    return False


def recover_from_auto_labeled(hashes: pd.Series) -> pd.DataFrame:
    """Map body_hash -> weak label from auto_labeled (fallback only)."""
    if not AUTO_LABELED.exists():
        return pd.DataFrame(columns=["body_hash", "recovered_label_auto"])
    usecols = ["body", "predicted_label"]
    auto = pd.read_csv(AUTO_LABELED, usecols=usecols, encoding="utf-8", low_memory=False)
    auto["body_hash"] = auto["body"].map(body_hash)
    auto["recovered_label_auto"] = auto["predicted_label"].map(normalize_label)
    auto = auto.dropna(subset=["recovered_label_auto"])
    auto = auto.drop_duplicates(subset=["body_hash"], keep="first")
    need = set(hashes.dropna().astype(str))
    auto = auto[auto["body_hash"].isin(need)]
    return auto[["body_hash", "recovered_label_auto"]]


def resolve_label(
    raw_label: str,
    label_original: Optional[str],
    auto_label: Optional[str],
) -> Tuple[Optional[str], str]:
    """
    Returns (canonical_label, recovery_source).
    recovery_source: raw | original | auto_weak | unrecoverable
    """
    for candidate, source in (
        (raw_label, "raw"),
        (label_original, "label_original"),
    ):
        if candidate is None or (isinstance(candidate, float) and pd.isna(candidate)):
            continue
        if is_broken_label(str(candidate)):
            continue
        cleaned = strip_label_wrappers(str(candidate))
        canon = normalize_label(cleaned)
        if canon:
            return canon, source

    if auto_label and not pd.isna(auto_label):
        canon = normalize_label(str(auto_label))
        if canon:
            return canon, "auto_weak"

    return None, "unrecoverable"


def run_recovery() -> None:
    source = find_source_file()
    print(f"Reading: {source}")
    df = pd.read_csv(source, encoding="utf-8", on_bad_lines="skip", low_memory=False)
    df["body"] = df["body"].fillna("").astype(str)
    df["body_hash"] = df["body"].map(body_hash)

    if "label_original" not in df.columns:
        df["label_original"] = df["label"]

    raw_labels = df["label"].astype(str)
    excel_error_count = raw_labels.str.fullmatch(EXCEL_ERROR_PATTERN, na=False).sum()
    print(f"\n=== Why #NAME? appears ===")
    print(
        "Excel treats cells starting with '=' as formulas. "
        "A label like '=Climate denial critique' becomes #NAME? because "
        "'Climate' is not a valid function. This is an Excel display/export issue; "
        "the CSV file on disk may still contain the text (or quoted text)."
    )
    print(f"Literal #NAME? in CSV: {excel_error_count}")
    quoted = raw_labels.str.match('^"').sum()
    eq_pref = raw_labels.str.match(r"^=").sum()
    print(f"Labels wrapped in extra quotes: {quoted}")
    print(f"Labels starting with '=': {eq_pref}")

    auto_map = recover_from_auto_labeled(df["body_hash"])
    if not auto_map.empty:
        df = df.merge(auto_map, on="body_hash", how="left")
    else:
        df["recovered_label_auto"] = None

    resolved = []
    for row in df.itertuples(index=False):
        canon, src = resolve_label(
            getattr(row, "label", ""),
            getattr(row, "label_original", None),
            getattr(row, "recovered_label_auto", None),
        )
        resolved.append({"label_canonical": canon, "recovery_source": src})

    res_df = pd.DataFrame(resolved)
    df = pd.concat([df.reset_index(drop=True), res_df], axis=1)

    recovered = df[df["label_canonical"].notna()].copy()
    broken = df[df["label_canonical"].isna()].copy()

    # Output columns for recovered set
    out_cols = [
        c
        for c in [
            "body_hash",
            "type",
            "id",
            "subreddit.name",
            "body",
            "sentiment",
            "score",
            "label",
            "label_original",
            "label_canonical",
            "recovery_source",
        ]
        if c in recovered.columns or c in ["label_canonical", "recovery_source"]
    ]
    keep = [c for c in out_cols if c in recovered.columns and c != "label_canonical"]
    recovered_out = recovered[keep].copy()
    recovered_out["label"] = recovered["label_canonical"].values

    RECOVERED_OUT.parent.mkdir(parents=True, exist_ok=True)
    recovered_out.to_csv(RECOVERED_OUT, index=False)
    broken.to_csv(BROKEN_OUT, index=False)

    # Update canonical manual file for pipeline
    clean = recovered[
        ["body_hash", "body", "label_canonical", "recovery_source"]
        + [c for c in df.columns if c.startswith("subreddit") or c in ("id", "type", "sentiment", "score", "permalink", "created_utc")]
    ].copy()
    clean = clean.rename(columns={"label_canonical": "label"})
    clean["label_original"] = df.loc[recovered.index, "label"].values
    clean.to_csv(CLEAN_MANUAL, index=False)
    if source.resolve() != CLEAN_MANUAL.resolve():
        clean.to_csv(source, index=False)

    # Print summary
    print("\n=== Final counts ===")
    print(f"Total rows:              {len(df)}")
    print(f"Valid labeled rows:      {len(recovered)}")
    print(f"Broken/unusable rows:    {len(broken)}")
    print(f"\nRecovery sources:")
    print(recovered["recovery_source"].value_counts().to_string())
    print(f"\nLabel distribution (canonical):")
    print(recovered["label_canonical"].value_counts().to_string())
    print(f"\nWrote: {RECOVERED_OUT}")
    print(f"Wrote: {BROKEN_OUT}")
    print(f"Wrote: {CLEAN_MANUAL}")


if __name__ == "__main__":
    run_recovery()
