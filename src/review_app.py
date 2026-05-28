#!/usr/bin/env python3
"""
Streamlit app for human review of weak-labeled Reddit comments.

Run:
  streamlit run src/review_app.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from label_utils import (
    AUTO_LABELED,
    HUMAN_VERIFIED,
    REVIEW_PROGRESS,
    body_hash,
    load_auto_labeled,
)
from taxonomy import CANONICAL_LABELS, PRIORITY_LABELS, RELATED_LABEL_GROUPS, TAXONOMY

st.set_page_config(page_title="Climate Label Review", layout="wide")


@st.cache_data
def load_queue() -> pd.DataFrame:
    df = load_auto_labeled()
    reviewed = set()
    if HUMAN_VERIFIED.exists():
        done = pd.read_csv(HUMAN_VERIFIED, encoding="utf-8")
        if not done.empty and "body_hash" in done.columns:
            reviewed = set(done["body_hash"].astype(str))
    df = df[~df["body_hash"].isin(reviewed)].copy()

    # Priority: target labels + low confidence + flagged review
    df["_priority"] = df["predicted_label"].isin(PRIORITY_LABELS).astype(int) * 3
    df["_priority"] += df["needs_human_review"].astype(str).str.lower().isin(
        ["true", "1", "yes"]
    ).astype(int)
    df["_priority"] += (df["confidence"] < 0.35).astype(int)
    df = df.sort_values(
        ["_priority", "confidence", "predicted_label"],
        ascending=[False, True, True],
    )
    return df


def append_verified(row: dict) -> None:
    HUMAN_VERIFIED.parent.mkdir(parents=True, exist_ok=True)
    if HUMAN_VERIFIED.exists():
        existing = pd.read_csv(HUMAN_VERIFIED, encoding="utf-8")
    else:
        existing = pd.DataFrame()
    existing = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    existing.to_csv(HUMAN_VERIFIED, index=False)


def save_progress(idx: int) -> None:
    REVIEW_PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    REVIEW_PROGRESS.write_text(json.dumps({"last_index": idx}), encoding="utf-8")


def load_progress() -> int:
    if REVIEW_PROGRESS.exists():
        return int(json.loads(REVIEW_PROGRESS.read_text()).get("last_index", 0))
    return 0


def main() -> None:
    st.title("Climate Opinion Label Review")
    st.caption(
        "Verify weak labels before training. **Priority:** Climate nihilism, "
        "climate anxiety, Climate denial (especially low confidence)."
    )

    with st.sidebar:
        st.header("Filters")
        label_filter = st.multiselect(
            "Predicted label",
            options=CANONICAL_LABELS,
            default=list(PRIORITY_LABELS),
        )
        conf_max = st.slider("Max confidence (review uncertain)", 0.0, 1.0, 0.45, 0.01)
        review_only = st.checkbox("Only needs_human_review", value=True)
        priority_only = st.checkbox("Only priority labels", value=False)

        if st.button("Reload queue"):
            st.cache_data.clear()

    df = load_queue()
    if label_filter:
        df = df[df["predicted_label"].isin(label_filter)]
    df = df[df["confidence"] <= conf_max]
    if review_only and "needs_human_review" in df.columns:
        df = df[df["needs_human_review"].astype(str).str.lower().isin(["true", "1", "yes"])]
    if priority_only:
        df = df[df["predicted_label"].isin(PRIORITY_LABELS)]

    verified_count = 0
    if HUMAN_VERIFIED.exists():
        verified_count = len(pd.read_csv(HUMAN_VERIFIED, encoding="utf-8"))

    st.metric("Queue remaining", len(df))
    st.metric("Already reviewed", verified_count)

    if df.empty:
        st.success("No comments match filters — try widening filters or reload.")
        return

    start_idx = st.number_input("Start at index", 0, max(0, len(df) - 1), load_progress())
    if start_idx >= len(df):
        start_idx = 0
    row = df.iloc[int(start_idx)]

    st.subheader(f"Comment {int(start_idx) + 1} / {len(df)}")
    st.info(f"**Predicted:** {row['predicted_label']}  |  **Confidence:** {row['confidence']:.3f}")

    with st.expander("Taxonomy distinction (anxiety vs nihilism vs nihilism critique)"):
        for group in RELATED_LABEL_GROUPS:
            st.markdown("**Related labels:**")
            for lab in group:
                st.markdown(f"- **{lab}:** {TAXONOMY[lab]}")

    st.text_area("Comment", row["body"], height=220, disabled=True)
    st.caption(f"Reason: {row.get('reason', '')}")
    if pd.notna(row.get("similar_example")):
        st.caption(f"Similar example: {str(row['similar_example'])[:300]}…")

    default_label = row["predicted_label"]
    if default_label not in CANONICAL_LABELS:
        default_label = CANONICAL_LABELS[0]
    chosen = st.selectbox(
        "Assign label",
        CANONICAL_LABELS,
        index=CANONICAL_LABELS.index(default_label),
    )
    st.caption(TAXONOMY[chosen])

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("Accept prediction", type="primary", use_container_width=True):
            append_verified(
                {
                    "body_hash": row["body_hash"],
                    "body": row["body"],
                    "predicted_label": row["predicted_label"],
                    "verified_label": row["predicted_label"],
                    "decision": "accept",
                    "confidence": row["confidence"],
                    "reviewed_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            save_progress(int(start_idx) + 1)
            st.rerun()
    with c2:
        if st.button("Change label", use_container_width=True):
            append_verified(
                {
                    "body_hash": row["body_hash"],
                    "body": row["body"],
                    "predicted_label": row["predicted_label"],
                    "verified_label": chosen,
                    "decision": "change",
                    "confidence": row["confidence"],
                    "reviewed_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            save_progress(int(start_idx) + 1)
            st.rerun()
    with c3:
        if st.button("Skip", use_container_width=True):
            append_verified(
                {
                    "body_hash": row["body_hash"],
                    "body": row["body"],
                    "predicted_label": row["predicted_label"],
                    "verified_label": "",
                    "decision": "skip",
                    "confidence": row["confidence"],
                    "reviewed_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            save_progress(int(start_idx) + 1)
            st.rerun()
    with c4:
        if st.button("Back", use_container_width=True):
            save_progress(max(0, int(start_idx) - 1))
            st.rerun()

    st.divider()
    st.download_button(
        "Download human_verified_labels.csv",
        data=HUMAN_VERIFIED.read_text(encoding="utf-8") if HUMAN_VERIFIED.exists() else "",
        file_name="human_verified_labels.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
