import os
import sys
from pathlib import Path

import datetime
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from demo.inference import (
    MISSING_MSG,
    load_climatebert_metrics,
    load_climatebert_model,
    predict_climatebert,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "subreddit_trends"

SUBREDDIT_COLORS = {
    "worldnews": "#FF4500",
    "politics":  "#5f99cf",
    "collapse":  "#46d160",
    "askreddit": "#fffa71",
}

# --- Page Configuration ---
st.set_page_config(page_title="Climate Nihilism Dashboard", layout="wide")


@st.cache_resource
def _climatebert_bundle():
    return load_climatebert_model()


@st.cache_data
def _climatebert_metrics():
    return load_climatebert_metrics()


@st.cache_data
def load_subreddit_data() -> pd.DataFrame:
    """Load and combine all subreddit CSVs from DATA_DIR."""
    files = list(DATA_DIR.glob("*.csv"))
    if not files:
        return pd.DataFrame()

    frames = []
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined["created_datetime"] = pd.to_datetime(
        combined["created_datetime"], errors="coerce"
    )
    combined = combined.dropna(subset=["created_datetime"])

    if "label_clean" in combined.columns:
        combined["label_norm"] = combined["label_clean"].str.strip().str.lower()
    elif "label" in combined.columns:
        combined["label_norm"] = (
            combined["label"].str.strip().str.strip('"').str.lower()
        )
    else:
        combined["label_norm"] = ""

    combined["is_nihilism"] = combined["label_norm"] == "climate nihilism"
    combined["subreddit"] = combined["subreddit.name"].str.strip()
    combined["year_month"] = combined["created_datetime"].dt.to_period("M")
    return combined


def compute_monthly_nihilism(
    df: pd.DataFrame,
    subreddits: list[str],
    start: datetime.date,
    end: datetime.date,
    min_posts: int,
) -> pd.DataFrame:
    """
    Return a long-format DataFrame with monthly nihilism rate per subreddit.
    Months with fewer than min_posts total posts are excluded (unreliable rates).
    Columns: Month, Subreddit, Nihilism Rate (%), Nihilism Posts, Total Posts
    """
    mask = (
        df["subreddit"].isin(subreddits)
        & (df["created_datetime"].dt.date >= start)
        & (df["created_datetime"].dt.date <= end)
    )
    filtered = df[mask].copy()

    if filtered.empty:
        return pd.DataFrame()

    monthly = (
        filtered.groupby(["subreddit", "year_month"])
        .agg(
            total=("label_norm", "count"),
            nihilism=("is_nihilism", "sum"),
        )
        .reset_index()
    )

    # Drop months below the minimum post threshold to avoid misleading spikes
    monthly = monthly[monthly["total"] >= min_posts].copy()

    if monthly.empty:
        return pd.DataFrame()

    monthly["nihilism_rate"] = monthly["nihilism"] / monthly["total"]
    monthly["Nihilism Rate (%)"] = (monthly["nihilism_rate"] * 100).round(1)
    monthly["Month"] = monthly["year_month"].dt.to_timestamp()
    monthly = monthly.rename(columns={
        "subreddit": "Subreddit",
        "nihilism": "Nihilism Posts",
        "total": "Total Posts",
    })

    return monthly[["Month", "Subreddit", "Nihilism Rate (%)", "Nihilism Posts", "Total Posts"]]


def build_chart(long_df: pd.DataFrame, start: datetime.date, end: datetime.date) -> alt.Chart:
    """Altair line chart with fully labelled tooltips and correct formatting.

    The x-axis domain is set explicitly from the slider values so that
    changing the date range immediately updates the visible window.
    (.interactive() is intentionally omitted — it locks the viewport on
    first render and ignores subsequent data changes.)
    """
    color_scale = alt.Scale(
        domain=list(SUBREDDIT_COLORS.keys()),
        range=list(SUBREDDIT_COLORS.values()),
    )

    base = alt.Chart(long_df).encode(
        x=alt.X(
            "Month:T",
            title="Month",
            axis=alt.Axis(format="%b %Y", labelAngle=-45),
            scale=alt.Scale(
                domain=[start.isoformat(), end.isoformat()],
            ),
        ),
        y=alt.Y(
            "Nihilism Rate (%):Q",
            title="Nihilism Rate (%)",
            scale=alt.Scale(domain=[0, 100]),
        ),
        color=alt.Color(
            "Subreddit:N",
            title="Subreddit",
            scale=color_scale,
        ),
        tooltip=[
            alt.Tooltip("Month:T",            title="Month",             format="%B %Y"),
            alt.Tooltip("Subreddit:N",         title="Subreddit"),
            alt.Tooltip("Nihilism Rate (%):Q", title="Nihilism Rate (%)", format=".1f"),
            alt.Tooltip("Nihilism Posts:Q",    title="Nihilism Posts"),
            alt.Tooltip("Total Posts:Q",       title="Total Posts"),
        ],
    )

    return (
        base.mark_line() + base.mark_point(filled=True, size=60)
    ).properties(height=380)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.subheader("ClimateBERT model")
    model_version = st.selectbox(
        "Checkpoint",
        options=["v2", "v1"],
        index=0,
        help="Uses saved .joblib weights from outputs/climatebert_v2/ (or v1/). Train once; no retrain per query.",
    )
    st.caption(
        f"Loads pretrained heads from `outputs/climatebert_{model_version}/` "
        "(plus Hugging Face ClimateBERT embeddings on first run)."
    )
    if st.button("Clear model cache"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🌍 Climate Nihilism Tracking Dashboard")
st.markdown("Monitor and analyze climate nihilism trends across Reddit communities.")

st.divider()

# ---------------------------------------------------------------------------
# MACRO VIEW — Subreddit Trends
# ---------------------------------------------------------------------------
st.header("Macro View: Subreddit Trends")
st.markdown(
    "Proportion of posts labelled **Climate nihilism** per month, "
    "filtered by subreddit and date range."
)

raw_df = load_subreddit_data()

if raw_df.empty:
    st.warning(
        f"No subreddit data found. Place your CSV files in `{DATA_DIR}` and reload the app."
    )
else:
    all_subreddits = sorted(raw_df["subreddit"].dropna().unique().tolist())

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        selected_subs = st.multiselect(
            "Select Subreddits",
            options=all_subreddits,
            default=all_subreddits,
        )
    with col2:
        overall_min = raw_df["created_datetime"].dt.date.min()
        overall_max = raw_df["created_datetime"].dt.date.max()
        date_range = st.slider(
            "Select Date Range",
            min_value=overall_min,
            max_value=overall_max,
            value=(overall_min, overall_max),
        )
    with col3:
        min_posts = st.number_input(
            "Min posts / month",
            min_value=1,
            max_value=20,
            value=5,
            help="Months with fewer posts than this are hidden — single-post months create misleading 100% or 0% spikes.",
        )

    if selected_subs:
        long_df = compute_monthly_nihilism(
            raw_df, selected_subs, date_range[0], date_range[1], min_posts
        )

        if long_df.empty:
            st.info(
                f"No months with ≥ {min_posts} posts found for the selected filters. "
                "Try lowering the minimum posts threshold."
            )
        else:
            st.altair_chart(build_chart(long_df, date_range[0], date_range[1]), use_container_width=True)
            st.caption(
                f"Y-axis: percentage of monthly posts classified as *Climate nihilism*. "
                f"Months with fewer than {min_posts} total posts are hidden to avoid "
                "misleading rates from low-volume months."
            )

        # --- Summary stats table ---
        st.subheader("Subreddit Summary")
        summary_rows = []
        for sub in selected_subs:
            sub_df = raw_df[
                (raw_df["subreddit"] == sub)
                & (raw_df["created_datetime"].dt.date >= date_range[0])
                & (raw_df["created_datetime"].dt.date <= date_range[1])
            ]
            total = len(sub_df)
            nihilism_count = int(sub_df["is_nihilism"].sum())
            rate = nihilism_count / total if total > 0 else 0
            summary_rows.append(
                {
                    "Subreddit": f"r/{sub}",
                    "Total Posts": total,
                    "Nihilism Posts": nihilism_count,
                    "Nihilism Rate": f"{rate:.1%}",
                }
            )
        st.dataframe(
            pd.DataFrame(summary_rows),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning("Select at least one subreddit to view trends.")

st.divider()

# ---------------------------------------------------------------------------
# MICRO VIEW — Model Comparison Analyzer
# ---------------------------------------------------------------------------
st.header("Micro View: Model Comparison Analyzer")
st.markdown(
    "Input custom text to see how our baseline TF-IDF model compares "
    "to the advanced ClimateBERT model."
)

user_text = st.text_area(
    "Enter a post or comment:",
    placeholder="e.g., What is the point of recycling if the corporations are just going to burn it all anyway?",
)

if st.button("Analyze Text"):
    if user_text:
        model_col1, model_col2 = st.columns(2)

        with model_col1:
            st.subheader("📊 TF-IDF Baseline")
            # TODO(Liu): connect TF-IDF inference here
            st.metric(label="Nihilism Score", value="45%")
            st.metric(label="Classification", value="Frustration")
            st.caption("Training F1-Score: 0.41")

        with model_col2:
            st.subheader("🧠 ClimateBERT")
            cb: dict = {"error": "Unknown error."}
            try:
                bundle = _climatebert_bundle()
                _climatebert_metrics()
                cb = predict_climatebert(user_text, model=bundle)
            except FileNotFoundError as exc:
                st.error(str(exc))
                st.code("python src/climatebert/train.py", language="bash")
                cb = {"error": str(exc)}

            if cb.get("error"):
                if MISSING_MSG not in cb.get("error", ""):
                    st.error(cb["error"])
            else:
                nih_pct = (
                    f"{cb['nihilism_probability'] * 100:.1f}%"
                    if cb.get("nihilism_probability") is not None
                    else "N/A"
                )
                st.metric(label="Nihilism probability", value=nih_pct)
                st.metric(
                    label="Predicted class (14-way)",
                    value=cb["predicted_label"],
                )
                if cb.get("predicted_label") == "Climate nihilism":
                    st.warning("⚠️ Predicted as **Climate nihilism**")
                f1 = cb.get("nihilism_f1_test")
                st.caption(
                    f"Model checkpoint: {model_version} · "
                    + (f"Test nihilism F1: {f1:.3f}" if f1 is not None else "see climatebert_metrics.json")
                )
                st.caption(f"Multiclass confidence: {cb['multiclass_confidence']:.2f}")
                if cb.get("taxonomy_definition"):
                    st.caption(cb["taxonomy_definition"])
                st.markdown("**Top 3 labels**")
                for lab, prob in cb["top_3_labels"]:
                    st.write(f"- {lab}: {prob:.1%}")

        st.success("Analysis complete.")
    else:
        st.warning("Please enter some text to analyze.")