import os
import sys
from pathlib import Path

import datetime
import numpy as np
import pandas as pd
import streamlit as st

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from common.paths import resolve_climatebert_version
from demo.inference import (
    MISSING_MSG,
    load_climatebert_metrics,
    load_climatebert_model,
    missing_model_message,
    predict_climatebert,
)

# --- Page Configuration ---
st.set_page_config(page_title="Climate Nihilism Dashboard", layout="wide")

_DEFAULT_VER = resolve_climatebert_version(os.environ.get("CLIMATEBERT_MODEL_VERSION"))


@st.cache_resource
def _climatebert_bundle(model_version: str):
    return load_climatebert_model(version=model_version)


@st.cache_data
def _climatebert_metrics(model_version: str):
    return load_climatebert_metrics(version=model_version)

# --- Header ---
st.title("🌍 Climate Nihilism Tracking Dashboard")
st.markdown("Monitor and analyze climate nihilism trends across social media platforms.")

with st.sidebar:
    st.subheader("ClimateBERT model")
    model_version = st.selectbox(
        "Checkpoint",
        options=["v2", "v1"],
        index=0 if _DEFAULT_VER == "v2" else 1,
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

st.divider()

# --- MACRO VIEW: Platform Trends ---
st.header("Macro View: Platform Trends")
st.markdown("Filter the scraped data to observe how nihilistic sentiment shifts over time.")

# Filters
col1, col2 = st.columns(2)
with col1:
    platforms = st.multiselect(
        "Select Platforms",
        ["Reddit", "X (Twitter)", "TikTok"],
        default=["Reddit", "X (Twitter)", "TikTok"]
    )
with col2:
    date_range = st.slider(
        "Select Date Range",
        min_value=datetime.date(2019, 1, 1),
        max_value=datetime.date(2025, 1, 1),
        value=(datetime.date(2019, 1, 1), datetime.date(2025, 1, 1))
    )

# --- Data Integration ---
# TODO: IMPLEMENT ACTUAL DATA LOADING HERE
# 1. Use pandas to load the clean dataset you saved (e.g., pd.read_pickle('Processed_Data/train.pkl'))
# 2. Ensure your DataFrame uses the master schema columns: 'platform', 'timestamp', and whatever column holds the calculated 'nihilism_score'
# 3. Filter the loaded DataFrame based on the 'date_range' and 'platforms' variables selected by the user above
# 4. Group the data by month/week and calculate the average nihilism score to feed into the chart

# --- Dummy Data Generation (Remove once real data is hooked up) ---
dates = pd.date_range(start=date_range[0], end=date_range[1], freq='ME')
np.random.seed(42) 
dummy_data = pd.DataFrame({
    "Reddit": np.random.normal(loc=0.5, scale=0.1, size=len(dates)),
    "X (Twitter)": np.random.normal(loc=0.6, scale=0.15, size=len(dates)),
    "TikTok": np.random.normal(loc=0.4, scale=0.12, size=len(dates))
}, index=dates).clip(0, 1)
# ------------------------------------------------------------------

# Render Chart
platform_colors = {
    "Reddit": "#FF4500",      # Reddit Orange
    "X (Twitter)": "#1DA1F2", # Twitter Blue
    "TikTok": "#000000"       # TikTok Black
}

if platforms:
    selected_colors = [platform_colors[plat] for plat in platforms]
    st.line_chart(dummy_data[platforms], color=selected_colors)
else:
    st.warning("Please select at least one platform to view trends.")

st.divider()

# --- MICRO VIEW: Model Comparison Analyzer ---
st.header("Micro View: Model Comparison Analyzer")
st.markdown("Input custom text to see how our baseline TF-IDF model compares to the advanced ClimateBERT model.")

user_text = st.text_area(
    "Enter a post or comment:", 
    placeholder="e.g., What is the point of recycling if the corporations are just going to burn it all anyway?"
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
                bundle = _climatebert_bundle(model_version)
                _climatebert_metrics(model_version)
                cb = predict_climatebert(user_text, model=bundle)
            except FileNotFoundError:
                st.error(missing_model_message(model_version))
                st.code(
                    f"python src/climatebert/train.py --dataset-version {model_version}",
                    language="bash",
                )
                cb = {"error": MISSING_MSG}

            if cb.get("error"):
                if "pretrained" not in cb.get("error", "").lower():
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
                if cb.get("is_climate_nihilism"):
                    st.warning("Predicted as **Climate nihilism**")
                f1 = cb.get("nihilism_f1_test")
                st.caption(
                    f"Model: {cb.get('model_version', model_version)} · "
                    f"Test nihilism F1: {f1:.3f}" if f1 is not None else f"Model: {model_version}"
                )
                st.caption(
                    f"Multiclass confidence: {cb['multiclass_confidence']:.2f}"
                )
                if cb.get("taxonomy_definition"):
                    st.caption(cb["taxonomy_definition"])
                st.markdown("**Top 3 labels**")
                for lab, prob in cb["top_3_labels"]:
                    st.write(f"- {lab}: {prob:.1%}")

        st.success("Analysis complete.")
    else:
        st.warning("Please enter some text to analyze.")