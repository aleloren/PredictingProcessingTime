import streamlit as st
from datetime import datetime
import pandas as pd
import numpy as np
import joblib  # <-- switched from pickle to joblib

# ─── Load preprocessor + model ─────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_objects():
    # Use joblib.load since these were saved via joblib.dump
    preprocessor = joblib.load("preprocessor.pkl")
    model        = joblib.load("gbr_model.pkl")
    return preprocessor, model

preprocessor, model = load_objects()

# ─── Keyword → Category mapping ────────────────────────────────────────────────
keyword_map = {
    1: ["rebuild"],
    2: ["replace", "replacement"],
    3: ["revision"],
    4: ["install"],
    5: ["repair"],
    6: ["remove"],
    7: ["alteration", "attach", "detach"],
    8: ["construct"],
    9: ["conversion", "convert"],
    10: ["remodel"],
    11: ["erect"],
    12: ["build"],
    13: ["demolish"],
}

def assign_category_for_input(desc: str) -> str | None:
    text = str(desc).lower()
    for cat, keywords in keyword_map.items():
        for kw in keywords:
            if kw in text:
                return f"WD{cat}"
    return None

# ─── Streamlit UI ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Permit Processing Time Predictor", layout="centered")
st.title("Permit Processing Time Predictor")
st.write("Fill out the form below and click **Predict** to see the estimated processing time (in days).")

with st.form(key="input_form"):
    # 1. Application Date
    app_date_str = st.text_input(
        label="Application Date (YYYY-MM-DD)",
        placeholder="e.g. 2025-06-03"
    )

    # 2. Reported Cost
    reported_cost = st.number_input(
        label="Reported Cost (≥ 0)",
        min_value=0.0,
        step=100.0,
        value=0.0
    )

    # 3. Community Area
    community_area_raw = st.text_input(
        label="Community Areas (e.g., 1, 10, 32, etc.)",
        placeholder="Enter a number from training set"
    )

    # 4. Permit Type
    valid_permit_types = [
        "PERMIT - NEW CONSTRUCTION",
        "PERMIT - WRECKING/DEMOLITION",
        "PERMIT - RENOVATION/ALTERATION"
    ]
    permit_type = st.selectbox(
        label="Permit Type",
        options=valid_permit_types
    )

    # 5. Work Description Keyword
    st.write("**Enter a keyword from the Work Description** (must match one of these):")
    for cat, kws in keyword_map.items():
        st.write(f"• Category {cat}: {', '.join(kws)}")
    work_desc_keyword = st.text_input(
        label="Work Description Keyword",
        placeholder="e.g. replace, install, demolish, etc."
    )

    submit_button = st.form_submit_button(label="Predict")

if submit_button:
    # ─── Validate & parse date ────────────────────────────────────────────────
    try:
        app_date = datetime.strptime(app_date_str.strip(), "%Y-%m-%d")
    except Exception:
        st.error("❌ Invalid date format. Please use YYYY-MM-DD.")
        st.stop()

    # ─── Encode WD_ONEHOTENCODED ───────────────────────────────────────────────
    wd_encoded = assign_category_for_input(work_desc_keyword.strip())
    if wd_encoded is None:
        st.error("❌ Keyword not found in any category. Please use exactly one of the keywords shown.")
        st.stop()

    # ─── Prefix Community Area ────────────────────────────────────────────────
    community_area = f"Community {community_area_raw.strip()}"

    # ─── Build raw‐input DataFrame ────────────────────────────────────────────
    raw_dict = {
        "APPLICATION_START_DATE": app_date,
        "REPORTED_COST": float(reported_cost),
        "Community Areas": community_area,
        "PERMIT_TYPE": permit_type,
        "WD_ONEHOTENCODED": wd_encoded
    }
    input_df = pd.DataFrame([raw_dict])

    # ─── Date feature engineering ─────────────────────────────────────────────
    date_col = "APPLICATION_START_DATE"
    input_df[date_col] = pd.to_datetime(input_df[date_col], errors="coerce")
    input_df["APPLICATION_START_DATE_month"] = input_df[date_col].dt.month
    input_df["APPLICATION_START_DATE_weekday"] = input_df[date_col].dt.weekday

    input_df["APPLICATION_START_DATE_month_sin"] = np.sin(
        2 * np.pi * input_df["APPLICATION_START_DATE_month"] / 12
    )
    input_df["APPLICATION_START_DATE_month_cos"] = np.cos(
        2 * np.pi * input_df["APPLICATION_START_DATE_month"] / 12
    )
    input_df["APPLICATION_START_DATE_weekday_sin"] = np.sin(
        2 * np.pi * input_df["APPLICATION_START_DATE_weekday"] / 7
    )
    input_df["APPLICATION_START_DATE_weekday_cos"] = np.cos(
        2 * np.pi * input_df["APPLICATION_START_DATE_weekday"] / 7
    )

    # Drop raw date/month/weekday
    input_df = input_df.drop(
        columns=[
            "APPLICATION_START_DATE",
            "APPLICATION_START_DATE_month",
            "APPLICATION_START_DATE_weekday",
        ],
        errors="ignore"
    )

    # ─── Reorder columns to match preprocessor’s expected order ──────────────
    feature_order = [
        "REPORTED_COST",
        "APPLICATION_START_DATE_month_sin",
        "APPLICATION_START_DATE_month_cos",
        "APPLICATION_START_DATE_weekday_sin",
        "APPLICATION_START_DATE_weekday_cos",
        "PERMIT_TYPE",
        "Community Areas",
        "WD_ONEHOTENCODED",
    ]
    input_df = input_df[feature_order]

    # ─── Preprocess + Predict ────────────────────────────────────────────────
    try:
        X_processed = preprocessor.transform(input_df)
        pred = model.predict(X_processed)
        st.success(f"✅ Estimated Processing Time: {pred[0]:.2f} days")
    except Exception as e:
        st.error(f"⚠️ Prediction error: {e}")
