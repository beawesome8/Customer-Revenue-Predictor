"""
Purchase Intent Predictor - interactive demo.

Exposes the six features that most move the prediction, with sensible
defaults for the rest, so a recruiter gets a meaningful prediction in
15 seconds instead of filling out a 17-field form.
"""
import os
import streamlit as st
from logic import build_payload, call_api, MONTHS, VISITOR_TYPES

API_URL = os.environ.get("API_URL", "http://localhost:8080")

st.set_page_config(page_title="Purchase Intent Predictor", page_icon="🛒")
st.title("🛒 Purchase Intent Predictor")
st.caption(
    "XGBoost model trained on the UCI Online Shoppers dataset, "
    "served via FastAPI on Cloud Run. [View source on GitHub]"
    "(https://github.com/beawesome8/Purchase-Intent-GCP)"
)

st.subheader("Session behavior")
col1, col2 = st.columns(2)
with col1:
    product_related = st.slider("Product pages viewed", 0, 100, 15)
    product_duration = st.slider("Time on product pages (seconds)", 0, 3000, 600)
    page_values = st.slider("Page value score", 0.0, 200.0, 12.5,
                              help="Google Analytics PageValues metric - higher means the pages "
                                   "viewed historically lead to purchases")
with col2:
    bounce_rate = st.slider("Bounce rate", 0.0, 1.0, 0.02)
    exit_rate = st.slider("Exit rate", 0.0, 1.0, 0.05)
    month = st.selectbox("Month", MONTHS, index=MONTHS.index("Nov"))

col3, col4 = st.columns(2)
with col3:
    visitor_type = st.selectbox("Visitor type", VISITOR_TYPES)
with col4:
    weekend = st.checkbox("Weekend session")

if st.button("Predict purchase likelihood", type="primary"):
    payload = build_payload(
        product_related, float(product_duration), bounce_rate, exit_rate,
        page_values, month, visitor_type, weekend,
    )
    try:
        with st.spinner("Calling the model API..."):
            result = call_api(payload, API_URL)
        proba = result["purchase_probability"]
        st.metric("Purchase probability", f"{proba:.1%}")
        if result["will_purchase"]:
            st.success("Prediction: likely to purchase")
        else:
            st.info("Prediction: unlikely to purchase")
        st.caption(f"Model version: {result['model_version']}")
    except Exception as e:
        st.error(f"Couldn't reach the prediction API: {e}")

with st.expander("How this works"):
    st.markdown(
        "This demo sends the values above to a live FastAPI service "
        "running an XGBoost classifier on Cloud Run. The model was "
        "benchmarked against a neural network and a logistic regression "
        "baseline before being selected - see `phase2_comparison.json` "
        "in the repo for the full comparison."
    )
