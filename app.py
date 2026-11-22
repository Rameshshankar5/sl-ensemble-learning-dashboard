import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from hybrid_model import RFANNHybridRegressor  # Required for joblib deserialization

st.set_page_config(
    page_title="Sri Lanka Vehicle Price Ensemble Dashboard",
    page_icon="🚗",
    layout="wide"
)

@st.cache_resource
def load_bundle():
    return joblib.load("model_bundle.joblib")

bundle = load_bundle()
preprocessor = bundle["preprocessor"]
models = bundle["models"]
metrics_df = bundle["metrics"]
feat_imp_df = bundle["feature_importance"]
brand_model_map = bundle["brand_model_map"]

st.title("🇱🇰 Sri Lankan Used Vehicle Valuation — Ensemble Learning Dashboard")
st.markdown(
    "Dynamic price prediction (LKR) and evaluation across **Bagging**, **Boosting**, "
    "**Stacking**, and **Hybrid (Random Forest + ANN)** architectures."
)

# ==========================================
# SIDEBAR: MODEL & VEHICLE SPECIFICATIONS
# ==========================================
st.sidebar.header("1. Choose Ensemble Algorithm")
selected_model_name = st.sidebar.selectbox(
    "Active Model", list(models.keys()), index=0
)

st.sidebar.header("2. Configure Vehicle Attributes")
user_input = {}

# Render categorical dropdowns with smart Brand -> Model filtering
if "brand" in bundle["cat_cols"]:
    selected_brand = st.sidebar.selectbox("Brand", bundle["ui_options"]["brand"])
    user_input["brand"] = selected_brand

if "model" in bundle["cat_cols"]:
    if brand_model_map and "brand" in user_input:
        model_choices = brand_model_map.get(user_input["brand"], bundle["ui_options"]["model"])
    else:
        model_choices = bundle["ui_options"]["model"]
    user_input["model"] = st.sidebar.selectbox("Model", model_choices)

for col in bundle["cat_cols"]:
    if col not in ["brand", "model"]:
        user_input[col] = st.sidebar.selectbox(
            col.replace("_", " ").title(), bundle["ui_options"][col]
        )

# Render numeric inputs
for col in bundle["num_cols"]:
    stats = bundle["num_defaults"][col]
    step = 1.0 if col == "year" else 500.0
    user_input[col] = st.sidebar.number_input(
        col.replace("_", " ").title(),
        min_value=stats["min"],
        max_value=stats["max"],
        value=stats["median"],
        step=step
    )

# ==========================================
# LIVE INFERENCE & CROSS-MODEL COMPARISON
# ==========================================
input_df = pd.DataFrame([user_input])
input_proc = preprocessor.transform(input_df)

active_pred = float(models[selected_model_name].predict(input_proc)[0])

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Estimated Market Value")
    st.metric(
        label=selected_model_name,
        value=f"LKR {active_pred:,.0f}",
        delta=f"≈ LKR {active_pred / 1e6:.2f} Million"
    )
    st.caption("Selected Vehicle Profile")
    st.dataframe(input_df.T.rename(columns={0: "Specification"}), use_container_width=True)

with col2:
    st.subheader("Cross-Architecture Prediction Comparison")
    pred_rows = [
        {"Model": name, "Predicted Price (LKR)": float(m.predict(input_proc)[0])}
        for name, m in models.items()
    ]
    pred_df = pd.DataFrame(pred_rows)

    fig_pred = px.bar(
        pred_df,
        x="Predicted Price (LKR)",
        y="Model",
        orientation="h",
        color="Model",
        text_auto=",.0f"
    )
    fig_pred.update_layout(showlegend=False, height=380)
    st.plotly_chart(fig_pred, use_container_width=True)

st.divider()

# ==========================================
# EVALUATION METRICS & FEATURE IMPORTANCE
# ==========================================
tab1, tab2 = st.tabs(["📊 Benchmark Evaluation Matrix", "🔍 Feature Importance"])

with tab1:
    st.subheader("Hold-Out Test Set Performance (20% Split)")
    st.dataframe(
        metrics_df.style.highlight_min(subset=["RMSE (LKR)", "MAE (LKR)"], color="#d4edda")
                        .highlight_max(subset=["R2 Score"], color="#d4edda"),
        use_container_width=True
    )
    fig_rmse = px.bar(
        metrics_df,
        x="Model",
        y="RMSE (LKR)",
        color="R2 Score",
        title="Model Error Comparison (Lower RMSE = Higher Accuracy)"
    )
    st.plotly_chart(fig_rmse, use_container_width=True)

with tab2:
    st.subheader("Key Pricing Drivers in Sri Lanka (Random Forest Feature Importance)")
    fig_imp = px.bar(
        feat_imp_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top 15 Most Influential Features"
    )
    fig_imp.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_imp, use_container_width=True)