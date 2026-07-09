import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Workplace Accident Risk Predictor",
    page_icon="🦺",
    layout="wide"
)

px.defaults.template = "plotly_white"

# ----------------------------
# Load Model Artifacts
# ----------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/best_model_Partofbodyaffected_logistic_regression.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder_Partofbodyaffected.pkl")
    trained_columns = joblib.load("models/training_columns_Partofbodyaffected.pkl")
    return model, scaler, label_encoder, trained_columns

model, scaler, label_encoder, trained_columns = load_artifacts()

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.header("🦺 Accident Risk Predictor")
    st.markdown(
        """
        **Purpose**  
        Predict the most likely *part of the body affected* in workplace accidents
        to support proactive safety decisions.
        """
    )

    uploaded_file = st.file_uploader(
        "Upload Accident Dataset (CSV)",
        type=["csv"]
    )

# ----------------------------
# Main Header
# ----------------------------
st.title("🦺 Workplace Accident Risk Prediction Dashboard")
st.markdown(
    """
    Transform accident data into **actionable safety insights** using
    predictive analytics.
    """
)

st.divider()

# ----------------------------
# Main App Logic
# ----------------------------
if uploaded_file:

    # ----------------------------
    # Load Data
    # ----------------------------
    df = pd.read_csv(uploaded_file)

    # ----------------------------
    # KPI Section
    # ----------------------------
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Total Incidents Analyzed", f"{len(df):,}")
    kpi2.metric("Prediction Coverage", "100%")
    kpi3.metric("Model Type", "Predictive ML Model")

    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    # ----------------------------
    # Data Preparation
    # ----------------------------
    df = df.loc[:, ~df.columns.duplicated()]
    categorical_cols = df.select_dtypes(include="object").columns

    df_encoded = pd.get_dummies(
        df,
        columns=categorical_cols,
        drop_first=True
    )

    df_encoded = df_encoded.reindex(
        columns=trained_columns,
        fill_value=0
    )

    X_new = scaler.transform(df_encoded)

    # ----------------------------
    # Prediction
    # ----------------------------
    predictions = model.predict(X_new)
    decoded_preds = label_encoder.inverse_transform(predictions)

    result_df = pd.DataFrame({
        "Predicted Part of Body Affected": decoded_preds
    })

    st.success("✅ Prediction completed successfully")

    # ----------------------------
    # Aggregations for Charts
    # ----------------------------
    risk_summary = (
        result_df["Predicted Part of Body Affected"]
        .value_counts()
        .reset_index()
    )

    risk_summary.columns = ["Body Part", "Incidents"]
    risk_summary["Percentage"] = (
        risk_summary["Incidents"] / risk_summary["Incidents"].sum() * 100
    )
    risk_summary["Cumulative %"] = risk_summary["Percentage"].cumsum()

    # ----------------------------
    # Layout for Charts
    # ----------------------------
    col1, col2 = st.columns(2)

    # ----------------------------
    # Chart 1: Top Risk Areas
    # ----------------------------
    with col1:
        st.subheader("🔝 Top Risk Areas")

        fig_top = px.bar(
            risk_summary.head(10),
            x="Incidents",
            y="Body Part",
            orientation="h",
            text="Incidents"
        )

        fig_top.update_layout(
            yaxis=dict(autorange="reversed"),
            showlegend=False
        )

        st.plotly_chart(fig_top, use_container_width=True)

    # ----------------------------
    # Chart 2: Injury Distribution (Donut)
    # ----------------------------
    with col2:
        st.subheader("🍩 Injury Distribution")

        fig_donut = px.pie(
            risk_summary,
            names="Body Part",
            values="Incidents",
            hole=0.5
        )

        fig_donut.update_traces(textinfo="percent+label")
        st.plotly_chart(fig_donut, use_container_width=True)

    st.divider()

    # ----------------------------
    # Chart 3: Pareto Analysis
    # ----------------------------
    st.subheader("📈 Pareto Analysis (80/20 Rule)")

    fig_pareto = px.bar(
        risk_summary,
        x="Body Part",
        y="Incidents",
        text="Incidents"
    )

    fig_pareto.add_scatter(
        x=risk_summary["Body Part"],
        y=risk_summary["Cumulative %"],
        mode="lines+markers",
        name="Cumulative %"
    )

    fig_pareto.update_layout(
        yaxis2=dict(
            overlaying="y",
            side="right",
            title="Cumulative %"
        )
    )

    st.plotly_chart(fig_pareto, use_container_width=True)

    # ----------------------------
    # Results Table
    # ----------------------------
    st.subheader("🧠 Prediction Results")
    st.dataframe(result_df, use_container_width=True)

    # ----------------------------
    # Download Section
    # ----------------------------
    st.download_button(
        "⬇ Download Predictions",
        result_df.to_csv(index=False),
        "accident_predictions.csv",
        "text/csv"
    )

    st.caption(
        "⚠️ Predictions are based on historical patterns and should support — not replace — professional safety judgment."
    )

else:
    st.info("👈 Upload a CSV file from the sidebar to begin analysis.")
