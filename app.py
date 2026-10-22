"""
app.py — House Price Prediction Web Application (Streamlit)
============================================================
A clean, modern, beginner-friendly interface that:
  • Loads/generates the dataset automatically (no manual CSV download).
  • Trains a Simple Linear Regression model.
  • Lets users enter a house area and see the predicted price in ₹.
  • Shows the scatter plot with regression line.
  • Displays evaluation metrics.

Run with:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Import our training module (same folder)
from train import load_dataset, train_model, predict_price, dataset_info_string

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Custom CSS for a modern, clean look
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    /* Card-like containers */
    .stMetric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 12px;
        color: white;
    }
    .stMetric label { color: #e0e0e0 !important; }
    .stMetric [data-testid="stMetricValue"] { color: white !important; }

    /* Prediction result box */
    .prediction-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 14px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(17,153,142,0.3);
    }
    .prediction-box h2 { margin: 0; font-size: 2rem; }
    .prediction-box p  { margin: 0.3rem 0 0; font-size: 1rem; opacity: 0.9; }

    /* Header accent */
    .header-accent {
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Source badge */
    .source-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .source-real      { background: #d4edda; color: #155724; }
    .source-synthetic  { background: #fff3cd; color: #856404; }

    div[data-testid="stVerticalBlock"] > div { gap: 0.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown("# 🏠 House Price Prediction")
st.markdown(
    "**Predict the price of a house based on its area in square feet**  \n"
    "Using **Simple Linear Regression** — a supervised machine learning algorithm."
)

# ---------------------------------------------------------------------------
# Load Data & Train Model  (cached so it only runs once)
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="Loading dataset and training model …")
def get_everything():
    """Load data + train model. Cached by Streamlit."""
    df, source = load_dataset()
    model, X_train, X_test, y_train, y_test, y_pred, metrics = train_model(df)
    return df, source, model, X_train, X_test, y_train, y_test, y_pred, metrics


df, source, model, X_train, X_test, y_train, y_test, y_pred, metrics = get_everything()

# ---------------------------------------------------------------------------
# Dataset Info Section
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown("## 📊 Dataset")

# Source badge
if source == "real":
    badge = '<span class="source-badge source-real">✅ Real-World Data (fetched from GitHub)</span>'
else:
    badge = '<span class="source-badge source-synthetic">⚙️ Synthetic Data (auto-generated)</span>'
st.markdown(badge, unsafe_allow_html=True)

if source == "synthetic":
    st.info(
        "The dataset was generated automatically using NumPy because the remote "
        "dataset could not be fetched. Predictions are illustrative and do not "
        "represent real market prices.",
        icon="ℹ️",
    )

col1, col2 = st.columns(2)
col1.metric("Total Samples", f"{len(df)}")
col2.metric("Features", "Area (sq ft) → Price (₹)")

with st.expander("🔍 Preview first 5 rows"):
    st.dataframe(df.head(), width="stretch")

with st.expander("ℹ️ Dataset Info"):
    st.text(dataset_info_string(df))

# ---------------------------------------------------------------------------
# Prediction Section
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown("## 🔮 Predict House Price")

area_input = st.slider(
    "Select house area (sq ft)",
    min_value=300,
    max_value=6000,
    value=1500,
    step=50,
    help="Drag the slider or type a value to set the house area.",
)

if st.button("🏷️ Predict Price", type="primary", width="stretch"):
    predicted = predict_price(model, area_input)
    st.markdown(
        f"""
        <div class="prediction-box">
            <p>Estimated Price for <strong>{area_input:,} sq ft</strong></p>
            <h2>₹ {predicted:,.0f}</h2>
            <p>Indian Rupees (approx.)</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.caption("👆 Click **Predict Price** to see the estimate.")

# ---------------------------------------------------------------------------
# Graphs Section
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown("## 📈 Visualisations")

tab1, tab2 = st.tabs(["Scatter + Regression Line", "Actual vs Predicted"])

with tab1:
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.scatter(
        df["Area (sq ft)"],
        df["Price (₹)"],
        alpha=0.45,
        color="#667eea",
        edgecolors="white",
        linewidth=0.5,
        label="Data points",
    )
    # Regression line across full range
    x_range = np.linspace(df["Area (sq ft)"].min(), df["Area (sq ft)"].max(), 300)
    y_line = model.predict(x_range.reshape(-1, 1))
    ax1.plot(x_range, y_line, color="#e63946", linewidth=2.5, label="Regression line")
    ax1.set_xlabel("Area (sq ft)", fontsize=12)
    ax1.set_ylabel("Price (₹)", fontsize=12)
    ax1.set_title("House Area vs Price", fontsize=14, fontweight="bold")
    ax1.legend()
    ax1.grid(alpha=0.2)
    fig1.tight_layout()
    st.pyplot(fig1)

with tab2:
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.scatter(
        y_test, y_pred, alpha=0.6, color="#38b000", edgecolors="white", linewidth=0.5
    )
    mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    ax2.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Perfect prediction")
    ax2.set_xlabel("Actual Price (₹)", fontsize=12)
    ax2.set_ylabel("Predicted Price (₹)", fontsize=12)
    ax2.set_title("Actual vs Predicted Prices", fontsize=14, fontweight="bold")
    ax2.legend()
    ax2.grid(alpha=0.2)
    fig2.tight_layout()
    st.pyplot(fig2)

# ---------------------------------------------------------------------------
# Model Metrics Section
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown("## 📐 Model Evaluation Metrics")

m1, m2, m3, m4 = st.columns(4)
m1.metric("MAE", metrics["MAE (₹)"])
m2.metric("MSE", metrics["MSE"])
m3.metric("RMSE", metrics["RMSE (₹)"])
m4.metric("R² Score", metrics["R² Score"])

with st.expander("📖 What do these metrics mean?"):
    st.markdown(
        """
| Metric | Full Name | Meaning |
|--------|-----------|---------|
| **MAE** | Mean Absolute Error | Average difference between predicted and actual prices. Lower is better. |
| **MSE** | Mean Squared Error | Average of squared differences. Penalises large errors more. |
| **RMSE** | Root Mean Squared Error | Square root of MSE — same unit as price, easier to interpret. |
| **R²** | R-squared (Coefficient of Determination) | How much variance is explained by the model. 1.0 = perfect. |
        """
    )

# ---------------------------------------------------------------------------
# Disclaimer
# ---------------------------------------------------------------------------

st.markdown("---")
st.caption(
    "⚠️ **Disclaimer:** Predictions are estimates based on a limited dataset and a "
    "simple linear model. Real house prices depend on many more factors such as "
    "location, number of rooms, age of the property, and market conditions. "
    "This project is for educational purposes only."
)

st.markdown(
    "<div style='text-align:center; opacity:0.5; font-size:0.8rem; margin-top:2rem;'>"
    "Built with ❤️ using Python, Scikit-learn & Streamlit — BTech CSE Predictive Analytics Project"
    "</div>",
    unsafe_allow_html=True,
)
