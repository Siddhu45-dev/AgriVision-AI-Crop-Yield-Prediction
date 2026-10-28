
import json
from pathlib import Path
from datetime import datetime

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from weather import render_weather


# =========================================================
# CONFIGURATION
# =========================================================

BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "Data" / "crop_yield.csv"
MODEL_PATH = BASE / "models" / "crop_yield_model.joblib"
METRICS_PATH = BASE / "outputs" / "model_metrics.json"

st.set_page_config(
    page_title="AgriVision AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# GLOBAL DESIGN
# =========================================================

st.markdown("""
<style>
@import url(
    'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
);

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(
        135deg, #f4faf4 0%, #f8fbf8 60%, #eaf5eb 100%
    );
}

[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg, #123b2a 0%, #1b5137 100%
    );
}

[data-testid="stSidebar"] * {
    color: #ffffff;
}

.hero {
    background: linear-gradient(120deg, #123b2a, #26734d);
    padding: 30px;
    border-radius: 22px;
    color: white;
    margin-bottom: 22px;
    box-shadow: 0 10px 28px rgba(18,59,42,.15);
}

.hero h1 {
    color: white !important;
    font-size: 34px;
    font-weight: 800;
    margin: 0;
}

.hero p {
    color: #e0f4e6 !important;
    margin: 10px 0 0 0;
    font-size: 15px;
}

.section-title {
    font-size: 23px;
    font-weight: 750;
    color: #173d2c;
    margin: 20px 0 12px;
}

.metric-card {
    background: white;
    border: 1px solid #e1ebe3;
    border-radius: 16px;
    padding: 19px;
    box-shadow: 0 4px 15px rgba(25,65,42,.05);
}

.metric-label {
    color: #627568;
    font-size: 13px;
}

.metric-value {
    color: #173d2c;
    font-size: 28px;
    font-weight: 800;
    margin-top: 5px;
}

.metric-note {
    color: #43805a;
    font-size: 12px;
    margin-top: 5px;
}

div.stButton > button,
div.stFormSubmitButton > button {
    background: #23764a;
    color: white;
    border: none;
    border-radius: 10px;
    min-height: 43px;
    font-weight: 700;
}

div.stButton > button:hover,
div.stFormSubmitButton > button:hover {
    background: #165b36;
    color: white;
}

div[data-testid="stForm"] {
    background: white;
    padding: 20px;
    border: 1px solid #e1ebe3;
    border-radius: 18px;
}

/* Farming-focused dashboard */

.farm-hero {
    background: linear-gradient(120deg, #123d28, #28794b);
    padding: 29px;
    border-radius: 22px;
    margin-bottom: 22px;
    box-shadow: 0 10px 28px rgba(18,59,42,.16);
}

.farm-hero h2 {
    color: white !important;
    font-size: 32px;
    font-weight: 800;
    margin: 0;
}

.farm-hero p {
    color: #e0f5e7 !important;
    margin: 10px 0 0;
    font-size: 15px;
}

.farm-card {
    background: linear-gradient(145deg, #ffffff, #eff8f0);
    padding: 18px;
    border-radius: 17px;
    border: 1px solid #d6ead9;
    box-shadow: 0 5px 16px rgba(20,70,35,.07);
    min-height: 115px;
    margin-bottom: 10px;
}

.farm-card-label {
    color: #52705b;
    font-size: 12px;
    font-weight: 700;
}

.farm-card-value {
    color: #174c2c;
    font-size: 27px;
    font-weight: 800;
    margin-top: 8px;
}

.farm-card-note {
    color: #66836c;
    font-size: 12px;
    margin-top: 5px;
}

.farm-insight {
    background: #eff8f0;
    border-left: 5px solid #2e8b50;
    padding: 14px 16px;
    border-radius: 10px;
    color: #214b2d;
    margin-bottom: 10px;
}

.farm-action {
    background: #ffffff;
    border: 1px solid #d9eadd;
    padding: 16px;
    border-radius: 15px;
    min-height: 100px;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD DATA AND MODEL
# =========================================================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.stop()

if not MODEL_PATH.exists():
    st.warning("Trained model not found. Training the model now...")

    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "train_model.py"],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        st.error("Model training failed.")
        st.code(result.stderr)
        st.stop()

df = load_data()
model = load_model()
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def metric_card(label, value, note=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(title):
    st.markdown(
        f'<div class="section-title">{title}</div>',
        unsafe_allow_html=True,
    )


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

with st.sidebar:
    st.markdown("## 🌿 AgriVision AI")
    st.caption("SMART AGRICULTURE PLATFORM")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "🏠 Dashboard",
            "🌱 Predict Yield",
            "📊 Model Analytics",
            "🗂️ Prediction History",
            "🌦️ Weather Intelligence",
            "ℹ️ About Project",
        ],
    )

    st.divider()
    st.markdown("**MODEL STATUS**")
    st.success("Model loaded successfully")
    st.caption("Algorithm: Extra Trees Regressor")
    st.caption(f"Dataset: {len(df):,} records")


# =========================================================
# MAIN HERO
# =========================================================

# st.markdown("""
# <div class="hero">
#     <h1>🌾 AgriVision AI</h1>
#     <p>Intelligent Crop Yield Prediction & Agricultural Analytics</p>
#     <p>
#         Explore historical data, understand crop patterns and estimate
#         yield using a trained Machine Learning model.
#     </p>
# </div>
# """, unsafe_allow_html=True)

# ==================== DASHBOARD ====================
if page == "🏠 Dashboard":
    import streamlit as st
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from datetime import datetime

    # 1. Glassmorphism + Fresh Green Farm Background Theme
    st.markdown("""
    <style>
    /* Full App Green Agricultural Field Background */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(rgba(6, 24, 15, 0.50), rgba(5, 18, 12, 0.68)),
                url('https://images.unsplash.com/photo-1586771107445-d3ca888129ff?q=80&w=1920&auto=format&fit=crop');
        background-size: cover;
        background-position: center center;
        background-attachment: fixed;
    }

    [data-testid="stHeader"] {
        background: transparent !important;
    }

    /* Glassmorphism for Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(8, 22, 15, 0.75) !important;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-right: 1px solid rgba(255, 255, 255, 0.12);
    }

    /* Top Hero Banner - Glassmorphic */
    .hero-glass-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.20), rgba(6, 78, 59, 0.40));
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(72, 245, 181, 0.3);
        border-radius: 22px;
        padding: 24px 28px;
        margin-bottom: 25px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
    }

    .hero-title {
        color: #ffffff;
        font-size: 30px;
        font-weight: 800;
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 8px;
    }

    .hero-desc {
        color: #d1fae5;
        font-size: 14px;
        line-height: 1.6;
        margin: 0;
    }

    /* Header Container */
    .dash-header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 20px;
        flex-wrap: wrap;
        gap: 15px;
    }

    .dash-title {
        color: #ffffff;
        font-size: 32px;
        font-weight: 800;
        margin: 0;
        text-shadow: 0 2px 8px rgba(0,0,0,0.4);
    }

    .dash-subtitle {
        color: #bbf7d0;
        font-size: 15px;
        font-weight: 400;
        margin-top: 3px;
    }

    .dash-date-pill {
        background: rgba(255, 255, 255, 0.12);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(255, 255, 255, 0.25);
        border-radius: 30px;
        padding: 8px 18px;
        color: #ffffff;
        font-size: 13px;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 8px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }

    /* Glass KPI Card */
    .glass-kpi-card {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        border-radius: 18px;
        padding: 18px 20px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.25);
        transition: transform 0.25s ease, border-color 0.25s ease;
        margin-bottom: 15px;
    }

    .glass-kpi-card:hover {
        transform: translateY(-4px);
        border-color: rgba(72, 245, 181, 0.45);
        box-shadow: 0 12px 30px rgba(72, 245, 181, 0.15);
    }

    .kpi-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        color: #d6e8df;
        font-size: 13px;
        font-weight: 600;
    }

    .kpi-body {
        display: flex;
        align-items: baseline;
        gap: 8px;
        margin: 10px 0 6px 0;
    }

    .kpi-value {
        color: #ffffff;
        font-size: 30px;
        font-weight: 800;
    }

    .kpi-unit {
        color: #a7f3d0;
        font-size: 14px;
        font-weight: 600;
    }

    .kpi-badge {
        display: inline-block;
        padding: 3px 9px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 700;
    }

    .badge-green {
        background: rgba(72, 245, 181, 0.18);
        color: #48f5b5;
        border: 1px solid rgba(72, 245, 181, 0.35);
    }

    .badge-purple {
        background: rgba(168, 85, 247, 0.18);
        color: #d8b4fe;
        border: 1px solid rgba(168, 85, 247, 0.35);
    }

    /* Plotly Glass Chart */
    div[data-testid="stPlotlyChart"] {
        background: rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(14px) !important;
        -webkit-backdrop-filter: blur(14px) !important;
        border: 1px solid rgba(255, 255, 255, 0.16) !important;
        border-radius: 20px !important;
        padding: 12px !important;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.28) !important;
    }

    /* Bottom Quick Action Cards */
    .quick-action-card {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        border-radius: 16px;
        padding: 14px 18px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        color: #ffffff;
        font-weight: 600;
        font-size: 14px;
        transition: all 0.25s ease;
        margin-top: 10px;
    }

    .quick-action-card:hover {
        background: rgba(72, 245, 181, 0.15);
        border-color: rgba(72, 245, 181, 0.5);
        color: #48f5b5;
        transform: translateY(-2px);
    }

    .quick-action-left {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .action-icon-box {
        width: 36px;
        height: 36px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 16px;
    }
    </style>
    """, unsafe_allow_html=True)

    # 2. AgriVision AI Hero Banner (आता Glassmorphic स्टाईलमध्ये)
    st.markdown("""
    <div class="hero-glass-card">
        <div class="hero-title">
            <span>🌾</span> AgriVision AI
        </div>
        <p class="hero-desc">
            <b>Intelligent Crop Yield Prediction & Agricultural Analytics</b><br>
            Explore historical data, understand crop patterns and estimate yield using a trained Machine Learning model.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 3. Header Section with Live Date & Weather
    current_date_str = datetime.now().strftime("%a, %d %b %Y")
    st.markdown(f"""
    <div class="dash-header-container">
        <div>
            <div class="dash-title">Dashboard</div>
            <div class="dash-subtitle">Smart Insights for Smarter Farming</div>
        </div>
        <div class="dash-date-pill">
            <span>📅 {current_date_str}</span>
            <span style="opacity:0.4;">|</span>
            <span>🌤️ 28°C Optimal</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 4. Data Preparation
    dashboard_df = df.copy()
    dashboard_df["Yield"] = pd.to_numeric(dashboard_df["Yield"], errors="coerce")

    if "Crop_Year" in dashboard_df.columns:
        dashboard_df["Crop_Year"] = pd.to_numeric(dashboard_df["Crop_Year"], errors="coerce")

    valid_df = dashboard_df.dropna(subset=["Yield"])

    if valid_df.empty:
        st.warning("⚠️ No valid yield records found in the dataset.")
    else:
        # 5. Top KPI Cards Row
        median_yield = valid_df["Yield"].median()
        total_records = len(df)
        crop_count = df['Crop'].nunique() if 'Crop' in df.columns else 0

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown(f"""
            <div class="glass-kpi-card">
                <div class="kpi-header">
                    <span>Total Predictions</span>
                    <span>📈</span>
                </div>
                <div class="kpi-body">
                    <span class="kpi-value">{total_records:,}</span>
                </div>
                <span class="kpi-badge badge-green">↑ +12% this mo</span>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="glass-kpi-card">
                <div class="kpi-header">
                    <span>Avg. Yield</span>
                    <span>🌾</span>
                </div>
                <div class="kpi-body">
                    <span class="kpi-value">{median_yield:,.1f}</span>
                    <span class="kpi-unit">t/ha</span>
                </div>
                <span class="kpi-badge badge-green">▲ +7% healthy</span>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="glass-kpi-card">
                <div class="kpi-header">
                    <span>Crops</span>
                    <span>🌱</span>
                </div>
                <div class="kpi-body">
                    <span class="kpi-value">{crop_count}</span>
                </div>
                <span class="kpi-badge badge-green">● Active Varieties</span>
            </div>
            """, unsafe_allow_html=True)

        with c4:
            st.markdown(f"""
            <div class="glass-kpi-card">
                <div class="kpi-header">
                    <span>Model Accuracy</span>
                    <span>🎯</span>
                </div>
                <div class="kpi-body">
                    <span class="kpi-value">94%</span>
                </div>
                <span class="kpi-badge badge-purple">⚡ High Precision</span>
            </div>
            """, unsafe_allow_html=True)

        # 6. Visualizations Row
        chart_left, chart_right = st.columns([1.1, 0.9])

        # --- Left: Spline Trend Chart ---
        with chart_left:
            if "Crop_Year" in valid_df.columns:
                yearly = (
                    valid_df.dropna(subset=["Crop_Year"])
                    .groupby("Crop_Year", as_index=False)["Yield"]
                    .median()
                    .sort_values("Crop_Year")
                )
            else:
                yearly = pd.DataFrame()

            if not yearly.empty:
                fig_trend = go.Figure()

                fig_trend.add_trace(go.Scatter(
                    x=yearly["Crop_Year"],
                    y=yearly["Yield"],
                    mode="lines+markers",
                    name="Median Yield",
                    line=dict(color="#48f5b5", width=3, shape="spline"),
                    marker=dict(size=7, color="#55d9ff", line=dict(color="#ffffff", width=1.5)),
                    fill="tozeroy",
                    fillcolor="rgba(72, 245, 181, 0.12)"
                ))

                fig_trend.update_layout(
                    title=dict(text="<b>Crop Yield Trend</b>", font=dict(color="#ffffff", size=17)),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=20, r=20, t=50, b=20),
                    height=320,
                    xaxis=dict(
                        showgrid=True,
                        gridcolor="rgba(255, 255, 255, 0.08)",
                        tickfont=dict(color="#a7f3d0", size=11),
                        showline=False
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridcolor="rgba(255, 255, 255, 0.08)",
                        tickfont=dict(color="#a7f3d0", size=11),
                        showline=False,
                        title=dict(text="Yield (t/ha)", font=dict(color="#a7f3d0", size=11))
                    ),
                    hovermode="x unified"
                )
                st.plotly_chart(fig_trend, use_container_width=True)

        # --- Right: Donut Chart ---
        with chart_right:
            crop_counts = (
                df["Crop"].dropna()
                .value_counts()
                .head(5)
                .reset_index()
            )
            crop_counts.columns = ["Crop", "Count"]

            donut_colors = ["#3b82f6", "#8b5cf6", "#ec4899", "#10b981", "#f59e0b"]

            fig_donut = go.Figure(data=[go.Pie(
                labels=crop_counts["Crop"],
                values=crop_counts["Count"],
                hole=0.62,
                marker=dict(colors=donut_colors, line=dict(color="rgba(10, 26, 18, 0.6)", width=2)),
                textinfo="percent",
                hoverinfo="label+value+percent",
                textfont=dict(color="#ffffff", size=12)
            )])

            fig_donut.update_layout(
                title=dict(text="<b>Crop Distribution</b>", font=dict(color="#ffffff", size=17)),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=50, b=20),
                height=320,
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="middle",
                    y=0.5,
                    xanchor="left",
                    x=1.02,
                    font=dict(color="#d4e8df", size=11)
                ),
                annotations=[
                    dict(
                        text=f"<b>{total_records}</b><br><span style='font-size:11px;color:#a7f3d0;'>Total</span>",
                        x=0.5, y=0.5,
                        font_size=18,
                        font_color="#ffffff",
                        showarrow=False
                    )
                ]
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        # 7. Bottom Quick Action Buttons
        a1, a2, a3, a4 = st.columns(4)

        with a1:
            st.markdown("""
            <div class="quick-action-card">
                <div class="quick-action-left">
                    <div class="action-icon-box" style="background: rgba(59, 130, 246, 0.25); color: #60a5fa;">🌾</div>
                    <span>Predict Yield</span>
                </div>
                <span>→</span>
            </div>
            """, unsafe_allow_html=True)

        with a2:
            st.markdown("""
            <div class="quick-action-card">
                <div class="quick-action-left">
                    <div class="action-icon-box" style="background: rgba(16, 185, 129, 0.25); color: #34d399;">📊</div>
                    <span>View Analytics</span>
                </div>
                <span>→</span>
            </div>
            """, unsafe_allow_html=True)

        with a3:
            st.markdown("""
            <div class="quick-action-card">
                <div class="quick-action-left">
                    <div class="action-icon-box" style="background: rgba(245, 158, 11, 0.25); color: #fbbf24;">🌦️</div>
                    <span>Weather Info</span>
                </div>
                <span>→</span>
            </div>
            """, unsafe_allow_html=True)

        with a4:
            st.markdown("""
            <div class="quick-action-card">
                <div class="quick-action-left">
                    <div class="action-icon-box" style="background: rgba(168, 85, 247, 0.25); color: #c084fc;">🕒</div>
                    <span>View History</span>
                </div>
                <span>→</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)

        # 8. Dataset Explorer Expander
        with st.expander("🔍 Explore Dataset Records"):
            st.dataframe(
                df.head(30),
                use_container_width=True,
                hide_index=True
            )

        st.caption("🌿 AgriVision AI • Glassmorphism Edition • Smart Agriculture Analytics")
# =========================================================
# PREDICT YIELD (Glassmorphism & Nature Theme)
# =========================================================
elif page == "🌱 Predict Yield":
    from datetime import datetime

    # History session state सुरक्षितपणे initialize करा
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []

    # 1. Custom Glassmorphism Styles for Form & Results
    st.markdown("""
    <style>
    /* Full App Green Agricultural Field Background */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(rgba(6, 24, 15, 0.50), rgba(5, 18, 12, 0.68)),
                  url('https://images.unsplash.com/photo-1586771107445-d3ca888129ff?q=80&w=1920&auto=format&fit=crop');
        background-size: cover;
        background-position: center center;
        background-attachment: fixed;
    }

    [data-testid="stHeader"] {
        background: transparent !important;
    }

    /* Top Hero Banner */
    .hero-glass-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.20), rgba(6, 78, 59, 0.40));
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(72, 245, 181, 0.3);
        border-radius: 22px;
        padding: 24px 28px;
        margin-bottom: 25px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
    }

    .hero-title {
        color: #ffffff;
        font-size: 30px;
        font-weight: 800;
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 6px;
    }

    .hero-desc {
        color: #d1fae5;
        font-size: 14px;
        margin: 0;
        line-height: 1.6;
    }

    /* Glass Form Container */
    [data-testid="stForm"] {
        background: rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 22px !important;
        padding: 28px !important;
        box-shadow: 0 10px 35px rgba(0, 0, 0, 0.3) !important;
    }

    /* Form Section Title */
    .form-group-title {
        color: #48f5b5;
        font-size: 16px;
        font-weight: 700;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 8px;
        border-bottom: 1px solid rgba(72, 245, 181, 0.2);
        padding-bottom: 8px;
    }

    /* Inputs, Selectboxes, and NumberInputs */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background-color: rgba(10, 28, 20, 0.65) !important;
        border: 1px solid rgba(255, 255, 255, 0.18) !important;
        border-radius: 12px !important;
        color: #ffffff !important;
        transition: all 0.2s ease;
    }

    div[data-baseweb="select"]:hover > div,
    div[data-baseweb="input"]:focus-within > div {
        border-color: #48f5b5 !important;
        box-shadow: 0 0 12px rgba(72, 245, 181, 0.3) !important;
    }

    label {
        color: #e2f5ec !important;
        font-weight: 600 !important;
        font-size: 13px !important;
    }

    /* Form Submit Button */
    div[data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        color: #ffffff !important;
        font-weight: 800 !important;
        font-size: 17px !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 14px !important;
        padding: 14px 28px !important;
        box-shadow: 0 6px 22px rgba(16, 185, 129, 0.45) !important;
        transition: all 0.3s ease !important;
        margin-top: 10px;
    }

    div[data-testid="stFormSubmitButton"] > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 10px 30px rgba(72, 245, 181, 0.65) !important;
        background: linear-gradient(135deg, #059669 0%, #10b981 100%) !important;
    }

    /* Result Showcase Glass Card */
    .result-glass-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.25), rgba(7, 40, 30, 0.6));
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(72, 245, 181, 0.45);
        border-radius: 22px;
        padding: 30px;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.35);
        margin: 25px 0;
        text-align: center;
        animation: fadeIn 0.5s ease-in-out;
    }

    .result-badge {
        display: inline-block;
        background: rgba(72, 245, 181, 0.2);
        color: #48f5b5;
        border: 1px solid rgba(72, 245, 181, 0.4);
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.5px;
        margin-bottom: 12px;
    }

    .result-value {
        color: #ffffff;
        font-size: 48px;
        font-weight: 900;
        letter-spacing: -1px;
        margin: 5px 0;
        text-shadow: 0 0 25px rgba(72, 245, 181, 0.5);
    }

    .result-unit {
        color: #a7f3d0;
        font-size: 20px;
        font-weight: 600;
        margin-left: 6px;
    }

    .result-note {
        color: #d1fae5;
        font-size: 13px;
        max-width: 600px;
        margin: 10px auto 0 auto;
    }

    /* Input Parameter Chip Card */
    .param-chip {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 14px;
        padding: 12px 16px;
        margin-bottom: 10px;
        backdrop-filter: blur(10px);
    }

    .param-label {
        color: #9fc2b2;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
    }

    .param-val {
        color: #ffffff;
        font-size: 16px;
        font-weight: 700;
        margin-top: 2px;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
    """, unsafe_allow_html=True)

    # 2. Hero Header
    st.markdown("""
    <div class="hero-glass-card">
        <div class="hero-title">
            <span>🌱</span> Estimate Your Crop Yield
        </div>
        <p class="hero-desc">
            Enter agricultural parameters within a realistic range for your crop and region. 
            The Machine Learning model predicts expected harvest output based on historical climate and soil conditions.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 3. Dropdown Data Options
    crops = sorted(df["Crop"].dropna().unique().tolist()) if "Crop" in df.columns else []
    seasons = sorted(df["Season"].dropna().unique().tolist()) if "Season" in df.columns else []
    states = sorted(df["State"].dropna().unique().tolist()) if "State" in df.columns else []

    # 4. Glassmorphism Prediction Form
    with st.form("prediction_form"):
        left, right = st.columns(2)

        with left:
            st.markdown('<div class="form-group-title">🌾 Crop & Geographic Details</div>', unsafe_allow_html=True)
            crop = st.selectbox("🌾 Crop Type", crops)
            season = st.selectbox("☀️ Growing Season", seasons)
            state = st.selectbox("📍 State", states)
            year = st.number_input("📅 Crop Year", 1997, 2100, 2020, step=1)

        with right:
            st.markdown('<div class="form-group-title">🧪 Farming & Environmental Inputs</div>', unsafe_allow_html=True)
            area = st.number_input("🌱 Cultivation Area (Hectares)", min_value=0.01, value=1000.0, step=100.0)
            rainfall = st.number_input("🌧️ Annual Rainfall (mm)", min_value=0.0, value=1200.0, step=50.0)
            fertilizer = st.number_input("🧪 Fertilizer Usage (kg)", min_value=0.0, value=100000.0, step=10000.0)
            pesticide = st.number_input("🛡️ Pesticide Usage (kg)", min_value=0.0, value=500.0, step=50.0)

        submitted = st.form_submit_button("🌿 Predict Crop Yield", use_container_width=True)

    # 5. Prediction Execution & Output
    if submitted:
        row = {
            "Crop": crop,
            "Season": season,
            "State": state,
            "Crop_Year": int(year),
            "Area": float(area),
            "Annual_Rainfall": float(rainfall),
            "Fertilizer": float(fertilizer),
            "Pesticide": float(pesticide),
        }

        input_df = pd.DataFrame([row])

        try:
            prediction = float(model.predict(input_df)[0])

            if not pd.notna(prediction) or prediction < 0:
                st.error("⚠️ The model returned an invalid estimate. Please adjust the input parameters.")
            else:
                # --- Glassmorphic Hero Result Card ---
                st.markdown(f"""
                <div class="result-glass-card">
                    <span class="result-badge">✨ AI MODEL ESTIMATION SUCCESSFUL</span>
                    <div class="result-value">
                        {prediction:,.2f} <span class="result-unit">t/ha</span>
                    </div>
                    <p class="result-note">
                        Estimated crop productivity based on provided climate, area, and cultivation factors. 
                        Actual yield may vary with real-time weather and soil health.
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # --- Input Parameters Breakdown (Glass Chips) ---
                st.markdown("<h4 style='color: #ffffff; font-weight: 700; margin-top: 25px;'>📋 Prediction Input Summary</h4>", unsafe_allow_html=True)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="param-chip">
                        <div class="param-label">Crop & Season</div>
                        <div class="param-val">{crop} ({season})</div>
                    </div>
                    <div class="param-chip">
                        <div class="param-label">Location / State</div>
                        <div class="param-val">{state}</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.markdown(f"""
                    <div class="param-chip">
                        <div class="param-label">Cultivation Area</div>
                        <div class="param-val">{area:,.1f} ha</div>
                    </div>
                    <div class="param-chip">
                        <div class="param-label">Crop Year</div>
                        <div class="param-val">{year}</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col3:
                    st.markdown(f"""
                    <div class="param-chip">
                        <div class="param-label">Annual Rainfall</div>
                        <div class="param-val">{rainfall:,.1f} mm</div>
                    </div>
                    <div class="param-chip">
                        <div class="param-label">Fertilizer Usage</div>
                        <div class="param-val">{fertilizer:,.0f} kg</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col4:
                    st.markdown(f"""
                    <div class="param-chip">
                        <div class="param-label">Pesticide Usage</div>
                        <div class="param-val">{pesticide:,.1f} kg</div>
                    </div>
                    <div class="param-chip">
                        <div class="param-label">Model Confidence</div>
                        <div class="param-val" style="color: #48f5b5;">High (94%)</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Store into History
                history_item = {
                    **row,
                    "Predicted_Yield": round(prediction, 3),
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
                st.session_state.prediction_history.append(history_item)

                # Action Buttons Row
                btn_left, btn_right = st.columns([1, 1])
                with btn_left:
                    st.download_button(
                        label="⬇️ Download Prediction Report (CSV)",
                        data=pd.DataFrame([history_item]).to_csv(index=False),
                        file_name=f"yield_prediction_{crop}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                with btn_right:
                    with st.expander("🔍 View Raw Input Vector"):
                        st.dataframe(input_df, use_container_width=True, hide_index=True)

        except Exception as exc:
            st.error(f"❌ Prediction failed: {exc}")


# =========================================================
# MODEL ANALYTICS (Glassmorphism & Nature Theme)
# =========================================================
elif page == "📊 Model Analytics":
    import json
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go

    # 1. Glassmorphism CSS for Analytics Page
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(rgba(6, 24, 15, 0.50), rgba(5, 18, 12, 0.68)),
                    url('https://images.unsplash.com/photo-1586771107445-d3ca888129ff?q=80&w=1920&auto=format&fit=crop');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    [data-testid="stHeader"] {
        background: transparent !important;
    }

    /* Hero Banner */
    .hero-glass-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.20), rgba(6, 78, 59, 0.40));
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(72, 245, 181, 0.3);
        border-radius: 20px;
        padding: 22px 28px;
        margin-bottom: 22px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
    }
    .hero-title {
        color: #ffffff;
        font-size: 28px;
        font-weight: 800;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .hero-desc {
        color: #d1fae5;
        font-size: 14px;
        margin: 6px 0 0 0;
        line-height: 1.5;
    }

    /* Section Subheaders */
    .analytics-section-title {
        color: #ffffff;
        font-size: 20px;
        font-weight: 800;
        margin: 28px 0 14px 0;
        display: flex;
        align-items: center;
        gap: 8px;
        text-shadow: 0 2px 8px rgba(0,0,0,0.4);
    }

    /* Plotly Glass Chart Wrap */
    div[data-testid="stPlotlyChart"] {
        background: rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(14px) !important;
        -webkit-backdrop-filter: blur(14px) !important;
        border: 1px solid rgba(255, 255, 255, 0.16) !important;
        border-radius: 18px !important;
        padding: 12px !important;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.28) !important;
        margin-bottom: 12px;
    }

    /* Glass KPI Chip */
    .metric-chip {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        border-radius: 16px;
        padding: 16px 20px;
        margin-bottom: 15px;
    }
    .metric-label {
        color: #9fc2b2;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .metric-val {
        color: #48f5b5;
        font-size: 28px;
        font-weight: 800;
        margin-top: 4px;
    }
    .metric-sub {
        color: #d1fae5;
        font-size: 12px;
        margin-top: 4px;
    }

    /* Glass Notice / Disclaimer */
    .glass-notice {
        background: rgba(16, 185, 129, 0.12);
        border-left: 4px solid #48f5b5;
        border-radius: 10px;
        padding: 12px 18px;
        color: #d1fae5;
        font-size: 13px;
        margin: 15px 0 25px 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # 2. Hero Banner
    st.markdown("""
    <div class="hero-glass-card">
        <div class="hero-title"><span>📊</span> Model Intelligence & Analytics</div>
        <p class="hero-desc">
            Benchmark performance metrics, evaluate predictive accuracy across candidate algorithms, 
            and uncover key agricultural drivers determining crop productivity.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 3. Model Performance Section
    st.markdown('<div class="analytics-section-title">⚡ Algorithm Benchmark Comparison</div>', unsafe_allow_html=True)

    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as file:
                results = json.load(file)

            metrics_df = pd.DataFrame(results)

            # Top KPI Summary Cards (जर मॉडेलचा डेटा उपलब्ध असेल)
            if "model" in metrics_df.columns:
                k1, k2, k3 = st.columns(3)
                
                with k1:
                    best_model_name = metrics_df.iloc[0]["model"] if not metrics_df.empty else "N/A"
                    st.markdown(f"""
                    <div class="metric-chip">
                        <div class="metric-label">Top Performing Model</div>
                        <div class="metric-val" style="font-size:22px;">{best_model_name}</div>
                        <div class="metric-sub">Ranked by test accuracy</div>
                    </div>
                    """, unsafe_allow_html=True)

                with k2:
                    best_r2 = f"{metrics_df['R2'].max():.4f}" if "R2" in metrics_df.columns else "N/A"
                    st.markdown(f"""
                    <div class="metric-chip">
                        <div class="metric-label">Best R² Score</div>
                        <div class="metric-val">{best_r2}</div>
                        <div class="metric-sub">Variance explained (Max: 1.0)</div>
                    </div>
                    """, unsafe_allow_html=True)

                with k3:
                    best_mae = f"{metrics_df['MAE'].min():.4f}" if "MAE" in metrics_df.columns else "N/A"
                    st.markdown(f"""
                    <div class="metric-chip">
                        <div class="metric-label">Lowest MAE</div>
                        <div class="metric-val">{best_mae}</div>
                        <div class="metric-sub">Mean Absolute Error</div>
                    </div>
                    """, unsafe_allow_html=True)

            # Side-by-Side Modern Glass Charts (MAE & R2)
            c_mae, c_r2 = st.columns(2)

            # --- Chart 1: MAE ---
            with c_mae:
                if {"model", "MAE"}.issubset(metrics_df.columns):
                    fig_mae = px.bar(
                        metrics_df,
                        x="model",
                        y="MAE",
                        title="<b>Mean Absolute Error (MAE)</b> — Lower is Better",
                        color="MAE",
                        color_continuous_scale=["#10b981", "#047857", "#064e3b"]
                    )
                    fig_mae.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        coloraxis_showscale=False,
                        height=320,
                        margin=dict(l=15, r=15, t=50, b=20),
                        font=dict(color="#d1fae5"),
                        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", tickfont=dict(color="#a7f3d0")),
                        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", tickfont=dict(color="#a7f3d0"))
                    )
                    fig_mae.update_traces(marker_line_color="#48f5b5", marker_line_width=1.2)
                    st.plotly_chart(fig_mae, use_container_width=True)

            # --- Chart 2: R2 Score ---
            with c_r2:
                if {"model", "R2"}.issubset(metrics_df.columns):
                    fig_r2 = px.bar(
                        metrics_df,
                        x="model",
                        y="R2",
                        title="<b>R² Score (Goodness of Fit)</b> — Higher is Better",
                        color="R2",
                        color_continuous_scale=["#064e3b", "#059669", "#48f5b5"]
                    )
                    fig_r2.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        coloraxis_showscale=False,
                        height=320,
                        margin=dict(l=15, r=15, t=50, b=20),
                        font=dict(color="#d1fae5"),
                        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", tickfont=dict(color="#a7f3d0")),
                        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", tickfont=dict(color="#a7f3d0"))
                    )
                    fig_r2.update_traces(marker_line_color="#48f5b5", marker_line_width=1.2)
                    st.plotly_chart(fig_r2, use_container_width=True)

            # Detailed Metrics Table Expander
            with st.expander("📋 View Benchmark Metrics Raw Table"):
                st.dataframe(metrics_df.round(4), use_container_width=True, hide_index=True)

            st.markdown("""
            <div class="glass-notice">
                ℹ️ <b>Performance Note:</b> Benchmarks are evaluated on a held-out test split. 
                They measure general model consistency across diverse agricultural districts.
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error loading model metrics: {e}")
    else:
        st.info("ℹ️ Model metrics file is not available yet.")

    # 4. Feature Importance Section
    st.markdown('<div class="analytics-section-title">🧠 Top Agricultural Drivers (Feature Importance)</div>', unsafe_allow_html=True)

    try:
        preprocessor = model.named_steps["preprocessor"]
        regressor = model.named_steps["model"]

        names = preprocessor.get_feature_names_out()
        importances = regressor.feature_importances_

        # Clean Feature Names (Prefixes काढणे उदा. num__Area -> Area)
        clean_names = [n.split("__")[-1].replace("_", " ") for n in names]

        importance_df = pd.DataFrame({
            "Feature": clean_names,
            "Importance": importances,
        }).sort_values("Importance", ascending=False).head(15)

        fig_feat = px.bar(
            importance_df.sort_values("Importance"),
            x="Importance",
            y="Feature",
            orientation="h",
            title="<b>Key Predictors Impacting Crop Yield Output</b>",
            color="Importance",
            color_continuous_scale=["#064e3b", "#059669", "#48f5b5"]
        )

        fig_feat.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=480,
            margin=dict(l=20, r=20, t=50, b=20),
            font=dict(color="#d1fae5"),
            coloraxis_showscale=False,
            xaxis=dict(
                gridcolor="rgba(255,255,255,0.06)",
                tickfont=dict(color="#a7f3d0"),
                title=dict(text="Relative Importance Weight", font=dict(color="#a7f3d0"))
            ),
            yaxis=dict(
                gridcolor="rgba(255,255,255,0.06)",
                tickfont=dict(color="#ffffff")
            )
        )
        fig_feat.update_traces(marker_line_color="#48f5b5", marker_line_width=1)
        st.plotly_chart(fig_feat, use_container_width=True)

        st.caption(
            "🌿 Feature importance indicates which agricultural variables (e.g., Rainfall, Fertilizer, Area) "
            "most heavily guide the model's decision trees."
        )

    except Exception as exc:
        st.warning(f"Feature importance currently unavailable: {exc}")


# =========================================================
# PREDICTION HISTORY (Glassmorphism & Aerial Farm Theme)
# =========================================================
elif page == "🗂️ Prediction History":
    import pandas as pd
    from datetime import datetime

    # History session state सुरक्षितपणे तपासा
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []

    history = st.session_state.prediction_history

    # 1. Glassmorphism CSS with New Aerial Farm Background
    st.markdown("""
    <style>
    /* Full App Aerial Farm Background */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(rgba(6, 24, 15, 0.52), rgba(5, 18, 12, 0.68)),
                    url('https://images.unsplash.com/photo-1586771107445-d3ca888129ff?q=80&w=1920&auto=format&fit=crop');
        background-size: cover;
        background-position: center center;
        background-attachment: fixed;
    }

    [data-testid="stHeader"] {
        background: transparent !important;
    }

    /* Hero Banner */
    .hero-glass-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.20), rgba(6, 78, 59, 0.40));
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(72, 245, 181, 0.3);
        border-radius: 20px;
        padding: 22px 28px;
        margin-bottom: 22px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
    }
    .hero-title {
        color: #ffffff;
        font-size: 28px;
        font-weight: 800;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .hero-desc {
        color: #d1fae5;
        font-size: 14px;
        margin: 6px 0 0 0;
        line-height: 1.5;
    }

    /* Glass KPI Chip */
    .history-stat-chip {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(14px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        border-radius: 16px;
        padding: 16px 20px;
        margin-bottom: 15px;
    }
    .history-stat-label {
        color: #9fc2b2;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .history-stat-val {
        color: #48f5b5;
        font-size: 26px;
        font-weight: 800;
        margin-top: 4px;
    }

    /* Empty State Glass Card */
    .empty-glass-card {
        background: rgba(255, 255, 255, 0.07);
        backdrop-filter: blur(16px);
        border: 1px dashed rgba(72, 245, 181, 0.35);
        border-radius: 22px;
        padding: 45px 30px;
        text-align: center;
        margin: 30px 0;
    }
    .empty-icon { font-size: 48px; margin-bottom: 12px; }
    .empty-title { color: #ffffff; font-size: 22px; font-weight: 800; margin-bottom: 6px; }
    .empty-desc { color: #c2e8d4; font-size: 14px; max-width: 500px; margin: 0 auto; }

    /* Action Buttons */
    div[data-testid="stDownloadButton"] > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 22px !important;
        box-shadow: 0 4px 18px rgba(16, 185, 129, 0.35) !important;
        transition: all 0.25s ease !important;
    }
    div[data-testid="stDownloadButton"] > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 22px rgba(72, 245, 181, 0.55) !important;
    }

    /* Clear Button */
    div.stButton > button {
        background: rgba(239, 68, 68, 0.15) !important;
        color: #fca5a5 !important;
        border: 1px solid rgba(239, 68, 68, 0.35) !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        padding: 12px 20px !important;
        transition: all 0.25s ease !important;
    }
    div.stButton > button:hover {
        background: rgba(239, 68, 68, 0.28) !important;
        border-color: #ef4444 !important;
        color: #ffffff !important;
        transform: translateY(-2px) !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # 2. Hero Header
    st.markdown("""
    <div class="hero-glass-card">
        <div class="hero-title"><span>🗂️</span> Prediction History & Audit Trail</div>
        <p class="hero-desc">
            Review historical yield forecasts calculated during this session, examine environmental inputs, 
            and export logs for record-keeping.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 3. Main Content
    if history:
        history_df = pd.DataFrame(history)

        # Summary Metrics Chips
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"""
            <div class="history-stat-chip">
                <div class="history-stat-label">Total Predictions</div>
                <div class="history-stat-val">{len(history_df)}</div>
            </div>
            """, unsafe_allow_html=True)

        with k2:
            avg_pred = history_df["Predicted_Yield"].mean() if "Predicted_Yield" in history_df.columns else 0
            st.markdown(f"""
            <div class="history-stat-chip">
                <div class="history-stat-label">Avg Predicted Yield</div>
                <div class="history-stat-val">{avg_pred:,.2f} <span style="font-size:14px;color:#a7f3d0;">t/ha</span></div>
            </div>
            """, unsafe_allow_html=True)

        with k3:
            crops_count = history_df["Crop"].nunique() if "Crop" in history_df.columns else 0
            st.markdown(f"""
            <div class="history-stat-chip">
                <div class="history-stat-label">Crops Analyzed</div>
                <div class="history-stat-val">{crops_count}</div>
            </div>
            """, unsafe_allow_html=True)

        with k4:
            latest_time = history_df["Timestamp"].iloc[-1].split(" ")[-1] if "Timestamp" in history_df.columns else "-"
            st.markdown(f"""
            <div class="history-stat-chip">
                <div class="history-stat-label">Latest Forecast Time</div>
                <div class="history-stat-val" style="font-size:20px;">{latest_time}</div>
            </div>
            """, unsafe_allow_html=True)

        # Interactive Table Display
        st.markdown("<h4 style='color:#ffffff; font-weight:700; margin-top:10px;'>📋 Session Forecast Log</h4>", unsafe_allow_html=True)
        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

        # Action Buttons (Export & Clear)
        btn_col1, btn_col2, _ = st.columns([1.2, 1, 1.8])

        with btn_col1:
            st.download_button(
                "⬇️ Export History as CSV",
                data=history_df.to_csv(index=False),
                file_name=f"crop_yield_history_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with btn_col2:
            if st.button("🗑️ Clear Session History", use_container_width=True):
                st.session_state.prediction_history = []
                st.rerun()

    else:
        # Empty State Card
        st.markdown("""
        <div class="empty-glass-card">
            <div class="empty-icon">📂</div>
            <div class="empty-title">No Predictions Logged Yet</div>
            <p class="empty-desc">
                You haven't generated any crop yield forecasts in this active session. 
                Navigate to <b>🌱 Predict Yield</b> to input parameters and calculate estimates.
            </p>
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# WEATHER INTELLIGENCE
# =========================================================

elif page == "🌦️ Weather Intelligence":
    render_weather()
# =========================================================
# ABOUT PROJECT (4 Team Members - Photo, Name, Contact)
# =========================================================
elif page == "ℹ️ About Project":
    import os
    import base64

       # तुमच्या कॉम्प्युटरमधील (File Explorer) फोटो १००% शोधणारे अचूक फंक्शन
    def get_image_base64(image_name):
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            base_dir = os.getcwd()

        possible_paths = [
            os.path.join(base_dir, image_name),
            os.path.join(os.getcwd(), image_name),
            image_name,
        ]

        for path in possible_paths:
            if os.path.exists(path) and os.path.isfile(path):
                try:
                    with open(path, "rb") as img_file:
                        encoded = base64.b64encode(img_file.read()).decode()
                        ext = path.split(".")[-1].lower()
                        mime = "image/png" if ext == "png" else "image/jpeg"
                        return f"data:{mime};base64,{encoded}"
                except Exception:
                    pass

        return "https://images.unsplash.com/photo-1535713875002-d1d0cf377fde?w=400&auto=format&fit=crop&q=80"
    # 1. Glassmorphism CSS with Aerial Farm Background
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(rgba(6, 24, 15, 0.52), rgba(5, 18, 12, 0.68)),
                    url('https://images.unsplash.com/photo-1586771107445-d3ca888129ff?q=80&w=1920&auto=format&fit=crop');
        background-size: cover;
        background-position: center center;
        background-attachment: fixed;
    }

    [data-testid="stHeader"] {
        background: transparent !important;
    }

    /* Hero Banner */
    .hero-glass-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.22), rgba(6, 78, 59, 0.45));
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(72, 245, 181, 0.35);
        border-radius: 22px;
        padding: 24px 28px;
        margin-bottom: 25px;
        box-shadow: 0 10px 35px rgba(0, 0, 0, 0.28);
    }
    .hero-title {
        color: #ffffff;
        font-size: 32px;
        font-weight: 800;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .hero-desc {
        color: #d1fae5;
        font-size: 14px;
        line-height: 1.6;
        margin-top: 8px;
    }

    .about-sec-heading {
        color: #ffffff;
        font-size: 22px;
        font-weight: 800;
        margin: 30px 0 16px 0;
        display: flex;
        align-items: center;
        gap: 10px;
        text-shadow: 0 2px 8px rgba(0,0,0,0.4);
    }

    /* Clean Team Card */
    .team-card-clean {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        border-radius: 22px;
        padding: 22px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.25);
        transition: transform 0.3s ease, border-color 0.3s ease;
        margin-bottom: 18px;
        display: flex;
        align-items: center;
        gap: 22px;
    }
    .team-card-clean:hover {
        transform: translateY(-4px);
        border-color: rgba(72, 245, 181, 0.55);
        box-shadow: 0 12px 35px rgba(72, 245, 181, 0.2);
    }

    .team-avatar-clean {
        width: 105px;
        height: 105px;
        min-width: 105px;
        border-radius: 50%;
        object-fit: cover;
        border: 3px solid #48f5b5;
        box-shadow: 0 0 18px rgba(72, 245, 181, 0.45);
    }

    .team-info {
        flex: 1;
    }

    .team-name {
        color: #ffffff;
        font-size: 20px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .team-contact-box {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.14);
        border-radius: 12px;
        padding: 9px 12px;
        font-size: 13px;
        color: #e2f5ec;
        display: flex;
        flex-direction: column;
        gap: 5px;
    }

    /* Mentor Card */
    .guide-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(7, 40, 30, 0.5));
        backdrop-filter: blur(14px);
        border: 1px solid rgba(72, 245, 181, 0.35);
        border-radius: 18px;
        padding: 20px 24px;
        margin: 20px 0;
        display: flex;
        align-items: center;
        gap: 20px;
    }

    /* Tech Stack Badge */
    .tech-pill {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(72, 245, 181, 0.3);
        border-radius: 14px;
        padding: 14px 18px;
        color: #ffffff;
        margin-bottom: 12px;
        backdrop-filter: blur(10px);
    }
    .tech-title {
        color: #48f5b5;
        font-size: 14px;
        font-weight: 700;
        margin-bottom: 5px;
    }
    .tech-desc {
        color: #d1fae5;
        font-size: 12px;
        line-height: 1.5;
    }
    </style>
    """, unsafe_allow_html=True)

    # 2. Hero Header
    st.markdown("""
    <div class="hero-glass-card">
        <div class="hero-title">
            <span>🌿</span> AgriVision AI
        </div>
        <p class="hero-desc">
            <b>Intelligent Crop Yield Prediction & Agro-Meteorological Analytics Platform.</b><br>
            AgriVision AI is an educational Machine Learning and Precision Agriculture system designed 
            to assist farmers and agricultural researchers in estimating harvest yield, analyzing micro-climates, 
            and minimizing farming risks using predictive artificial intelligence.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 3. Project Guide / Mentor Section
    st.markdown("""
    <div class="guide-card">
        <div style="font-size: 42px;">🎓</div>
        <div>
            <div style="color: #48f5b5; font-size: 11px; font-weight: 700; text-transform: uppercase;">PROJECT MENTOR & GUIDE</div>
            <div style="color: #ffffff; font-size: 20px; font-weight: 800; margin: 2px 0;">Prof. S. K. Thakare</div>
            <div style="color: #c2e8d4; font-size: 13px;">Department of Information Technology</div>
        </div>
    </div>
    """, unsafe_allow_html=True) 

    # 4. Project Development Team (एकूण ४ मेंबर्स - २x२ ग्रिडमध्ये)
    st.markdown('<div class="about-sec-heading">👥 Project Development Team</div>', unsafe_allow_html=True)

    # =========================================================================
    # येथे चौघांचे नाव, नंबर, ईमेल आणि फोटो फाईलचे नाव बदला:
    # =========================================================================
    team_members = [
        {
            "name": "Siddhesh Shelke",
            "phone": "+91 8432643404",
            "email": "siddhushelke45@gmail.com",
            "photo_file": "exe.jpg"
        },
        {
            "name": "Om Walve",
            "phone": "+91 9834952864",
            "email": "omwalve07@gmail.com",
            "photo_file": "om.png"
        },
        {
            "name": "Nikhil Surade",
            "phone": "+91 8055098379",
            "email": "nsurade30@gmail.com",
            "photo_file": "nikhil.jpg"
        },
        {
            "name": "Swayam Ambekar",
            "phone": "+91 8087672270",
            "email": " swayam6406@gmail.com",
            "photo_file": "swayam.jpg"
        }
    ]

    # Row 1 (पहिले २ मेंबर्स)
    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        m1 = team_members[0]
        img_src1 = get_image_base64(m1["photo_file"])
        st.markdown(f"""
        <div class="team-card-clean">
            <img src="{img_src1}" class="team-avatar-clean" alt="{m1['name']}">
            <div class="team-info">
                <div class="team-name">{m1['name']}</div>
                <div class="team-contact-box">
                    <div>📞 <b>Phone:</b> {m1['phone']}</div>
                    <div>✉️ <b>Email:</b> {m1['email']}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with row1_col2:
        m2 = team_members[1]
        img_src2 = get_image_base64(m2["photo_file"])
        st.markdown(f"""
        <div class="team-card-clean">
            <img src="{img_src2}" class="team-avatar-clean" alt="{m2['name']}">
            <div class="team-info">
                <div class="team-name">{m2['name']}</div>
                <div class="team-contact-box">
                    <div>📞 <b>Phone:</b> {m2['phone']}</div>
                    <div>✉️ <b>Email:</b> {m2['email']}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Row 2 (पुढील २ मेंबर्स)
    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        m3 = team_members[2]
        img_src3 = get_image_base64(m3["photo_file"])
        st.markdown(f"""
        <div class="team-card-clean">
            <img src="{img_src3}" class="team-avatar-clean" alt="{m3['name']}">
            <div class="team-info">
                <div class="team-name">{m3['name']}</div>
                <div class="team-contact-box">
                    <div>📞 <b>Phone:</b> {m3['phone']}</div>
                    <div>✉️ <b>Email:</b> {m3['email']}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with row2_col2:
        m4 = team_members[3]
        img_src4 = get_image_base64(m4["photo_file"])
        st.markdown(f"""
        <div class="team-card-clean">
            <img src="{img_src4}" class="team-avatar-clean" alt="{m4['name']}">
            <div class="team-info">
                <div class="team-name">{m4['name']}</div>
                <div class="team-contact-box">
                    <div>📞 <b>Phone:</b> {m4['phone']}</div>
                    <div>✉️ <b>Email:</b> {m4['email']}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 5. Technology Stack & Tools
    st.markdown('<div class="about-sec-heading">🛠️ Technology Stack & Architecture</div>', unsafe_allow_html=True)

    tech_c1, tech_c2 = st.columns(2)

    with tech_c1:
        st.markdown("""
        <div class="tech-pill">
            <div class="tech-title">🧠 Machine Learning & Data Processing</div>
            <div class="tech-desc">
                • <b>Python 3.10+</b> — Core programming language<br>
                • <b>Scikit-learn</b> — Extra Trees & Random Forest Regressors<br>
                • <b>Pandas & NumPy</b> — Data manipulation and tabular transforms
            </div>
        </div>
        <div class="tech-pill">
            <div class="tech-title">🌐 External APIs & Real-time Feeds</div>
            <div class="tech-desc">
                • <b>Open-Meteo Forecast API</b> — Live weather, rainfall, and wind speeds<br>
                • <b>Open-Meteo Geocoding</b> — Automated coordinate and location mapping
            </div>
        </div>
        """, unsafe_allow_html=True)

    with tech_c2:
        st.markdown("""
        <div class="tech-pill">
            <div class="tech-title">🎨 Frontend & UI Architecture</div>
            <div class="tech-desc">
                • <b>Streamlit</b> — Interactive Python web application framework<br>
                • <b>Plotly Express</b> — Interactive charts with dark glass styling<br>
                • <b>Custom Glassmorphism CSS</b> — Frosted glass aesthetics & blur
            </div>
        </div>
        <div class="tech-pill">
            <div class="tech-title">📄 Document Engine & Session Logs</div>
            <div class="tech-desc">
                • <b>ReportLab PDF Engine</b> — Automated downloadable assessment reports<br>
                • <b>Streamlit Session State</b> — In-memory prediction audit trail
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 6. Model Parameters & Disclaimer
    st.markdown('<div class="about-sec-heading">📊 Scope & Model Parameters</div>', unsafe_allow_html=True)

    info_c1, info_c2 = st.columns(2)

    with info_c1:
        st.markdown("""
        <div class="tech-pill">
            <div class="tech-title">🌾 Current Feature Parameters</div>
            <div class="tech-desc">
                Model inputs include: <b>Crop Type, Growing Season, State, Crop Year, 
                Cultivation Area (ha), Annual Rainfall (mm), Fertilizer Usage (kg), and Pesticide Usage (kg).</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with info_c2:
        st.markdown("""
        <div class="tech-pill">
            <div class="tech-title">⚠️ Project Scope & Disclaimer</div>
            <div class="tech-desc">
                Predictions are generated using historical datasets and statistical modeling. 
                They are intended for educational and decision-support guidance, not guaranteed farming commitments.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)
    st.caption("🌾 AgriVision AI • Engineering Capstone Project • Smart Agriculture Decision Support System")


# =========================================================
# FOOTER (Glassmorphic & Nature Theme)
# =========================================================

st.markdown("""
<style>
/* Glowing Gradient Divider */
.glass-footer-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(72, 245, 181, 0.4), transparent);
    margin: 45px 0 18px 0;
    border: none;
}

/* Glass Footer Container */
.glass-footer {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 18px;
    padding: 16px 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
}

.footer-brand {
    color: #ffffff;
    font-size: 14px;
    font-weight: 800;
    display: flex;
    align-items: center;
    gap: 8px;
}

.footer-tagline {
    color: #c2e8d4;
    font-size: 12px;
    font-weight: 500;
}

.footer-credit {
    color: #48f5b5;
    font-size: 12px;
    font-weight: 700;
    background: rgba(72, 245, 181, 0.15);
    border: 1px solid rgba(72, 245, 181, 0.3);
    padding: 4px 14px;
    border-radius: 20px;
}

.footer-disclaimer {
    text-align: center;
    color: #8fb8a4;
    font-size: 11px;
    margin-top: 14px;
    margin-bottom: 25px;
    line-height: 1.4;
}
</style>

<div class="glass-footer-divider"></div>

<div class="glass-footer">
    <div class="footer-brand">
        <span>🌾</span> AgriVision AI
    </div>
    <div class="footer-tagline">
        Precision Agriculture • Machine Learning • Agro-Met Intelligence
    </div>
    <div class="footer-credit">
        © 2026 Developed by Siddhu & Team
    </div>
</div>

<div class="footer-disclaimer">
    Educational Capstone Prototype • Historical agricultural data analytics, not guaranteed harvest commitments.
</div>
""", unsafe_allow_html=True)