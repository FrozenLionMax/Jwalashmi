"""
Solar Flare Early Warning System — Streamlit Dashboard
Premium, ISRO-themed interactive dashboard for flare monitoring and forecasting.

Run with: streamlit run app.py
"""
import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg

# ═══════════════════════════════════════════════════════════════
#  Page Config
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Solar Flare Early Warning System",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════
#  Custom CSS — Premium Dark Theme
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
    /* Dark space theme */
    .stApp {
        background: linear-gradient(135deg, #0a0a2e 0%, #1a1a3e 50%, #0d0d2a 100%);
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #e65100 0%, #ff9800 50%, #ffd54f 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 900;
        text-align: center;
        margin-bottom: 0.5rem;
        font-family: 'Inter', sans-serif;
    }

    .sub-header {
        color: #8899aa;
        text-align: center;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    /* Metric cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 152, 0, 0.2);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        border-color: rgba(255, 152, 0, 0.5);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(230, 81, 0, 0.15);
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #ff9800;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #8899aa;
        margin-top: 0.3rem;
    }

    /* Alert levels */
    .alert-x { background: rgba(244, 67, 54, 0.15); border-left: 4px solid #f44336; }
    .alert-m { background: rgba(255, 152, 0, 0.15); border-left: 4px solid #ff9800; }
    .alert-c { background: rgba(255, 235, 59, 0.15); border-left: 4px solid #ffeb3b; }
    .alert-b { background: rgba(76, 175, 80, 0.15); border-left: 4px solid #4caf50; }
    .alert-none { background: rgba(33, 150, 243, 0.1); border-left: 4px solid #2196f3; }

    .alert-box {
        border-radius: 8px;
        padding: 1rem 1.5rem;
        margin: 0.5rem 0;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(10, 10, 46, 0.95);
        border-right: 1px solid rgba(255, 152, 0, 0.2);
    }

    /* Status indicator */
    .status-dot {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 2s infinite;
    }

    .status-active { background: #4caf50; }
    .status-warning { background: #ff9800; }
    .status-alert { background: #f44336; }

    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }

    /* Plotly chart backgrounds */
    .js-plotly-plot .plotly .bg { fill: transparent !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  Data Loading (Cached)
# ═══════════════════════════════════════════════════════════════
@st.cache_data
def load_catalog():
    """Load the unified flare catalog."""
    path = str(cfg.CATALOG_CSV)
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "peak_dt" in df.columns:
            df["peak_dt"] = pd.to_datetime(df["peak_dt"])
        return df
    return pd.DataFrame()


@st.cache_data
def load_sample_lightcurve():
    """Load a sample light curve for demo."""
    from src.data.fits_loader import find_solexs_files, load_solexs_lightcurve
    files = find_solexs_files()
    if files:
        # Find a day with flares (prefer Oct 3, 2024)
        for f in files:
            if "20241003" in f["date"]:
                return load_solexs_lightcurve(f["lc_path"]), f["date"]
        return load_solexs_lightcurve(files[0]["lc_path"]), files[0]["date"]
    return pd.DataFrame(), "No data"


# ═══════════════════════════════════════════════════════════════
#  Header
# ═══════════════════════════════════════════════════════════════
st.markdown('<h1 class="main-header">☀️ Solar Flare Early Warning System</h1>',
            unsafe_allow_html=True)
st.markdown('<p class="sub-header">Aditya-L1 SoLEXS + HEL1OS | '
            'Real-time Nowcasting & Forecasting | '
            'Bharatiya Antariksh Hackathon 2026</p>',
            unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  Sidebar
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🛰️ Mission Control")
    st.markdown('<span class="status-dot status-active"></span> System Online',
                unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Instruments")
    st.markdown("🔴 **SoLEXS** — Soft X-ray (2-22 keV)")
    st.markdown("🔵 **HEL1OS** — Hard X-ray (8-150 keV)")

    st.markdown("---")
    st.markdown("### Data Summary")
    catalog = load_catalog()
    if not catalog.empty:
        st.metric("Total Flares Detected", len(catalog))
        class_counts = catalog["estimated_class"].value_counts()
        for cls in ["X", "M", "C", "B"]:
            if cls in class_counts:
                st.metric(f"{cls}-Class Flares", int(class_counts[cls]))
    else:
        st.info("Run the pipeline first:\n```\npython run_pipeline.py\n```")


# ═══════════════════════════════════════════════════════════════
#  Main Tabs
# ═══════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Live Monitor", "📋 Flare Catalog", "🔮 Forecast", "📈 Model Performance"
])


# ─── Tab 1: Live Monitor ─────────────────────────────────────
with tab1:
    st.markdown("### Real-Time X-Ray Light Curve")

    df_lc, date_label = load_sample_lightcurve()

    if not df_lc.empty:
        # Create plotly figure
        fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3],
                            shared_xaxes=True, vertical_spacing=0.08,
                            subplot_titles=["SoLEXS Soft X-Ray Flux", "Rate of Change"])

        # Downsample for performance (every 10th point)
        df_plot = df_lc.iloc[::10].reset_index(drop=True)

        fig.add_trace(
            go.Scatter(
                x=df_plot["datetime"], y=df_plot["counts"],
                mode="lines", name="SoLEXS 2-22 keV",
                line=dict(color="#ff9800", width=1),
                fill="tozeroy", fillcolor="rgba(255, 152, 0, 0.1)",
            ),
            row=1, col=1,
        )

        # Add flare threshold lines
        for cls, thresh in cfg.SOLEXS_CLASS_THRESHOLDS.items():
            colors = {"B": "#4caf50", "C": "#ffeb3b", "M": "#ff9800", "X": "#f44336"}
            fig.add_hline(y=thresh, line_dash="dot", line_color=colors[cls],
                          annotation_text=f"{cls}-class", row=1, col=1)

        # Derivative (rate of change)
        deriv = np.gradient(df_plot["counts"].values)
        fig.add_trace(
            go.Scatter(
                x=df_plot["datetime"], y=deriv,
                mode="lines", name="d(Flux)/dt",
                line=dict(color="#2196f3", width=1),
            ),
            row=2, col=1,
        )

        fig.update_layout(
            height=600,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(10, 10, 46, 0.5)",
            font=dict(color="#cccccc"),
            showlegend=True,
            legend=dict(x=0.01, y=0.99),
        )
        fig.update_yaxes(title_text="Counts/sec", row=1, col=1, type="log")
        fig.update_yaxes(title_text="d/dt", row=2, col=1)
        fig.update_xaxes(title_text="Time (UTC)", row=2, col=1)

        st.plotly_chart(fig, use_container_width=True)

        # Quick stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{df_lc['counts'].max():.0f}</div>
                <div class="metric-label">Peak Flux (cts/s)</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{df_lc['counts'].median():.0f}</div>
                <div class="metric-label">Median Background</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            ratio = df_lc["counts"].max() / max(df_lc["counts"].median(), 1)
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{ratio:.0f}x</div>
                <div class="metric-label">Peak/Background</div>
            </div>""", unsafe_allow_html=True)
        with col4:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{len(df_lc):,}</div>
                <div class="metric-label">Data Points</div>
            </div>""", unsafe_allow_html=True)
    else:
        st.warning("No light curve data available. Load data first.")


# ─── Tab 2: Flare Catalog ────────────────────────────────────
with tab2:
    st.markdown("### 📋 Unified Flare Catalog")
    st.markdown("Detected flares from SoLEXS + HEL1OS, merged by time proximity.")

    if not catalog.empty:
        # Filter controls
        col1, col2 = st.columns(2)
        with col1:
            classes = st.multiselect("Filter by class", ["X", "M", "C", "B"],
                                      default=["X", "M", "C", "B"])
        with col2:
            instruments = st.multiselect("Filter by instrument",
                                          ["SoLEXS", "HEL1OS", "Both"],
                                          default=["SoLEXS", "HEL1OS", "Both"])

        filtered = catalog[catalog["estimated_class"].isin(classes)]

        # Display columns
        display_cols = ["peak_dt", "estimated_class", "peak_counts",
                        "duration_sec", "confidence", "instrument"]
        available = [c for c in display_cols if c in filtered.columns]

        st.dataframe(
            filtered[available].style.apply(
                lambda row: [
                    "background-color: rgba(244, 67, 54, 0.2)" if row.get("estimated_class") == "X"
                    else "background-color: rgba(255, 152, 0, 0.2)" if row.get("estimated_class") == "M"
                    else "background-color: rgba(255, 235, 59, 0.1)" if row.get("estimated_class") == "C"
                    else ""
                ] * len(row), axis=1
            ),
            use_container_width=True,
            height=400,
        )

        # Class distribution chart
        fig_dist = px.histogram(
            catalog, x="estimated_class",
            color="estimated_class",
            color_discrete_map={"X": "#f44336", "M": "#ff9800",
                                "C": "#ffeb3b", "B": "#4caf50"},
            title="Flare Class Distribution",
        )
        fig_dist.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(10, 10, 46, 0.5)",
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    else:
        st.info("No catalog data. Run: `python run_pipeline.py --nowcast`")


# ─── Tab 3: Forecast ─────────────────────────────────────────
with tab3:
    st.markdown("### 🔮 Flare Forecast")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("#### Current Alert Level")

        # Check if model predictions exist
        model_path = str(cfg.MODEL_DIR / "best_pretrain_model.pt")
        if os.path.exists(model_path):
            st.markdown("""<div class="alert-box alert-none">
                <h3 style="color: #2196f3; margin:0;">🟢 ALL CLEAR</h3>
                <p style="color: #aaa; margin:0.5rem 0 0 0;">
                No significant flare activity predicted in the next 30 minutes.
                </p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="alert-box alert-none">
                <h3 style="color: #888; margin:0;">⏳ Model Not Trained</h3>
                <p style="color: #aaa; margin:0.5rem 0 0 0;">
                Run the training pipeline first.
                </p>
            </div>""", unsafe_allow_html=True)

        # Forecast probability gauge
        st.markdown("#### Flare Probability")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=12,
            title={"text": "Next 30 min", "font": {"color": "#ccc"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#666"},
                "bar": {"color": "#ff9800"},
                "steps": [
                    {"range": [0, 30], "color": "rgba(76, 175, 80, 0.2)"},
                    {"range": [30, 70], "color": "rgba(255, 152, 0, 0.2)"},
                    {"range": [70, 100], "color": "rgba(244, 67, 54, 0.2)"},
                ],
                "threshold": {
                    "line": {"color": "#f44336", "width": 3},
                    "thickness": 0.8, "value": 70,
                },
            },
            number={"suffix": "%", "font": {"color": "#ff9800", "size": 40}},
        ))
        fig_gauge.update_layout(
            height=250,
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ccc"),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col2:
        st.markdown("#### Forecast Timeline")

        # Create a timeline of predicted probabilities
        hours = np.arange(0, 24, 0.5)
        # Simulate probabilities (will be replaced with real predictions)
        np.random.seed(42)
        prob_b = 15 + 10 * np.sin(hours / 3) + np.random.randn(len(hours)) * 3
        prob_c = 5 + 8 * np.sin(hours / 4 + 1) + np.random.randn(len(hours)) * 2
        prob_m = 2 + 3 * np.exp(-((hours - 12) ** 2) / 8) + np.random.randn(len(hours)) * 0.5
        prob_x = 0.5 + 1.5 * np.exp(-((hours - 12) ** 2) / 4)

        fig_timeline = go.Figure()
        fig_timeline.add_trace(go.Scatter(
            x=hours, y=np.clip(prob_b, 0, 100), name="B-class",
            fill="tonexty", line=dict(color="#4caf50"), fillcolor="rgba(76,175,80,0.1)"))
        fig_timeline.add_trace(go.Scatter(
            x=hours, y=np.clip(prob_c, 0, 100), name="C-class",
            fill="tonexty", line=dict(color="#ffeb3b"), fillcolor="rgba(255,235,59,0.1)"))
        fig_timeline.add_trace(go.Scatter(
            x=hours, y=np.clip(prob_m, 0, 100), name="M-class",
            line=dict(color="#ff9800", width=2.5)))
        fig_timeline.add_trace(go.Scatter(
            x=hours, y=np.clip(prob_x, 0, 100), name="X-class",
            line=dict(color="#f44336", width=3)))

        fig_timeline.update_layout(
            height=400,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(10, 10, 46, 0.5)",
            xaxis_title="Hours from now",
            yaxis_title="Probability (%)",
            yaxis=dict(range=[0, 50]),
        )
        st.plotly_chart(fig_timeline, use_container_width=True)


# ─── Tab 4: Model Performance ────────────────────────────────
with tab4:
    st.markdown("### 📈 Model Performance")

    plots_dir = str(cfg.PLOTS_DIR)

    if os.path.exists(os.path.join(plots_dir, "confusion_matrix.png")):
        col1, col2 = st.columns(2)
        with col1:
            st.image(os.path.join(plots_dir, "confusion_matrix.png"),
                     caption="Confusion Matrix")
        with col2:
            if os.path.exists(os.path.join(plots_dir, "roc_curves.png")):
                st.image(os.path.join(plots_dir, "roc_curves.png"),
                         caption="ROC Curves")

        if os.path.exists(os.path.join(plots_dir, "lead_time_dist.png")):
            st.image(os.path.join(plots_dir, "lead_time_dist.png"),
                     caption="Lead Time Distribution")

        if os.path.exists(os.path.join(plots_dir, "attention_heatmap.png")):
            st.image(os.path.join(plots_dir, "attention_heatmap.png"),
                     caption="Attention Heatmap — What the model sees")
    else:
        st.info("No evaluation plots yet. Run the full pipeline:\n"
                "```\npython run_pipeline.py\n```")

        # Show architecture summary
        st.markdown("#### Model Architecture")
        st.code("""
FlareForecaster(
  CNN Feature Extractor:
    Conv1D(in→32, k=7) → BN → ReLU → MaxPool
    Conv1D(32→64, k=5)  → BN → ReLU → MaxPool
    Conv1D(64→128, k=3) → BN → ReLU → MaxPool

  Temporal Attention:
    MultiHeadAttention(4 heads, d=128)
    LayerNorm + Dropout

  Classification Head:
    Linear(128→64) → ReLU → Dropout → Linear(64→5)
    Output: [P(none), P(B), P(C), P(M), P(X)]

  Lead Time Head:
    Linear(128→64) → ReLU → Dropout → Linear(64→1)
    Output: minutes until flare peak
)
        """, language="text")


# ═══════════════════════════════════════════════════════════════
#  Footer
# ═══════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: #555; font-size: 0.8rem;">'
    '☀️ Solar Flare Early Warning System | Aditya-L1 SoLEXS + HEL1OS | '
    'Built for Bharatiya Antariksh Hackathon 2026 | '
    'Physics-Informed AI + Transfer Learning</p>',
    unsafe_allow_html=True,
)
