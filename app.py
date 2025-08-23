import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import model_from_json
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ================= Configuration =================
st.set_page_config(page_title="Slope Stability Analysis", layout="wide", initial_sidebar_state="collapsed")

# ================= CSS =================
st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    .stApp {
        background-color: #ffffff;
    }
    
    h1 {
        color: #2c3e50;
        font-size: 2.5rem !important;
        font-weight: 600;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .parameter-container {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border: 1px solid #e9ecef;
    }
    
    .slider-label {
        font-size: 1.1rem;
        font-weight: 500;
        color: #495057;
        margin-bottom: 0.5rem;
    }
    
    .fs-display {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 2rem 0;
    }
    
    .fs-value {
        font-size: 3rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    
    .fs-status {
        font-size: 1.2rem;
        margin-top: 1rem;
    }
    
    .ml-predictions {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 2rem;
    }
    
    .prediction-item {
        display: flex;
        justify-content: space-between;
        padding: 0.5rem 0;
        border-bottom: 1px solid #dee2e6;
    }
    
    .prediction-item:last-child {
        border-bottom: none;
    }
    
    .critical-label {
        color: #dc3545;
        font-weight: bold;
    }
    
    .stable-label {
        color: #28a745;
        font-weight: bold;
    }
    
    .marginal-label {
        color: #ffc107;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True
)

# ================= Título =================
st.markdown("<h1>Virtual Slope Stability Laboratory</h1>", unsafe_allow_html=True)

# ================= Layout =================
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Slope Parameters")
    
    # Height slider
    st.markdown('<div class="slider-label">Height (H)</div>', unsafe_allow_html=True)
    height = st.slider('', min_value=1.0, max_value=20.0, value=10.0, step=0.5, key='height')
    st.markdown(f'<div style="text-align: right; margin-top: -10px; color: #666;">{height} m</div>', unsafe_allow_html=True)
    
    # Slope angle slider
    st.markdown('<div class="slider-label">Slope angle (β)</div>', unsafe_allow_html=True)
    slope_inclination = st.slider('', min_value=10.0, max_value=80.0, value=45.0, step=1.0, key='slope')
    st.markdown(f'<div style="text-align: right; margin-top: -10px; color: #666;">{slope_inclination} °</div>', unsafe_allow_html=True)
    
    # Cohesion slider
    st.markdown('<div class="slider-label">Cohesion (c)</div>', unsafe_allow_html=True)
    cohesion = st.slider('', min_value=0.0, max_value=50.0, value=15.0, step=1.0, key='cohesion')
    st.markdown(f'<div style="text-align: right; margin-top: -10px; color: #666;">{cohesion} kPa</div>', unsafe_allow_html=True)
    
    # Friction angle slider
    st.markdown('<div class="slider-label">Friction angle (φ)</div>', unsafe_allow_html=True)
    friction_angle = st.slider('', min_value=10.0, max_value=50.0, value=25.0, step=1.0, key='friction')
    st.markdown(f'<div style="text-align: right; margin-top: -10px; color: #666;">{friction_angle} °</div>', unsafe_allow_html=True)
    
    # Unit weight slider
    st.markdown('<div class="slider-label">Unit weight (γ)</div>', unsafe_allow_html=True)
    unit_weight = st.slider('', min_value=10.0, max_value=25.0, value=18.0, step=0.5, key='unit_weight')
    st.markdown(f'<div style="text-align: right; margin-top: -10px; color: #666;">{unit_weight} kN/m³</div>', unsafe_allow_html=True)
    
    # Seismic coefficient (hidden for now to match the design)
    kh = 0.0

# ================= Cargar modelos =================
@st.cache_resource
def load_models():
    scaler = joblib.load('scaler.pkl')
    rf_model = joblib.load('rf_model.pkl')
    svm_model = joblib.load('svm_model.pkl')
    
    with open("nn_model.json", "r") as json_file:
        nn_json = json_file.read()
    nn_model = model_from_json(nn_json)
    nn_model.load_weights("nn_model.weights.h5")
    nn_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    
    return scaler, rf_model, svm_model, nn_model

scaler, rf_model, svm_model, nn_model = load_models()

# ================= Real-time calculation =================
def calculate_fs():
    X_input = np.array([[unit_weight, friction_angle, cohesion, height, slope_inclination, kh]])
    X_scaled = scaler.transform(X_input)
    
    fs_rf = rf_model.predict(X_scaled)[0]
    fs_svm = svm_model.predict(X_scaled)[0]
    fs_nn = nn_model.predict(X_scaled)[0][0]
    
    # Use Random Forest as primary prediction (you can change this)
    return fs_rf, fs_svm, fs_nn

def get_fs_status(fs):
    if fs < 1.0:
        return "Crítico", "critical-label"
    elif fs < 1.25:
        return "Inestable", "marginal-label"
    elif fs < 1.5:
        return "Marginal", "marginal-label"
    else:
        return "Estable", "stable-label"

def create_gauge(fs_value):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = fs_value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "FS"},
        delta = {'reference': 1.5},
        gauge = {
            'axis': {'range': [None, 3]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 1], 'color': "red"},
                {'range': [1, 1.25], 'color': "orange"},
                {'range': [1.25, 1.5], 'color': "yellow"},
                {'range': [1.5, 3], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 1.0
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig

# Calculate current FS values
fs_rf, fs_svm, fs_nn = calculate_fs()
primary_fs = fs_rf  # Using Random Forest as primary
status_text, status_class = get_fs_status(primary_fs)

with col2:
    # Factor of Safety Display
    st.markdown(f"""
    <div class="fs-display">
        <div style="font-size: 1.2rem;">Factor de Seguridad</div>
        <div class="fs-value">FS = {primary_fs:.2f}</div>
        <div class="fs-status {status_class}">
            Estado del Talud:<br/>
            <strong>{status_text}</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Gauge
    gauge_fig = create_gauge(primary_fs)
    st.plotly_chart(gauge_fig, use_container_width=True)

# ================= Slope Visualization =================
def create_slope_plot():
    # Geometry
    beta = np.deg2rad(slope_inclination)
    L = height / np.tan(beta)
    
    # Slope coordinates
    x_slope = [0, L, L + height/np.tan(np.pi/2 - beta), 0, 0]
    y_slope = [0, 0, height, height, 0]
    
    # Critical failure surface (simplified circular arc)
    scale = max(0.5, 2.0 - primary_fs)
    R = height * scale
    xc = L/2
    yc = -R*0.3
    
    theta = np.linspace(-np.pi/4, 3*np.pi/4, 100)
    xs = xc + R*np.cos(theta)
    ys = yc + R*np.sin(theta)
    
    # Filter points within slope domain
    mask = (xs >= 0) & (xs <= L + height/np.tan(np.pi/2 - beta)) & (ys >= 0)
    xs_filtered = xs[mask]
    ys_filtered = ys[mask]
    
    fig = go.Figure()
    
    # Add slope
    fig.add_trace(go.Scatter(
        x=x_slope, y=y_slope,
        fill='toself',
        fillcolor='lightgray',
        line=dict(color='black', width=3),
        name='Slope',
        showlegend=False
    ))
    
    # Add critical failure surface
    if len(xs_filtered) > 0:
        fig.add_trace(go.Scatter(
            x=xs_filtered, y=ys_filtered,
            line=dict(color='red', width=3, dash='dash'),
            name='Critical failure surface',
            showlegend=True
        ))
    
    # Add coordinate axes
    fig.add_annotation(
        x=L + height/np.tan(np.pi/2 - beta) + 1,
        y=0,
        text="x",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        ax=-20,
        ay=0
    )
    
    fig.add_annotation(
        x=0,
        y=height + 1,
        text="y",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        ax=0,
        ay=-20
    )
    
    # Add height dimension
    fig.add_trace(go.Scatter(
        x=[-2, -2], y=[0, height],
        line=dict(color='blue', width=2),
        showlegend=False
    ))
    
    fig.add_annotation(
        x=-3,
        y=height/2,
        text=f"{height} m",
        showarrow=False,
        textangle=90,
        font=dict(color='blue', size=12)
    )
    
    fig.update_layout(
        title="",
        xaxis=dict(
            showgrid=False,
            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=2,
            range=[-4, L + height/np.tan(np.pi/2 - beta) + 2]
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=2,
            range=[-1, height + 2]
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=400,
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=True,
        legend=dict(x=0.02, y=0.98)
    )
    
    return fig

# ================= Display slope visualization =================
st.markdown("---")
col3, col4 = st.columns([1, 1])

with col3:
    st.plotly_chart(create_slope_plot(), use_container_width=True)

with col4:
    st.markdown("### Predicciones de Modelos ML")
    
    # ML Predictions display
    st.markdown(f"""
    <div class="ml-predictions">
        <div class="prediction-item">
            <span><strong>Random Forest:</strong></span>
            <span class="{get_fs_status(fs_rf)[1]}">FS = {fs_rf:.4f}</span>
        </div>
        <div class="prediction-item">
            <span><strong>SVM:</strong></span>
            <span class="{get_fs_status(fs_svm)[1]}">FS = {fs_svm:.4f}</span>
        </div>
        <div class="prediction-item">
            <span><strong>Neural Network:</strong></span>
            <span class="{get_fs_status(fs_nn)[1]}">FS = {fs_nn:.4f}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Safety interpretation
    if primary_fs < 1.0:
        interpretation = "El talud es inestable y puede fallar. Se requieren medidas correctivas inmediatas."
    elif primary_fs < 1.25:
        interpretation = "El talud está en condición crítica. Se recomienda monitoreo continuo y medidas preventivas."
    elif primary_fs < 1.5:
        interpretation = "El talud está en condición marginal. Se sugiere evaluación adicional."
    else:
        interpretation = "El talud es estable bajo las condiciones actuales."
    
    st.markdown(f"""
    <div style="margin-top: 1rem; padding: 1rem; background-color: #f8f9fa; border-radius: 8px; border-left: 4px solid #007bff;">
        <strong>Interpretación:</strong><br/>
        {interpretation}
    </div>
    """, unsafe_allow_html=True)
