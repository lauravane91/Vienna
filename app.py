import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import model_from_json
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ================= Configuration =================
st.set_page_config(
    page_title="Slope Stability Analysis", 
    layout="wide", 
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

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
        color: #2c3e50 !important;
        font-size: 2.5rem !important;
        font-weight: 600 !important;
        margin-bottom: 2rem !important;
        text-align: center !important;
        padding: 1rem 0 !important;
        display: block !important;
        visibility: visible !important;
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
st.title("Slope Stability Factor of Safety Prediction")
st.markdown("""
<div style="text-align: center; padding: 0.5rem; background-color: #f8f9fa; border-radius: 10px; margin-bottom: 2rem;">
    <h2 style="color: #2c3e50; margin: 0; font-size: 1.8rem;">Virtual Slope Stability Laboratory</h2>
</div>
""", unsafe_allow_html=True)

# ================= Layout =================
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Slope Parameters")
    
    # Height slider
    st.markdown('<div class="slider-label">Height (H)</div>', unsafe_allow_html=True)
    height = st.slider('', min_value=1.0, max_value=20.0, value=10.0, step=0.5, key='height')
    st.markdown(f'<div style="text-align: right; margin-top: -10px; color: #666;">{height} m</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size: 0.85rem; color: #888; margin-bottom: 1rem;">Height of the slope from base to top. Taller slopes tend to be less stable.</div>', unsafe_allow_html=True)
    
    # Slope angle slider
    st.markdown('<div class="slider-label">Slope angle (β)</div>', unsafe_allow_html=True)
    slope_inclination = st.slider('', min_value=10.0, max_value=80.0, value=45.0, step=1.0, key='slope')
    st.markdown(f'<div style="text-align: right; margin-top: -10px; color: #666;">{slope_inclination} °</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size: 0.85rem; color: #888; margin-bottom: 1rem;">Inclination angle of the slope. Steeper slopes are usually less stable.</div>', unsafe_allow_html=True)
    
    # Cohesion slider
    st.markdown('<div class="slider-label">Cohesion (c)</div>', unsafe_allow_html=True)
    cohesion = st.slider('', min_value=0.0, max_value=50.0, value=15.0, step=1.0, key='cohesion')
    st.markdown(f'<div style="text-align: right; margin-top: -10px; color: #666;">{cohesion} kPa</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size: 0.85rem; color: #888; margin-bottom: 1rem;">Force of adhesion between soil particles. Higher cohesion helps maintain slope stability.</div>', unsafe_allow_html=True)
    
    # Friction angle slider
    st.markdown('<div class="slider-label">Friction angle (φ)</div>', unsafe_allow_html=True)
    friction_angle = st.slider('', min_value=10.0, max_value=50.0, value=25.0, step=1.0, key='friction')
    st.markdown(f'<div style="text-align: right; margin-top: -10px; color: #666;">{friction_angle} °</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size: 0.85rem; color: #888; margin-bottom: 1rem;">Internal friction angle of the soil. Higher angle means more resistance to sliding.</div>', unsafe_allow_html=True)
    
    # Unit weight slider
    st.markdown('<div class="slider-label">Unit weight (γ)</div>', unsafe_allow_html=True)
    unit_weight = st.slider('', min_value=10.0, max_value=25.0, value=18.0, step=0.5, key='unit_weight')
    st.markdown(f'<div style="text-align: right; margin-top: -10px; color: #666;">{unit_weight} kN/m³</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size: 0.85rem; color: #888; margin-bottom: 1rem;">Weight of the soil per unit volume. Heavier soils increase the load on the slope.</div>', unsafe_allow_html=True)
    
    # Seismic coefficient (hidden for now to match the design)
    kh = 0.0

# ================= Cargar modelos =================
@st.cache_resource
def load_models():
    """Load and cache ML models for performance"""
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

# ================= Real-time calculation with caching =================
@st.cache_data
def calculate_fs(_scaler, _rf_model, _svm_model, _nn_model, unit_weight, friction_angle, cohesion, height, slope_inclination, kh):
    """Calculate factor of safety with caching to improve performance"""
    X_input = np.array([[unit_weight, friction_angle, cohesion, height, slope_inclination, kh]])
    X_scaled = _scaler.transform(X_input)
    
    fs_rf = _rf_model.predict(X_scaled)[0]
    fs_svm = _svm_model.predict(X_scaled)[0]
    fs_nn = _nn_model.predict(X_scaled)[0][0]
    
    return fs_rf, fs_svm, fs_nn

def get_fs_status(fs):
    if fs < 1.0:
        return "Critical", "critical-label"
    elif fs < 1.25:
        return "Unstable", "marginal-label"
    elif fs < 1.5:
        return "Marginal", "marginal-label"
    else:
        return "Stable", "stable-label"

def create_gauge(fs_value):
    """
    Crea un gráfico de tipo gauge (medidor circular) para visualizar el Factor de Seguridad (FS).
    
    Este gráfico circular muestra:
    - El valor actual del FS en el centro
    - Bandas de colores que indican diferentes niveles de estabilidad:
      * Rojo (0-1): Crítico - el talud puede fallar
      * Naranja (1-1.25): Inestable - condición crítica
      * Amarillo (1.25-1.5): Marginal - requiere evaluación adicional  
      * Verde (1.5-3): Estable - condición segura
    - Una línea de umbral roja en FS = 1.0 que marca el límite de estabilidad
    - Un delta que compara el valor actual con el valor de referencia (1.5)
    
    Args:
        fs_value (float): El valor del Factor de Seguridad a mostrar
        
    Returns:
        plotly.graph_objects.Figure: El gráfico gauge configurado
    """
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",  # Mostrar medidor + número + delta de referencia
        value = fs_value,             # Valor actual del FS a mostrar
        domain = {'x': [0, 1], 'y': [0, 1]},  # Usar todo el espacio disponible
        title = {'text': "FS"},       # Título del medidor
        delta = {'reference': 1.5},   # Valor de referencia para mostrar diferencia
        gauge = {
            'axis': {'range': [None, 3]},  # Rango del medidor: 0 a 3
            'bar': {'color': "darkblue"},   # Color de la aguja indicadora
            # Definir las bandas de colores según criterios de estabilidad geotécnica:
            'steps': [
                {'range': [0, 1], 'color': "red"},        # FS < 1.0: Crítico/Falla inminente
                {'range': [1, 1.25], 'color': "orange"},  # 1.0 ≤ FS < 1.25: Inestable
                {'range': [1.25, 1.5], 'color': "yellow"}, # 1.25 ≤ FS < 1.5: Marginal
                {'range': [1.5, 3], 'color': "green"}     # FS ≥ 1.5: Estable/Seguro
            ],
            # Línea de umbral crítico en FS = 1.0 (límite entre estable/inestable)
            'threshold': {
                'line': {'color': "red", 'width': 4},  # Línea roja gruesa
                'thickness': 0.75,                      # Grosor relativo de la línea
                'value': 1.0                           # Posición del umbral crítico
            }
        }
    ))
    
    # Configurar el diseño del gráfico
    fig.update_layout(
        height=300,                                    # Altura fija del gráfico
        margin=dict(l=20, r=20, t=20, b=20),          # Márgenes mínimos
        paper_bgcolor="rgba(0,0,0,0)",                # Fondo transparente
        plot_bgcolor="rgba(0,0,0,0)"                  # Área de trazado transparente
    )
    return fig

with col2:
    # Calculate current FS values (cached for performance)
    fs_rf, fs_svm, fs_nn = calculate_fs(scaler, rf_model, svm_model, nn_model, unit_weight, friction_angle, cohesion, height, slope_inclination, kh)
    primary_fs = fs_rf  # Using Random Forest as primary
    status_text, status_class = get_fs_status(primary_fs)
    
    # Factor of Safety Display
    st.markdown("### Factor of Safety Results")
    
    # Display all three FS values with methodology identification
    col2a, col2b, col2c = st.columns(3)
    
    with col2a:
        rf_status, rf_class = get_fs_status(fs_rf)
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; background-color: #f8f9fa; border-radius: 10px; margin-bottom: 0.5rem;">
            <div style="font-size: 0.9rem; color: #666; margin-bottom: 0.5rem;">Random Forest</div>
            <div style="font-size: 1.8rem; font-weight: bold; color: #2c3e50;">FS = {fs_rf:.3f}</div>
            <div class="{rf_class}" style="font-size: 0.8rem; margin-top: 0.5rem;">{rf_status}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2b:
        svm_status, svm_class = get_fs_status(fs_svm)
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; background-color: #f8f9fa; border-radius: 10px; margin-bottom: 0.5rem;">
            <div style="font-size: 0.9rem; color: #666; margin-bottom: 0.5rem;">SVM</div>
            <div style="font-size: 1.8rem; font-weight: bold; color: #2c3e50;">FS = {fs_svm:.3f}</div>
            <div class="{svm_class}" style="font-size: 0.8rem; margin-top: 0.5rem;">{svm_status}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2c:
        nn_status, nn_class = get_fs_status(fs_nn)
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; background-color: #f8f9fa; border-radius: 10px; margin-bottom: 0.5rem;">
            <div style="font-size: 0.9rem; color: #666; margin-bottom: 0.5rem;">Neural Network</div>
            <div style="font-size: 1.8rem; font-weight: bold; color: #2c3e50;">FS = {fs_nn:.3f}</div>
            <div class="{nn_class}" style="font-size: 0.8rem; margin-top: 0.5rem;">{nn_status}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Primary status display (using Random Forest as reference)
    st.markdown(f"""
    <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
         border-radius: 10px; color: white; margin-top: 1rem;">
        <div style="font-size: 1.1rem;">Overall Slope Status (RF-based)</div>
        <div style="font-size: 1.5rem; font-weight: bold; margin-top: 0.5rem;">{status_text}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Gauge (using Random Forest as primary)
    # Comentado para que no se muestre el gráfico de gauge  
    #gauge_fig = create_gauge(primary_fs)
    #st.plotly_chart(gauge_fig, use_container_width=True)

# ================= Calculate FS values for visualization (cached) =================
fs_rf_viz, fs_svm_viz, fs_nn_viz = calculate_fs(scaler, rf_model, svm_model, nn_model, unit_weight, friction_angle, cohesion, height, slope_inclination, kh)
primary_fs_viz = fs_rf_viz

# ================= Slope Visualization with caching =================
@st.cache_data
def create_slope_plot(fs_value, _height, _slope_inclination):
    # Geometry - Invertida verticalmente para que la base esté abajo
    beta = np.deg2rad(_slope_inclination)
    L = _height / np.tan(beta)
    
    # Slope coordinates - Geometría invertida verticalmente
    # Base en y=0, talud hacia arriba
    x_slope = [0, L, L + _height/np.tan(np.pi/2 - beta), 0, 0]
    y_slope = [_height, _height, 0, 0, _height]  # Invertido: base arriba -> base abajo
    
    # Critical failure surface (simplified circular arc) - Invertida
    scale = max(0.5, 2.0 - fs_value)
    R = _height * scale
    xc = L/2
    yc = _height + R*0.3  # Invertido: centro arriba del talud
    
    theta = np.linspace(-3*np.pi/4, np.pi/4, 100)  # Rango invertido para arco hacia abajo
    xs = xc + R*np.cos(theta)
    ys = yc + R*np.sin(theta)
    
    # Filter points within slope domain - Ajustado para geometría invertida
    mask = (xs >= 0) & (xs <= L + _height/np.tan(np.pi/2 - beta)) & (ys <= _height)
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
    # commented for now to match the design
    #if len(xs_filtered) > 0:
    #    fig.add_trace(go.Scatter(
    #        x=xs_filtered, y=ys_filtered,
    #        line=dict(color='red', width=3, dash='dash'),
    #        name='Critical failure surface',
    #        showlegend=True
    #    ))
    
    # Add coordinate axes
    fig.add_annotation(
        x=L + _height/np.tan(np.pi/2 - beta) + 1,
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
        y=_height + 1,
        text="y",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        ax=0,
        ay=-20
    )
    
    # Add slope angle annotation (β) at the intersection point
    angle_arc_radius = min(L/6, _height/6, 2)  # Radio más pequeño para mejor ajuste
    
    # Posición en la intersección de la base horizontal y la pendiente inclinada
    # Esta es la esquina inferior derecha donde se forma el ángulo β
    intersection_x = L + _height/np.tan(np.pi/2 - beta)  # Punto donde la pendiente toca la base
    intersection_y = 0  # Base del talud
    
    # Dibujar arco desde la horizontal hacia la pendiente inclinada
    # Ángulo desde la horizontal (0°) hacia la pendiente (β medido desde vertical)
    slope_angle_from_horizontal = np.pi/2 - beta  # Ángulo de la pendiente desde horizontal
    angle_theta = np.linspace(0, slope_angle_from_horizontal, 20)
    
    arc_x = intersection_x - angle_arc_radius * np.cos(angle_theta)
    arc_y = intersection_y + angle_arc_radius * np.sin(angle_theta)
    
    fig.add_trace(go.Scatter(
        x=arc_x, y=arc_y,
        line=dict(color='black', width=2),
        showlegend=False,
        mode='lines'
    ))
    
    # Posición del texto del ángulo
    text_x = intersection_x - angle_arc_radius * 1.8
    text_y = intersection_y + angle_arc_radius * 0.8
    
    # Agregar texto del ángulo
    fig.add_annotation(
        x=text_x,
        y=text_y,
        text=f"β = {_slope_inclination}°",
        showarrow=False,
        font=dict(color='black', size=12),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="black",
        borderwidth=1
    )
    
    # Add height dimension
    fig.add_trace(go.Scatter(
        x=[-2, -2], y=[0, _height],
        line=dict(color='blue', width=2),
        showlegend=False
    ))
    
    fig.add_annotation(
        x=-3,
        y=_height/2,
        text=f"{_height} m",
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
            range=[-4, L + _height/np.tan(np.pi/2 - beta) + 2]
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=2,
            range=[-2, _height + 2]
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

with col2:
    st.plotly_chart(create_slope_plot(primary_fs_viz, height, slope_inclination), use_container_width=True)
