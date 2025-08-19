import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import model_from_json
import os

# ================= CSS para agrandar toda la app =================
st.markdown(
    """
    <style>
    /* Título principal */
    h1 {
        font-size: 48px !important;
    }
    /* Labels de los inputs */
    label {
        font-size: 24px !important;
    }
    /* Botones */
    .stButton>button {
        font-size: 24px !important;
    }
    /* Tooltips */
    .css-1d391kg p {
        font-size: 20px !important;
    }
    /* Resultados / textos mostrados con st.markdown */
    .stMarkdown, .stText {
        font-size: 28px !important;
    }
    </style>
    """, unsafe_allow_html=True
)

# ================= Título =================
st.markdown("<h1>Slope Stability Factor of Safety Prediction</h1>", unsafe_allow_html=True)

# ================= Entradas del usuario con tooltips =================
unit_weight = st.number_input(
    'Unit weight (kN/m³)', 
    value=18.0, 
    format="%.2f",
    help="Weight of the soil per unit volume. Heavier soils increase the load on the slope."
)

friction_angle = st.number_input(
    'Friction angle (°)',
    value=30.0,
    format="%.2f",
    help="Internal friction angle of the soil. Higher angle means more resistance to sliding."
)

cohesion = st.number_input(
    'Cohesion (kPa)',
    value=10.0,
    format="%.2f",
    help="Force of adhesion between soil particles. Higher cohesion helps maintain slope stability."
)

height = st.number_input(
    'Height H (m)',
    value=5.0,
    format="%.2f",
    help="Height of the slope from base to top. Taller slopes tend to be less stable."
)

slope_inclination = st.number_input(
    'Slope inclination (°)',
    value=45.0,
    format="%.2f",
    help="Inclination of the slope. Steeper slopes are usually less stable."
)

kh = st.number_input(
    'kh (horizontal seismic coefficient)',
    value=0.0,
    format="%.2f",
    help="Horizontal seismic coefficient. Represents horizontal ground acceleration during an earthquake; higher values increase seismic effect on slope stability."
)

# ================= Cargar modelos y scaler =================
scaler = joblib.load('scaler.pkl')
rf_model = joblib.load('rf_model.pkl')
svm_model = joblib.load('svm_model.pkl')

with open("nn_model.json", "r") as json_file:
    nn_json = json_file.read()
nn_model = model_from_json(nn_json)
nn_model.load_weights("nn_model.weights.h5")
nn_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# ================= Predicción =================
if st.button('Predict FS'):
    # Crear vector de entrada
    X_input = np.array([[unit_weight, friction_angle, cohesion, height, slope_inclination, kh]])
    X_scaled = scaler.transform(X_input)

    # Random Forest
    fs_rf = rf_model.predict(X_scaled)[0]

    # SVM
    fs_svm = svm_model.predict(X_scaled)[0]

    # Red Neuronal
    fs_nn = nn_model.predict(X_scaled)[0][0]

    # Mostrar resultados
    st.markdown(f"<p>Random Forest: {fs_rf:.4f}</p>", unsafe_allow_html=True)
    st.markdown(f"<p>SVM: {fs_svm:.4f}</p>", unsafe_allow_html=True)
    st.markdown(f"<p>Neural Network: {fs_nn:.4f}</p>", unsafe_allow_html=True)
