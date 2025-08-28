import streamlit as st

# Test simple title display
st.set_page_config(page_title="Test App", layout="wide")

# Simple CSS
st.markdown("""
<style>
.stApp { background-color: white; }
</style>
""", unsafe_allow_html=True)

# Test title with multiple methods
st.title("Slope Stability Factor of Safety Prediction")

st.markdown("# Alternative Title Method")

st.markdown("""
<h1 style="text-align: center; color: #2c3e50; padding: 20px;">
HTML Title Method
</h1>
""", unsafe_allow_html=True)

# Simple content to verify app works
st.write("✅ If you can see this text, the app is working!")

col1, col2 = st.columns(2)

with col1:
    st.write("### Left Column")
    height = st.slider("Test Slider", 1, 10, 5)
    
with col2:
    st.write("### Right Column")
    st.metric("Test Metric", f"{height}")

st.success("Application loaded successfully!")
