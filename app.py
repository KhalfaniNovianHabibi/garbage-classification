import streamlit as st

from home import show_home
from eda import show_eda
from prediction import show_prediction

# =====================================================
# Page Configuration
# =====================================================
st.set_page_config(
    page_title="Garbage Classification using CNN",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# Sidebar
# =====================================================
st.sidebar.title("♻️ Garbage Classification")
st.sidebar.caption("CNN Image Classification")

menu = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📊 EDA",
        "🤖 Prediction"
    ]
)

st.sidebar.divider()

st.sidebar.markdown(
"""
### About

Computer Vision project menggunakan CNN untuk melakukan klasifikasi sampah ke dalam lima kategori.

**Framework**
- TensorFlow
- Streamlit

**Author**
- Khalfani Novian Habibi
"""
)

# =====================================================
# Routing
# =====================================================
if menu == "🏠 Home":
    show_home()

elif menu == "📊 EDA":
    show_eda()

elif menu == "🤖 Prediction":
    show_prediction()