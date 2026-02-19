import streamlit as st
import sys
import os


# Add project root to path
ROOT_DIR = os.path.join(os.path.dirname(__file__), '..')
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import logging
from src.logger import setup_logging

logger = setup_logging(__name__)

from ui_components import (
    render_styles,
    render_home_page,
    render_data_suite_page,
    render_analytics_page,
    render_prediction_page,
    render_login_page,
    render_footer
)


# Page configuration
st.set_page_config(
    page_title="Student Dropout Prediction | AUCA",
    layout="wide"
)

# Initialize Session State
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

# Render Styles (Global)
render_styles()

# Define Navigation logic
if not st.session_state['logged_in']:
    # Show ONLY the login page as landing and DISABLE sidebar
    login_page = st.Page(render_login_page, title="Login", icon=":material/login:")
    pg = st.navigation([login_page], position="hidden")
    
    # Force hide sidebar via CSS just in case
    st.markdown("<style> [data-testid='stSidebar'] { display: none; } </style>", unsafe_allow_html=True)
else:
    # Full Protected Navigation
    pages = {
        "Overview": [
            st.Page(render_home_page, title="Home", icon=":material/home:"),
        ],
        "Data Center": [
            st.Page(render_data_suite_page, title="Data Suite", icon=":material/cleaning_services:"),
            st.Page(render_analytics_page, title="Analytics", icon=":material/bar_chart:"),
        ],
        "Intelligence": [
            st.Page(render_prediction_page, title="Risk Prediction", icon=":material/psychology:"),
        ]
    }
    pg = st.navigation(pages)
    
    # Add Logout to sidebar
    import ui_components
    ui_components.render_logout()

# Execute selected page
pg.run()

# Render Footer (Global)
render_footer()
