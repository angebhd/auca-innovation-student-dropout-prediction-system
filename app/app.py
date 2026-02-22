import sys
import os

# Add project root to Python path so src/ module can be found
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_processing import clean_student_data
from src.data_generation import generate_student_dummy_data
import yaml
import logging
from src.logger import setup_logging
import bcrypt
import json

logger = setup_logging(__name__)

# Load config
try:
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    RAW_DATA_PATH = config['paths']['raw_dir']
    CLEANED_DATA_PATH = config['paths'].get('cleaned_dir', 'data/cleaned')      
except Exception:
    RAW_DATA_PATH = "data/raw"
    CLEANED_DATA_PATH = "data/cleaned"

USERS_FILE = "users.json"

def load_users():
    """Load users from JSON file."""
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    """Save users to JSON file."""
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f)

def hash_password(password):
    """Hash a password."""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def check_password(password, hashed):
    """Check password against hash."""
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def ensure_raw_dir():
    """Ensures that the raw data directory exists."""
    if not os.path.exists(RAW_DATA_PATH):
        os.makedirs(RAW_DATA_PATH)

def ensure_cleaned_dir():
    """Ensures that the cleaned data directory exists."""
    if not os.path.exists(CLEANED_DATA_PATH):
        os.makedirs(CLEANED_DATA_PATH)

def render_styles():
    """Renders custom CSS for the application with smooth animations."""
    st.markdown("""
        <style>
        /* Animations */
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        @keyframes slideInUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes slideInLeft {
            from { opacity: 0; transform: translateX(-20px); }
            to { opacity: 1; transform: translateX(0); }
        }
        @keyframes slideInRight {
            from { opacity: 0; transform: translateX(20px); }
            to { opacity: 1; transform: translateX(0); }
        }

        /* Main container styling */
        .stApp {
            background: transparent;
        }

        /* Buttons */
        .stButton>button {
            width: 100%;
            border-radius: 8px;
            height: 3.2em;
            background-color: #3b82f6;
            color: white;
            border: none;
            transition: all 0.3s ease;
            font-weight: 600;
            animation: slideInUp 0.5s ease-out;
        }
        .stButton>button:hover {
            background-color: #2563eb;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            transform: translateY(-2px);
        }

        /* Headers */
        .header {
            font-size: 2.8rem;
            font-weight: 800;
            background: linear-gradient(90deg, #3b82f6, #60a5fa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
            animation: slideInLeft 0.6s ease-out;
        }
        .subheader {
            font-size: 1.1rem;
            color: #94a3b8;
            margin-bottom: 2.5rem;
            font-weight: 400;
            letter-spacing: 0.5px;
        }

        /* Chart Container Animations */
        .chart-container {
            padding: 1.5rem;
            border-radius: 12px;
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(59, 130, 246, 0.1);
            animation: slideInUp 0.6s ease-out;
            transition: all 0.3s ease;
            height: 100%;
            min-height: 520px;
            display: flex;
            flex-direction: column;
        }
        .chart-container:hover {
            border-color: rgba(59, 130, 246, 0.3);
            box-shadow: 0 8px 16px rgba(59, 130, 246, 0.1);
        }

        /* Chart titles and captions - optimized spacing */
        .chart-title {
            font-size: 1.1rem !important;
            font-weight: 700 !important;
            color: #1e3a8a !important;
            margin: 0 0 0.3rem 0 !important;
            padding: 0 !important;
            letter-spacing: 0.3px !important;
        }
        .chart-caption {
            font-size: 0.8rem !important;
            color: #64748b !important;
            margin: 0 0 0.75rem 0 !important;
            padding: 0 !important;
            line-height: 1.3 !important;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-image: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
            border-right: 1px solid #334155;
        }
        [data-testid="stSidebar"] * {
            color: #f8fafc !important;
        }

        /* Sidebar Navigation Items */
        [data-testid="stSidebarNav"] {
            background-color: transparent !important;
            padding-top: 1rem;
        }

        /* Branding */
        .sidebar-title {
            padding: 1.5rem 1rem;
            font-size: 1.6rem;
            font-weight: 900;
            color: #3b82f6 !important;
            letter-spacing: -1px;
            border-bottom: 1px solid #334155;
            margin-bottom: 1.5rem;
            text-transform: uppercase;
        }

        /* Metrics Cards */
        [data-testid="metric-container"] {
            animation: fadeIn 0.6s ease-out;
            background: rgba(59, 130, 246, 0.05) !important;
            border-radius: 12px !important;
            border: 1px solid rgba(59, 130, 246, 0.1) !important;
            padding: 1.5rem !important;
        }

        /* Dataframes & Cards */
        .stDataFrame {
            border: 1px solid #334155 !important;
            border-radius: 10px;
        }

        </style>
    """, unsafe_allow_html=True)
    st.sidebar.markdown('<div class="sidebar-title">📊 AUCA System</div>', unsafe_allow_html=True)

def render_header():
    """Renders the main headers of the application."""
    st.markdown('<div class="header">Student Dropout Prediction System</div>', unsafe_allow_html=True)
    st.markdown('<div class="subheader">Professional Data Management & Analytics Suite</div>', unsafe_allow_html=True)

def render_home_page():
    """Renders the Home page with project overview."""
    render_header()
    st.markdown("## AUCA Innovation Lab: Big Data Student Success System")
    st.info("Advanced analytics initiative focused on predictive modeling for student retention at the Adventist University of Central Africa.")
    st.markdown("""
    ### Problem Statement
    Higher education institutions face challenges in identifying students who are at risk of dropping out or failing courses at an early stage. Academic performance data, attendance records, and learning behavior are often analyzed too late, resulting in delayed interventions and increased failure rates.
    
    ### Proposed ML-Based Solution
    We are developing a machine learning–based prediction system that analyzes historical student data such as grades, attendance, and course engagement to classify students into risk categories. The system will generate early warning indicators to support timely academic interventions by lecturers and academic administrators.
    
    ### System Architecture & Pipeline
    1. **Data Engineering**: Automated ingestion and cleaning of disparate student records from backend systems.
    2. **Exploratory Data Analysis**: Real-time visualization of academic trends using Big Data visualization techniques.
    3. **Predictive Modeling**: Implementation of classification algorithms to identify churn risk before it becomes critical.
    4. **Intervention Tracking**: Monitoring the impact of administrative support on student outcomes.
    """)

def render_data_acquisition_column():
    """Renders the column for data acquisition (Generation and Upload)."""      
    ensure_raw_dir()
    st.subheader("1. Data Acquisition")

    # Dummy Data Generation
    st.info("No data? Generate a synthetic 'dirty' dataset to test the system.")
    gen_col1, gen_col2 = st.columns([2, 1])
    n_samples = gen_col1.number_input("Number of samples", min_value=100, max_value=10000, value=500, key="gen_samples")
    file_name = gen_col2.text_input("File Name", value="dummy_data", key="gen_name")
    
    if st.button("Generate & Save"):
        df_dummy = generate_student_dummy_data(n_samples=n_samples)
        full_path = os.path.join(RAW_DATA_PATH, f"{file_name}.csv")
        df_dummy.to_csv(full_path, index=False)
        st.success(f"Generated and saved to {full_path}")
        st.session_state['selected_file'] = f"{file_name}.csv"

    st.divider()

    # CSV Upload
    st.info("Upload your own student dataset (CSV format).")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        file_save_name = st.text_input("Save As (Optional)", value=uploaded_file.name, key="upload_name")
        if st.button("Save Uploaded"):
            try:
                df_uploaded = pd.read_csv(uploaded_file)
                full_path = os.path.join(RAW_DATA_PATH, file_save_name if file_save_name.endswith(".csv") else f"{file_save_name}.csv")
                df_uploaded.to_csv(full_path, index=False)
                st.success(f"Uploaded file saved to {full_path}")
                st.session_state['selected_file'] = os.path.basename(full_path)
            except Exception as e:
                st.error(f"Error reading CSV: {e}")

def render_processing_export_column():
    """Renders the column for processing data and exporting results."""
    ensure_raw_dir()
    ensure_cleaned_dir()
    st.subheader("2. Processing & Export")

    # File Selector
    raw_files = [f for f in os.listdir(RAW_DATA_PATH) if f.endswith(".csv")] if os.path.exists(RAW_DATA_PATH) else []

    if not raw_files:
        st.warning("No files found in data/raw/. Please acquire a dataset first.")
        return

    # Use session state to keep track of selection
    default_index = 0
    if 'selected_file' in st.session_state and st.session_state['selected_file'] in raw_files:
        default_index = raw_files.index(st.session_state['selected_file'])

    selected_file = st.selectbox("Select File to Clean", raw_files, index=default_index)
    st.session_state['selected_file'] = selected_file

    df_to_clean = pd.read_csv(os.path.join(RAW_DATA_PATH, selected_file))
    st.write(f"Selected Data: **{selected_file}** ({len(df_to_clean)} rows)")

    save_to_cleaned = st.checkbox("Save cleaned file to data/cleaned", value=True)

    if st.button("Clean Selected Data"):
        with st.spinner(f"Cleaning {selected_file}..."):
            cleaned_df, stats = clean_student_data(df_to_clean)
            st.session_state['cleaned_df'] = cleaned_df
            st.session_state['stats'] = stats

            if save_to_cleaned:
                base_name = os.path.splitext(selected_file)[0]
                cleaned_path = os.path.join(CLEANED_DATA_PATH, f"{base_name}_cleaned.csv")
                cleaned_df.to_csv(cleaned_path, index=False)
                st.session_state['cleaned_path'] = cleaned_path

        st.success("Dataset cleaned and validated!")

        if save_to_cleaned and 'cleaned_path' in st.session_state:
            st.info(f"Saved cleaned file to {st.session_state['cleaned_path']}")

        # Display stats
        s1, s2, s3 = st.columns(3)
        s1.metric("Final Rows", stats['final_rows'])
        s2.metric("Duplicates Removed", stats['duplicates_removed'])
        s3.metric("Retention Rate", f"{stats['retention_rate']}%")

        st.dataframe(cleaned_df.head(5))

    if 'cleaned_df' in st.session_state:
        st.divider()
        st.subheader("Download Results")

        csv = st.session_state['cleaned_df'].to_csv(index=False).encode('utf-8')
        base_name = os.path.splitext(selected_file)[0]
        st.download_button(
            label="Download Cleaned CSV",
            data=csv,
            file_name=f"{base_name}_cleaned.csv",
            mime="text/csv",
        )

def render_data_suite_page():
    """Orchestrates the Data Suite page layout."""
    render_header()
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        render_data_acquisition_column()
    with col2:
        render_processing_export_column()

def load_cleaned_data():
    """Load cleaned data from the cleaned directory."""
    try:
        if os.path.exists(CLEANED_DATA_PATH):
            files = [f for f in os.listdir(CLEANED_DATA_PATH) if f.endswith('.csv')]
            if files:
                latest_file = max([os.path.join(CLEANED_DATA_PATH, f) for f in files], key=os.path.getctime)
                return pd.read_csv(latest_file)
    except Exception as e:
        st.error(f"Error loading cleaned data: {e}")
    return None

def plot_dropout_risk_distribution(df):
    """Chart 1: Dropout Risk Distribution - Pie Chart"""
    if 'dropout_risk' not in df.columns:
        st.warning("Column 'dropout_risk' not found in data")
        return
    
    fig, ax = plt.subplots(figsize=(5, 4.5))
    
    # Remove NaN values and get value counts
    risk_counts = df['dropout_risk'].dropna().value_counts()
    
    # Define colors for different risk levels
    color_map = {'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e74c3c'}
    colors = [color_map.get(str(risk), '#3498db') for risk in risk_counts.index]
    
    # Only plot if we have data
    if len(risk_counts) == 0:
        st.warning("No valid dropout risk data available")
        return
    
    wedges, texts, autotexts = ax.pie(risk_counts.values, 
                                        labels=risk_counts.index,
                                        autopct='%1.1f%%',
                                        colors=colors,
                                        startangle=90,
                                        textprops={'fontsize': 10, 'weight': 'bold'})
    
    ax.set_title('Student Dropout Risk Distribution', fontsize=12, fontweight='bold', pad=10)
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(9)
    
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

def plot_gpa_attendance_scatter(df):
    """Chart 2: GPA vs Attendance Rate by Dropout Status"""
    required_cols = ['attendance_rate', 'current_gpa', 'actual_dropout']
    if not all(col in df.columns for col in required_cols):
        st.warning(f"Missing columns: {set(required_cols) - set(df.columns)}")
        return
    
    fig, ax = plt.subplots(figsize=(5, 4.5))
    
    dropout_groups = df['actual_dropout'].unique()
    colors_map = {0: '#2ecc71', 1: '#e74c3c'}
    labels_map = {0: 'Continuing', 1: 'Dropped Out'}
    
    for dropout in sorted(dropout_groups):
        mask = df['actual_dropout'] == dropout
        ax.scatter(df[mask]['attendance_rate'], 
                df[mask]['current_gpa'],
                alpha=0.5,
                s=40,
                label=labels_map.get(dropout, str(dropout)),
                color=colors_map.get(dropout, '#3498db'),
                edgecolors='black',
                linewidth=0.3)
    
    ax.set_xlabel('Attendance Rate (%)', fontsize=9, fontweight='bold')
    ax.set_ylabel('Current GPA', fontsize=9, fontweight='bold')
    ax.set_title('GPA vs Attendance by Student Status', fontsize=11, fontweight='bold', pad=8)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.15, linestyle='--')
    
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

def plot_gpa_by_risk_level(df):
    """Chart 3: Average GPA by Dropout Risk Level"""
    required_cols = ['dropout_risk', 'current_gpa']
    if not all(col in df.columns for col in required_cols):
        st.warning(f"Missing columns: {set(required_cols) - set(df.columns)}")
        return
    
    fig, ax = plt.subplots(figsize=(5, 4.5))
    
    gpa_by_risk = df.groupby('dropout_risk')['current_gpa'].agg(['mean', 'std'])
    
    # Sort by risk level order
    risk_order = ['Low', 'Medium', 'High']
    existing_risks = [risk for risk in risk_order if risk in gpa_by_risk.index]
    gpa_by_risk = gpa_by_risk.reindex(existing_risks)
    
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    bars = ax.bar(range(len(gpa_by_risk)), gpa_by_risk['mean'], 
            color=colors[:len(gpa_by_risk)], edgecolor='black', linewidth=1.5, alpha=0.8)
    
    # Add error bars
    ax.errorbar(range(len(gpa_by_risk)), gpa_by_risk['mean'], 
                yerr=gpa_by_risk['std'], fmt='none', color='black', 
                capsize=5, capthick=2, linewidth=2)
    
    ax.set_xticks(range(len(gpa_by_risk)))
    ax.set_xticklabels(gpa_by_risk.index, fontsize=10, fontweight='bold')
    ax.set_ylabel('Average GPA', fontsize=10, fontweight='bold')
    ax.set_xlabel('Dropout Risk Level', fontsize=10, fontweight='bold')
    ax.set_title('Average GPA by Dropout Risk Level', fontsize=12, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.2, axis='y', linestyle='--')
    
    # Add value labels on bars
    for bar, val in zip(bars, gpa_by_risk['mean']):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.1, f'{val:.2f}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

def plot_feature_importance(df):
    """Chart 4: Feature Importance for Dropout Prediction"""
    key_features = [
        'age', 'admission_grade', 'current_gpa', 'failed_courses', 
        'attendance_rate', 'absences_count', 'online_portal_logins', 
        'library_visits', 'participation_score', 'distance_from_campus'
    ]
    
    # Filter columns that exist in the dataframe
    available_features = [col for col in key_features if col in df.columns]
    
    if 'actual_dropout' not in df.columns or len(available_features) < 2:
        st.warning("Missing required columns for feature importance analysis")
        return
    
    # Calculate importance based on correlation with dropout outcome
    importances = []
    feature_names = []
    
    for feature in available_features:
        if df[feature].dtype in [float, int]:
            # Calculate absolute correlation with dropout
            corr = abs(df[feature].corr(df['actual_dropout']))
            importances.append(corr)
            feature_names.append(feature.replace('_', ' ').title())
    
    # Sort by importance
    sorted_indices = np.argsort(importances)[::-1]
    sorted_features = np.array(feature_names)[sorted_indices]
    sorted_importances = np.array(importances)[sorted_indices]
    
    # Create color gradient based on importance
    colors = plt.cm.RdYlGn_r(sorted_importances / max(sorted_importances))
    
    fig, ax = plt.subplots(figsize=(5, 4.5))
    
    bars = ax.barh(range(len(sorted_features)), sorted_importances, color=colors, edgecolor='black', linewidth=1)
    
    ax.set_yticks(range(len(sorted_features)))
    ax.set_yticklabels(sorted_features, fontsize=9)
    ax.set_xlabel('Importance Score', fontsize=10, fontweight='bold')
    ax.set_title('Feature Importance for Dropout Prediction', fontsize=12, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.2, axis='x', linestyle='--')
    ax.set_xlim(0, max(sorted_importances) * 1.15)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, sorted_importances)):
        ax.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

def render_analytics_page():
    """Renders the Analytics Dashboard with 4 key charts in a perfect 2x2 grid."""
    render_header()
    st.markdown("<h2 style='text-align: center; margin-bottom: 2rem;'>Analytics Dashboard</h2>", unsafe_allow_html=True)
    
    # Load cleaned data
    df = load_cleaned_data()
    
    if df is None:
        st.warning("⚠️ No cleaned data available. Please clean a dataset in the Data Suite first.")
        return
    
    # Display summary metrics with tight spacing
    st.markdown("<div style='margin-bottom: 1.5rem;'></div>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4, gap="medium")
    with col1:
        st.metric("Total Students", len(df))
    with col2:
        dropout_count = (df['actual_dropout'] == 1).sum() if 'actual_dropout' in df.columns else 0
        st.metric("Dropped Out", dropout_count)
    with col3:
        if 'current_gpa' in df.columns:
            st.metric("Avg GPA", f"{df['current_gpa'].mean():.2f}")
    with col4:
        if 'attendance_rate' in df.columns:
            st.metric("Avg Attendance", f"{df['attendance_rate'].mean():.1f}%")
    
    # Spacing before grid
    st.markdown("<div style='margin-top: 2rem; margin-bottom: 1rem;'></div>", unsafe_allow_html=True)
    
    # First row of charts - perfectly aligned 2x2 grid
    col1, col2 = st.columns(2, gap="medium")
    
    with col1:
        st.markdown('<div style="min-height: 50px;"><h3 class="chart-title">Dropout Risk Distribution</h3></div>', unsafe_allow_html=True)
        plot_dropout_risk_distribution(df)
    
    with col2:
        st.markdown('<div style="min-height: 50px;"><h3 class="chart-title">GPA vs Attendance</h3></div>', unsafe_allow_html=True)
        plot_gpa_attendance_scatter(df)
    
    # Spacing between rows
    st.markdown("<div style='margin-top: 1.5rem; margin-bottom: 1rem;'></div>", unsafe_allow_html=True)
    
    # Second row of charts
    col3, col4 = st.columns(2, gap="medium")
    
    with col3:
        st.markdown('<div style="min-height: 50px;"><h3 class="chart-title">GPA by Risk Level</h3></div>', unsafe_allow_html=True)
        plot_gpa_by_risk_level(df)
    
    with col4:
        st.markdown('<div style="min-height: 50px;"><h3 class="chart-title">Feature Importance</h3></div>', unsafe_allow_html=True)
        plot_feature_importance(df)

def render_prediction_page():
    """Placeholder for the AI Prediction page."""
    render_header()
    st.header("Predictive Modeling Engine")
    st.info("Coming Soon: ML-based student risk prediction model")

def render_login_page():
    """Renders a professional login and signup page."""
    st.markdown("""
        <style>
        .login-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 3rem;
            border-radius: 12px;
            background-color: #1e293b;
            border: 1px solid #334155;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
            margin-top: 5rem;
        }
        </style>
    """, unsafe_allow_html=True)

    _, col2, _ = st.columns([1, 2, 1])

    with col2:
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        st.markdown('<div class="header" style="text-align: center; font-size: 2.2rem;">AUCA Platform</div>', unsafe_allow_html=True)
        st.markdown('<div class="subheader" style="text-align: center; margin-bottom: 2rem;">Secure Institutional Access</div>', unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["Login", "Sign Up"])
        
        with tab1:
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Sign In")
                
                if submitted:
                    users = load_users()
                    if username in users and check_password(password, users[username]):
                        st.session_state['logged_in'] = True
                        st.session_state['username'] = username
                        st.success("Access Granted ✓")
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password")
        
        with tab2:
            with st.form("signup_form"):
                new_username = st.text_input("Username")
                new_password = st.text_input("Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                submitted_signup = st.form_submit_button("Sign Up")
                
                if submitted_signup:
                    if new_password != confirm_password:
                        st.error("Passwords do not match")
                    elif len(new_password) < 6:
                        st.error("Password must be at least 6 characters")
                    else:
                        users = load_users()
                        if new_username in users:
                            st.error("Username already exists")
                        else:
                            users[new_username] = hash_password(new_password)
                            save_users(users)
                            st.success("Account created successfully! Please login.")
        
        st.markdown('</div>', unsafe_allow_html=True)

def render_logout():
    """Handles logout logic."""
    st.sidebar.markdown(f"Logged in as: **{st.session_state['username']}**")
    if st.sidebar.button("� Logout", key="logout_btn", type="secondary"):
        st.session_state['logged_in'] = False
        st.session_state['username'] = None
        st.rerun()

def render_footer():
    """Renders the application footer."""
    st.markdown("---")
    st.caption("Developed by AUCA Innovation Lab | © 2026 | Student Dropout Prediction System v1.0")

def main():
    """Main Streamlit application."""
    # Page configuration
    st.set_page_config(
        page_title="Student Dropout Prediction | AUCA",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize Session State
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    if 'username' not in st.session_state:
        st.session_state['username'] = None

    # Render Styles (Global)
    render_styles()

    # Define Navigation logic
    if not st.session_state['logged_in']:
        # Show ONLY the login page as landing
        render_login_page()
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
        render_logout()

        # Execute selected page
        pg.run()

    # Render Footer (Global)
    render_footer()

if __name__ == "__main__":
    main()