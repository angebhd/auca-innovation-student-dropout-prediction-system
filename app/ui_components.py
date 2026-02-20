import streamlit as st
import pandas as pd
import os
from src.data_processing import clean_student_data
from src.data_generation import generate_student_dummy_data

import yaml

# Load config
try:
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    RAW_DATA_PATH = config['paths']['raw_dir']
    CLEANED_DATA_PATH = config['paths'].get('cleaned_dir', 'data/cleaned')
except Exception:
    RAW_DATA_PATH = "data/raw"
    CLEANED_DATA_PATH = "data/cleaned"

def ensure_raw_dir():
    """Ensures that the raw data directory exists."""
    if not os.path.exists(RAW_DATA_PATH):
        os.makedirs(RAW_DATA_PATH)

def ensure_cleaned_dir():
    """Ensures that the cleaned data directory exists."""
    if not os.path.exists(CLEANED_DATA_PATH):
        os.makedirs(CLEANED_DATA_PATH)

def render_styles():
    """Renders custom CSS for the application."""
    st.markdown("""
        <style>
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
        }
        .stButton>button:hover {
            background-color: #2563eb;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            transform: translateY(-1px);
        }
        
        /* Headers */
        .header {
            font-size: 2.8rem;
            font-weight: 800;
            background: linear-gradient(90deg, #3b82f6, #60a5fa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }
        .subheader {
            font-size: 1.1rem;
            color: #94a3b8;
            margin-bottom: 2.5rem;
            font-weight: 400;
            letter-spacing: 0.5px;
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
        
        /* Umoja Branding */
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
        
        /* Dataframes & Cards */
        .stDataFrame {
            border: 1px solid #334155 !important;
            border-radius: 10px;
        }
        
        </style>
    """, unsafe_allow_html=True)
    st.sidebar.markdown('<div class="sidebar-title">Umoja Team</div>', unsafe_allow_html=True)

def render_header():
    """Renders the main headers of the application."""
    st.markdown('<div class="header">Student Dropout Prediction System</div>', unsafe_allow_html=True)
    st.markdown('<div class="subheader">Professional Data Management & Cleaning Suite</div>', unsafe_allow_html=True)

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
    raw_files = [f for f in os.listdir(RAW_DATA_PATH) if f.endswith(".csv")]
    
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

def render_home_page():
    """Renders the Home page with project overview."""
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

def render_analytics_page():
    """Placeholder for the Analytics page."""
    st.header("Exploratory Data Analysis")
    st.image("https://via.placeholder.com/800x400.png?text=Big+Data+Analytics+Dashboard", width='stretch')

def render_prediction_page():
    
    import joblib
    import pandas as pd
    import numpy as np

    st.markdown("## AI Dropout Risk Prediction ")
    st.markdown("Upload cleaned student data or evaluate a single student profile.")

   
    try:
        model = joblib.load("models/risk_model.pkl")
        feature_cols = joblib.load("models/feature_columns.pkl")
    except Exception:
        st.error(" Model not found. Please train the model first.")
        return

    option = st.radio("Choose Prediction Mode:", ["Upload CSV", "Single Student Input"])

   
    if option == "Upload CSV":

        uploaded_file = st.file_uploader("Upload Cleaned CSV File", type="csv")

        if uploaded_file:
            df = pd.read_csv(uploaded_file)

            st.write("Preview:", df.head())

          
            drop_cols = [col for col in ["student_id", "actual_dropout", "dropout_risk"] if col in df.columns]
            df = df.drop(columns=drop_cols, errors="ignore")

            df_encoded = pd.get_dummies(df)
            df_encoded = df_encoded.reindex(columns=feature_cols, fill_value=0)

            if st.button("Run Prediction"):
                predictions = model.predict(df_encoded)
                probabilities = model.predict_proba(df_encoded)

                df["Predicted_Risk"] = predictions
                df["Confidence_%"] = np.round(np.max(probabilities, axis=1) * 100, 2)

                st.success("Prediction completed successfully.")
                st.dataframe(df.head())

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Predictions",
                    data=csv,
                    file_name="predicted_dropout_risk.csv",
                    mime="text/csv",
                )

    else:

        st.subheader("Enter Student Profile")

        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", 16, 60, 20)
            admission_grade = st.number_input("Admission Grade", 0.0, 20.0, 12.0)
            semester_1_gpa = st.number_input("Semester 1 GPA", 0.0, 20.0, 12.0)
            semester_2_gpa = st.number_input("Semester 2 GPA", 0.0, 20.0, 12.0)
            semester_3_gpa = st.number_input("Semester 3 GPA", 0.0, 20.0, 12.0)

        with col2:
            current_gpa = st.number_input("Current GPA", 0.0, 20.0, 12.0)
            attendance_rate = st.slider("Attendance Rate (%)", 0, 100, 75)
            failed_courses = st.number_input("Failed Courses", 0, 20, 0)
            absences_count = st.number_input("Absences Count", 0, 100, 5)
            late_submissions = st.number_input("Late Submissions", 0, 50, 2)

        if st.button("Predict Student Risk"):

            input_dict = {
                "age": age,
                "admission_grade": admission_grade,
                "semester_1_gpa": semester_1_gpa,
                "semester_2_gpa": semester_2_gpa,
                "semester_3_gpa": semester_3_gpa,
                "current_gpa": current_gpa,
                "attendance_rate": attendance_rate,
                "failed_courses": failed_courses,
                "absences_count": absences_count,
                "late_submissions": late_submissions,
            }

            input_df = pd.DataFrame([input_dict])

            input_df = pd.get_dummies(input_df)
            input_df = input_df.reindex(columns=feature_cols, fill_value=0)

            prediction = model.predict(input_df)[0]
            probabilities = model.predict_proba(input_df)[0]
            confidence = round(max(probabilities) * 100, 2)

            if prediction == "High":
                st.error(f" High Dropout Risk ({confidence}% confidence)")
            elif prediction == "Medium":
                st.warning(f" Medium Dropout Risk ({confidence}% confidence)")
            else:
                st.success(f" Low Dropout Risk ({confidence}% confidence)")
                st.write("### Prediction Probabilities")





    

def render_login_page():
    """Renders a professional mocked login landing page."""
    # Custom styling for login
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
        st.markdown('<div class="header" style="text-align: center; font-size: 2.2rem;">Umoja Platform</div>', unsafe_allow_html=True)
        st.markdown('<div class="subheader" style="text-align: center; margin-bottom: 2rem;">Secure Institutional Access</div>', unsafe_allow_html=True)
        
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username", placeholder="admin")
            password = st.text_input("Password", type="password", placeholder="admin")
            submitted = st.form_submit_button("Sign In")
            
            if submitted:
                if username == "admin" and password == "admin":
                    st.session_state['logged_in'] = True
                    st.success("Access Granted")
                    st.rerun()
                else:
                    st.error("Authentication failed")
        st.markdown('</div>', unsafe_allow_html=True)

def render_logout():
    """Handles logout logic."""
    if st.sidebar.button("Logout", key="logout_btn", type="secondary"):
        st.session_state['logged_in'] = False
        st.rerun()

def render_footer():
    """Renders the application footer."""
    st.markdown("---")
    st.caption("Developed by Umoja Team | © 2026")

def render_data_suite_page():
    """Orchestrates the Data Suite page layout."""
    render_header()
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        render_data_acquisition_column()
    with col2:
        render_processing_export_column()
