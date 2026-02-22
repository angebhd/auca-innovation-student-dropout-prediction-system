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
            
            # Log to history
            from src.history import log_data_clean
            user = st.session_state.get('user', {}).get('username', 'anonymous')
            log_data_clean(selected_file, stats, user=user)

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
    """Render the Analytics page with interactive charts."""
    from src.analytics import analytics
    from src.prediction import predictor
    
    st.markdown('<div class="header">Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="subheader">Interactive visualization of student performance metrics</div>', unsafe_allow_html=True)
    
    # Data source selection
    ensure_cleaned_dir()
    cleaned_files = [f for f in os.listdir(CLEANED_DATA_PATH) if f.endswith(".csv")]
    
    if not cleaned_files:
        st.warning("No cleaned datasets found. Please process a dataset in Data Suite first.")
        st.info("Navigate to **Data Suite** → Clean a dataset → Return here for analytics.")
        return
    
    selected_file = st.selectbox("Select Dataset for Analysis", cleaned_files, key="analytics_file")
    df = pd.read_csv(os.path.join(CLEANED_DATA_PATH, selected_file))
    
    # Check if predictions exist, if not run them
    if 'risk_category' not in df.columns and predictor.is_trained:
        with st.spinner("Running risk predictions..."):
            df = predictor.predict(df)
    
    # Overview metrics
    st.markdown("### 📊 Overview Metrics")
    stats = analytics.get_overview_stats(df)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Students", stats['total_students'])
    col2.metric("Average GPA", stats.get('avg_gpa', 'N/A'))
    col3.metric("Avg Attendance", f"{stats.get('avg_attendance', 0):.1f}%")
    col4.metric("Students with Failures", stats.get('students_with_failures', 0))
    
    st.divider()
    
    # Charts section
    st.markdown("### 📈 Performance Analytics")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        gpa_chart = analytics.create_gpa_distribution_chart(df)
        if gpa_chart:
            st.plotly_chart(gpa_chart, use_container_width=True)
    
    with chart_col2:
        attendance_chart = analytics.create_attendance_distribution_chart(df)
        if attendance_chart:
            st.plotly_chart(attendance_chart, use_container_width=True)
    
    # Second row
    chart_col3, chart_col4 = st.columns(2)
    
    with chart_col3:
        semester_chart = analytics.create_semester_gpa_trend(df)
        if semester_chart:
            st.plotly_chart(semester_chart, use_container_width=True)
    
    with chart_col4:
        failed_chart = analytics.create_failed_courses_chart(df)
        if failed_chart:
            st.plotly_chart(failed_chart, use_container_width=True)
    
    st.divider()
    
    # Risk analysis section (if predictions exist)
    if 'risk_category' in df.columns:
        st.markdown("### 🎯 Risk Analysis")
        
        risk_col1, risk_col2 = st.columns(2)
        
        with risk_col1:
            risk_pie = analytics.create_risk_distribution_chart(df)
            if risk_pie:
                st.plotly_chart(risk_pie, use_container_width=True)
        
        with risk_col2:
            scatter = analytics.create_gpa_vs_attendance_scatter(df)
            if scatter:
                st.plotly_chart(scatter, use_container_width=True)
    
    # Engagement section
    st.markdown("### 👥 Engagement Analysis")
    
    eng_col1, eng_col2 = st.columns(2)
    
    with eng_col1:
        radar = analytics.create_engagement_radar(df)
        if radar:
            st.plotly_chart(radar, use_container_width=True)
    
    with eng_col2:
        corr = analytics.create_correlation_heatmap(df)
        if corr:
            st.plotly_chart(corr, use_container_width=True)
    
    # Data preview
    st.divider()
    st.markdown("### 📋 Data Preview")
    with st.expander("View Raw Data"):
        st.dataframe(df.head(50), use_container_width=True)

def render_prediction_page():
    """Render the AI Prediction page with model training and predictions."""
    from src.prediction import predictor
    from src.analytics import analytics
    
    st.markdown('<div class="header">Risk Prediction Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="subheader">ML-powered dropout risk assessment and early warning system</div>', unsafe_allow_html=True)
    
    # Tabs for different functions
    tab1, tab2, tab3 = st.tabs(["🎓 Train Model", "🔮 Batch Prediction", "👤 Individual Prediction"])
    
    # ==================== TAB 1: TRAIN MODEL ====================
    with tab1:
        st.markdown("### Train Prediction Model")
        st.info("Train a Random Forest classifier on cleaned student data to predict dropout risk.")
        
        ensure_cleaned_dir()
        cleaned_files = [f for f in os.listdir(CLEANED_DATA_PATH) if f.endswith(".csv")]
        
        if not cleaned_files:
            st.warning("No cleaned datasets available. Please clean a dataset in Data Suite first.")
        else:
            training_file = st.selectbox("Select Training Dataset", cleaned_files, key="train_file")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                if st.button("🚀 Train Model", type="primary", use_container_width=True):
                    with st.spinner("Training model... This may take a moment."):
                        df = pd.read_csv(os.path.join(CLEANED_DATA_PATH, training_file))
                        metrics = predictor.train(df)
                        st.session_state['training_metrics'] = metrics
                        
                        # Log to history
                        from src.history import log_model_train
                        user = st.session_state.get('user', {}).get('username', 'anonymous')
                        log_model_train(metrics, user=user)
                    st.success("Model trained successfully!")
            
            with col2:
                if predictor.is_trained:
                    st.success("✅ Model Ready")
                else:
                    st.warning("⚠️ No Model")
            
            # Show training metrics
            if 'training_metrics' in st.session_state:
                metrics = st.session_state['training_metrics']
                st.markdown("#### Training Results")
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
                m2.metric("Precision", f"{metrics['precision']*100:.1f}%")
                m3.metric("Recall", f"{metrics['recall']*100:.1f}%")
                m4.metric("F1 Score", f"{metrics['f1_score']*100:.1f}%")
                
                # Feature importance
                importance_df = predictor.get_feature_importance()
                if importance_df is not None:
                    st.markdown("#### Feature Importance")
                    fig = analytics.create_feature_importance_chart(importance_df)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
    
    # ==================== TAB 2: BATCH PREDICTION ====================
    with tab2:
        st.markdown("### Batch Prediction")
        st.info("Run predictions on an entire dataset to identify at-risk students.")
        
        if not predictor.is_trained:
            st.warning("Please train a model first in the 'Train Model' tab.")
        else:
            cleaned_files = [f for f in os.listdir(CLEANED_DATA_PATH) if f.endswith(".csv")]
            
            if cleaned_files:
                pred_file = st.selectbox("Select Dataset for Prediction", cleaned_files, key="pred_file")
                
                if st.button("🔮 Run Predictions", type="primary"):
                    with st.spinner("Running predictions..."):
                        df = pd.read_csv(os.path.join(CLEANED_DATA_PATH, pred_file))
                        results = predictor.predict(df)
                        st.session_state['prediction_results'] = results
                        st.session_state['risk_summary'] = predictor.get_risk_summary(results)
                        
                        # Log to history
                        from src.history import log_batch_prediction
                        user = st.session_state.get('user', {}).get('username', 'anonymous')
                        summary = st.session_state['risk_summary']
                        log_batch_prediction(pred_file, summary.get('total_students', 0), summary.get('high_risk', 0), user=user)
                    st.success("Predictions complete!")
                
                # Show results
                if 'prediction_results' in st.session_state:
                    results = st.session_state['prediction_results']
                    summary = st.session_state.get('risk_summary', {})
                    
                    st.markdown("#### Risk Summary")
                    s1, s2, s3, s4 = st.columns(4)
                    s1.metric("Total Students", summary.get('total_students', 0))
                    s2.metric("🔴 High Risk", summary.get('high_risk', 0), 
                             f"{summary.get('high_risk_pct', 0):.1f}%")
                    s3.metric("🟡 Medium Risk", summary.get('medium_risk', 0),
                             f"{summary.get('medium_risk_pct', 0):.1f}%")
                    s4.metric("🟢 Low Risk", summary.get('low_risk', 0),
                             f"{summary.get('low_risk_pct', 0):.1f}%")
                    
                    # Risk distribution chart
                    risk_chart = analytics.create_risk_distribution_chart(results)
                    if risk_chart:
                        st.plotly_chart(risk_chart, use_container_width=True)
                    
                    # Top at-risk students
                    st.markdown("#### 🚨 Top At-Risk Students")
                    top_risk = analytics.get_top_at_risk_students(results, n=10)
                    if not top_risk.empty:
                        st.dataframe(top_risk, use_container_width=True)
                    
                    # Recommendations
                    st.markdown("#### 📋 Intervention Recommendations")
                    recommendations = analytics.get_intervention_recommendations(summary)
                    for rec in recommendations:
                        st.markdown(f"- {rec}")
                    
                    # Download results
                    st.divider()
                    csv = results.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download Predictions CSV",
                        csv,
                        f"predictions_{pred_file}",
                        "text/csv",
                        use_container_width=True
                    )
    
    # ==================== TAB 3: INDIVIDUAL PREDICTION ====================
    with tab3:
        st.markdown("### Individual Student Assessment")
        st.info("Enter student data to get an instant risk prediction.")
        
        if not predictor.is_trained:
            st.warning("Please train a model first in the 'Train Model' tab.")
        else:
            with st.form("individual_prediction"):
                st.markdown("#### Student Information")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    age = st.number_input("Age", min_value=16, max_value=60, value=20)
                    gender = st.selectbox("Gender", ["M", "F", "Other"])
                    admission_grade = st.number_input("Admission Grade", min_value=0.0, max_value=100.0, value=75.0)
                
                with col2:
                    current_gpa = st.number_input("Current GPA", min_value=0.0, max_value=20.0, value=12.0)
                    attendance_rate = st.number_input("Attendance Rate (%)", min_value=0.0, max_value=100.0, value=80.0)
                    failed_courses = st.number_input("Failed Courses", min_value=0, max_value=10, value=0)
                
                with col3:
                    absences_count = st.number_input("Absences Count", min_value=0, max_value=50, value=5)
                    late_submissions = st.number_input("Late Submissions", min_value=0, max_value=20, value=2)
                    participation_score = st.number_input("Participation Score", min_value=0.0, max_value=10.0, value=6.0)
                
                submitted = st.form_submit_button("🔮 Predict Risk", use_container_width=True, type="primary")
                
                if submitted:
                    student_data = {
                        'age': age,
                        'gender': gender,
                        'admission_grade': admission_grade,
                        'current_gpa': current_gpa,
                        'semester_1_gpa': current_gpa,
                        'semester_2_gpa': current_gpa,
                        'semester_3_gpa': current_gpa,
                        'attendance_rate': attendance_rate,
                        'failed_courses': failed_courses,
                        'absences_count': absences_count,
                        'late_submissions': late_submissions,
                        'participation_score': participation_score,
                        'average_grade': current_gpa,
                        'library_visits': 10,
                        'online_portal_logins': 50,
                        'extracurricular_activities': 1,
                        'distance_from_campus': 10,
                        'scholarship_status': 'No',
                        'financial_aid': 'No',
                        'accommodation_type': 'Home',
                        'previous_education': 'Public'
                    }
                    
                    result = predictor.predict_single(student_data)
                    
                    st.markdown("---")
                    st.markdown("### Prediction Result")
                    
                    risk_color = {
                        'Low': '🟢',
                        'Medium': '🟡', 
                        'High': '🔴'
                    }
                    
                    r1, r2, r3 = st.columns(3)
                    r1.metric("Risk Category", f"{risk_color.get(result['risk_category'], '')} {result['risk_category']}")
                    r2.metric("Risk Score", f"{result['risk_score']:.1f}%")
                    r3.metric("Dropout Probability", f"{result['dropout_probability']*100:.1f}%")
                    
                    if result['risk_category'] == 'High':
                        st.error("⚠️ This student is at HIGH risk of dropout. Immediate intervention recommended.")
                    elif result['risk_category'] == 'Medium':
                        st.warning("⚡ This student shows MEDIUM risk. Monitor closely and consider support.")
                    else:
                        st.success("✅ This student is at LOW risk. Continue regular monitoring.")

def render_login_page():
    """Wrapper that calls the new auth page."""
    # Import inside function to avoid circular import
    from auth_ui import render_auth_page
    render_auth_page()


def _deprecated_render_login_page():
    """DEPRECATED: Old login page - kept for reference."""
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
    from app.auth_ui import render_logout_button, render_user_profile_sidebar
    render_user_profile_sidebar()
    render_logout_button()

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
