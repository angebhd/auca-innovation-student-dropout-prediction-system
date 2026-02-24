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
        .main .block-container {
            padding-top: 1rem;
            max-width: 1200px;
        }
        
        /* Buttons */
        .stButton>button {
            border-radius: 6px;
            height: 2.8em;
            background-color: #3b82f6;
            color: white;
            border: none;
            font-weight: 500;
            transition: all 0.2s ease;
        }
        .stButton>button:hover {
            background-color: #2563eb;
        }
        
        /* Page Title */
        .page-title {
            font-size: 1.75rem;
            font-weight: 600;
            color: #f1f5f9;
            margin-bottom: 0.25rem;
        }
        .page-subtitle {
            font-size: 0.95rem;
            color: #64748b;
            margin-bottom: 1.5rem;
        }
        
        /* Metric Cards */
        .metric-card {
            background: linear-gradient(145deg, #1e293b, #0f172a);
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 1.25rem;
            text-align: center;
        }
        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: #f1f5f9;
        }
        .metric-label {
            font-size: 0.85rem;
            color: #64748b;
            margin-top: 0.25rem;
        }
        .metric-delta-positive { color: #22c55e; }
        .metric-delta-negative { color: #ef4444; }
        .metric-delta-neutral { color: #f59e0b; }
        
        /* Section Headers */
        .section-header {
            font-size: 1.1rem;
            font-weight: 600;
            color: #e2e8f0;
            margin: 1.5rem 0 1rem 0;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid #334155;
        }
        
        /* Insight Box */
        .insight-box {
            background: rgba(59, 130, 246, 0.1);
            border-left: 3px solid #3b82f6;
            padding: 1rem;
            border-radius: 0 8px 8px 0;
            margin: 1rem 0;
        }
        .insight-box p {
            color: #cbd5e1;
            margin: 0;
            font-size: 0.9rem;
            line-height: 1.5;
        }
        
        /* Warning Box */
        .warning-box {
            background: rgba(239, 68, 68, 0.1);
            border-left: 3px solid #ef4444;
            padding: 1rem;
            border-radius: 0 8px 8px 0;
            margin: 1rem 0;
        }
        
        /* Success Box */
        .success-box {
            background: rgba(34, 197, 94, 0.1);
            border-left: 3px solid #22c55e;
            padding: 1rem;
            border-radius: 0 8px 8px 0;
            margin: 1rem 0;
        }
        
        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
            border-right: 1px solid #1e293b;
        }
        [data-testid="stSidebar"] * {
            color: #e2e8f0 !important;
        }
        
        .sidebar-brand {
            padding: 1.25rem 1rem;
            font-size: 1.25rem;
            font-weight: 700;
            color: #3b82f6 !important;
            letter-spacing: -0.5px;
            border-bottom: 1px solid #334155;
            margin-bottom: 1rem;
        }
        
        /* DataFrames */
        .stDataFrame {
            border: 1px solid #334155 !important;
            border-radius: 8px;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            background: transparent;
            border-radius: 6px;
            padding: 0.5rem 1rem;
        }
        
        </style>
    """, unsafe_allow_html=True)
    st.sidebar.markdown('<div class="sidebar-brand">UMOJA</div>', unsafe_allow_html=True)

def render_header():
    """Renders the main headers of the application."""
    st.markdown('<p class="page-title">Data Management Suite</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Generate, upload, clean, and export student datasets</p>', unsafe_allow_html=True)

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
    """Renders the Dashboard/Home page with key metrics and insights."""
    from src.prediction import predictor
    from src.analytics import analytics
    
    st.markdown('<p class="page-title">Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Student retention analytics and risk monitoring</p>', unsafe_allow_html=True)
    
    # Check for available data
    ensure_cleaned_dir()
    cleaned_files = [f for f in os.listdir(CLEANED_DATA_PATH) if f.endswith(".csv")]
    
    if not cleaned_files:
        st.info("Welcome to the Student Success Prediction Platform. To get started:")
        st.markdown("""
        1. Navigate to **Data Suite** to upload or generate student data
        2. Clean and process the dataset
        3. Go to **Predictions** to train the model
        4. Return here to view the dashboard
        """)
        return
    
    # Load most recent data
    latest_file = cleaned_files[-1]
    df = pd.read_csv(os.path.join(CLEANED_DATA_PATH, latest_file))
    
    # Run predictions if model is trained
    if predictor.is_trained and 'risk_category' not in df.columns:
        df = predictor.predict(df)
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    total_students = len(df)
    avg_gpa = df['current_gpa'].mean() if 'current_gpa' in df.columns else 0
    avg_attendance = df['attendance_rate'].mean() if 'attendance_rate' in df.columns else 0
    high_risk_count = len(df[df['risk_category'] == 'High']) if 'risk_category' in df.columns else 0
    
    col1.metric("Total Students", f"{total_students:,}")
    col2.metric("Average GPA", f"{avg_gpa:.2f}")
    col3.metric("Attendance Rate", f"{avg_attendance:.1f}%")
    col4.metric("High Risk", high_risk_count, delta=f"{high_risk_count/total_students*100:.1f}%" if total_students > 0 else "0%", delta_color="inverse")
    
    st.markdown('<p class="section-header">Risk Overview</p>', unsafe_allow_html=True)
    
    if 'risk_category' in df.columns:
        risk_col1, risk_col2 = st.columns([1.5, 2])
        
        with risk_col1:
            # Risk breakdown
            risk_counts = df['risk_category'].value_counts()
            
            for risk_level in ['High', 'Medium', 'Low']:
                count = risk_counts.get(risk_level, 0)
                pct = count / total_students * 100 if total_students > 0 else 0
                color = {'High': '#ef4444', 'Medium': '#f59e0b', 'Low': '#22c55e'}[risk_level]
                
                st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.75rem 0; border-bottom: 1px solid #334155;">
                        <span style="color: {color}; font-weight: 500;">{risk_level} Risk</span>
                        <span style="color: #e2e8f0; font-weight: 600;">{count} <small style="color: #64748b;">({pct:.1f}%)</small></span>
                    </div>
                """, unsafe_allow_html=True)
            
            # Interpretation
            high_pct = risk_counts.get('High', 0) / total_students * 100 if total_students > 0 else 0
            if high_pct > 20:
                st.markdown("""
                    <div class="warning-box">
                        <p><strong>Alert:</strong> High-risk students exceed 20% of the population. 
                        Immediate intervention programs are recommended to prevent dropout.</p>
                    </div>
                """, unsafe_allow_html=True)
            elif high_pct > 10:
                st.markdown("""
                    <div class="insight-box">
                        <p><strong>Note:</strong> High-risk population is moderate. 
                        Consider targeted support for students with GPA below 12 or attendance below 70%.</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class="success-box">
                        <p><strong>Good:</strong> Risk levels are within acceptable range. 
                        Continue monitoring and maintain current support programs.</p>
                    </div>
                """, unsafe_allow_html=True)
        
        with risk_col2:
            risk_chart = analytics.create_risk_distribution_chart(df)
            if risk_chart:
                risk_chart.update_layout(height=300, margin=dict(t=30, b=30, l=30, r=30))
                st.plotly_chart(risk_chart, use_container_width=True)
    else:
        st.warning("Train a prediction model to view risk analysis.")
    
    # Performance Insights
    st.markdown('<p class="section-header">Academic Performance</p>', unsafe_allow_html=True)
    
    perf_col1, perf_col2 = st.columns(2)
    
    with perf_col1:
        gpa_chart = analytics.create_gpa_distribution_chart(df)
        if gpa_chart:
            gpa_chart.update_layout(height=280, margin=dict(t=40, b=40, l=40, r=40))
            st.plotly_chart(gpa_chart, use_container_width=True)
        
        # GPA Interpretation
        if 'current_gpa' in df.columns:
            below_threshold = len(df[df['current_gpa'] < 12])
            st.markdown(f"""
                <div class="insight-box">
                    <p><strong>Interpretation:</strong> {below_threshold} students ({below_threshold/total_students*100:.1f}%) 
                    have GPA below 12.0, indicating academic difficulty. 
                    These students may benefit from tutoring or academic counseling.</p>
                </div>
            """, unsafe_allow_html=True)
    
    with perf_col2:
        attendance_chart = analytics.create_attendance_distribution_chart(df)
        if attendance_chart:
            attendance_chart.update_layout(height=280, margin=dict(t=40, b=40, l=40, r=40))
            st.plotly_chart(attendance_chart, use_container_width=True)
        
        # Attendance Interpretation
        if 'attendance_rate' in df.columns:
            low_attendance = len(df[df['attendance_rate'] < 70])
            st.markdown(f"""
                <div class="insight-box">
                    <p><strong>Interpretation:</strong> {low_attendance} students ({low_attendance/total_students*100:.1f}%) 
                    have attendance below 70%. Low attendance is strongly correlated with dropout risk. 
                    Consider attendance monitoring interventions.</p>
                </div>
            """, unsafe_allow_html=True)
    
    # Quick Actions
    st.markdown('<p class="section-header">Quick Actions</p>', unsafe_allow_html=True)
    
    action_col1, action_col2, action_col3 = st.columns(3)
    
    with action_col1:
        if st.button("View Full Analytics", use_container_width=True):
            st.switch_page("pages/analytics")
    
    with action_col2:
        if st.button("Run New Predictions", use_container_width=True):
            st.switch_page("pages/predictions")
    
    with action_col3:
        if st.button("Process New Data", use_container_width=True):
            st.switch_page("pages/data_suite")

def render_analytics_page():
    """Render the Analytics page with interactive charts and interpretations."""
    from src.analytics import analytics
    from src.prediction import predictor
    
    st.markdown('<p class="page-title">Exploratory Data Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Interactive visualization and interpretation of student performance metrics</p>', unsafe_allow_html=True)
    
    # Data source selection
    ensure_cleaned_dir()
    cleaned_files = [f for f in os.listdir(CLEANED_DATA_PATH) if f.endswith(".csv")]
    
    if not cleaned_files:
        st.warning("No cleaned datasets found. Please process a dataset in Data Suite first.")
        st.info("Navigate to **Data Suite** to clean a dataset, then return here for analytics.")
        return
    
    selected_file = st.selectbox("Select Dataset for Analysis", cleaned_files, key="analytics_file")
    df = pd.read_csv(os.path.join(CLEANED_DATA_PATH, selected_file))
    
    # Check if predictions exist, if not run them
    if 'risk_category' not in df.columns and predictor.is_trained:
        with st.spinner("Running risk predictions..."):
            df = predictor.predict(df)
    
    total_students = len(df)
    
    # Overview metrics
    st.markdown('<p class="section-header">Overview Metrics</p>', unsafe_allow_html=True)
    stats = analytics.get_overview_stats(df)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Students", stats['total_students'])
    col2.metric("Average GPA", stats.get('avg_gpa', 'N/A'))
    col3.metric("Avg Attendance", f"{stats.get('avg_attendance', 0):.1f}%")
    col4.metric("Students with Failures", stats.get('students_with_failures', 0))
    
    st.divider()
    
    # ==================== GPA DISTRIBUTION ====================
    st.markdown('<p class="section-header">GPA Distribution Analysis</p>', unsafe_allow_html=True)
    
    chart_col1, chart_col2 = st.columns([1.5, 1])
    
    with chart_col1:
        gpa_chart = analytics.create_gpa_distribution_chart(df)
        if gpa_chart:
            gpa_chart.update_layout(height=320, margin=dict(t=40, b=40, l=40, r=40))
            st.plotly_chart(gpa_chart, use_container_width=True)
    
    with chart_col2:
        if 'current_gpa' in df.columns:
            avg_gpa = df['current_gpa'].mean()
            median_gpa = df['current_gpa'].median()
            below_12 = len(df[df['current_gpa'] < 12])
            above_15 = len(df[df['current_gpa'] >= 15])
            
            st.markdown("**How to Read This Chart**")
            st.caption("This histogram shows the distribution of Grade Point Averages across all students. The x-axis represents GPA (0-20 scale), while the y-axis shows the count of students.")
            
            st.markdown("**Key Findings**")
            findings_df = pd.DataFrame({
                'Metric': ['Average GPA', 'Median GPA', 'Below 12.0 (at risk)', 'Above 15.0 (high performing)'],
                'Value': [f"{avg_gpa:.2f}", f"{median_gpa:.2f}", f"{below_12} ({below_12/total_students*100:.1f}%)", f"{above_15} ({above_15/total_students*100:.1f}%)"]
            })
            st.dataframe(findings_df, hide_index=True, use_container_width=True)
            
            if below_12 > 0:
                st.info(f"**Recommendation:** Focus attention on the {below_12} students below 12.0 GPA who may need academic support.")
    
    st.divider()
    
    # ==================== ATTENDANCE DISTRIBUTION ====================
    st.markdown('<p class="section-header">Attendance Rate Analysis</p>', unsafe_allow_html=True)
    
    att_col1, att_col2 = st.columns([1.5, 1])
    
    with att_col1:
        attendance_chart = analytics.create_attendance_distribution_chart(df)
        if attendance_chart:
            attendance_chart.update_layout(height=320, margin=dict(t=40, b=40, l=40, r=40))
            st.plotly_chart(attendance_chart, use_container_width=True)
    
    with att_col2:
        if 'attendance_rate' in df.columns:
            avg_att = df['attendance_rate'].mean()
            below_70 = len(df[df['attendance_rate'] < 70])
            below_50 = len(df[df['attendance_rate'] < 50])
            
            st.markdown("**How to Read This Chart**")
            st.caption("This distribution shows class attendance percentages. Research shows attendance below 70% correlates strongly with dropout risk.")
            
            st.markdown("**Key Findings**")
            att_df = pd.DataFrame({
                'Metric': ['Average Attendance', 'Below 70% (warning)', 'Below 50% (critical)'],
                'Value': [f"{avg_att:.1f}%", f"{below_70} ({below_70/total_students*100:.1f}%)", f"{below_50} ({below_50/total_students*100:.1f}%)"]
            })
            st.dataframe(att_df, hide_index=True, use_container_width=True)
            
            if below_50 > 0:
                st.error(f"**Urgent:** {below_50} students with attendance below 50% require immediate outreach.")
            elif below_70 > 0:
                st.warning(f"**Monitor:** {below_70} students below 70% attendance need attention.")
    
    st.divider()
    
    # ==================== SEMESTER TRENDS & FAILURES ====================
    st.markdown('<p class="section-header">Academic Trends</p>', unsafe_allow_html=True)
    
    trend_col1, trend_col2 = st.columns(2)
    
    with trend_col1:
        semester_chart = analytics.create_semester_gpa_trend(df)
        if semester_chart:
            semester_chart.update_layout(height=300, margin=dict(t=40, b=40, l=40, r=40))
            st.plotly_chart(semester_chart, use_container_width=True)
        
        st.caption("**Interpretation:** The box plot shows GPA variation across semesters. Wide boxes indicate high variance (inconsistent performance). Declining medians suggest systemic issues requiring curriculum review.")
    
    with trend_col2:
        failed_chart = analytics.create_failed_courses_chart(df)
        if failed_chart:
            failed_chart.update_layout(height=300, margin=dict(t=40, b=40, l=40, r=40))
            st.plotly_chart(failed_chart, use_container_width=True)
        
        if 'failed_courses' in df.columns:
            no_failures = len(df[df['failed_courses'] == 0])
            multiple_failures = len(df[df['failed_courses'] >= 3])
            st.caption(f"**Interpretation:** {no_failures} students ({no_failures/total_students*100:.1f}%) have no failures. However, {multiple_failures} students have 3+ failures and are at high dropout risk.")
    
    st.divider()
    
    # ==================== RISK ANALYSIS ====================
    if 'risk_category' in df.columns:
        st.markdown('<p class="section-header">Risk Analysis</p>', unsafe_allow_html=True)
        
        risk_col1, risk_col2 = st.columns(2)
        
        with risk_col1:
            risk_pie = analytics.create_risk_distribution_chart(df)
            if risk_pie:
                risk_pie.update_layout(height=350, margin=dict(t=40, b=40, l=40, r=40))
                st.plotly_chart(risk_pie, use_container_width=True)
            
            risk_counts = df['risk_category'].value_counts()
            high_risk = risk_counts.get('High', 0)
            medium_risk = risk_counts.get('Medium', 0)
            low_risk = risk_counts.get('Low', 0)
            
            st.markdown("**Risk Distribution Summary**")
            risk_df = pd.DataFrame({
                'Category': ['High Risk', 'Medium Risk', 'Low Risk'],
                'Count': [high_risk, medium_risk, low_risk],
                'Percentage': [f"{high_risk/total_students*100:.1f}%", f"{medium_risk/total_students*100:.1f}%", f"{low_risk/total_students*100:.1f}%"]
            })
            st.dataframe(risk_df, hide_index=True, use_container_width=True)
            
            if high_risk / total_students > 0.2:
                st.error(f"**Critical Alert:** {high_risk} students ({high_risk/total_students*100:.1f}%) are classified as high risk. This exceeds the 20% threshold. Immediate intervention programs are strongly recommended.")
        
        with risk_col2:
            scatter = analytics.create_gpa_vs_attendance_scatter(df)
            if scatter:
                scatter.update_layout(height=350, margin=dict(t=40, b=40, l=40, r=40))
                st.plotly_chart(scatter, use_container_width=True)
            
            st.markdown("**How to Read This Chart**")
            st.caption("Each dot represents a student. Position shows their GPA (y-axis) and attendance (x-axis). Color indicates risk level.")
            st.markdown("**Pattern to Watch**")
            st.caption("High-risk students (red) typically cluster in the lower-left quadrant (low GPA + low attendance). Students in upper-right are generally safe.")
        
        st.divider()
    
    # ==================== ENGAGEMENT ANALYSIS ====================
    st.markdown('<p class="section-header">Engagement Analysis</p>', unsafe_allow_html=True)
    
    eng_col1, eng_col2 = st.columns(2)
    
    with eng_col1:
        radar = analytics.create_engagement_radar(df)
        if radar:
            radar.update_layout(height=380, margin=dict(t=50, b=50, l=50, r=50))
            st.plotly_chart(radar, use_container_width=True)
        
        st.caption("**Interpretation:** The radar chart compares average engagement metrics. A balanced shape indicates healthy engagement. Collapsed areas highlight metrics needing improvement.")
    
    with eng_col2:
        corr = analytics.create_correlation_heatmap(df)
        if corr:
            corr.update_layout(height=380, margin=dict(t=50, b=50, l=50, r=50))
            st.plotly_chart(corr, use_container_width=True)
        
        st.caption("**Interpretation:** The correlation matrix shows relationships between variables. Dark red indicates strong positive correlation; dark blue indicates strong negative correlation. Use this to identify which factors most influence each other.")
    
    st.divider()
    
    # ==================== DATA PREVIEW ====================
    st.markdown('<p class="section-header">Data Preview</p>', unsafe_allow_html=True)
    with st.expander("View Raw Data"):
        st.dataframe(df.head(50), use_container_width=True)

def render_prediction_page():
    """Render the AI Prediction page with model training and predictions."""
    from src.prediction import predictor
    from src.analytics import analytics
    
    st.markdown('<p class="page-title">Risk Prediction Engine</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">ML-powered dropout risk assessment and early warning system</p>', unsafe_allow_html=True)
    
    # Tabs for different functions
    tab1, tab2, tab3 = st.tabs(["Train Model", "Batch Prediction", "Individual Assessment"])
    
    # ==================== TAB 1: TRAIN MODEL ====================
    with tab1:
        st.markdown('<p class="section-header">Train Prediction Model</p>', unsafe_allow_html=True)
        
        st.markdown("""
            <div class="insight-box">
                <p>Train a Random Forest classifier on cleaned student data to predict dropout risk. 
                The model uses 16 features including GPA, attendance, and engagement metrics.</p>
            </div>
        """, unsafe_allow_html=True)
        
        ensure_cleaned_dir()
        cleaned_files = [f for f in os.listdir(CLEANED_DATA_PATH) if f.endswith(".csv")]
        
        if not cleaned_files:
            st.warning("No cleaned datasets available. Please clean a dataset in Data Suite first.")
        else:
            training_file = st.selectbox("Select Training Dataset", cleaned_files, key="train_file")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                if st.button("Train Model", type="primary", use_container_width=True):
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
                    st.markdown("""
                        <div class="success-box" style="padding: 0.5rem 1rem; margin: 0;">
                            <p style="margin: 0;"><strong>Model Ready</strong></p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                        <div class="warning-box" style="padding: 0.5rem 1rem; margin: 0;">
                            <p style="margin: 0;"><strong>No Model</strong></p>
                        </div>
                    """, unsafe_allow_html=True)
            
            # Show training metrics
            if 'training_metrics' in st.session_state:
                metrics = st.session_state['training_metrics']
                st.markdown('<p class="section-header">Training Results</p>', unsafe_allow_html=True)
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
                m2.metric("Precision", f"{metrics['precision']*100:.1f}%")
                m3.metric("Recall", f"{metrics['recall']*100:.1f}%")
                m4.metric("F1 Score", f"{metrics['f1_score']*100:.1f}%")
                
                st.markdown("""
                    <div class="insight-box">
                        <p><strong>Metric Interpretation:</strong></p>
                        <ul style="color: #94a3b8; margin-left: 1rem;">
                            <li><strong>Accuracy:</strong> Overall correct predictions</li>
                            <li><strong>Precision:</strong> When predicting "at risk", how often is it correct</li>
                            <li><strong>Recall:</strong> Of actual at-risk students, how many were identified</li>
                            <li><strong>F1 Score:</strong> Balance between precision and recall</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)
                
                # Feature importance
                importance_df = predictor.get_feature_importance()
                if importance_df is not None:
                    st.markdown('<p class="section-header">Feature Importance</p>', unsafe_allow_html=True)
                    fig = analytics.create_feature_importance_chart(importance_df)
                    if fig:
                        fig.update_layout(height=400, margin=dict(t=40, b=40, l=40, r=40))
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("""
                        <div class="insight-box">
                            <p><strong>Interpretation:</strong> Features at the top have the strongest influence on 
                            dropout predictions. Focus intervention resources on improving these factors 
                            for maximum impact on student retention.</p>
                        </div>
                    """, unsafe_allow_html=True)
    
    # ==================== TAB 2: BATCH PREDICTION ====================
    with tab2:
        st.markdown('<p class="section-header">Batch Prediction</p>', unsafe_allow_html=True)
        
        st.markdown("""
            <div class="insight-box">
                <p>Run predictions on an entire dataset to identify and prioritize at-risk students 
                for intervention programs.</p>
            </div>
        """, unsafe_allow_html=True)
        
        if not predictor.is_trained:
            st.markdown("""
                <div class="warning-box">
                    <p>Please train a model first in the <strong>Train Model</strong> tab before running predictions.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            cleaned_files = [f for f in os.listdir(CLEANED_DATA_PATH) if f.endswith(".csv")]
            
            if cleaned_files:
                pred_file = st.selectbox("Select Dataset for Prediction", cleaned_files, key="pred_file")
                
                if st.button("Run Predictions", type="primary"):
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
                    
                    st.markdown('<p class="section-header">Risk Summary</p>', unsafe_allow_html=True)
                    s1, s2, s3, s4 = st.columns(4)
                    s1.metric("Total Students", summary.get('total_students', 0))
                    s2.metric("High Risk", summary.get('high_risk', 0), 
                             f"{summary.get('high_risk_pct', 0):.1f}%")
                    s3.metric("Medium Risk", summary.get('medium_risk', 0),
                             f"{summary.get('medium_risk_pct', 0):.1f}%")
                    s4.metric("Low Risk", summary.get('low_risk', 0),
                             f"{summary.get('low_risk_pct', 0):.1f}%")
                    
                    # Risk distribution chart
                    chart_col1, chart_col2 = st.columns([1.5, 1])
                    
                    with chart_col1:
                        risk_chart = analytics.create_risk_distribution_chart(results)
                        if risk_chart:
                            risk_chart.update_layout(height=320, margin=dict(t=40, b=40, l=40, r=40))
                            st.plotly_chart(risk_chart, use_container_width=True)
                    
                    with chart_col2:
                        high_pct = summary.get('high_risk_pct', 0)
                        if high_pct > 20:
                            st.markdown(f"""
                                <div class="warning-box">
                                    <p><strong>Critical:</strong> {high_pct:.1f}% of students are high-risk. 
                                    This exceeds the 20% threshold. Prioritize intervention resources immediately.</p>
                                </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                                <div class="success-box">
                                    <p><strong>Status:</strong> High-risk population at {high_pct:.1f}% is within 
                                    manageable levels. Continue monitoring and targeted interventions.</p>
                                </div>
                            """, unsafe_allow_html=True)
                    
                    # Top at-risk students
                    st.markdown('<p class="section-header">Students Requiring Attention</p>', unsafe_allow_html=True)
                    top_risk = analytics.get_top_at_risk_students(results, n=10)
                    if not top_risk.empty:
                        st.dataframe(top_risk, use_container_width=True)
                    
                    # Recommendations
                    st.markdown('<p class="section-header">Intervention Recommendations</p>', unsafe_allow_html=True)
                    recommendations = analytics.get_intervention_recommendations(summary)
                    for rec in recommendations:
                        st.markdown(f"- {rec}")
                    
                    # Download results
                    st.divider()
                    csv = results.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download Predictions CSV",
                        csv,
                        f"predictions_{pred_file}",
                        "text/csv",
                        use_container_width=True
                    )
    
    # ==================== TAB 3: INDIVIDUAL PREDICTION ====================
    with tab3:
        st.markdown('<p class="section-header">Individual Student Assessment</p>', unsafe_allow_html=True)
        
        st.markdown("""
            <div class="insight-box">
                <p>Enter student metrics below to get an instant risk prediction. 
                The model will assess dropout probability based on academic and engagement indicators.</p>
            </div>
        """, unsafe_allow_html=True)
        
        if not predictor.is_trained:
            st.markdown("""
                <div class="warning-box">
                    <p>Please train a model first in the <strong>Train Model</strong> tab before running predictions.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            with st.form("individual_prediction"):
                st.markdown("**Student Information**")
                
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
                
                submitted = st.form_submit_button("Predict Risk", use_container_width=True, type="primary")
                
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
                    
                    st.divider()
                    st.markdown('<p class="section-header">Prediction Result</p>', unsafe_allow_html=True)
                    
                    r1, r2, r3 = st.columns(3)
                    r1.metric("Risk Category", result['risk_category'])
                    r2.metric("Risk Score", f"{result['risk_score']:.1f}%")
                    r3.metric("Dropout Probability", f"{result['dropout_probability']*100:.1f}%")
                    
                    # Show interpretation based on risk level
                    if result['risk_category'] == 'High':
                        st.markdown(f"""
                            <div class="warning-box">
                                <p><strong>HIGH RISK Assessment</strong></p>
                                <p>This student shows significant indicators of dropout risk with a 
                                {result['dropout_probability']*100:.1f}% probability. Immediate intervention is recommended.</p>
                            </div>
                        """, unsafe_allow_html=True)
                        st.markdown("""
                        **Recommended Actions:**
                        - Schedule immediate one-on-one counseling session
                        - Review academic support resources availability  
                        - Assess financial situation and scholarship eligibility
                        - Connect with peer mentoring program
                        """)
                    elif result['risk_category'] == 'Medium':
                        st.markdown(f"""
                            <div class="insight-box">
                                <p><strong>MEDIUM RISK Assessment</strong></p>
                                <p>This student shows some concerning indicators with a 
                                {result['dropout_probability']*100:.1f}% dropout probability. Close monitoring advised.</p>
                            </div>
                        """, unsafe_allow_html=True)
                        st.markdown("""
                        **Recommended Actions:**
                        - Schedule follow-up meeting within 2 weeks
                        - Review course load and academic progress
                        - Offer study skills workshop enrollment
                        - Monitor attendance patterns
                        """)
                    else:
                        st.markdown(f"""
                            <div class="success-box">
                                <p><strong>LOW RISK Assessment</strong></p>
                                <p>This student shows healthy academic indicators with only a 
                                {result['dropout_probability']*100:.1f}% dropout probability. Continue standard monitoring.</p>
                            </div>
                        """, unsafe_allow_html=True)
                        st.markdown("""
                        **Recommended Actions:**
                        - Maintain regular check-ins
                        - Encourage continued engagement
                        - Consider for peer tutoring opportunities
                        """)

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
