from pyexpat import model

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

# def render_prediction_page():

#     import joblib
#     import pandas as pd
#     import numpy as np
#     st.markdown("##  AI Dropout Risk Prediction")
#     st.markdown("Evaluate student dropout risk using the trained machine learning model.")

    
#     try:
#         model = joblib.load("models/risk_model.pkl")
#         feature_cols = joblib.load("models/feature_columns.pkl")
#     except Exception:
#         st.error(" Model not found. Please train the model first.")
#         return

#     option = st.radio("Choose Prediction Mode:", ["Upload CSV", "Single Student Input"])

#     if option == "Upload CSV":

#         uploaded_file = st.file_uploader("Upload Cleaned CSV File", type="csv")

#         if uploaded_file:
#             df = pd.read_csv(uploaded_file)
#             st.write("Preview of Uploaded Data:", df.head())

#             drop_cols = ["student_id", "actual_dropout", "dropout_risk"]
#             df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

#             df_encoded = pd.get_dummies(df)
#             df_encoded = df_encoded.reindex(columns=feature_cols, fill_value=0)

#             if st.button("Run Prediction"):

#                 predictions = model.predict(df_encoded)
#                 probabilities = model.predict_proba(df_encoded)

#                 df["Predicted_Risk"] = predictions
#                 df["Confidence_%"] = np.round(np.max(probabilities, axis=1) * 100, 2)

#                 st.success(" Prediction completed successfully.")
#                 st.dataframe(df.head())

#                 csv = df.to_csv(index=False).encode("utf-8")
#                 st.download_button(
#                     label="Download Predictions",
#                     data=csv,
#                     file_name="predicted_dropout_risk.csv",
#                     mime="text/csv",
#                 )

   
#     else:

#         st.subheader("Enter Student Profile")

#         col1, col2 = st.columns(2)

#         with col1:
#             age = st.number_input("Age", 16, 60, 20)
#             admission_grade = st.number_input("Admission Grade", 0.0, 20.0, 12.0)
#             semester_1_gpa = st.number_input("Semester 1 GPA", 0.0, 20.0, 12.0)
#             semester_2_gpa = st.number_input("Semester 2 GPA", 0.0, 20.0, 12.0)
#             semester_3_gpa = st.number_input("Semester 3 GPA", 0.0, 20.0, 12.0)

#         with col2:
#             current_gpa = st.number_input("Current GPA", 0.0, 20.0, 12.0)
#             attendance_rate = st.slider("Attendance Rate (%)", 0, 100, 75)
#             failed_courses = st.number_input("Failed Courses", 0, 20, 0)
#             absences_count = st.number_input("Absences Count", 0, 100, 5)
#             late_submissions = st.number_input("Late Submissions", 0, 50, 2)

#         if st.button("Predict Student Risk"):

#             input_dict = {
#                 "age": age,
#                 "admission_grade": admission_grade,
#                 "semester_1_gpa": semester_1_gpa,
#                 "semester_2_gpa": semester_2_gpa,
#                 "semester_3_gpa": semester_3_gpa,
#                 "current_gpa": current_gpa,
#                 "attendance_rate": attendance_rate,
#                 "failed_courses": failed_courses,
#                 "absences_count": absences_count,
#                 "late_submissions": late_submissions,
#             }

#             input_df = pd.DataFrame([input_dict])
#             input_df = pd.get_dummies(input_df)
#             input_df = input_df.reindex(columns=feature_cols, fill_value=0)

#             prediction = model.predict(input_df)[0]
#             probabilities = model.predict_proba(input_df)[0]
#             confidence = round(max(probabilities) * 100, 2)

          
#             if prediction == "High":
#                 st.error(f" High Dropout Risk ({confidence}% confidence)")
#             elif prediction == "Medium":
#                 st.warning(f" Medium Dropout Risk ({confidence}% confidence)")
#             else:
#                 st.success(f" Low Dropout Risk ({confidence}% confidence)")

           
#             if prediction == "High":
#                 st.info("Immediate academic intervention recommended.")
#             elif prediction == "Medium":
#                 st.info("Monitor academic performance closely.")
#             else:
#                 st.info("Student currently stable. Continue standard monitoring.")

           
#             st.markdown("###  Prediction Probability Breakdown")

#             prob_df = pd.DataFrame({
#                 "Risk Level": model.classes_,
#                 "Probability (%)": np.round(probabilities * 100, 2)
#             })

#             st.bar_chart(prob_df.set_index("Risk Level"))

            
#             st.markdown("###  Top Influential Factors")

#             importances = model.feature_importances_

#             feature_importance_df = pd.DataFrame({
#                 "Feature": feature_cols,
#                 "Importance": importances
#             }).sort_values(by="Importance", ascending=False).head(8)

#             st.dataframe(feature_importance_df)


def render_prediction_page():
    import joblib
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    render_header()

   
    st.markdown("""
        <div style="margin-bottom: 1.5rem;">
            <h2 style="margin:0; color:#1e3a8a; font-size:1.6rem;"> Student Risk Assessment</h2>
            <p style="color:#64748b; margin-top:0.3rem;">
                Enter a student's information below to instantly assess their dropout risk
                and receive personalised support recommendations.
            </p>
        </div>
    """, unsafe_allow_html=True)

   
    try:
        model        = joblib.load("models/risk_model.pkl")
        feature_cols = joblib.load("models/feature_columns.pkl")
        st.success(" Prediction  is ready.")
    except Exception:
        st.error(
            " Sorry for incovenience ,the system is having issues."
        )
        return

    mode = st.radio(
        "How would you like to assess students?",
        [" Assess One Student", " Assess Multiple Students (Upload File)"],
        horizontal=True
    )

    st.divider()

  
    if mode == " Assess One Student":

        st.markdown("### Student Information")
        st.caption("Fill in the fields below. All fields are required for an accurate assessment.")

        
        st.markdown("####  Academic Performance")
        a1, a2, a3 = st.columns(3)
        admission_grade = a1.number_input("Admission Score",       0.0, 20.0, 12.0, step=0.1,
                                           help="Score the student received when they joined (0–20 scale)")
        current_gpa     = a2.number_input("Current Overall Grade", 0.0, 20.0, 12.0, step=0.1,
                                           help="The student's average grade right now (0–20 scale)")
        failed_courses  = a3.number_input("Courses Failed",        0,   20,   0,
                                           help="How many courses has this student failed so far?")

        a4, a5, a6 = st.columns(3)
        semester_1_gpa  = a4.number_input("Semester 1 Grade", 0.0, 20.0, 12.0, step=0.1)
        semester_2_gpa  = a5.number_input("Semester 2 Grade", 0.0, 20.0, 12.0, step=0.1)
        semester_3_gpa  = a6.number_input("Semester 3 Grade", 0.0, 20.0, 12.0, step=0.1)

        st.markdown("####  Attendance")
        b1, b2, b3 = st.columns(3)
        attendance_rate  = b1.slider("Class Attendance (%)", 0, 100, 75,
                                      help="What percentage of classes does this student attend?")
        absences_count   = b2.number_input("Total Absences", 0, 100, 5,
                                            help="Total number of classes missed this semester")
        late_submissions = b3.number_input("Late Assignments", 0, 50, 2,
                                            help="How many assignments were submitted late?")

        st.markdown("####  Active Participation")
        c1, c2, c3 = st.columns(3)
        online_portal_logins = c1.number_input("Online Platform Logins", 0, 500, 30,
                                                help="How often does the student log into the learning portal?")
        library_visits       = c2.number_input("Library Visits",          0, 200, 10,
                                                help="Number of times the student has visited the library")
        participation_score  = c3.number_input("Class Participation Score", 0.0, 10.0, 6.0, step=0.1,
                                                help="Lecturer's score for how actively this student participates (0–10)")

       
        st.markdown("####  Student Background")
        d1, d2, d3 = st.columns(3)
        age                  = d1.number_input("Age", 16, 60, 20)
        distance_from_campus = d2.slider("Distance from Campus (km)", 0, 100, 15)
        scholarship_status   = d3.selectbox("Has a Scholarship?", ["No", "Yes"])

        st.divider()

      
        if st.button(" Assess This Student", type="primary", use_container_width=True):

            input_dict = {
                "age":                  age,
                "admission_grade":      admission_grade,
                "semester_1_gpa":       semester_1_gpa,
                "semester_2_gpa":       semester_2_gpa,
                "semester_3_gpa":       semester_3_gpa,
                "current_gpa":          current_gpa,
                "attendance_rate":      attendance_rate,
                "failed_courses":       failed_courses,
                "absences_count":       absences_count,
                "late_submissions":     late_submissions,
                "online_portal_logins": online_portal_logins,
                "library_visits":       library_visits,
                "participation_score":  participation_score,
                "distance_from_campus": distance_from_campus,
            }

            input_df = pd.DataFrame([input_dict])
            input_df = pd.get_dummies(input_df)
            input_df = input_df.reindex(columns=feature_cols, fill_value=0)

            prediction    = model.predict(input_df)[0]
            probabilities = model.predict_proba(input_df)[0]
            classes       = list(model.classes_)
            prob_dict     = {cls: round(prob * 100, 1) for cls, prob in zip(classes, probabilities)}
            confidence    = round(max(probabilities) * 100, 1)

            risk_config = {
                "High":   {"color": "#e74c3c", "bg": "#fef2f2", "emoji": "🚨",
                            "label": "HIGH RISK",
                            "tagline": "This student needs immediate support."},
                "Medium": {"color": "#f39c12", "bg": "#fffbeb", "emoji": "⚠️",
                            "label": "MEDIUM RISK",
                            "tagline": "This student should be monitored closely."},
                "Low":    {"color": "#2ecc71", "bg": "#f0fdf4", "emoji": "✅",
                            "label": "LOW RISK",
                            "tagline": "This student is on a good track."},
            }
            cfg = risk_config.get(prediction, risk_config["Medium"])

            st.markdown(f"""
                <div style="
                    background:{cfg['bg']}; border:2px solid {cfg['color']};
                    border-radius:16px; padding:2rem; text-align:center; margin:1.5rem 0;
                ">
                    <div style="font-size:3rem;">{cfg['emoji']}</div>
                    <div style="font-size:2.2rem; font-weight:900; color:{cfg['color']}; margin:0.3rem 0;">
                        {cfg['label']}
                    </div>
                    <div style="font-size:1rem; color:#475569;">
                        {cfg['tagline']} &nbsp;|&nbsp; Model confidence: <strong>{confidence}%</strong>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("####  Risk Breakdown")
            st.caption("How likely is this student to fall into each category?")
            bar_colors = {"High": "#e74c3c", "Medium": "#f39c12", "Low": "#2ecc71"}
            for level in ["Low", "Medium", "High"]:
                if level in prob_dict:
                    pct = prob_dict[level]
                    color = bar_colors[level]
                    st.markdown(f"""
                        <div style="margin-bottom:0.6rem;">
                            <div style="display:flex; justify-content:space-between; margin-bottom:3px;">
                                <span style="font-weight:600; color:#1e293b;">{level} Risk</span>
                                <span style="font-weight:700; color:{color};">{pct}%</span>
                            </div>
                            <div style="background:#e2e8f0; border-radius:99px; height:12px;">
                                <div style="
                                    width:{pct}%; background:{color}; border-radius:99px;
                                    height:12px; transition: width 0.5s ease;
                                "></div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

            st.divider()

            st.markdown("####  Personalised Recommendations")
            st.caption("These suggestions are based on this specific student's data.")

            recommendations = []

          
            if attendance_rate < 60:
                recommendations.append({
                     "icon": "",
                    "area": "Attendance",
                    "issue": f"Very low class attendance ({attendance_rate}%)",
                    "action": "Schedule an urgent meeting with this student. Investigate barriers to attending — transport issues, work commitments, or personal challenges. A minimum of 80% attendance is strongly advised."
                })
            elif attendance_rate < 75:
                recommendations.append({
                     "icon": "",
                    "area": "Attendance",
                    "issue": f"Below-average attendance ({attendance_rate}%)",
                    "action": "Send a friendly reminder about the importance of attending classes regularly. Consistent attendance is one of the strongest predictors of academic success."
                })

           
            if current_gpa < 10:
                recommendations.append({
                     "icon": "📉",
                    "area": "Academic Performance",
                    "issue": f"Current grade is critically low ({current_gpa}/20)",
                    "action": "Connect this student with a tutor or academic support programme immediately. Consider reviewing their course load — they may be overwhelmed."
                })
            elif current_gpa < 12:
                recommendations.append({
                     "icon": "",
                    "area": "Academic Performance",
                    "issue": f"Current grade is below the passing threshold ({current_gpa}/20)",
                    "action": "Encourage the student to attend extra revision sessions and visit lecturers during office hours. A study group may also help improve their grades."
                })

          
            if failed_courses >= 3:
                recommendations.append({
                     "icon": "",
                    "area": "Course Failures",
                    "issue": f"Failed {failed_courses} course(s) — a significant concern",
                    "action": "Consider a course load review with the academic advisor. Repeated failures indicate the student may need to retake certain prerequisites before progressing."
                })
            elif failed_courses >= 1:
                recommendations.append({
                     "icon": "",
                    "area": "Course Failures",
                    "issue": f"Failed {failed_courses} course(s) this semester",
                    "action": "Discuss a study plan with this student and identify which topics they are struggling with. Early tutoring support can prevent further failures."
                })

            if late_submissions >= 5:
                recommendations.append({
                     "icon": "",
                    "area": "Assignment Submission",
                    "issue": f"Submitted {late_submissions} assignments late",
                    "action": "Talk to the student about time management. They may benefit from a weekly planner or a mentoring session to help them stay organised and meet deadlines."
                })

           
            if online_portal_logins < 15:
                recommendations.append({
                     "icon": "",
                    "area": "Online Engagement",
                    "issue": "Very little use of the online learning platform",
                    "action": "Ensure the student knows how to access learning materials online. Low portal usage often means they are missing important course content and announcements."
                })

         
            if participation_score < 4:
                recommendations.append({
                     "icon": "",
                    "area": "Class Participation",
                    "issue": f"Very low class participation score ({participation_score}/10)",
                    "action": "Encourage lecturers to actively involve this student in discussions. Low participation can indicate disengagement, lack of confidence, or language barriers."
                })

           
            gpas = [semester_1_gpa, semester_2_gpa, semester_3_gpa]
            if semester_3_gpa > 0 and semester_2_gpa > 0 and gpas[-1] < gpas[-2] < gpas[-3]:
                recommendations.append({
                     "icon": "",
                    "area": "Grade Trend",
                    "issue": "Grades have been declining each semester",
                    "action": "This downward trend is a warning sign. Meet with the student to understand what changed. Personal, financial, or mental health issues may be contributing."
                })

            # Distance
            if distance_from_campus > 40:
                recommendations.append({
                     "icon": "",
                    "area": "Commute & Logistics",
                    "issue": f"Student lives far from campus ({distance_from_campus} km away)",
                    "action": "Explore whether campus accommodation or transport support is available. Long commutes are a common hidden cause of absenteeism and fatigue."
                })

            
            if scholarship_status == "No" and current_gpa < 12:
                recommendations.append({
                     "icon": "",
                    "area": "Financial Support",
                    "issue": "No scholarship and struggling academically",
                    "action": "Check if this student is eligible for financial aid or a bursary. Financial stress is a major driver of dropout and can affect study focus significantly."
                })

           
            if not recommendations:
                st.success(" This student shows no major warning signs. Encourage them to keep up the great work!")
            else:
                for i, rec in enumerate(recommendations):
                    color = "#fef2f2" if prediction == "High" else "#fffbeb" if prediction == "Medium" else "#f0fdf4"
                    border = "#e74c3c" if prediction == "High" else "#f39c12" if prediction == "Medium" else "#2ecc71"
                    st.markdown(f"""
                        <div style="
                            background:{color}; border-left:4px solid {border};
                            border-radius:8px; padding:1rem 1.2rem; margin-bottom:0.8rem;
                        ">
                            <div style="font-size:1rem; font-weight:700; color:#1e293b; margin-bottom:0.3rem;">
                                {rec['icon']} {rec['area']}
                            </div>
                            <div style="font-size:0.85rem; color:#64748b; margin-bottom:0.4rem;">
                                <em>{rec['issue']}</em>
                            </div>
                            <div style="font-size:0.9rem; color:#1e293b;">
                                <strong>What to do:</strong> {rec['action']}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

            st.divider()
            st.markdown("####  Summary for Academic Advisor")
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Overall Grade",    f"{current_gpa}/20")
            col_b.metric("Attendance",       f"{attendance_rate}%")
            col_c.metric("Courses Failed",   failed_courses)
            col_d, col_e, col_f = st.columns(3)
            col_d.metric("Late Submissions", late_submissions)
            col_e.metric("Platform Logins",  online_portal_logins)
            col_f.metric("Participation",    f"{participation_score}/10")


    else:
        st.markdown("### Upload Student List")
        st.info(
            "Upload a CSV file with student data. The file should include columns like: "
            "`current_gpa`, `attendance_rate`, `failed_courses`, `late_submissions`, etc."
        )

        uploaded_file = st.file_uploader("Choose your CSV file", type="csv")

        if uploaded_file:
            df = pd.read_csv(uploaded_file)

            drop_cols = ["student_id", "actual_dropout", "dropout_risk", "dropout_probability"]
            df_input  = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
            df_input  = df_input.fillna(0)

            st.write(f"**{len(df_input)} students** loaded. Preview:")
            st.dataframe(df_input.head(5), use_container_width=True)

            if st.button(" Run Assessment for All Students", type="primary", use_container_width=True):
                with st.spinner("Analysing students..."):
                    df_enc        = pd.get_dummies(df_input)
                    df_enc        = df_enc.reindex(columns=feature_cols, fill_value=0)
                    predictions   = model.predict(df_enc)
                    probabilities = model.predict_proba(df_enc)

                    result_df = df.copy().fillna("—")   
                    result_df["Risk Level"]    = predictions
                    result_df["Confidence (%)"]= np.round(np.max(probabilities, axis=1) * 100, 1)

             
                st.success(f" Assessment complete for **{len(result_df)}** students.")
                counts = result_df["Risk Level"].value_counts()
                total  = len(result_df)

                m1, m2, m3 = st.columns(3)
                m1.metric(" High Risk Students",
                          f"{counts.get('High', 0)}",
                          delta=f"{counts.get('High',0)/total*100:.1f}% of total",
                          delta_color="inverse")
                m2.metric(" Medium Risk Students",
                          f"{counts.get('Medium', 0)}",
                          delta=f"{counts.get('Medium',0)/total*100:.1f}% of total",
                          delta_color="off")
                m3.metric(" Low Risk Students",
                          f"{counts.get('Low', 0)}",
                          delta=f"{counts.get('Low',0)/total*100:.1f}% of total")

                
                fig, ax = plt.subplots(figsize=(5, 3))
                colors  = {"High": "#e74c3c", "Medium": "#f39c12", "Low": "#2ecc71"}
                for level in ["High", "Medium", "Low"]:
                    ax.bar(level, counts.get(level, 0),
                           color=colors[level], edgecolor="black", linewidth=0.8)
                ax.set_ylabel("Number of Students", fontweight="bold")
                ax.set_title("Students by Risk Level", fontweight="bold")
                ax.grid(True, axis="y", alpha=0.2, linestyle="--")
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)

           
                st.markdown("#### Student Results")
                st.caption("Students are sorted from highest to lowest risk.")

                display_cols = (
                    ["student_id"] if "student_id" in result_df.columns else []
                ) + [
                    c for c in ["current_gpa", "attendance_rate", "failed_courses",
                                "late_submissions", "Risk Level", "Confidence (%)"]
                    if c in result_df.columns
                ]

               
                rename_map = {
                    "current_gpa":      "Current Grade",
                    "attendance_rate":  "Attendance (%)",
                    "failed_courses":   "Courses Failed",
                    "late_submissions": "Late Submissions",
                    "student_id":       "Student ID",
                }
                display_df = (
                    result_df[display_cols]
                    .rename(columns=rename_map)
                    .sort_values("Risk Level",
                                 key=lambda x: x.map({"High":0,"Medium":1,"Low":2}))
                    .reset_index(drop=True)
                )
                st.dataframe(display_df, use_container_width=True)

                
                csv_out = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Full Results as CSV",
                    data=csv_out,
                    file_name="student_risk_assessment.csv",
                    mime="text/csv",
                )




def render_login_page():
    """Renders a professional mocked login landing page."""
  
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
