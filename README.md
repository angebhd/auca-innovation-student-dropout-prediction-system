# 🎓 Student Dropout Prediction System

An advanced data analytics and machine learning system for predicting student dropout risk using comprehensive academic, engagement, and socioeconomic metrics.

## 📋 Project Overview

This system analyzes student data to:

- Identify at-risk students before dropout occurs
- Understand key factors contributing to student retention
- Provide actionable insights through interactive analytics
- Support institutional intervention strategies

## 🎯 Key Components

- **Interactive Streamlit Dashboard**: Real-time visualizations and analytics
- **Data Processing Pipeline**: Automated cleaning, preprocessing, and validation
- **Exploratory Data Analysis**: Jupyter notebook with 40+ analysis cells
- **24 Key Metrics**: Academic, engagement, and socioeconomic factors
- **Model Training Suite**: Risk classification and dropout prediction

## 📁 Project Structure

```
auca-innovation-student-dropout-prediction-system/
├── app/
│   ├── app.py                          # Streamlit application with analytics dashboard
│   └── __pycache__/
├── src/
│   ├── data_loader.py                 # Data loading utilities
│   ├── preprocessing.py                # Data preprocessing
│   ├── training.py                    # Model training
│   ├── data_processing.py              # Data processing workflows
│   ├── data_generation.py              # Synthetic data generation
│   ├── logger.py                       # Logging configuration
│   └── __pycache__/
├── data/
│   ├── raw/
│   │   └── dummy_data.csv              # Input dataset
│   └── cleaned/                        # Processed data
├── exploratory_data_analysis.ipynb    # EDA with 40+ cells of analysis
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
└── ANALYTICS_SETUP.md                 # Detailed analytics guide
```

## 📊 Dataset Features (24 Metrics)

### Demographic Information

- `student_id`: Unique identifier
- `age`: Student age
- `gender`: Male/Female/Other

### Academic Performance (Core Predictors)

- `admission_grade`: Entrance exam score
- `semester_1_gpa`, `semester_2_gpa`, `semester_3_gpa`: GPA by semester
- `current_gpa`: Overall GPA (0.0-4.0)
- `average_grade`: Mean of all course grades
- `failed_courses`: Number of failed courses
- `previous_education`: High school type/quality

### Attendance Data

- `attendance_rate`: Class attendance percentage (0-100%)
- `absences_count`: Total absences
- `late_submissions`: Number of late assignments

### Engagement Metrics

- `library_visits`: Per semester visits
- `online_portal_logins`: LMS platform usage
- `participation_score`: Class participation (0-10)
- `extracurricular_activities`: Number of activities

### Socioeconomic Factors

- `scholarship_status`: Yes/No or Full/Partial/None
- `financial_aid`: Yes/No
- `distance_from_campus`: Kilometers from campus
- `accommodation_type`: Campus/Home/Rented

### Target Variables

- `dropout_risk`: High/Medium/Low risk category
- `actual_dropout`: Binary (1=Dropped, 0=Continuing)

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Place your CSV dataset at: `data/raw/dummy_data.csv`

### 3. Run the Application

```bash
streamlit run app/app.py
```

The app opens automatically at **http://localhost:8501**

## 🛠️ Technology Stack

- **Frontend/Backend**: Streamlit (Python)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Analysis**: Jupyter Notebook (EDA)
- **Logging**: Python logging module

## 📈 Key Insights

The analysis reveals:

1. **GPA is a Strong Predictor**: Lower GPAs correlate with higher dropout rates
2. **Attendance Matters**: Strong correlation between attendance and retention
3. **Engagement is Critical**: Online activities and library visits indicate engagement
4. **Socioeconomic Factors Impact Risk**: Scholarship and accommodation types influence outcomes
5. **Early Warning Signs**: Semester 1-2 GPA trends predict future dropout risk

## � Important Notes

**Data Requirements:**

- Minimum 100+ student records recommended
- Ensure `student_id` is unique in your dataset
- All numeric columns should be properly formatted

**First Time Setup:**
If you encounter any issues with running the app, ensure:

1. Python 3.8 or higher is installed
2. All dependencies from `requirements.txt` are installed
3. Your data file exists at `data/raw/dummy_data.csv`

## 📖 Documentation

- **[ANALYTICS_SETUP.md](ANALYTICS_SETUP.md)**: Complete guide to the analytics dashboard, configuration, and troubleshooting
- **[exploratory_data_analysis.ipynb](exploratory_data_analysis.ipynb)**: Full exploratory data analysis with 40+ visualization cells

## 🤝 Contributing

Contributions are welcome! Areas for enhancement:

- [ ] Add more prediction models (Logistic Regression, Random Forest, etc.)
- [ ] Export reports to PDF
- [ ] Student-level drill-down reports
- [ ] Time-series analysis for trends
- [ ] Mobile app version
- [ ] Real-time data integration

## 📄 License

This project is part of the AUCA Innovation program.

## 👥 Team

Developed at AUCA Innovation for the Student Dropout Prediction System.

## 📞 Support

For issues or questions:

1. Check the troubleshooting section
2. Review [ANALYTICS_SETUP.md](ANALYTICS_SETUP.md)
3. Check browser console for errors (F12)
4. Verify data file and format

---

**Last Updated**: February 2026  
**Version**: 1.0.0  
**Status**: Production Ready
