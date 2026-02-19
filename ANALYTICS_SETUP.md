# Analytics Dashboard Setup Guide

## Overview

The analytics dashboard displays **4 key visualizations** from the exploratory data analysis of the student dropout prediction dataset. The dashboard features:

- **Left Sidebar**: Key metrics and statistics about the dataset
- **Main Content Area**: 4 descriptive charts with detailed insights

## The 4 Key Charts

### 1. **Dropout Risk Distribution** (Pie Chart)

- **Purpose**: Shows the percentage breakdown of students across risk categories
- **Categories**: Low Risk (Green), Medium Risk (Yellow), High Risk (Red)
- **Insight**: Identifies what proportion of the student body falls into each risk category
- **Use Case**: Strategic planning and resource allocation for intervention programs

### 2. **GPA vs Attendance Rate by Student Status** (Scatter Plot)

- **Purpose**: Reveals the relationship between two critical academic metrics
- **Dimensions**:
  - X-axis: Attendance Rate (%)
  - Y-axis: Current GPA
  - Color: Student Status (Continuing vs Dropped Out)
- **Insight**: Shows correlation between attendance and GPA, segmented by dropout status
- **Use Case**: Identify patterns that distinguish continuing students from those who dropped out

### 3. **GPA Performance by Risk Level** (Bar Chart with Error Bars)

- **Purpose**: Compares average GPA across dropout risk categories
- **Dimensions**: Risk Level (Low, Medium, High) vs Average GPA
- **Insight**: High-risk students typically have lower GPA, confirming GPA as a predictor
- **Use Case**: Validates the relationship between academic performance and dropout risk

### 4. **Correlation Matrix of Key Metrics** (Heatmap)

- **Purpose**: Shows how different student metrics correlate with each other
- **Includes**: Age, Admission Grade, GPA, Failed Courses, Attendance, Absences, Portal Logins, Library Visits, Participation Score, Distance from Campus
- **Color Scale**: Red (negative correlation) to Blue (positive correlation)
- **Insight**: Identifies which metrics are most strongly related to each other
- **Use Case**: Feature engineering and understanding multi-factor relationships in dropout prediction

## Sidebar Metrics

The left sidebar displays key summary statistics:

- **Total Students**: Overall dataset size
- **Students Dropped Out**: Absolute count and percentage
- **Average GPA**: Mean GPA across the cohort
- **Average Attendance Rate**: Mean attendance percentage
- **Retention Rate**: Percentage of students who didn't drop out

## Installation & Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Ensure Data File Exists

The app expects data at: `data/raw/dummy_data.csv`

Required columns:

- Student demographics: `student_id`, `age`, `gender`, `admission_grade`, `previous_education`
- Academic: `semester_1_gpa`, `semester_2_gpa`, `semester_3_gpa`, `current_gpa`, `failed_courses`, `average_grade`
- Attendance: `attendance_rate`, `absences_count`, `late_submissions`
- Engagement: `library_visits`, `online_portal_logins`, `participation_score`, `extracurricular_activities`
- Socioeconomic: `scholarship_status`, `financial_aid`, `distance_from_campus`, `accommodation_type`
- Target: `dropout_risk` (High/Medium/Low), `actual_dropout` (0/1)

### 3. Run the Application

```bash
cd app
python app.py
```

### 4. Access the Dashboard

- **Home**: http://localhost:5000/
- **Analytics Dashboard**: http://localhost:5000/analytics
- **API Endpoint**: http://localhost:5000/api/analytics (JSON data)

## Features

✅ **Dynamic Chart Generation**: Charts are generated on-demand from the data
✅ **Responsive Design**: Works on desktop and tablet devices
✅ **Data Caching**: Charts are cached in memory for performance
✅ **Refresh Capability**: Manual refresh button to reload data
✅ **Color-Coded Metrics**: Risk levels use intuitive color coding
✅ **Professional Styling**: Clean, modern UI with proper data visualization best practices

## Color Scheme

- **Green (#6bcf7f)**: Low Risk / Positive
- **Yellow (#ffd93d)**: Medium Risk / Warning
- **Red (#ff6b6b)**: High Risk / Danger
- **Blue (#3498db)**: Primary Actions / Information

## API Usage

Get analytics data in JSON format:

```bash
curl http://localhost:5000/api/analytics
```

Response:

```json
{
  "timestamp": "2024-02-19 10:30:45",
  "total_students": 450,
  "dropout_count": 67,
  "dropout_rate": "14.9%",
  "avg_gpa": "3.45",
  "avg_attendance": "87.3%"
}
```

## Customization

### Modify Chart Colors

Edit the color dictionaries in `app.py`:

- `colors` list in `generate_dropout_risk_distribution()`
- `colors_map` dict in `generate_gpa_attendance_scatter()`

### Change Chart Parameters

Edit the respective chart generation functions:

- `generate_*()` functions control chart appearance
- Modify `figsize`, `fontsize`, and colors as needed

### Add More Charts

1. Create a new `generate_*_chart()` function
2. Add to `generate_analytics_data()`
3. Update `analytics.html` template to display

## Troubleshooting

**"Error loading analytics data"**

- Ensure `data/raw/dummy_data.csv` exists
- Check that all required columns are present
- Verify file is not corrupted

**Charts not displaying**

- Check browser console for errors
- Verify matplotlib is properly installed
- Try refreshing the page

**Port 5000 already in use**

- Change port in `app.py`: `app.run(port=5001)`

## Performance Notes

- Initial load generates all 4 charts (takes ~2-3 seconds)
- Charts are cached in memory for subsequent requests
- Refresh clears cache and regenerates charts
- Suitable for datasets up to 100,000 rows

## Future Enhancements

- [ ] Interactive charts with hover tooltips (Plotly/Bokeh)
- [ ] Filter data by risk category or demographics
- [ ] Export reports as PDF
- [ ] Time-series analysis for semester trends
- [ ] Predictive model integration
- [ ] Student-level drill-down reports
