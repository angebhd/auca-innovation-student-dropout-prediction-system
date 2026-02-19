# Student Dropout Prediction System - Analytics Dashboard Guide

## Overview

The **Analytics Dashboard** is an interactive visualization tool that helps educators and administrators understand student dropout patterns at AUCA (Adventist University of Central Africa). It presents **4 key charts** that analyze relationships between student academic performance, attendance, engagement, and dropout risk.

---

## Dashboard Metrics (Top Summary Cards)

Before diving into charts, here are the summary metrics displayed at the top:

| Metric             | Meaning                                         | Example | Interpretation                                          |
| ------------------ | ----------------------------------------------- | ------- | ------------------------------------------------------- |
| **Total Students** | Number of students in the dataset               | 514     | Dataset size - how many student records we're analyzing |
| **Dropped Out**    | Students who did not complete their program     | 203     | 203 students left before finishing their degree         |
| **Avg GPA**        | Average Grade Point Average across all students | 13.53   | On a scale of 0-20, the mean performance level          |
| **Avg Attendance** | Average attendance rate across all students     | 70.3%   | On average, how regularly students attend classes       |

---

## The 4 Analytics Charts Explained

### Chart 1: **Dropout Risk Distribution** (Pie Chart)

**What it shows:** A breakdown of how many students fall into each risk category.

**The Categories:**

- **Low Risk** (Green): Students likely to succeed and graduate
- **Medium Risk** (Yellow): Students who need monitoring and support
- **High Risk** (Red): Students at significant risk of dropping out

**What the numbers mean:**

- Each percentage represents what fraction of your student population is in that category
- Example: If "Low: 45.9%", approximately 236 students (out of 514) are in good standing

**Why this matters:**

- Helps allocate resources to at-risk students
- Shows the overall health of your student retention
- Identifies if intervention programs are working

**Key Insight:** If most students are "Low Risk", retention is strong. If many are "High Risk", urgent action is needed.

---

### Chart 2: **GPA vs Attendance Rate by Student Status** (Scatter Plot)

**What it shows:** How attendance and academic performance relate to each other, colored by whether students stayed or dropped out.

**The Axes:**

- **X-axis (Horizontal)**: Attendance Rate (%) - how often students come to class
  - Range: 0-100%
  - 80%+ = Very consistent attendance
  - 50-80% = Moderate attendance
  - <50% = Frequent absences

- **Y-axis (Vertical)**: Current GPA - academic performance
  - Range: 0-20
  - 15-20 = Excellent performance
  - 10-15 = Good/average performance
  - <10 = Below average/struggling

**The Colors:**

- **Green dots**: Students who continued their studies (stayed)
- **Red dots**: Students who dropped out (left)

**What to look for:**

- **Bottom-left cluster (Red)**: Students with low attendance AND low GPA who dropped out
- **Top-right cluster (Green)**: Students with high attendance AND high GPA who stayed
- **Clear separation**: If red and green are clearly separated, these metrics are strong predictors

**Key Insight:**

> Students who attend classes regularly AND maintain good grades almost never drop out. Low attendance is often the first warning sign of dropout.

**Real Example:**

- A student with 40% attendance and GPA 8.5 → Very likely to drop out
- A student with 90% attendance and GPA 17 → Very likely to graduate

---

### Chart 3: **GPA Performance by Risk Level** (Bar Chart with Error Bars)

**What it shows:** The average GPA for students in each risk category, with uncertainty ranges.

**The Columns:**

- **Low Risk** (Green bar): Average GPA ≈ 17.23
- **Medium Risk** (Yellow bar): Average GPA ≈ 13.38
- **High Risk** (Red bar): Average GPA ≈ 10.53

**The Black Lines (Error Bars):**

- Show the **range of variation** around the average
- Indicates how spread out the GPAs are within each group
- Smaller lines = more consistent
- Larger lines = more variation

**What the numbers mean:**

- Low-risk students average 17.23/20 GPA (very good)
- High-risk students average 10.53/20 GPA (struggling)
- The difference (≈6.7 points) is **huge** - showing GPA strongly predicts dropout

**Why this matters:**

- **Validates our risk assessment**: Risk categories correctly reflect academic performance
- **Guides intervention**: High-risk students need academic support urgently
- **Identifies thresholds**: If a student's GPA drops below ~12, they may shift to higher risk

**Key Insight:**

> The clearer the separation between bars, the better our risk classification model is working.

---

### Chart 4: **Feature Importance for Dropout Prediction** (Horizontal Bar Chart)

**What it shows:** Which student factors are the strongest predictors of dropout.

**The Bars:**

- **Horizontal bars** ranked from most to least important
- **Bar length** = Importance score (correlation with actual dropout)
- **Colors**: Red/Orange (High Importance) → Yellow → Green (Low Importance)

**Key Metrics Ranked:**

| Factor                   | What It Measures                  | Why It Matters              | Predictive Power |
| ------------------------ | --------------------------------- | --------------------------- | ---------------- |
| **Current GPA**          | Academic performance              | Strongest dropout predictor | Very High        |
| **Failed Courses**       | Number of courses not passed      | Academic struggle           | Very High        |
| **Admission Grade**      | Entry exam/score                  | Academic readiness          | High             |
| **Attendance Rate**      | How often student attends         | Engagement with studies     | High             |
| **Absences Count**       | Number of missed classes          | Class participation         | High             |
| **Online Portal Logins** | How often student accesses system | Digital engagement          | Medium           |
| **Library Visits**       | Times student visited library     | Study effort                | Medium           |
| **Participation Score**  | Class participation level         | Active learning             | Medium           |
| **Distance from Campus** | km from campus                    | Travel burden               | Low              |
| **Age**                  | Student's age                     | Maturity, life experience   | Low              |

**What to look for:**

1. **Top bars (most important):**
   - GPA, Failed Courses, Admission Grade, Attendance
   - These are your intervention triggers

2. **Middle bars (moderate importance):**
   - Online engagement metrics, Library visits, Participation
   - Support factors that correlate with success

3. **Bottom bars (least important):**
   - Distance from campus, Age
   - Still matter, but less predictive than others

**Key Insight:**

> Focus interventions on factors with longest bars. A student with low GPA + high failed courses faces highest risk. Monitor these metrics first!

---

## Understanding the 24 Student Factors

The system tracks **24 different factors** about each student:

### Demographics (4 factors)

- Student ID, Age, Gender, Admission Grade

### Academic Performance (6 factors)

- Semester 1 GPA, Semester 2 GPA, Semester 3 GPA, Current GPA, Failed Courses, Average Grade

### Attendance & Engagement (6 factors)

- Attendance Rate (%), Absences Count, Late Submissions, Library Visits, Online Portal Logins, Participation Score

### Behavioral (2 factors)

- Extracurricular Activities, Distance from Campus

### Socioeconomic (4 factors)

- Scholarship Status, Financial Aid, Accommodation Type

### Risk & Outcome (2 factors)

- Dropout Risk Category (Low/Medium/High)
- Actual Dropout Status (0=Stayed, 1=Dropped Out)

---

## How to Read the Dashboard

### Step 1: Check Summary Cards

- If **Avg Attendance < 60%**, you have an engagement problem
- If **Avg GPA < 12**, students are struggling academically
- If **% Dropped Out > 30%**, retention rates are concerning

### Step 2: Review Pie Chart

- Aim for **>50% Low Risk** students (healthy cohort)
- If >30% are High Risk, implement support programs immediately

### Step 3: Analyze Scatter Plot

- Look for the **green cluster in the top-right** (ideal students)
- Any red dots in top-left = students we almost lost despite good attendance
- Any green dots in bottom-right = high performers who still graduated (unlikely dropout)

### Step 4: Examine Bar Chart

- If **Low Risk bar >> High Risk bar**, model is working well
- If they're similar heights, need better risk assessment

### Step 5: Study Feature Importance Chart

- **Look at the top bars** - these are strongest predictors
- Highest bars = Most important factors for dropout prediction
- Use these insights to prioritize interventions

---

## Quick Decision Guide

| What You See                             | What It Means                        | Action Required               |
| ---------------------------------------- | ------------------------------------ | ----------------------------- |
| >50% Low Risk, high attendance, high GPA | Students doing well                  | Maintain current approach     |
| 30-40% Medium Risk, moderate attendance  | Some students at risk                | Offer tutoring/counseling     |
| >20% High Risk, low attendance, low GPA  | Serious concern                      | Launch retention program      |
| Red dots throughout scatter plot         | High dropout rate                    | Review curriculum and support |
| GPA has high importance score            | GPA strongly predicts dropout        | Monitor GPA closely           |
| Attendance has high importance score     | Attendance strongly predicts dropout | Track attendance regularly    |

---

## Installation & Running

### Prerequisites

- Python 3.13+
- Streamlit 1.54.0
- All dependencies from `requirements.txt`

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Run the Application

```bash
streamlit run app/app.py
```

### Step 3: Access Dashboard

Open your browser to: `http://localhost:8501`

### Navigation

- **Home**: Welcome page with system overview
- **Data Suite**: Upload/generate data and clean it
- **Analytics** (this dashboard): View all 4 charts
- **Risk Prediction**: ML model predictions (coming soon)

---

## FAQ - Frequently Asked Questions

**Q: Why is "NaN" showing in charts?**
A: This was caused by invalid data. It's been fixed - the system now filters out missing values automatically.

**Q: What GPA range indicates risk?**
A: Typically:

- > 15 = Low Risk
- 10-15 = Medium Risk
- <10 = High Risk

**Q: Does attendance alone predict dropout?**
A: Highly correlated but not the only factor. Combine with GPA and engagement for best prediction.

**Q: How are students categorized as Low/Medium/High risk?**
A: Using a weighted algorithm:

- GPA: 40% weight
- Attendance: 30% weight
- Academic Trend: 20% weight
- Engagement Metrics: 10% weight

**Q: Can I export these visualizations?**
A: Yes - use Streamlit's built-in download buttons at the top-right of each chart.

---

## Color Scheme

- **Green (#2ecc71)**: Low Risk / Positive / Success
- **Yellow (#f39c12)**: Medium Risk / Warning / Caution
- **Red (#e74c3c)**: High Risk / Danger / Intervention Needed
- **Orange/Red**: High Importance / Strong predictors

---

## Technical Details

### Data Processing

- **Missing values**: Handled using mean/median imputation
- **Outliers**: Removed using Interquartile Range (IQR) method
- **Standardization**: All metrics normalized to 0-20 or 0-100 scale

### Performance

- Initial load generates all 4 charts (1-2 seconds)
- Suitable for datasets up to 100,000 rows
- Charts refresh in real-time as data updates

---

## Future Enhancements

- Predictive model integration for individual risk scores
- Student-level drill-down reports
- Time-series analysis for semester trends
- Intervention recommendation engine
- Automated high-risk student alerts

---

## Support

For questions or issues:

1. Check the README.md for general setup
2. Review config/config.yaml for configuration options
3. Contact the development team for technical support
