# Umoja Team: Student Dropout Prediction System

## Problem Statement

Higher education institutions face challenges in identifying students who are at risk of dropping out or failing courses at an early stage. Academic performance data, attendance records, and learning behavior are often analyzed too late, resulting in delayed interventions and increased failure rates.

## Proposed ML-Based Solution

This project implements a machine learning–based prediction system that analyzes historical student data such as grades, attendance, and course engagement to classify students into risk categories. The system generates early warning indicators to support timely academic interventions by lecturers and academic administrators.

---

## Team & Responsibilities

| Member      | Responsibility                | Status         |
| :---------- | :---------------------------- | :------------- |
| **Ange**    | Dataset, Clean up & Filtering | ✅ Implemented |
| **Arnaud**  | Dashboard UI                  | ✅ Implemented |
| **Raissa**  | File Upload                   | ✅ Implemented |
| **Ines**    | Sign up / Log in / Logout     | ✅ Implemented |
| **Jessica** | Charts / Analytics            | ✅ Implemented |
| **Mugeni**  | History                       | ✅ Implemented |
| **Ritha**   | AI Prediction                 | ✅ Implemented |
| **Zakaria** | Code Review                   | In Progress    |

---

## Technical Architecture & Logic

The system is built with a strong focus on **Separation of Concerns** to allow the team to work concurrently.

### 1. Logic & Core Engine Layer (`src/`)

- `data_processing.py`: Contains the core logic for cleaning (imputation, duplicate removal, outlier detection). Isolated from the UI so it can be used by the AI engine later.
- `data_generation.py`: Generates "dirty" synthetic data to test the robustness of the cleaning pipeline.

### 2. UI Layer (`app/`)

- `app.py`: The entry point. It handles orchestration but _no_ rendering logic.
- `ui_components.py`: Contains all Streamlit rendering functions (styles, headers, columns). This allows UI designers (Arnaud) to work without breaking the core logic.

### 4. File Architecture

```text
.
├── app/                    # Streamlit UI & Orchestration
│   ├── app.py              # Entry point
│   ├── ui_components.py    # UI Components & Styling
│   ├── auth_ui.py          # Authentication UI
│   └── history_ui.py       # History/Activity UI
├── config/                 # Environment Configuration
│   └── config.yaml         # Centralized Parameters
├── data/                   # Persistent Storage
│   ├── cleaned/            # Processed Datasets
│   ├── raw/                # Original Datasets
│   ├── users.json          # User Database
│   └── history.json        # Activity Logs
├── models/                 # ML Model Serialized Files
│   └── dropout_model.pkl   # Trained prediction model
├── src/                    # Core Logic & Engine
│   ├── analytics.py        # Charts & visualization
│   ├── auth.py             # Authentication service
│   ├── data_generation.py  # Synthetic data generator
│   ├── data_processing.py  # Data cleaning pipeline
│   ├── database.py         # JSON database operations
│   ├── history.py          # Activity tracking
│   ├── logger.py           # Logging configuration
│   └── prediction.py       # ML prediction engine
├── tests/                  # Unit & Integration Tests
└── requirements.txt        # Project Dependencies
```

---

## Feature Status Matrix

- [x] Data Acquisition: Persistent storage in data/raw/.
- [x] Data Cleaning: Modularized and SonarLint compliant.
- [x] Authentication: Login with secure password hashing.
- [x] Interactive Charts: Plotly-powered analytics dashboard.
- [x] AI Predictions: Random Forest classifier for dropout risk.
- [x] History Tracking: Activity logging for all operations.

---

## Getting Started

### Prerequisites

- Python 3.10+
- Virtual environment

### Installation & Run

```bash
# 1. Setup
git clone https://github.com/angebhd/auca-innovation-student-dropout-prediction-system.git
cd auca-innovation-student-dropout-prediction-system
python -m venv .venv

# 2. Activate (Linux/macOS)
source .venv/bin/activate
# Or (Windows)
.venv\Scripts\activate

# 2. Dependencies
pip install -r requirements.txt

# 3. Launch
streamlit run app/app.py
```

---

**Developed by Umoja Team | © 2026**
