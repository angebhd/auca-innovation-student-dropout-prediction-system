# Umoja Team: Student Dropout Prediction System

## Problem Statement

Higher education institutions face challenges in identifying students who are at risk of dropping out or failing courses at an early stage. Academic performance data, attendance records, and learning behavior are often analyzed too late, resulting in delayed interventions and increased failure rates.

## Proposed ML-Based Solution

This project implements a machine learning–based prediction system that analyzes historical student data such as grades, attendance, and course engagement to classify students into risk categories. The system generates early warning indicators to support timely academic interventions by lecturers and academic administrators.

---

## Team & Responsibilities

| Member      | Responsibility                | Status      |
| :---------- | :---------------------------- | :---------- |
| **Ange**    | Dataset, Clean up & Filtering | Implemented |
| **Arnaud**  | Dashboard UI                  | Not Started |
| **Raissa**  | File Upload                   | Not Started |
| **Ines**    | Sign up / Log in / Logout     | Not Started |
| **Jessica** | Charts / Analytics            | Not Started |
| **Mugeni**  | History                       | Not Started |
| **Ritha**   | AI Prediction                 | Not Started |
| **Zakaria** | Code Review                   | Not Started |

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
├── app/                # Streamlit UI & Orchestration
│   ├── app.py          # Entry point
│   └── ui_components.py# UI Components & Styling
├── config/             # Environment Configuration
│   └── config.yaml     # Centralized Parameters
├── data/               # Persistent Storage
│   ├── cleaned/        # Processed Datasets
│   └── raw/            # Original Datasets
├── models/             # ML Model Serialized Files
├── src/                # Core Logic & Engine
│   ├── data_generation.py
│   └── data_processing.py
├── tests/              # Unit & Integration Tests
└── requirements.txt    # Project Dependencies
```

---

## Feature Status Matrix

- [x] Data Acquisition: Persistent storage in data/raw/.
- [x] Data Cleaning: Modularized and SonarLint compliant.
- [ ] Authentication: To be implemented.
- [ ] Interactive Charts: To be implemented.
- [ ] AI Predictions: To be implemented.

---

## Getting Started

### Prerequisites

- Python 3.14+
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
