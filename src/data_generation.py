import pandas as pd
import numpy as np
from typing import List
import logging
from src.logger import setup_logging

logger = setup_logging(__name__)

def generate_student_dummy_data(n_samples: int = 500, seed: int = None) -> pd.DataFrame:
    """
    Generate a realistic dummy student dataset for testing.
    
    Args:
        n_samples: Number of records to generate.
        seed: Random seed for reproducibility.
        
    Returns:
        Generated DataFrame.
    """
    rng = np.random.default_rng(seed)
    
    data = {
        'student_id': [f'STU{str(i).zfill(6)}' for i in range(1, n_samples + 1)],
        'age': rng.integers(18, 35, n_samples),
        'gender': rng.choice(['M', 'F', 'Other'], n_samples, p=[0.45, 0.45, 0.1]),
        'admission_grade': rng.uniform(60, 95, n_samples),
        'previous_education': rng.choice(['Public', 'Private', 'International'], n_samples),
        
        'semester_1_gpa': rng.uniform(7.0, 20.0, n_samples),
        'semester_2_gpa': rng.uniform(7.0, 20.0, n_samples),
        'semester_3_gpa': rng.uniform(7.0, 20.0, n_samples),
        'current_gpa': rng.uniform(7.0, 20.0, n_samples),
        'failed_courses': rng.choice([0, 1, 2, 3], n_samples, p=[0.7, 0.15, 0.1, 0.05]),
        'average_grade': rng.uniform(10, 20, n_samples),
        
        'attendance_rate': rng.uniform(40, 100, n_samples),
        'absences_count': rng.integers(0, 25, n_samples),
        'late_submissions': rng.integers(0, 10, n_samples),
        
        'library_visits': rng.integers(0, 30, n_samples),
        'online_portal_logins': rng.integers(10, 200, n_samples),
        'participation_score': rng.uniform(0, 10, n_samples),
        'extracurricular_activities': rng.integers(0, 5, n_samples),
        
        'scholarship_status': rng.choice(['Yes', 'No'], n_samples),
        'financial_aid': rng.choice(['Yes', 'No'], n_samples),
        'distance_from_campus': rng.uniform(0, 60, n_samples),
        'accommodation_type': rng.choice(['Campus', 'Home', 'Rented'], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # --- Inject "Noise" for Cleaning Verification ---
    
    # 1. Inject Duplicates (Random amount between 2% and 10%)
    if n_samples > 10:
        dup_ratio = rng.uniform(0.02, 0.10)
        n_duplicates = int(n_samples * dup_ratio)
        dup_indices = rng.choice(df.index, size=n_duplicates, replace=False)
        df = pd.concat([df, df.iloc[dup_indices]], ignore_index=True)
        logger.info(f"Injected {n_duplicates} duplicate rows ({dup_ratio*100:.1f}%).")

    # 2. Inject Missing Values (NAs) (Random amount between 3% and 8% per column)
    cols_to_nullify = ['age', 'current_gpa', 'attendance_rate', 'gender', 'scholarship_status']
    for col in cols_to_nullify:
        if col in df.columns:
            null_ratio = rng.uniform(0.03, 0.08)
            n_nulls = int(len(df) * null_ratio)
            null_indices = rng.choice(df.index, size=n_nulls, replace=False)
            df.loc[null_indices, col] = np.nan
            logger.info(f"Injected {n_nulls} NAs into {col} ({null_ratio*100:.1f}%).")

    # 3. Inject Invalid Values (Out-of-range/Anomaly)
    if 'age' in df.columns:
        invalid_age_indices = rng.choice(df.index, size=5, replace=False)
        df.loc[invalid_age_indices[0:3], 'age'] = -5  # Negative age
        df.loc[invalid_age_indices[3:5], 'age'] = 150 # Impossible age
        
    if 'current_gpa' in df.columns:
        invalid_gpa_indices = rng.choice(df.index, size=5, replace=False)
        df.loc[invalid_gpa_indices, 'current_gpa'] = 25.5 # GPA above 20.0
        
    if 'attendance_rate' in df.columns:
        invalid_att_indices = rng.choice(df.index, size=5, replace=False)
        df.loc[invalid_att_indices, 'attendance_rate'] = 120 # Over 100%

    # 4. Inject Outliers (Valid but extreme)
    # (Already handled by range logic usually, but let's ensure some exist)
    
    # Shuffle the dirty data
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Simple risk modeling (after noise, risk might be weird but that's fine for cleaning test)
    risk_scores = (
        (20.0 - df['current_gpa']) * 4 + 
        (df['failed_courses'] * 15) + 
        (100 - df['attendance_rate']) * 0.5
    )
    
    df['actual_dropout'] = (risk_scores > 50).astype(int)
    df['dropout_risk'] = pd.cut(risk_scores, 
                               bins=[-float('inf'), 30, 60, float('inf')], 
                               labels=['Low', 'Medium', 'High'])
    
    return df

if __name__ == "__main__":
    import yaml
    import os
    
    # Load config
    try:
        with open("config/config.yaml", "r") as f:
            cfg = yaml.safe_load(f)
        output_dir = cfg['paths']['raw_dir']
    except Exception:
        output_dir = "data/raw"
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    df = generate_student_dummy_data()
    output_path = os.path.join(output_dir, "students_raw.csv")
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} records to {output_path}")
