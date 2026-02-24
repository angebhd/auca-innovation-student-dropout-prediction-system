import pandas as pd
import numpy as np
import random


def generate_student_dummy_data(n_samples: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic student data for testing the dropout prediction system.
    
    Parameters:
    -----------
    n_samples : int
        Number of student records to generate (default: 500)
    seed : int
        Random seed for reproducibility (default: 42)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing synthetic student data
    """
    np.random.seed(seed)
    random.seed(seed)
    
    # Define risk levels
    risk_levels = ['Low', 'Medium', 'High']
    
    # Generate base data
    data = {
        'student_id': [f'STU{str(i).zfill(5)}' for i in range(1, n_samples + 1)],
        'age': np.random.randint(18, 35, n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'admission_grade': np.random.uniform(60, 100, n_samples).round(2),
        'current_gpa': np.random.uniform(1.5, 4.0, n_samples).round(2),
        'failed_courses': np.random.randint(0, 8, n_samples),
        'attendance_rate': np.random.uniform(50, 100, n_samples).round(1),
        'absences_count': np.random.randint(0, 20, n_samples),
        'online_portal_logins': np.random.randint(0, 50, n_samples),
        'library_visits': np.random.randint(0, 30, n_samples),
        'participation_score': np.random.uniform(0, 100, n_samples).round(1),
        'distance_from_campus': np.random.uniform(0.5, 50, n_samples).round(1),
        'parent_education_level': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
        'scholarship_status': np.random.choice(['None', 'Partial', 'Full'], n_samples, p=[0.4, 0.35, 0.25]),
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Generate dropout_risk based on weighted factors
    dropout_risk = []
    actual_dropout = []
    
    for idx in range(n_samples):
        # Calculate risk score based on multiple factors
        risk_score = 0
        
        # Low GPA increases risk
        if df.loc[idx, 'current_gpa'] < 2.0:
            risk_score += 3
        elif df.loc[idx, 'current_gpa'] < 2.5:
            risk_score += 2
        elif df.loc[idx, 'current_gpa'] < 3.0:
            risk_score += 1
        
        # Failed courses increase risk
        risk_score += df.loc[idx, 'failed_courses'] * 0.5
        
        # Low attendance increases risk
        if df.loc[idx, 'attendance_rate'] < 60:
            risk_score += 3
        elif df.loc[idx, 'attendance_rate'] < 75:
            risk_score += 2
        elif df.loc[idx, 'attendance_rate'] < 90:
            risk_score += 1
        
        # Low participation increases risk
        if df.loc[idx, 'participation_score'] < 30:
            risk_score += 2
        elif df.loc[idx, 'participation_score'] < 50:
            risk_score += 1
        
        # Many absences increase risk
        if df.loc[idx, 'absences_count'] > 10:
            risk_score += 2
        elif df.loc[idx, 'absences_count'] > 5:
            risk_score += 1
        
        # Low online engagement
        if df.loc[idx, 'online_portal_logins'] < 5:
            risk_score += 1
        
        # Determine risk level
        if risk_score >= 5:
            risk = 'High'
        elif risk_score >= 2:
            risk = 'Medium'
        else:
            risk = 'Low'
        
        dropout_risk.append(risk)
        
        # Determine actual dropout (correlated with risk but with some noise)
        if risk == 'High':
            actual = 1 if random.random() < 0.7 else 0
        elif risk == 'Medium':
            actual = 1 if random.random() < 0.35 else 0
        else:
            actual = 1 if random.random() < 0.1 else 0
        
        actual_dropout.append(actual)
    
    df['dropout_risk'] = dropout_risk
    df['actual_dropout'] = actual_dropout
    
    # Add some missing values to simulate dirty data
    n_missing = int(n_samples * 0.05)  # 5% missing values
    
    for _ in range(n_missing):
        col = random.choice(['current_gpa', 'attendance_rate', 'failed_courses'])
        row = random.randint(0, n_samples - 1)
        df.loc[row, col] = np.nan
    
    # Add some duplicate rows
    n_duplicates = int(n_samples * 0.02)  # 2% duplicates
    duplicate_indices = np.random.choice(n_samples, n_duplicates, replace=True)
    duplicates = df.iloc[duplicate_indices].copy()
    df = pd.concat([df, duplicates], ignore_index=True)
    
    # Shuffle the DataFrame
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    return df
