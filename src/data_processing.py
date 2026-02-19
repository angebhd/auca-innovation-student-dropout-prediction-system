import pandas as pd
import numpy as np


def clean_student_data(df: pd.DataFrame) -> tuple:
    """
    Clean and validate student data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw student data DataFrame
        
    Returns:
    --------
    tuple: (cleaned_df, stats_dict)
        - cleaned_df: Cleaned pandas DataFrame
        - stats_dict: Dictionary with cleaning statistics
    """
    # Make a copy to avoid modifying the original
    df_cleaned = df.copy()
    
    # Store initial row count
    initial_rows = len(df_cleaned)
    
    # Remove duplicate rows
    duplicates_removed = df_cleaned.duplicated().sum()
    df_cleaned = df_cleaned.drop_duplicates()
    
    # Handle missing values
    # Fill numeric columns with median
    numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if df_cleaned[col].isnull().sum() > 0:
            df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
    
    # Fill categorical columns with mode
    categorical_columns = df_cleaned.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if df_cleaned[col].isnull().sum() > 0:
            mode_value = df_cleaned[col].mode()
            if len(mode_value) > 0:
                df_cleaned[col].fillna(mode_value[0], inplace=True)
    
    # Remove rows with critical missing values (e.g., missing dropout_risk)
    critical_columns = ['dropout_risk', 'actual_dropout']
    for col in critical_columns:
        if col in df_cleaned.columns:
            df_cleaned = df_cleaned.dropna(subset=[col])
    
    # Validate and fix data types
    # Ensure attendance_rate is between 0 and 100
    if 'attendance_rate' in df_cleaned.columns:
        df_cleaned['attendance_rate'] = df_cleaned['attendance_rate'].clip(0, 100)
    
    # Ensure GPA is between 0 and 4 (or 0-100 based on system)
    if 'current_gpa' in df_cleaned.columns:
        max_gpa = df_cleaned['current_gpa'].max()
        if max_gpa > 10:  # Assuming 100-based grading
            df_cleaned['current_gpa'] = df_cleaned['current_gpa'].clip(0, 100)
        else:  # Assuming 4-based grading
            df_cleaned['current_gpa'] = df_cleaned['current_gpa'].clip(0, 4)
    
    # Ensure failed_courses and absences_count are non-negative
    if 'failed_courses' in df_cleaned.columns:
        df_cleaned['failed_courses'] = df_cleaned['failed_courses'].clip(lower=0)
    
    if 'absences_count' in df_cleaned.columns:
        df_cleaned['absences_count'] = df_cleaned['absences_count'].clip(lower=0)
    
    # Final row count
    final_rows = len(df_cleaned)
    
    # Calculate retention rate
    if initial_rows > 0:
        retention_rate = (final_rows / initial_rows) * 100
    else:
        retention_rate = 0
    
    # Create stats dictionary
    stats = {
        'final_rows': final_rows,
        'duplicates_removed': int(duplicates_removed),
        'retention_rate': f"{retention_rate:.1f}"
    }
    
    return df_cleaned, stats


def validate_data(df: pd.DataFrame) -> bool:
    """
    Validate that the DataFrame has required columns for the prediction system.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to validate
        
    Returns:
    --------
    bool: True if valid, False otherwise
    """
    required_columns = [
        'age', 'admission_grade', 'current_gpa', 'failed_courses',
        'attendance_rate', 'dropout_risk'
    ]
    
    return all(col in df.columns for col in required_columns)
