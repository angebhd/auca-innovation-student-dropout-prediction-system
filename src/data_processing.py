
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
import logging
from src.logger import setup_logging

logger = setup_logging(__name__)

def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to snake_case."""
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("-", "_")
    return df

def _impute_numerical(df: pd.DataFrame, stats: Dict[str, Any]) -> pd.DataFrame:
    """Handle numerical missing values with smart strategies."""
    imputation_map = {
        'age': 'median',
        'distance_from_campus': 'median',
        'admission_grade': 'mean',
        'semester_1_gpa': 'mean',
        'semester_2_gpa': 'mean',
        'semester_3_gpa': 'mean',
        'current_gpa': 'mean',
        'average_grade': 'mean',
        'attendance_rate': 'mean',
        'participation_score': 'mean',
        'failed_courses': 0,
        'absences_count': 0,
        'late_submissions': 0,
        'library_visits': 0,
        'online_portal_logins': 0,
        'extracurricular_activities': 0
    }
    
    for col, strategy in imputation_map.items():
        if col not in df.columns:
            continue
        
        null_count = df[col].isnull().sum()
        if null_count == 0:
            continue

        if isinstance(strategy, (int, float)):
            df[col] = df[col].fillna(strategy)
        elif strategy == 'mean':
            df[col] = df[col].fillna(df[col].mean())
        elif strategy == 'median':
            df[col] = df[col].fillna(df[col].median())
        
        stats['missing_values_imputed'][col] = int(null_count)
    return df

def _impute_categorical(df: pd.DataFrame, stats: Dict[str, Any]) -> pd.DataFrame:
    """Handle categorical missing values using mode."""
    categorical_cols = ['gender', 'previous_education', 'scholarship_status', 'financial_aid', 'accommodation_type']
    for col in categorical_cols:
        if col not in df.columns:
            continue
            
        null_count = df[col].isnull().sum()
        if null_count > 0:
            mode_val = df[col].mode()
            if not mode_val.empty:
                df[col] = df[col].fillna(mode_val[0])
                stats['missing_values_imputed'][col] = int(null_count)
    return df

def _impute_missing_values(df: pd.DataFrame, stats: Dict[str, Any]) -> pd.DataFrame:
    """Main imputation entry point."""
    df = _impute_numerical(df, stats)
    df = _impute_categorical(df, stats)
    return df

def _convert_types(df: pd.DataFrame) -> pd.DataFrame:
    """Convert columns to appropriate data types."""
    int_cols = [
        'age', 'failed_courses', 'absences_count', 'late_submissions',
        'library_visits', 'online_portal_logins', 'extracurricular_activities'
    ]
    float_cols = [
        'distance_from_campus', 'admission_grade', 'semester_1_gpa',
        'semester_2_gpa', 'semester_3_gpa', 'current_gpa', 'average_grade',
        'attendance_rate', 'participation_score'
    ]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def _validate_domain(df: pd.DataFrame, stats: Dict[str, Any]) -> pd.DataFrame:
    """Validate domain-specific constraints (ranges, etc)."""
    invalid_counts = {}

    if 'age' in df.columns:
        invalid = ~df['age'].between(15, 70, inclusive="both") & df['age'].notna()
        if invalid.any():
            invalid_counts['age'] = int(invalid.sum())
            df.loc[invalid, 'age'] = np.nan

    gpa_cols = ['semester_1_gpa', 'semester_2_gpa', 'semester_3_gpa', 'current_gpa']
    for col in gpa_cols:
        if col in df.columns:
            invalid = ~df[col].between(0, 20.0, inclusive="both") & df[col].notna()
            if invalid.any():
                invalid_counts[col] = int(invalid.sum())
                df.loc[invalid, col] = np.nan

    if 'attendance_rate' in df.columns:
        invalid = ~df['attendance_rate'].between(0, 100, inclusive="both") & df['attendance_rate'].notna()
        if invalid.any():
            invalid_counts['attendance_rate'] = int(invalid.sum())
            df.loc[invalid, 'attendance_rate'] = np.nan

    stats['invalid_values_set_na'] = invalid_counts
    return df


def _remove_outliers(df: pd.DataFrame, stats: Dict[str, Any]) -> pd.DataFrame:
    """Remove outliers using IQR method."""
    outlier_cols = ['age', 'attendance_rate', 'current_gpa', 'average_grade']
    for col in outlier_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            if pd.isna(IQR) or IQR == 0:
                continue
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

            rows_before = len(df)
            keep_mask = df[col].between(lower, upper, inclusive="both") | df[col].isna()
            df = df[keep_mask]
            stats['outliers_removed'] += (rows_before - len(df))
    return df

def _standardize_values(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize categorical string values."""
    if 'gender' in df.columns:
        gender_map = {'MALE': 'M', 'M': 'M', 'FEMALE': 'F', 'F': 'F', 'OTHER': 'Other'}
        df['gender'] = df['gender'].astype(str).str.upper().map(gender_map).fillna('Other')
    
    bool_map = {'YES': 'Yes', 'Y': 'Yes', '1': 'Yes', 'NO': 'No', 'N': 'No', '0': 'No'}
    for col in ['scholarship_status', 'financial_aid']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.upper().map(bool_map).fillna('No')

    if 'dropout_risk' in df.columns:
        df['dropout_risk'] = df['dropout_risk'].astype(str).str.capitalize()
    return df

def clean_student_data(df: pd.DataFrame, verbose: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Comprehensive data cleaning and filtering for student dropout prediction dataset.
    """
    df = df.copy()
    stats = {
        'initial_rows': len(df),
        'duplicates_removed': 0, 'rows_removed': 0,
        'missing_values_imputed': {}, 'outliers_removed': 0,
        'invalid_values_set_na': {}
    }

    df = _standardize_columns(df)
    df = _standardize_values(df)

    duplicates = df.duplicated().sum()
    df = df.drop_duplicates()
    stats['duplicates_removed'] = int(duplicates)

    df = _convert_types(df)
    df = _validate_domain(df, stats)
    df = _impute_missing_values(df, stats)
    df = _remove_outliers(df, stats)

    stats['final_rows'] = len(df)
    if stats['initial_rows'] > 0:
        stats['retention_rate'] = round((stats['final_rows'] / stats['initial_rows']) * 100, 2)
    else:
        stats['retention_rate'] = 0

    if verbose:
        logger.info("Cleaning Complete. Stats: %s", stats)

    return df, stats
