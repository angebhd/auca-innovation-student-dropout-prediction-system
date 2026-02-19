"""
Data Cleaning Pipeline for Student Dropout Prediction Dataset
This script handles:
- Removing duplicate records
- Handling missing values
- Data validation
- Standardization of formats
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataCleaner:
    """Clean and validate student dropout prediction data"""
    
    def __init__(self, raw_data_path, cleaned_data_path):
        self.raw_data_path = raw_data_path
        self.cleaned_data_path = cleaned_data_path
        self.df = None
        self.original_rows = 0
        self.cleaned_rows = 0
    
    def load_raw_data(self):
        """Load raw data from CSV"""
        try:
            self.df = pd.read_csv(self.raw_data_path)
            self.original_rows = len(self.df)
            logger.info(f"✓ Loaded {self.original_rows} rows from {self.raw_data_path}")
            return self
        except Exception as e:
            logger.error(f"✗ Error loading data: {e}")
            raise
    
    def remove_duplicates(self):
        """Remove duplicate records based on student_id"""
        if 'student_id' in self.df.columns:
            duplicate_count = self.df.duplicated(subset=['student_id']).sum()
            self.df = self.df.drop_duplicates(subset=['student_id'], keep='first')
            logger.info(f"✓ Removed {duplicate_count} duplicate records")
        else:
            # If no student_id, remove complete duplicates
            duplicate_count = self.df.duplicated().sum()
            self.df = self.df.drop_duplicates(keep='first')
            logger.info(f"✓ Removed {duplicate_count} complete duplicate rows")
        return self
    
    def handle_missing_values(self):
        """Handle missing values appropriately"""
        missing_counts = self.df.isnull().sum()
        
        if missing_counts.sum() > 0:
            logger.info("Handling missing values...")
            
            # For numeric columns, fill with median
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if self.df[col].isnull().sum() > 0:
                    median_val = self.df[col].median()
                    self.df[col].fillna(median_val, inplace=True)
                    logger.info(f"  - Filled {col} with median: {median_val}")
            
            # For categorical columns, fill with mode
            categorical_cols = self.df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if self.df[col].isnull().sum() > 0:
                    mode_val = self.df[col].mode()[0] if not self.df[col].mode().empty else 'Unknown'
                    self.df[col].fillna(mode_val, inplace=True)
                    logger.info(f"  - Filled {col} with mode: {mode_val}")
        
        logger.info(f"✓ Handled all missing values")
        return self
    
    def validate_data(self):
        """Validate data ranges and types"""
        logger.info("Validating data ranges...")
        
        # GPA should be 0-4.0
        gpa_cols = [col for col in self.df.columns if 'gpa' in col.lower()]
        for col in gpa_cols:
            out_of_range = ((self.df[col] < 0) | (self.df[col] > 4.0)).sum()
            if out_of_range > 0:
                self.df[col] = self.df[col].clip(0, 4.0)
                logger.info(f"  - Clipped {col} to [0, 4.0] ({out_of_range} values)")
        
        # Attendance rate should be 0-100%
        if 'attendance_rate' in self.df.columns:
            out_of_range = ((self.df['attendance_rate'] < 0) | (self.df['attendance_rate'] > 100)).sum()
            if out_of_range > 0:
                self.df['attendance_rate'] = self.df['attendance_rate'].clip(0, 100)
                logger.info(f"  - Clipped attendance_rate to [0, 100] ({out_of_range} values)")
        
        # Admission grade 0-100
        if 'admission_grade' in self.df.columns:
            out_of_range = ((self.df['admission_grade'] < 0) | (self.df['admission_grade'] > 100)).sum()
            if out_of_range > 0:
                self.df['admission_grade'] = self.df['admission_grade'].clip(0, 100)
                logger.info(f"  - Clipped admission_grade to [0, 100] ({out_of_range} values)")
        
        # Age should be reasonable (15-70)
        if 'age' in self.df.columns:
            out_of_range = ((self.df['age'] < 15) | (self.df['age'] > 70)).sum()
            if out_of_range > 0:
                self.df['age'] = self.df['age'].clip(15, 70)
                logger.info(f"  - Clipped age to [15, 70] ({out_of_range} values)")
        
        logger.info(f"✓ Data validation complete")
        return self
    
    def standardize_data(self):
        """Standardize data formats"""
        logger.info("Standardizing data...")
        
        # Standardize categorical values
        if 'gender' in self.df.columns:
            self.df['gender'] = self.df['gender'].str.upper().str.strip()
            logger.info(f"  - Standardized gender values")
        
        if 'dropout_risk' in self.df.columns:
            valid_risks = ['Low', 'Medium', 'High']
            self.df['dropout_risk'] = self.df['dropout_risk'].str.capitalize().str.strip()
            invalid = self.df[~self.df['dropout_risk'].isin(valid_risks)]
            if len(invalid) > 0:
                self.df.loc[~self.df['dropout_risk'].isin(valid_risks), 'dropout_risk'] = 'Medium'
                logger.info(f"  - Standardized dropout_risk values ({len(invalid)} corrected)")
        
        if 'scholarship_status' in self.df.columns:
            self.df['scholarship_status'] = self.df['scholarship_status'].str.capitalize().str.strip()
            logger.info(f"  - Standardized scholarship_status values")
        
        if 'accommodation_type' in self.df.columns:
            self.df['accommodation_type'] = self.df['accommodation_type'].str.capitalize().str.strip()
            logger.info(f"  - Standardized accommodation_type values")
        
        logger.info(f"✓ Data standardization complete")
        return self
    
    def remove_outliers(self):
        """Remove statistical outliers using IQR method"""
        logger.info("Removing outliers...")
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        rows_before = len(self.df)
        
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Keep rows where value is within 3 IQR from quartiles
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            outlier_count = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
            if outlier_count > 0:
                self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
                logger.info(f"  - Removed {outlier_count} outliers from {col}")
        
        rows_removed = rows_before - len(self.df)
        if rows_removed > 0:
            logger.info(f"✓ Removed {rows_removed} rows with outliers")
        else:
            logger.info(f"✓ No significant outliers detected")
        return self
    
    def save_cleaned_data(self):
        """Save cleaned data to CSV"""
        try:
            # Create directory if it doesn't exist
            Path(self.cleaned_data_path).parent.mkdir(parents=True, exist_ok=True)
            
            self.df.to_csv(self.cleaned_data_path, index=False)
            self.cleaned_rows = len(self.df)
            logger.info(f"✓ Saved {self.cleaned_rows} cleaned rows to {self.cleaned_data_path}")
            return self
        except Exception as e:
            logger.error(f"✗ Error saving data: {e}")
            raise
    
    def generate_report(self):
        """Generate cleaning report"""
        report = {
            'original_rows': self.original_rows,
            'cleaned_rows': self.cleaned_rows,
            'rows_removed': self.original_rows - self.cleaned_rows,
            'removal_percentage': round(((self.original_rows - self.cleaned_rows) / self.original_rows * 100) if self.original_rows > 0 else 0, 2),
            'columns': len(self.df.columns),
            'column_names': list(self.df.columns),
            'dtypes': self.df.dtypes.to_dict(),
            'data_shape': self.df.shape,
            'missing_values': self.df.isnull().sum().to_dict(),
        }
        
        logger.info("\n" + "="*60)
        logger.info("DATA CLEANING REPORT")
        logger.info("="*60)
        logger.info(f"Original Records: {report['original_rows']}")
        logger.info(f"Cleaned Records:  {report['cleaned_rows']}")
        logger.info(f"Records Removed:  {report['rows_removed']} ({report['removal_percentage']}%)")
        logger.info(f"Total Columns:    {report['columns']}")
        logger.info("="*60 + "\n")
        
        return report
    
    def clean(self):
        """Execute full cleaning pipeline"""
        logger.info("="*60)
        logger.info("STARTING DATA CLEANING PIPELINE")
        logger.info("="*60 + "\n")
        
        (self.load_raw_data()
         .remove_duplicates()
         .handle_missing_values()
         .validate_data()
         .standardize_data()
         .remove_outliers()
         .save_cleaned_data())
        
        report = self.generate_report()
        logger.info("✓ DATA CLEANING COMPLETE!")
        logger.info("="*60 + "\n")
        
        return report


def clean_data():
    """Main function to clean data"""
    raw_path = 'data/raw/dummy_data.csv'
    cleaned_path = 'data/cleaned/student_data_cleaned.csv'
    
    cleaner = DataCleaner(raw_path, cleaned_path)
    report = cleaner.clean()
    
    return report


if __name__ == '__main__':
    clean_data()
