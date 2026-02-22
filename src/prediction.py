"""
Machine Learning Prediction Module for Student Dropout Risk.
Provides model training, prediction, and risk assessment.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
import os
import pickle
from datetime import datetime
import logging
from src.logger import setup_logging

logger = setup_logging(__name__)

# Model storage path
MODELS_DIR = "models"
DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, "dropout_model.pkl")


class DropoutPredictor:
    """
    Machine Learning predictor for student dropout risk.
    Uses Random Forest classifier for predictions.
    """
    
    # Features used for prediction
    FEATURE_COLUMNS = [
        'age', 'admission_grade', 'semester_1_gpa', 'semester_2_gpa', 
        'semester_3_gpa', 'current_gpa', 'failed_courses', 'average_grade',
        'attendance_rate', 'absences_count', 'late_submissions',
        'library_visits', 'online_portal_logins', 'participation_score',
        'extracurricular_activities', 'distance_from_campus'
    ]
    
    # Categorical columns to encode
    CATEGORICAL_COLUMNS = [
        'gender', 'previous_education', 'scholarship_status', 
        'financial_aid', 'accommodation_type'
    ]
    
    # Risk thresholds
    RISK_THRESHOLDS = {
        'low': 0.3,
        'medium': 0.6,
        'high': 1.0
    }
    
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        """Initialize the predictor."""
        self.model_path = model_path
        self.model = None
        self.feature_columns = []
        self.is_trained = False
        self._ensure_models_dir()
        self._load_model()
    
    def _ensure_models_dir(self) -> None:
        """Ensure models directory exists."""
        if not os.path.exists(MODELS_DIR):
            os.makedirs(MODELS_DIR)
            logger.info(f"Created models directory: {MODELS_DIR}")
    
    def _load_model(self) -> bool:
        """Load existing model if available."""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    saved_data = pickle.load(f)
                    self.model = saved_data['model']
                    self.feature_columns = saved_data['feature_columns']
                    self.is_trained = True
                    logger.info("Loaded existing model")
                    return True
            except Exception as e:
                logger.error(f"Error loading model: {e}")
        return False
    
    def _save_model(self) -> bool:
        """Save the trained model."""
        try:
            saved_data = {
                'model': self.model,
                'feature_columns': self.feature_columns,
                'saved_at': datetime.now().isoformat()
            }
            with open(self.model_path, 'wb') as f:
                pickle.dump(saved_data, f)
            logger.info(f"Model saved to {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for training/prediction."""
        df = df.copy()
        
        # Select numeric features
        features = pd.DataFrame()
        
        for col in self.FEATURE_COLUMNS:
            if col in df.columns:
                features[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Encode categorical columns
        for col in self.CATEGORICAL_COLUMNS:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                features = pd.concat([features, dummies], axis=1)
        
        # Fill any remaining NaN values
        features = features.fillna(0)
        
        return features
    
    def train(self, df: pd.DataFrame, target_column: str = 'actual_dropout') -> Dict[str, Any]:
        """
        Train the dropout prediction model.
        
        Args:
            df: Training dataframe with features and target.
            target_column: Name of the target column.
            
        Returns:
            Training metrics dictionary.
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        logger.info("Starting model training...")
        
        # Prepare features
        X = self._prepare_features(df)
        self.feature_columns = X.columns.tolist()
        
        # Get target - create if not exists
        if target_column in df.columns:
            y = df[target_column].astype(int)
        else:
            # Create synthetic target based on risk factors
            y = self._create_synthetic_target(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': round(accuracy_score(y_test, y_pred), 4),
            'precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
            'recall': round(recall_score(y_test, y_pred, zero_division=0), 4),
            'f1_score': round(f1_score(y_test, y_pred, zero_division=0), 4),
            'roc_auc': round(roc_auc_score(y_test, y_prob), 4),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'features_used': len(self.feature_columns),
            'trained_at': datetime.now().isoformat()
        }
        
        # Get feature importance
        importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        metrics['top_features'] = importance.head(10).to_dict('records')
        
        self.is_trained = True
        self._save_model()
        
        logger.info(f"Model trained. Accuracy: {metrics['accuracy']}")
        return metrics
    
    def _create_synthetic_target(self, df: pd.DataFrame) -> pd.Series:
        """Create synthetic dropout target based on risk factors."""
        risk_score = pd.Series(0.0, index=df.index)
        
        # GPA factors (lower GPA = higher risk)
        if 'current_gpa' in df.columns:
            risk_score += (20 - df['current_gpa'].fillna(10)) * 3
        
        # Failed courses
        if 'failed_courses' in df.columns:
            risk_score += df['failed_courses'].fillna(0) * 15
        
        # Attendance (lower = higher risk)
        if 'attendance_rate' in df.columns:
            risk_score += (100 - df['attendance_rate'].fillna(70)) * 0.5
        
        # Absences
        if 'absences_count' in df.columns:
            risk_score += df['absences_count'].fillna(0) * 2
        
        # Late submissions
        if 'late_submissions' in df.columns:
            risk_score += df['late_submissions'].fillna(0) * 3
        
        # Engagement (lower = higher risk)
        if 'participation_score' in df.columns:
            risk_score += (10 - df['participation_score'].fillna(5)) * 2
        
        # Normalize and threshold
        risk_score = (risk_score - risk_score.min()) / (risk_score.max() - risk_score.min() + 0.001)
        
        return (risk_score > 0.5).astype(int)
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict dropout risk for students.
        
        Args:
            df: DataFrame with student data.
            
        Returns:
            DataFrame with predictions and risk scores.
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        # Prepare features
        X = self._prepare_features(df)
        
        # Ensure columns match training
        missing_cols = set(self.feature_columns) - set(X.columns)
        for col in missing_cols:
            X[col] = 0
        X = X[self.feature_columns]
        
        # Get predictions
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]
        
        # Create results dataframe
        results = df.copy()
        results['dropout_prediction'] = predictions
        results['dropout_probability'] = np.round(probabilities, 4)
        results['risk_score'] = np.round(probabilities * 100, 2)
        results['risk_category'] = results['dropout_probability'].apply(self._categorize_risk)
        
        logger.info(f"Predictions made for {len(df)} students")
        return results
    
    def _categorize_risk(self, probability: float) -> str:
        """Categorize risk based on probability."""
        if probability < self.RISK_THRESHOLDS['low']:
            return 'Low'
        elif probability < self.RISK_THRESHOLDS['medium']:
            return 'Medium'
        else:
            return 'High'
    
    def predict_single(self, student_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict dropout risk for a single student.
        
        Args:
            student_data: Dictionary with student information.
            
        Returns:
            Prediction results dictionary.
        """
        df = pd.DataFrame([student_data])
        results = self.predict(df)
        
        return {
            'dropout_prediction': int(results['dropout_prediction'].iloc[0]),
            'dropout_probability': float(results['dropout_probability'].iloc[0]),
            'risk_score': float(results['risk_score'].iloc[0]),
            'risk_category': results['risk_category'].iloc[0]
        }
    
    def get_risk_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary statistics of risk predictions.
        
        Args:
            df: DataFrame with predictions.
            
        Returns:
            Summary statistics dictionary.
        """
        if 'risk_category' not in df.columns:
            df = self.predict(df)
        
        risk_counts = df['risk_category'].value_counts().to_dict()
        
        summary = {
            'total_students': len(df),
            'high_risk': risk_counts.get('High', 0),
            'medium_risk': risk_counts.get('Medium', 0),
            'low_risk': risk_counts.get('Low', 0),
            'high_risk_pct': round(risk_counts.get('High', 0) / len(df) * 100, 2),
            'medium_risk_pct': round(risk_counts.get('Medium', 0) / len(df) * 100, 2),
            'low_risk_pct': round(risk_counts.get('Low', 0) / len(df) * 100, 2),
            'avg_risk_score': round(df['risk_score'].mean(), 2),
            'median_risk_score': round(df['risk_score'].median(), 2)
        }
        
        return summary
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Get feature importance from trained model."""
        if not self.is_trained or self.model is None:
            return None
        
        importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance


# Singleton instance
predictor = DropoutPredictor()
